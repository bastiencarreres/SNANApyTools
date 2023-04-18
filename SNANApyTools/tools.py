import pandas as pd
import numpy as np
import geopandas as gpd
import re
from numba import njit, guvectorize
from shapely import geometry as shp_geo
from shapely import ops as shp_ops
from astropy.table import Table
from . import utils as ut

class SNANAsim:
    def __init__(self, sim_name, sim_path='./'):
        self.sim_name = sim_name
        self.sim_path = sim_path
        self.lcs = self.init_lcs_table()
        self.params = self.init_par_table()
        
    def init_lcs_table(self):
        df = Table.read(self.sim_path + self.sim_name + '_PHOT.FITS').to_pandas()
        sepidx = [0, *df.index[df.MJD == -777]]
        
        df['ID'] = 0
        df['epochs'] = 0
        for i, (i1, i2) in enumerate(zip(sepidx[:-1], sepidx[1:])):
            if i1 > 0:
                i1 += 1
            df.loc[i1:(i2 - 1), 'ID'] = i
            df.loc[i1:(i2 - 1), 'epochs'] = np.arange(0, i2 - i1)
        return df.drop(index=sepidx[1:]).set_index(['ID', 'epochs'])
    
    def init_par_table(self):
        df = Table.read(self.sim_path + self.sim_name + '_HEAD.FITS').to_pandas()
        df.index.name = 'ID'
        df['SNID'] = df['SNID'].map(lambda x: int(x))
        pointsRA = df.RA + 360 * (df.RA < 0) 
        geo = gpd.points_from_xy(np.radians(pointsRA), np.radians(df.DEC))
        return gpd.GeoDataFrame(data=df, geometry=geo)
        

class SNANA_simlib:
    _default_keys = ['MJD','IDEXPT','FLT','GAIN','NOISE','SKYSIG','PSF1','PSF2','RATIO','ZPTAVG','ZPTSIG','MAG']
    _default_typ = ['float','int','str','float','float','float','float','float','float','float','float','float']
    
    _default_headtypes = {'LIBID': 'int', 
                         'RA': 'float', 
                         'DEC': 'float', 
                         'MWEBV': 'float', 
                         'NOBS': 'int', 
                         'PIXSIZE': 'float',
                         'REDSHIFT': 'float', 
                         'PEAKMJD': 'float'}

    def __init__(self, name, path='./', keys=None, key_types=None, add_head={}):
        self._head_typdic = {**self._default_headtypes,
                             **add_head}
        if keys is None:
            self.keys = self._default_keys
        else:
            self.keys = keys.copy()
        
        if key_types is None:
            self.key_types = self._default_typ
        else:
            self.key_types = key_types.copy()
        
        self.name = name
        self.path = path
        self.simlib_dic, self.data = self.read_simlib()
        
    def read_simlib(self):
        f = open('./ERROR_EXAMPLE/' + 'LOWZ_JRK07.SIMLIB', "r")
        lines = np.array(f.readlines())
        lines = lines[~ut.vstartswith(lines, '#')]

        lib_idx = np.arange(len(lines))[ut.vstartswith(lines, 'LIBID')]
        lib_idx = np.append(lib_idx, len(lines))

        libdic_list = []
        dflist = []
        simlib_dic = {'header': lines[:lib_idx[0]]}

        libdic_list = []
        dflist = []
        simlib_dic = {'header': lines[:lib_idx[0] - 1]}

        for i1, i2 in zip(lib_idx[:-1], lib_idx[1:]):
            libdic = {}

            sublines = lines[i1: i2]
            sublines_format = sublines[sublines != '\n']
            end_idx = np.arange(len(sublines_format))[ut.vcontains(sublines_format, 'END_LIBID')][0]
            sublines_format = sublines_format[:end_idx]
            table_lines = ut.vstartswith(sublines_format, 'S:')
            
            key_val = []
            for l in sublines_format[~table_lines]:
                l = re.sub("[\(\[].*?[\)\]]", "", l)
                splitlist = l.replace('\n', '').split('#')[0].split(':')
                key_val.extend(np.concatenate([[e for e in s.split(' ') if e!=''] for s in splitlist ]))

            for i in range(0, len(key_val), 2):
                libdic[key_val[i]] = ut.typer(key_val[i + 1], self._head_typdic[key_val[i]])

            simlib_dic[libdic['LIBID']] = sublines

            data = {k: [] for k in self.keys}

            for s in sublines_format[table_lines]:
                vals = [e for e in s.replace('S:', '').replace('\n', '').split(' ') if e!='']
                for k, v, ktyp in zip(self.keys, vals, self.key_types):
                    data[k].append(ut.typer(v, ktyp))
            df = pd.DataFrame(data)
            df['LIBID'] = libdic['LIBID']

            dflist.append(df)
            libdic_list.append(libdic)

            
        df = pd.concat(dflist)
        df.attrs = {l['LIBID']:l for l in libdic_list}
        df.set_index('LIBID', inplace=True)
        return simlib_dic, df
    
    def get(self, key):
        return np.array([self.data.attrs[k][key] for k in self.data.attrs])
    
    def compute_geo(self, field_size_rad, coord_keys={'ra': 'RA', 'dec': 'DEC'}):
        ra = self.get(coord_keys['ra'])
        dec = self.get(coord_keys['dec'])
        N = len(ra)
        sub_fields_corners = {0: np.array([[-field_size_rad[0] / 2,  field_size_rad[1] / 2],
                                           [ field_size_rad[0] / 2,  field_size_rad[1] / 2],
                                           [ field_size_rad[0] / 2, -field_size_rad[1] / 2],
                                           [-field_size_rad[0] / 2, -field_size_rad[1] / 2]])}
        
        field_corners = np.broadcast_to(sub_fields_corners[0], (N, 4, 2))

        corner = {}
        fieldRA = np.radians(ra)
        fieldDec = np.radians(dec)

        for i in range(4):
            corner[i] = ut.new_coord_on_fields(field_corners[:, i].T, 
                                               [fieldRA, fieldDec])

        corner = ut._format_corner(corner, fieldRA)

        geometry = [ut._compute_polygon([[corner[i][0][j], corner[i][1][j]] for i in range(4)]) 
                                         for j in range(N)]
        
        self.fields = gpd.GeoSeries({ID: geo for ID, geo in zip(self.get('LIBID'), geometry)})
        self.footprint = self.fields.unary_union
