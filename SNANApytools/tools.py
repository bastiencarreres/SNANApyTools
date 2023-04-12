import pandas as pd
import numpy as np
import geopandas as gpd
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
        return df
    

class SNANA_simlib:
    _head_typdic = {'LIBID': lambda x: int(x), 'RA': lambda x: float(x), 'DEC': lambda x: float(x), 
                    'MWEBV': lambda x: float(x), 'NOBS': lambda x: int(x), 'PIXSIZE': lambda x: float(x),
                    'REDSHIFT': lambda x: float(x), 'PEAKMJD': lambda x: float(x)}
    
    _columns_typdic  = {'MJD': lambda x: float(x), 'IDEXPT': lambda x: float(x), 'BAND': lambda x: str(x), 
                        'GAIN': lambda x: float(x), 'RDNOISE': lambda x: float(x), 'SKYSIG': lambda x: float(x),
                        'PSF1': lambda x: float(x), 'PSF2': lambda x: float(x), 'PSFRAT': lambda x: float(x), 
                         'ZP': lambda x: float(x), 'ZPERR': lambda x: float(x), 'MAG': lambda x: float(x)}
    
    def __init__(self, name, path='./'):
        self.name = name
        self.path = path
        self.data = self.read_simlib()
        
    def read_simlib(self):
        f = open(self.path + self.name, "r")
        lines = np.array(f.readlines())
        lib_idx = np.where(lines == '#--------------------------------------------\n')[0]
        lib_idx = np.append(lib_idx, len(lines))
        libdic_list = []
        dflist = []

        for i1, i2 in zip(lib_idx[:-1], lib_idx[1:]):
            sublines = lines[i1+1: i2]

            libdic = {}
            key_val = []
            for i in range(len(sublines)):
                l = sublines[i]
                if l=='\n':
                    idx_end_header =  i
                    break
                splitlist = l.replace('\n', '').split('#')[0].split(':')
                key_val.extend(np.concatenate([[e for e in s.split(' ') if e!=''] for s in splitlist ]))

            for i in range(0, len(key_val), 2):
                libdic[key_val[i]] = self._head_typdic[key_val[i]](key_val[i + 1])


            columns = [e for e in sublines[idx_end_header + 1].replace('#', '').replace('\n', '').split(' ') if e!='']
            data = {k: [] for k in columns}


            for s in sublines[idx_end_header + 2:idx_end_header + 2 + libdic['NOBS']]:
                vals = [e for e in s.replace('S:', '').replace('\n', '').split(' ') if e!='']
                for k, v in zip(columns, vals):
                    data[k].append(self._columns_typdic[k](v))
            df = pd.DataFrame(data)
            df['LIBID'] = libdic['LIBID']

            dflist.append(df)
            libdic_list.append(libdic)
            
        df = pd.concat(dflist)
        df.attrs = {l['LIBID']:l for l in libdic_list}
        df.set_index('LIBID', inplace=True)
        return df
    
    def get(self, key):
        return np.array([self.data.attrs[k][key] for k in self.data.attrs])
    
    def compute_geo(self, field_size_rad):
        N = len(self.get('RA'))
        sub_fields_corners = {0: np.array([[-field_size_rad[0] / 2,  field_size_rad[1] / 2],
                                           [ field_size_rad[0] / 2,  field_size_rad[1] / 2],
                                           [ field_size_rad[0] / 2, -field_size_rad[1] / 2],
                                           [-field_size_rad[0] / 2, -field_size_rad[1] / 2]])}
        
        field_corners = np.broadcast_to(sub_fields_corners[0], (N, 4, 2))

        corner = {}
        fieldRA = np.radians(self.get('RA'))
        fieldDec = np.radians(self.get('DEC'))

        for i in range(4):
            corner[i] = ut.new_coord_on_fields(field_corners[:, i].T, 
                                               [fieldRA, fieldDec])

        corner = ut._format_corner(corner, fieldRA)

        geometry = [ut._compute_polygon([[corner[i][0][j], corner[i][1][j]] for i in range(4)]) 
                                         for j in range(N)]
        
        self.fields = gpd.GeoSeries({ID: geo for ID, geo in zip(self.get('LIBID'), geometry)})
        self.footprint = self.fields.unary_union
