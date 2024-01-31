import pandas as pd
import time
import numpy as np
import geopandas as gpd
import re
import os
from multiprocessing import Pool
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
    
    _default_headtypes = {
                        'LIBID': 'int', 
                        'RA': 'float', 
                        'DEC': 'float', 
                        'MWEBV': 'float', 
                        'NOBS': 'int', 
                        'PIXSIZE': 'float',
                        'REDSHIFT': 'float', 
                        'PEAKMJD': 'float',
                        }

    def __init__(self, name, path='./', keys=None, key_types=None, add_head={}, nworker=1):
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
        self.simlib_dic, self.data = self.read_simlib(nworker=nworker)
    
    def read_libentry(self, lines):
        libdic = {}
        
        lines_format = lines[lines != '\n']
        end_idx = np.arange(len(lines_format))[ut.vcontains(lines_format, 'END_LIBID')][0]
        lines_format = lines_format[:end_idx]
        table_lines = ut.vstartswith(lines_format, 'S:')
        
        
        key_val = []
        
        for l in lines_format[~table_lines]:
            l = re.sub("[\(\[].*?[\)\]]", "", l)
            splitlist = l.replace('\n', '').split('#')[0].split(':')
            key_val.extend(np.concatenate([[e for e in s.split(' ') if e!=''] for s in splitlist ]))

        for i in range(0, len(key_val), 2):
            libdic[key_val[i]] = ut.typer(key_val[i + 1], self._head_typdic[key_val[i]])

        data = {k: [] for k in self.keys}

        for s in lines_format[table_lines]:
            vals = [e for e in s.replace('S:', '').replace('\n', '').split(' ') if e!='']
            for k, v, ktyp in zip(self.keys, vals, self.key_types):
                data[k].append(ut.typer(v, ktyp))
    
        df = pd.DataFrame(data)
        df['LIBID'] = libdic['LIBID']
        df.attrs = {'libdic': libdic, 'lines': lines}
        return df
        
    def read_simlib(self, nworker=1):
        f = open(self.path + self.name, "r")
        lines = np.array(f.readlines())
        lines = lines[~ut.vstartswith(lines, '#')]

        lib_idx = np.arange(len(lines))[ut.vstartswith(lines, 'LIBID')]
        lib_idx = np.append(lib_idx, len(lines))

        libdic_list = []
        dflist = []
        simlib_dic = {'header': lines[:lib_idx[0]]}
        
        batchs = [lines[i1:i2] for i1, i2 in zip(lib_idx[:-1], lib_idx[1:])]
        
        with Pool(nworker) as p:
            dflist = p.map(self.read_libentry, batchs)
        
        df = pd.concat(dflist)
        df.attrs = {l.attrs['libdic']['LIBID']: l.attrs['libdic'] for l in dflist}
        lines_dic = {l.attrs['libdic']['LIBID']: l.attrs['lines'] for l in dflist}
        simlib_dic ={**simlib_dic, **lines_dic}
        df.set_index('LIBID', inplace=True)
        return simlib_dic, df
    
    def get(self, key):
        return np.array([self.data.attrs[k][key] for k in self.data.attrs])
    
    def compute_geo(self, field_size, rad=False, coord_keys={'ra': 'RA', 'dec': 'DEC'}):
        if not rad:
            field_size = np.radians(field_size)
        ra = self.get(coord_keys['ra'])
        dec = self.get(coord_keys['dec'])
        N = len(ra)
        sub_fields_corners = {0: np.array([[-field_size[0] / 2,  field_size[1] / 2],
                                           [ field_size[0] / 2,  field_size[1] / 2],
                                           [ field_size[0] / 2, -field_size[1] / 2],
                                           [-field_size[0] / 2, -field_size[1] / 2]])}
        
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
    
    def plot_fields(self, ax, color='k', **kwargs):
        if not isinstance(color, str):
            color = iter(color)
        
        for f in self.fields:
            if not isinstance(color, str):
                c=next(color)
            else:
                c=color
            
            if f.geom_type =='Polygon':
                geoms = [f]
            else:
                geoms = f.geoms
                
            for g in geoms:
                ra, dec = g.boundary.coords.xy
                ax.plot(np.array(ra) - np.pi, dec, color=c, **kwargs)
    
    def group_host(self, host, hostlib_name=None, host_pqsave=False, 
                   return_hostdf=False, key_mapper={}, host_deg=False):
        """Create a SIMLIB and HOSTLIB from host dataframe.

        Parameters
        ----------
        host : pandas.Dataframe
            A datframe that contains informations about host
        hostlib_name : str, optional
            Name of the HOSTLIB file, by default SIMLIB name + _HOST

        Notes
        -----
        Can take few minutes to run.
        """        
        t0 = time.time()
        if hostlib_name is None:
            hostlib_name = os.path.splitext(self.name)[0] + '_HOST' 
        
        # Make a copy to not modify input table
        host = host.copy()
        
        if host_deg:
            host['ra'] = np.radians(host['ra'])
            host['dec'] = np.radians(host['dec'])

        host.ra += 2 * np.pi * (host.ra < 0)

        hostPos = gpd.GeoDataFrame(index=host.index, 
                                   geometry=gpd.points_from_xy(host.ra, host.dec))
        
        hf_join = hostPos.sjoin(gpd.GeoDataFrame(geometry=self.fields), 
                                how="inner", 
                                predicate="intersects")
        
        host['ra'] = np.degrees(host['ra'])
        host['dec'] = np.degrees(host['dec'])

        
        # Can take few minutes for few 10 millions of hosts
        grp_id = hf_join.index_right.sort_values().astype('str').groupby(level=0).agg('-'.join)

        host_infield = host.loc[grp_id.index]
        host_infield['groupid'] = host_infield.index.map(grp_id.to_dict())

        # Take the max libid to avoid grp with same id than existing libid
        idxmax = self.data.index.max()

        # Create strid_grpid map between '-' spearated string to integers group id
        # Also create a map between libid and possible groupid to be written in SIMLIB
        strid_grpid = {}
        libgrp_map = {i:[] for i in self.data.index.unique()}
        compt = idxmax * 10
        for gid in grp_id.unique():
            if '-' in gid:
                strid_grpid[gid] = compt
                compt += 1
            else:
                strid_grpid[gid] = int(gid)
            
            idlist = gid.split('-')
            for i in idlist:
                libgrp_map[int(i)].append(str(strid_grpid[gid]))


        # Create and write the new simlib
        new_simlib = ''.join(self.simlib_dic['header']) + '\n\n'
        for i in self.data.index.unique():
            new_simlib += '#--------------------------------------------\n'
            new_simlib += self.simlib_dic[i][0] + 'HOSTLIB_GROUPID: {}'.format(','.join(libgrp_map[i]) + '\n')
            new_simlib += ''.join(self.simlib_dic[i][1:])
        name, ext = os.path.splitext(self.name)
        newf = open(self.path +  name + '_GRP' + ext, 'w')
        newf.write(new_simlib)
        newf.close()

        # Change the grpid to number after write the simlib
        host_infield['groupid'] = host_infield.groupid.map(strid_grpid)
        ut.create_hostlib(host_infield, self.path + hostlib_name + '.HOSTLIB', key_mapper=key_mapper)

        if host_pqsave:
            host_infield.to_parquet(self.path + hostlib_name + '.parquet')
        dtime = time.time() - t0

        print(("Write:\n"
               f"- SIMLIB file {self.path + 'GRPID_' +  self.name}\n"
               f"- HOSTLIB file {self.path + hostlib_name + '.HOSTLIB'}"))

        print(f'Finished in {dtime // 60:.0f}min {dtime % 60:.0f}s')
        
        if return_hostdf:
            return host_infield

def read_wgtmap(file):
    f = open(file, "r")
    lines = np.array(f.readlines())
    lines = lines[~ut.vstartswith(lines, '#')]

    wgt_keys = lines[ut.vstartswith(lines, 'VARNAMES_WGTMAP')][0].replace('\n', '').strip().split(' ')[1:]
    wgt_idx = np.arange(len(lines))[ut.vstartswith(lines, 'WGT')]

    wgt_map = {k: [] for k in wgt_keys}
    for l in lines[wgt_idx]:
        l = l.replace('\n', '').split(' ')[1:]
        for k, e in zip(wgt_keys, l):
            wgt_map[k].append(float(e))

    return pd.DataFrame(wgt_map)


class SNANA_PDF:
    
    def __init__(pdf_dic):
        self.dic = pdf_dic
        self.keys = list(pdf_dic.keys())
    
    @classmethod
    def initfromFile(cls, file):
        return cls(cls.read_pdffile(file))
    
    @staticmethod
    def read_pdffile(file):
        f = open(file, 'r')
        lines = np.array(f.readlines())
        lines = lines[~ut.vstartswith(lines, '#')]
        varline = lines[ia].replace('VARNAMES:', '').replace('\n', '').strip().split(' ')
        dic = {k: [] for k in varline}
        for l in lines[ia+1:ib]:
            if l != '\n':
                vals = l.replace('PDF:', '').replace('\n', '').strip().split(' ')
                for k, v in zip(dic, vals):
                    dic[k].append(float(v))
        pdf_dic[varline[0]] = pd.DataFrame(dic) 
        return pdf_dic
    
    
