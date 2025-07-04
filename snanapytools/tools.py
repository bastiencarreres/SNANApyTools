import pandas as pd
import time
import yaml
import numpy as np
import geopandas as gpd
import re
import copy
import os
import shutil
import gzip
from multiprocessing import Pool
from astropy.table import Table
from pathlib import Path
from . import utils as ut

def dataframecol_decoder(df):
    for col, dtype in df.dtypes.items():
        if dtype == object:  # Only process object columns.
            # decode, or return original value if decode return Nan
            df[col] = df[col].str.decode('utf-8').fillna(df[col]) 

def read_SNANAfits(file, **kwargs):
    df = Table.read(file, **kwargs).to_pandas()
    dataframecol_decoder(df)
    return df

def read_biascor_input_doc(file):
    yml_text = ''
    with open(file) as f:
        yml_write = True
        for l in f.readlines():
            if '#END_YAML' in l:
                yml_write = False
            if yml_write:
                yml_text += l
            if l.startswith('simfile_biascor'):
                biascor_files = l.split('=')[-1]
                biascor_files = [Path(l.strip()) for l in biascor_files.split(',')]
                for i, b in enumerate(biascor_files):
                    while b.parent.name != '5_MERGE':
                        b = b.parent
                    biascor_files[i] = b
                break
                
    config = yaml.safe_load(yml_text)['CONFIG']
    
    return config, biascor_files

def trace_lcfit_from_merge(merge_dir):
    config_file = Path(merge_dir) / 'config.yml'
    with open(config_file) as f:
       fitlc_parent = yaml.safe_load(f)['OUTPUT']['lcfit_name']
        
    return fitlc_parent
    
def print_directory_tree(root_dir, indent="", nofile=False, filters=None):
    for item in root_dir.iterdir():
        if not nofile and item.is_file():
            continue
        print(indent + "├── " + item.name)
        if item.is_dir():
            print_directory_tree(item, indent=indent + "│   ")

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

def write_wgtmap(path, header_dic, var_dic):
    file = open(path, 'w')
    DocStr = "DOCUMENTATION:\n"
    for k in header_dic:
        DocStr += f"    {k.upper()}: {header_dic[k]}\n"
    DocStr += "DOCUMENTATION_END: \n"
    file.write(DocStr)
    file.write("VARNAMES_WGTMAP: " + " ".join([k for k in var_dic]) + "\n")
    
    lines = '\n'.join(ut.GalLine(*[var_dic[k] for k in var_dic], first='WGT: '))
    file.write(lines)
    file.close()
    
def read_wgtmap(file):
    file = Path(file)
    if file.suffix == '.gz':
        f = gzip.open(file,'rb')
        lines = np.array([l.decode() for l in f.readlines()])
    else:
        f = open(file, "r")
        lines = np.array(f.readlines())

    lines = lines[~ut.vstartswith(lines, '#')]
    varnames_mask = ut.vstartswith(lines, 'VARNAMES_WGTMAP')
    if not varnames_mask.any():
        varnames_mask = ut.vstartswith(lines, 'VARNAMES')
    
    varname_lines = np.where(varnames_mask)[0]    
    vardic = {}
    for i in range(len(varname_lines)):
        varl = lines[varname_lines[i]]
        wgt_keys = varl.replace('\n', '').split(' ')[1:]
        wgt_keys = list(filter(('').__ne__, wgt_keys))

        idxmin = varname_lines[i]
        if i + 1 < len(varname_lines):
            idxmax = varname_lines[i+1]
        else:
            idxmax = len(lines) + 1
            
        wgt_idx = np.arange(idxmin, idxmin + len(lines[idxmin: idxmax]))[ut.vstartswith(lines[idxmin: idxmax], 'WGT')]
        if not wgt_idx.any():
            wgt_idx = np.arange(idxmin, idxmin + len(lines[idxmin: idxmax]))[ut.vstartswith(lines[idxmin: idxmax], 'PDF')]
        wgt_map = {k: [] for k in wgt_keys}
        for l in lines[wgt_idx]:
            l = l.partition('#')[0]
            l = l.replace('\n', '').split(' ')[1:]
            l = list(filter(('').__ne__, l))
            for k, e in zip(wgt_keys, l):
                wgt_map[k].append(float(e))
        vardic[wgt_keys[0]] = pd.DataFrame(wgt_map)
    return vardic

def read_hostlib(file):
    f = open(file, 'r')
    i = 0
    for l in f:
        if l.startswith('VARNAMES'):
            break
        i+=1
    return pd.read_table(file, skiprows=i, delimiter=' ')

class SNANA_PDF:
    def __init__(self, pdf_dic):
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
        varlines = np.where(ut.vstartswith(lines, 'VARNAMES'))[0]
        pdf_dic = {}
        for i, ia in enumerate(varlines):
            if i < len(varlines) - 1:
                ib = varlines[i + 1]
            else:
                ib = len(lines)
            varnames = lines[ia].replace('VARNAMES:', '').replace('\n', '').strip().split(' ')
            dic = {k: [] for k in varnames}
            vallines = lines[ia+1:ib]
            vallines = vallines[ut.vstartswith(vallines, 'PDF:')]
            for l in vallines:
                if l != '\n':
                    vals = l.replace('PDF:', '').replace('\n', '').strip().split(' ')
                    for k, v in zip(dic, vals):
                        dic[k].append(float(v))
            pdf_dic[varnames[0]] = pd.DataFrame(dic) 
        return pdf_dic
    
    
def apply_selec(selec_prob, head_df, phot_df):
    # Main function
    mask = np.zeros(len(head_df), dtype='bool')
    photmask = np.zeros(len(phot_df), dtype='bool')
    for i in range(len(head_df)):
        sn_prop = head_df.iloc[i]
        sn_phot = phot_df.iloc[sn_prop['PTROBS_MIN']-1:sn_prop['PTROBS_MAX']]
        snr_mask = sn_phot['FLUXCAL']/sn_phot['FLUXCALERR'] > 5
        band_mask = np.isin(sn_phot['BAND'].map(lambda x: x.strip()), selec_prob['bands'])
        idx_minmag = sn_phot['SIM_MAGOBS'][snr_mask & band_mask].argmin()        
        min_mag = sn_phot['SIM_MAGOBS'].iloc[idx_minmag]
        p = selec_prob['prob'](min_mag)
        if np.random.random() < p:
            mask[i] = True
            photmask[sn_prop['PTROBS_MIN']-1:sn_prop['PTROBS_MAX']] = True
    return mask, photmask

def apply_selec_pippin_dir(pip_dir, out_dir, selec_prob, suffix='', exclude=[], only_include=[], 
                           compress=True, overwrite=False, random_seed=None):

    __PIPPIN_OUTPUT__ = os.getenv('PIPPIN_OUTPUT')
    PIP_DIR = Path(__PIPPIN_OUTPUT__ + '/' + pip_dir)

    loglines =  "SELECTION LOGS\n"
    loglines += "==============\n\n"
    loglines += f"PIPPIN_DIR: {PIP_DIR}\n"

    print(f"$PIPPIN_OUTPUT = {__PIPPIN_OUTPUT__}\n\n")
    print(f"READING {PIP_DIR}\n")
    
    SIM_DIR = PIP_DIR / '1_SIM'
    FIT_DIR = PIP_DIR / '2_LCFIT'
    
    OUT_DIR = Path(out_dir + f'/{pip_dir}_{suffix}')
    SIM_OUT_DIR = OUT_DIR / '1_SIM'
    FIT_OUT_DIR = OUT_DIR / '2_LCFIT'

    print(f'WRITE IN {OUT_DIR.stem}\n')
    
    SIM_DIRS = sorted(SIM_DIR.glob('*'))
    FIT_DIRS = sorted(FIT_DIR.glob('*'))
    
    SeedSeq = np.random.SeedSequence(random_seed)
    Seeds = np.random.SeedSequence(random_seed).spawn(len(SIM_DIRS))

    loglines += f"RANDOM SEED: {SeedSeq.entropy}\n" 
    loglines += " ".join([s.name for s in SIM_DIRS]) + "\n"
    
    for seed, sd, fd in zip(Seeds, SIM_DIRS, FIT_DIRS):
        if sd.name in exclude:
            print(f"{sd.name} EXCLUDED\n")
            loglines += f"{sd.name}: EXCLUDED\n"
            continue
        elif (len(only_include) > 0) and (sd.name not in only_include):
            print(f"{sd.name} EXCLUDED\n")
            loglines += f"{sd.name}: EXCLUDED\n"
            continue
        else:
            loglines += f"{sd.name}: PROCESSED\n"

        print(f"PROCESSING {sd.name}\n")

        head_files = sorted(sd.glob('*/*_HEAD*'))
        phot_files = sorted(sd.glob('*/*_PHOT*'))
        
        fitfiles = sorted(fd.glob('output/*/*.FITRES*'))
        
        VERSION_SIM_OUT_DIR = Path(SIM_OUT_DIR / sd.name)
        VERSION_FIT_OUT_DIR = Path(FIT_OUT_DIR / fd.name)
        VERSION_SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)
        (VERSION_FIT_OUT_DIR / 'output').mkdir(parents=True, exist_ok=True)


        # Copy Pippin needed files
        # SIM
        shutil.copy2(sd / 'config.yml', VERSION_SIM_OUT_DIR)
        shutil.copy2(list(sd.glob('PIP*.input'))[0], VERSION_SIM_OUT_DIR)
        shutil.copy2(list(sd.glob('PIP*.LOG'))[0], VERSION_SIM_OUT_DIR)
        shutil.copytree(sd / 'LOGS', VERSION_SIM_OUT_DIR / 'LOGS', dirs_exist_ok=True)
        # FIT
        shutil.copy2(fd / 'config.yml', VERSION_FIT_OUT_DIR)
        shutil.copy2(list(fd.glob('FIT_PIP*.nml'))[0], VERSION_FIT_OUT_DIR)
        shutil.copy2(list(fd.glob('FIT_PIP*.LOG'))[0], VERSION_FIT_OUT_DIR)
        shutil.copy2(list(fd.glob('output/ALL.DONE'))[0], VERSION_FIT_OUT_DIR / 'output/ALL.DONE')
        shutil.copy2(list(fd.glob('output/MERGE.LOG'))[0], VERSION_FIT_OUT_DIR / 'output/MERGE.LOG')
        shutil.copy2(list(fd.glob('output/SUBMIT.INFO'))[0], VERSION_FIT_OUT_DIR / 'output/SUBMIT.INFO')

        SN_IDs = []
        for hf, pf in zip(head_files, phot_files):
            
            # Define new files
            new_head_dir = VERSION_SIM_OUT_DIR / hf.parent.name
            new_phot_dir = VERSION_SIM_OUT_DIR / pf.parent.name 
            new_head_dir.mkdir(parents=True, exist_ok=True)
            new_phot_dir.mkdir(parents=True, exist_ok=True)
            
            new_head_file = new_head_dir / hf.stem.replace('HEAD',f'{suffix.upper()}' + '_HEAD')
            new_phot_file = new_phot_dir / pf.stem.replace('PHOT',f'{suffix.upper()}' + '_PHOT')
            
            if compress:
                new_head_file = new_head_file.with_suffix('.FITS.gz')
                new_phot_file = new_phot_file.with_suffix('.FITS.gz')
                
            # Open files
            header = fits.open(hf)
            phot = fits.open(pf)
            phot_df = Table(phot[1].data).to_pandas()
            head_df = Table(header[1].data)[['SNID', 'PTROBS_MIN', 'PTROBS_MAX', 'NOBS', 'SIM_REDSHIFT_CMB']].to_pandas()
            
            # Apply selec
            mask, photmask = apply_selec(selec_prob, head_df, phot_df)

            # Copy phot and head fits files
            new_phot = copy.deepcopy(phot)
            new_head = copy.deepcopy(header)
            
            # Change header data + header
            new_head[0].header.set('photfile', new_phot_file.name)
            new_head[1].data = new_head[1].data[mask]
            
            # Change phot data + header
            new_phot[1].data = new_phot[1].data[photmask]
            new_phot[1].header.set('naxis2', len(new_phot[1].data))
            new_phot[0].header.set('photfile', new_phot_file.name)
            
            # Write 
            fits.HDUList(new_head).writeto(new_head_file, overwrite=overwrite)
            fits.HDUList(new_phot).writeto(new_phot_file, overwrite=overwrite)
            
            SN_IDs.append(new_head[1].data['SNID'])
            
        SN_IDs = np.hstack(SN_IDs).astype('int')
        loglines += f"N REMAINING SNe FROM {sd.name}: {len(SN_IDs)}\n"
        
        print(f'PROCESSING {fd.name}\n')
        for fitresf in sorted(fd.glob('output/*/*.FITRES*')):
            # Define new fit files
            new_fit_dir = VERSION_FIT_OUT_DIR / fitresf.parents[1].name / fitresf.parents[0].name
            new_fit_dir.mkdir(parents=True, exist_ok=True)
            new_fit_file = new_fit_dir / fitresf.stem
                        
            # Capture comments
            comments = []
            with gzip.GzipFile(fitresf, 'r') as f:
                i = 0
                while i <= 100:
                    l = f.readline()
                    if l.startswith(b'SN:'):
                        break
                    comments.append(l)
                    i+= 1
                if i == 100:
                    raise ValueError('VARNAMES not found')    
            comments = [''.join([c.decode("utf-8") for c in comments]).rstrip()]
            
            # Read data
            fitres = Table.read(fitresf, format='ascii')
            fitres.meta['comments'] = comments
            mask = np.isin(fitres['CID'], SN_IDs)
            ascii.write(fitres[mask], output=new_fit_file, 
                        overwrite=overwrite, comment='', format="no_header")
            if compress:
                with open(new_fit_file, 'rb') as f_in:
                    with gzip.open(new_fit_file.with_suffix('.FITRES.gz'),'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                new_fit_file.unlink()
                
    with open(OUT_DIR / "SELECTION.LOG", "w") as f:
        f.write(loglines)
