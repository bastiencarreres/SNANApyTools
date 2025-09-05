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


def write_pdf(path, par_dic, 
                 docstr="DOCUMENTATION:\nDOCUMENTATION_END:",
                 format_dic={}):
    
    file = open(path, 'w')
    file.write(docstr + '\n\n')

    for k in par_dic:
        if isinstance(par_dic[k], list):
            if k not in format_dic:
                format_dic[k] = '.3f'
            vals = " ".join([f"{v:{format_dic[k]}}" for v in  par_dic[k]])
            file.write(k + ': ' + vals)
        else:
            file.write("VARNAMES: " + " ".join([v for v in par_dic[k]]) + "\n")
            if k not in format_dic:
                format_dic[k] = {}
            for v in par_dic[k]:
                if v not in format_dic[k]:
                    format_dic[k][v] = '.3f'
            line = "PDF: " 
            for v in par_dic[k]:
                line += "{:" + f"{format_dic[k][v]}" + "} "
            line = line.strip()
            lines = '\n'.join([line.format(*r[1]) for r in par_dic[k].iterrows()])
            file.write(lines)
        file.write('\n\n')
    file.close()
    
def read_pdf(file):
    file = Path(file)
    if file.suffix == '.gz':
        f = gzip.open(file,'rb')
    else:
        f = open(file, "r")

    line = ut.line_cleaner(f.readline())
    doc_header = """DOCUMENTATION:
                       NODOC
                    DOCUMENTATION_END:
                 """
    if "DOCUMENTATION" in line:
        doc_header = line + '\n'
        line = ut.line_cleaner(f.readline())
        while "DOCUMENTATION_END" not in line:
            doc_header += '    ' + line
            doc_header += '\n'
            line = ut.line_cleaner(f.readline())
        doc_header += 'DOCUMENTATION_END:'  
    vardic = {} 
    while line:
        line = ut.line_cleaner(f.readline())
        if line.startswith('VARNAMES'):
            splitted = line.split(' ')
            var = splitted[1:]
            var_val = []
            line = ut.line_cleaner(f.readline())
            while line.startswith('PDF'):
                splitted = line.split(' ')
                var_val.append([float(v) for v in splitted[1:]])
                line = ut.line_cleaner(f.readline())
            var_val = np.array(var_val).T
            vardic[var[0]] = pd.DataFrame({v: val for v, val in zip(var, var_val)})
        
        elif line.strip() != '':
            splitted = line.replace(':', '').split(' ')
            vardic[splitted[0]] = [float(v) for v in splitted[1:]]
    return vardic, doc_header


def read_WGTMAP(file):
    """Read a SNANA HOSTLIB WGTMAP

    Parameters
    ----------
    file : str
        path to WGTMAP

    Returns
    -------
    list(str), dict
        list of varnames, WGTMAP as a dict

    Raises
    ------
    ValueError
        No WGT key
    """
    if file[-2:] == "gz":
        f = gzip.open(file, "rt", encoding="utf-8")
    else:
        f = open(file, "r")

    data_starts = False
    for l in f:
        if "VARNAMES_WGTMAP" in l:
            var_names = l.split()[1:]
            data = {k: [] for k in var_names}
            data_starts = True
        elif data_starts and "WGT:" in l:
            for k, val in zip(var_names, l.split()[1:]):
                data[k].append(float(val))
    f.close()
    for k in data:
        data[k] = np.array(data[k])

    if "WGT" in var_names:
        var_names.remove("WGT")
    else:
        raise ValueError("WGTMAP require 'WGT' key")
    if "SNMAGSHIFT" in var_names:
        var_names.remove("SNMAGSHIFT")
    return var_names, data


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
