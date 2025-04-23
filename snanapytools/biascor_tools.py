"""This module contains tools relative to SNANA SALT2mu.exe output."""
import re
from pathlib import Path, PosixPath
from . import tools as tls


class SNANA_BIASCOR:
    """Python wrapper for SNANA SALT2mu.exe output
    """    
    def __init__(self, biascor_dir: str | PosixPath, load_data=True):
        """Init SNANA_BIASCOR

        Parameters
        ----------
        biascor_dir : str | PosixPath
            path to SALT2mu.exe output dir
        load_data : bool, optional
            load or not data tables, by default True
        """        
        self.dir = Path(biascor_dir)
        if not self.dir.exists():
            raise FileNotFoundError(self.path)

        self.data_files = {f.name[:re.search('.FITRES', f.name).start()]: f  for f in self.dir.glob('*.FITRES*')}
        self.data = {k: 'NotLoaded' for k in self.data_files if not k.startswith('INPUT')}

        available_fitopt = []
        available_muopt = []

        for f in self.data:
            if not f.startswith('INPUT_'):
                fitopt_match = re.search('FITOPT', f).end()
                muopt_match = re.search('MUOPT', f).end()
                
                fitopt = f[fitopt_match:fitopt_match+3]
                muopt = f[muopt_match:muopt_match+3]
                
                available_fitopt.append(fitopt)
                available_muopt.append(muopt)
                
                if load_data:
                    self.data[f] = self.get_bbcdata(fitopt, muopt)
        
        self.available_fitopt = set(available_fitopt)
        self.available_muopt = set(available_muopt)
        
    def get_bbcdata(self, fitopt: str | int, muopt: str | int = 0):
        """Returns BBC data for the corresponding fitopt, muopt.

        Parameters
        ----------
        fitopt : str | int
            the fitopt number or string.
            
        muopt : str | int
            the muopt number or string.
            
        Returns
        -------
        pandas.DataFrame
            Results of SALT2mu.exe fit.
        """        
        if isinstance(fitopt, int):
            fitopt = str(fitopt)
        fitopt = 'FITOPT' + '0' * (3 - len(fitopt)) + fitopt
            
        if isinstance(muopt, int):
            muopt = str(muopt)
        muopt = 'MUOPT' + '0' * (3 - len(muopt)) + muopt
        
        file_name = f'{fitopt}_{muopt}'
        if file_name not in self.data_files:
            raise ValueError(f'{fitopt} not available, check self.data_files')
        
        return tls.read_SNANAfits(self.data_files[file_name], format='ascii')
      