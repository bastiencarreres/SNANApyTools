"""This module contains tools relative to SNANA snlc_fit.exe output."""
import re
from pathlib import Path
from . import tools as tls


class SNANA_FIT:
    """Python wrapper for SNANA snlc_fit.exe output.
    """    
    def __init__(self, fit_dir: str, load_data=True):
        """Init SNANA_FIT.

        Parameters
        ----------
        fit_dir : str
            path to snlc_fit.exe output dir.
        load_data : bool, optional
            _description_, by default True
        
        """        
        self.dir = Path(fit_dir) 
        self.data_files = {f.name[:re.search('.FITRES', f.name).start()]: f  for f in self.dir.glob('*.FITRES*')}
        self.data = {k: 'NotLoaded' for k in self.data_files}

        if load_data:
            for k in self.data:
                self.data[k] = self.get_fitdata(k)
                
    def get_fitdata(self, fitopt: str | int ):
        """Returns fit data for the corresponding fitopt.

        Parameters
        ----------
        fitopt : str | int
            the fitopt number or string.

        Returns
        -------
        pandas.DataFrame
            Results of lc fit.
        """        
        if isinstance(fitopt, int):
            fitopt = str(fitopt)
            fitopt = 'FITOPT' + '0' * (3 - len(fitopt)) + fitopt
            
        if fitopt not in self.data_files:
            raise ValueError(f'{fitopt} not available, check self.data_files')
        
        return tls.read_SNANAfits(self.data_files[fitopt], format='ascii')
    