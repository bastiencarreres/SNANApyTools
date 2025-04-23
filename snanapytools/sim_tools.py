"""This module contains tools relative to SNANA snlc_sim.exe output."""
from pathlib import Path, PosixPath
from . import tools as tls
import pandas as pd


class SNANA_SIM:
    """Python wrapper for SNANA snlc_sim.exe output.
    """  
    def __init__(self, sim_dir: str | PosixPath, load_head=True, load_phot=False):
        """Init SNANA_SIM.

        Parameters
        ----------
        sim_dir : str | PosixPath
            Path to SNANA snlc_sum.exe output.
        load_head : bool, optional
            if True load head data, by default True
        load_phot : bool, optional
            if True load phot data, by default False (not implemented yet)
        """        
        self.path = Path(sim_dir)

        if not self.path.exists():
            raise FileNotFoundError(self.path)
        
        self._head_files = sorted(self.path.glob('*_HEAD*'))
        self._phot_files = sorted(self.path.glob('*_PHOT*')) 
        
        self.head = None
        self.phot = None

        if load_head:
            self.load_head()
        if load_phot:
            self.load_phot()

    def load_head(self):
        """Load header data into self.head
        """        
        if self.head is not None:
            print('Head already loaded')

        table_list = []
        for i, h in enumerate(self._head_files):
            df_tmp = tls.read_SNANAfits(h)
            df_tmp['PHOT_FILE'] = i
            table_list.append(df_tmp)
            
        self.head = pd.concat(table_list)
        self.head['SNID'] = self.head['SNID'].astype('int')
        
        self.head.set_index('SNID', inplace=True)

    def _read_lcs_table(self):
        raise NotImplementedError
        
    def load_phot(self):
        raise NotImplementedError

    def get_sn_phot(self, snid: int):
        """Return phot data of the LC corresponding to snid.

        Parameters
        ----------
        snid : int
            SNID of the LC.

        Returns
        -------
        pandas.DataFrame
            Sim data.
        """        
        file, PTROBS_MIN, PTROBS_MAX = self.head.loc[snid, ['PHOT_FILE', 'PTROBS_MIN', 'PTROBS_MAX']]
        return tls.read_SNANAfits(self._phot_files[file]).iloc[PTROBS_MIN-1:PTROBS_MAX]
        

