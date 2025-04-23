
"""This module contains tools relative to PIPPIN output."""
import os
from pathlib import Path, PosixPath
from .sim_tools import SNANA_SIM
from .fit_tools import SNANA_FIT
from .biascor_tools import SNANA_BIASCOR
from . import tools as tls

class PIPPIN_READER: 
    """Python wrapper for SNANA output run by PIPPIN.
    """      
    def __init__(self, name: str, pippin_output: str=None):
        """Init PIPPIN_READER.

        Parameters
        ----------
        name : str
            name of the PIPPIN dir
        pippin_output : _type_, optional
            PIPPIN_OUTPUT dir, by default env $PIPPIN_OUTPUT
        Raises
        ------
        FileNotFoundError
            _description_
        """        
        if pippin_output is None:
            self.__PIPPIN__OUTPUT__ = Path(os.environ['PIPPIN_OUTPUT'])
        else: 
            try:
                self.__PIPPIN__OUTPUT__ = Path(os.getenv('PIPPIN_OUTPUT'))
            except TypeError:
                raise FileNotFoundError("No PIPPIN_OUTPUT variable found. Please define 'pippin_output'")

        self.name = name     
        self.path = self.__PIPPIN__OUTPUT__ / name
        
        self.sim_path = self.path / '1_SIM'
        self.fitlc_path = self.path / '2_LCFIT'
        self.biascor_path = self.path / '6_BIASCOR'
        self.createcov_path = self.path / '7_CREATE_COV'

        self.available_sim = self.up_dir('sim')
        self.available_fitlc = self.up_dir('fitlc')
        self.available_biascor = self.up_dir('biascor')
        self.available_createcov = self.up_dir('cov')
        
    def up_dir(self, kind: str) -> set:
        """Get list of outputs of some kind. kind = 'sim', 'fitlc', 'biascor', 'cov'

        Parameters
        ----------
        kind : str
            sim, fitlc, biascor, cov

        Returns
        -------
        set
            Available outputs.
        """        
        if kind == 'sim':
            return set(d.name for d in self.sim_path.iterdir())
        if kind == 'fitlc':
            return set(d.name for d in self.fitlc_path.iterdir())
        if kind == 'biascor':
            return set(d.name for d in self.biascor_path.iterdir())
        if kind == 'cov':
            return set(d.name for d in self.createcov_path.iterdir())
 
    def get_sim(self, sim_name: str | PosixPath,  **kwargs) -> SNANA_SIM:
        """Get SNANA_SIM object of the simulation.

        Parameters
        ----------
        sim_name : str | PosixPath
            Simulation name

        Returns
        -------
        SNANA_SIM
            The SNANA_SIM object of the SIM.
        """        
        if sim_name not in self.available_sim:
            raise ValueError(f"{sim_name} is not available. Check self.available_sim. Maybe refresh with self.up_dir('sim')")
        sim_dir =  self.sim_path / sim_name
        sim_dir = list(d for d in sim_dir.iterdir() if (d.is_dir() and 'PIP_' in d.name))
        if len(sim_dir) > 1:
            raise ValueError(f'Multiple sims in {sim_dir}')
        return SNANA_SIM(sim_dir[0], **kwargs)

    def get_fit(self, fitlc_name: str | PosixPath, **kwargs) -> SNANA_FIT:
        """Get SNANA_FIT object of the given FITLC.

        Parameters
        ----------
        fitlc_name : str | PosixPath
            FITLC name

        Returns
        -------
        SNANA_FIT
            The SNANA_FIT object of the FITLC.
        """        
        if fitlc_name not in self.available_fitlc:
            raise ValueError(f"{fitlc_name} is not available. Check self.available_fitlc. Maybe refresh with self.up_dir('fitlc')")
        fit_dir = self.fitlc_path / fitlc_name / 'output'
        fit_dir = list(d for d in fit_dir.iterdir() if (d.is_dir() and 'PIP_' in d.name))
        if len(fit_dir) > 1:
            raise ValueError(f'Multiple fitlc directories in {fit_dir}')

        return SNANA_FIT(fit_dir[0], **kwargs)
    
    def get_biascor(self, biascor_name: str | PosixPath, **kwargs) -> SNANA_BIASCOR:
        """Get SNANA_BIASCOR object of the given BIASCOR

        Parameters
        ----------
        biascor_name : str | PosixPath
            BIASCOR name

        Returns
        -------
        SNANA_BIASCOR
            The SNANA_BIASCOR object of the BIASCOR.
        """        
        if biascor_name not in self.available_biascor:
            raise ValueError(f"{biascor_name} is not available. Check self.available_biascor. Maybe refresh with self.up_dir('biascor')")
        biascor_dir =  self.biascor_path / biascor_name / 'output/OUTPUT_BBCFIT'
        return SNANA_BIASCOR(biascor_dir, **kwargs)

    def print_tree(self, nofile: bool = False):
        """Print PIPPIN dir tree.

        Parameters
        ----------
        nofile : bool, optional
            Eiher or not to print file in the tree, by default False
        filters : _type_, optional
            _description_, by default None (Not Implemented Yet)
        """        
        print(self.name)
        tls.print_directory_tree(self.path, nofile=nofile)

