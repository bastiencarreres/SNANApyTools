
"""This module contains tools relative to PIPPIN output."""
import os
import yaml
from pathlib import Path, PosixPath
from .sim_tools import SNANA_SIM
from .fit_tools import SNANA_FIT
from .biascor_tools import SNANA_BIASCOR
from . import tools as tls
try:
    import graphviz
    graphviz_enabled = True
except ImportError:
    graphviz_enabled = False


class PIPPIN_READER: 
    """Python wrapper for SNANA output run by PIPPIN.
    """      
    _STEPS = ['dataprep', 'sim', 'lcfit', 'clas', 'agg', 'merge', 'biascor', 'create_cov']
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
        
        for i, s in enumerate(self._STEPS):
            path = self.path / f'{i}_{s.upper()}'
            print(path)
            if path.exists():
                setattr(self, s + '_path', path)
                setattr(self, 'available_' + s, set(d.name for d in getattr(self, s + '_path').iterdir()))
            else:
                setattr(self, s + '_path', None)
                setattr(self, 'available_' + s, None)
            

        self.tree, self.fitopt_map = self.build_tree()
        

 
    def build_tree(self):
        """Genrate the PIPPIN tree

        Returns
        -------
        dict, dict
            PIPPIN TREE, FITOPT_MAP
        """        
        sim_tree = {}
        fit_tree = {}
        biascor_tree = {}
        
        fitopt_map = {}
        for sim_name in self.available_sim:
            sim_dir =  self.sim_path / sim_name
            
            sim_tree[sim_name] = {
                'files': set(d for d in sim_dir.iterdir() if (d.is_dir() and 'PIP_' in d.name)),            
                'parents': set()
                }
            
        for fit_name in self.available_lcfit:
            fit_dir = self.lcfit_path / fit_name / 'output'
            fit_tree[fit_name] = {
                'files': set(d for d in fit_dir.iterdir() if (d.is_dir() and 'PIP_' in d.name)),
                'parents': set()
            }
            
            with open(self.lcfit_path / fit_name / 'config.yml') as f:
                output = yaml.safe_load(f)['OUTPUT']
                fit_tree[fit_name]['parents'].add(output['sim_name'])
                fit_tree[fit_name]['is_data'] = output['is_data']

            fitopt_map[fit_name] = {k: f'{i:03d}' for i, k in enumerate(output['fitopt_map'].keys())}
            
        for biascor_name in self.available_biascor:
            biascor_dir =  self.biascor_path / biascor_name / 'output'
            biascor_tree[biascor_name] = {'files': set(d for d in biascor_dir.iterdir() if (d.is_dir() and 'OUTPUT_BBCFIT' in d.name))}

            biascor_summary, biascor_files = tls.read_biascor_input_doc(self.biascor_path / biascor_name / f'{biascor_name}.input')

            fit_parents = []
            biascor_parents = []
            for mf in biascor_summary['INPDIR+']:
                merge_dir = Path(mf).parent 
                fit_parents.append(tls.trace_lcfit_from_merge(merge_dir))

            for mf in biascor_files:
                biascor_parents.append(tls.trace_lcfit_from_merge(mf))
                
            biascor_tree[biascor_name]['parents'] = set(fit_parents)
            biascor_tree[biascor_name]['biascor_parents'] = set(biascor_parents)

        PIPPIN_TREE = {'SIM': sim_tree, 'LCFIT': fit_tree, 'BIASCOR': biascor_tree}
        return PIPPIN_TREE, fitopt_map
    
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
            print('Multiple sims found') 
            return {s.name: SNANA_SIM(s, **kwargs) for s in sim_dir}
        
        return SNANA_SIM(sim_dir[0], **kwargs)

    def get_fit(self, lcfit_name: str | PosixPath, **kwargs) -> SNANA_FIT:
        """Get SNANA_FIT object of the given lcfit.

        Parameters
        ----------
        lcfit_name : str | PosixPath
            LCFIT name

        Returns
        -------
        SNANA_FIT
            The SNANA_FIT object of the LCFIT.
        """        
        if lcfit_name not in self.available_lcfit:
            raise ValueError(f"{lcfit_name} is not available. Check self.available_lcfit. Maybe refresh with self.up_dir('lcfit')")
        fit_dir = self.lcfit_path / lcfit_name / 'output'
        fit_dir = list(d for d in fit_dir.iterdir() if (d.is_dir() and 'PIP_' in d.name))
        if len(fit_dir) > 1:
            print('Multiple fit found')
            return {f.name: SNANA_FIT(f, **kwargs) for f in fit_dir}
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
        biascor_dir =  self.biascor_path / biascor_name / 'output'
        biascor_dir = list(d for d in biascor_dir.iterdir() if (d.is_dir() and 'OUTPUT_BBCFIT' in d.name))
        if len(biascor_dir) > 1:
            print('Multiple biascor found')
            return {b.name: SNANA_BIASCOR(b, **kwargs) for b in biascor_dir}
        return SNANA_BIASCOR(biascor_dir[0], **kwargs)

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

    def print_graph(self, name_filter='full'):
        """Plot PIPPIN graph.

        Parameters
        ----------
        name_filter : str, optional
            filters name to display, by default 'full'_
        """        
        if not graphviz_enabled:
            raise ImportError('Please install graphviz before using this function.')
        
        G = graphviz.Digraph('SNANA', comment='SNANA pipeline') 
        G.attr(rankdir='LR', style='rounded')
        
        with G.subgraph(name='cluster_0') as c:
            c.node_attr['style'] = 'filled'
            c.attr(label='DATAPREP')

            for s in self.available_dataprep:
                if name_filter == 'full' or name_filter in s:
                    c.node('DATAPREP_' + s, label=s)
        
        with G.subgraph(name='cluster_1') as c:
            c.node_attr['style'] = 'filled'
            c.attr(label='SIM')

            for s in self.available_sim:
                if name_filter == 'full' or name_filter in s:
                    c.node('SIM_' + s, label=s)
            
        with G.subgraph(name='cluster_2') as c:
            c.node_attr['style'] = 'filled'
            c.attr(label='LCFIT')
            for f in self.available_lcfit:
                if name_filter == 'full' or name_filter in f:
                    c.node('FIT_' + f)
        
        with G.subgraph(name='cluster_3') as c:
            c.node_attr['style'] = 'filled'
            c.attr(label='BIASCOR')
            for b in self.available_biascor:
                if name_filter == 'full' or (name_filter in b):
                    c.node('BIASCOR_'+ b, label=b)
                
        for f in self.available_lcfit:
            if f in self.tree['LCFIT'] and (name_filter == 'full' or (name_filter in f)):
                for p in  self.tree['LCFIT'][f]['parents']:
                    if self.tree['LCFIT'][f]['is_data']:
                         G.edge('DATAPREP_' + p, 'FIT_' + f, label='LC fit')  
                    else:
                        G.edge('SIM_' + p, 'FIT_' + f, label='LC fit')  
                    
        for f in self.available_biascor:
            if f in self.tree['BIASCOR'] and (name_filter == 'full' or (name_filter in f)):
                for p in  self.tree['BIASCOR'][f]['parents']:
                    G.edge('FIT_' + p, 'BIASCOR_'+ f, label='SALT2mu')
                for p in self.tree['BIASCOR'][f]['biascor_parents']:
                    G.edge('FIT_' + p, 'BIASCOR_'+ f, label='BiasCor SIM', style='dashed')

        return G