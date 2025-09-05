import numpy as np
import datetime
from pathlib import Path

class HOSTLIB_writer:
    def __init__(self, host_df, out_path, author_name=None):
        
        self.host_df = host_df
        self.out_path = Path(out_path)
        self.date_time = datetime.datetime.now()
        self.author_name = author_name
        

    def get_HOSTLIB_doc(self):
        """Give docstring for HOSTLIB file.

        Returns
        -------
        str
            Docstring of the HOSTLIB file
        """
        doc = (
            "DOCUMENTATION:\n"
            "PURPOSE: HOSTLIB for LSST based on mock opsim\n"
            "    VERSIONS:\n"
            f"    - DATE : {self.date_time}\n"
            f"    - ASSOCIATED SIMLIB : {self.out_path.absolute()}\n"
            f"    AUTHORS : {self.author_name}\n"
            "DOCUMENATION_END:\n"
            "# ========================\n"
        )
        return doc 

    def get_HOSTLIB_header(self):
        """Give HOSTLIB header.

        Parameters
        ----------
        hostdf : pandas.DataFrame
            Hosts dataframe

        Returns
        -------
        str
            Header of HSOTLIB file
        """
        header = (
            f"# Z_MIN={self.host_df.ZTRUE_CMB.min()} Z_MAX={self.host_df.ZTRUE_CMB.max()}\n\n"
            "VPECERR: 0\n\n"
        )
        return header
    
    def write_HOSTLIB(self, buffer_size=8192):
        """Write the HOSTLIB file. Called in write_SIMLIB.

        Parameters
        ----------
        buffer_size : int
            buffering option for open() function
        """
        import csv

        VARNAMES = "VARNAMES:"
        for k in self.host_df.columns:
            VARNAMES += f" {k}"

        with open(
            self.out_path.with_suffix(".HOSTLIB"), "w", buffering=buffer_size
        ) as hostf:
            hostf.write(self.get_HOSTLIB_doc())
            hostf.write(self.get_HOSTLIB_header())
            columns = self.host_df.columns.values
            columns = np.insert(columns, 0, "VARNAMES: ")
            self.host_df["VARNAMES: "] = "GAL: "
            self.host_df[columns].to_csv(
                hostf, sep=" ", index=False, quoting=csv.QUOTE_NONE, escapechar=" "
            )