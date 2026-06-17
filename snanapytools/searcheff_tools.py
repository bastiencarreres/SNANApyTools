import datetime
from pathlib import Path
# WILL ADD SEACHEFF_PIPLINE AND HOSTZ LATER

class SEARCHEFF_SPEC_writer:
    def __init__(self, speceff_df, out_path, var_name, survey_name,author_name=None):
        """
        Parameters
        ----------
        speceff_df : pandas.DataFrame
            Two-column DataFrame: [var_name, SPECEFF]
        var_name : str
            Name of columns accepted by SNANA (i.e g r i, g-r, r-i, REDSHIFT, etc.)
        """
        self.speceff_df = speceff_df
        self.out_path = Path(out_path)
        self.var_name = var_name
        self.date_time = datetime.datetime.now()
        self.author_name = author_name
        self.survey = survey_name

    def get_SEARCHEFF_SPEC_doc(self):
        """Return the DOCUMENTATION block string.

        Returns
        -------
        str
        """
        doc = (
            "DOCUMENTATION:\n"
            "    PURPOSE: Spectroscopic confirmation efficiency\n"
            "    INTENT:  Nominal\n"
            "    USAGE_KEY:  SEARCHEFF_SPEC_FILE\n"
            "    USAGE_CODE: snlc_sim.exe\n"
            "    VERSIONS:\n"
            f"    - DATE: {self.date_time}\n"
            f"      AUTHORS: {self.author_name}\n"
            "DOCUMENTATION_END:\n\n"
        )
        return doc

    def write_SEARCHEFF_SPEC(self):
        """Write the SEARCHEFF_SPEC .DAT file."""
        out_file = self.out_path / f"SEARCHEFF_SPEC_{self.survey}.DAT"

        with open(out_file, "w") as f:
            f.write(self.get_SEARCHEFF_SPEC_doc())
            f.write(f"VARNAMES: {self.var_name} SPECEFF\n")
            for _, row in self.speceff_df.iterrows():
                f.write(f"SPECEFF: {row[self.var_name]:.6g} {row['SPECEFF']:.6g}\n")
