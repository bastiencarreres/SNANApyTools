import datetime
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.affinity as shp_aff
import shapely.geometry as shp_geo

from . import tools as tls
from . import utils as ut


class SIMLIB_writer:
    _column_keys = {
        "MJD",
        "IDEXPT",
        "FLT",
        "GAIN",
        "NOISE",
        "SKYSIG",
        "PSF1",
        "PSF2",
        "RATIO",
        "ZPTAVG",
        "ZPTSIG",
        "MAG",
    }

    _head_keys = {
        'LIBID',
        'RA',
        'DEC',
        'MWEBV',
        'NOBS',
        'PIXSIZE',
        'PEAKMJD',
                }

    def __init__(
        self,
        simlib_libentry_headers,
        simlib_libentry_obs,
        *,
        survey_name="NONE",
        survey_filters="NONE",
        psf_unit=None,
        author_name=None,
        skysig_unit=None,
        file_suffix="",
        documentation_notes={},
        nexpose=False,
    ):

        self.simlib_libentry_headers = simlib_libentry_headers
        self.simlib_libentry_obs = simlib_libentry_obs
        self.author_name = author_name
        self.survey_name = survey_name
        self.skysig_unit = skysig_unit
        self.psf_unit = psf_unit
        self.survey_filters = survey_filters
        self.documentation_notes = documentation_notes
        self.nexpose = nexpose
        self.survey_hosts = None

        self.date_time = datetime.datetime.now()

        self.lib_dataline = np.vectorize(self._lib_dataline)

    def _init_out_path(self, out_path):
        """Format output path for SIMLIB and HOSTLIB.

        Parameters
        ----------
        out_path : str
            Output directory or file.

        Returns
        -------
        pathlib.Path
            Path to the output SIMLIB.
        """
        if out_path is None:
            out_path = "./NONAME.SIMLIB"

        out_path = Path(out_path)

        return out_path

    def get_SIMLIB_doc(self, out_path):
        """Give the DOCUMENTATION string for SIMLIB.

        Returns
        -------
        str
            DOCUMENTATION string
        """
        doc = "DOCUMENTATION:\n"
        doc += "    USAGE_KEY: SIMLIB_FILE\n"
        doc += "    USAGE_CODE: snlc_sim.exe\n"
        doc += "    NOTES: \n"
        if self.survey_hosts is not None:
            doc += "        ASSOCIATED HOSTLIB: {}\n".format(
                out_path.with_suffix(".HOSTLIB").absolute()
            )
        for notes, val in self.documentation_notes.items():
            doc += f"        {notes.upper()}: {val}\n"
        doc += "    VERSIONS:\n"
        doc += f'    - DATE : {self.date_time.strftime(format="%y-%m-%d")}\n'
        doc += f"    AUTHORS : {self.author_name}\n"
        doc += "DOCUMENTATION_END:\n"
        return doc

    def get_SIMLIB_header(self, comments="\n"):
        """Give the SIMLIB header string.

        Parameters
        ----------
        saturation_flag : int, optional
            The flag corresponding to saturated obs, by default 1024
        comments : str, optional
            Comments to add to the header, by default '\\n'

        Returns
        -------
        str
            The SIMLIB header string.
        """
        try:
            user = os.getlogin()
        except:
            user = "NONE"
        try:
            host = os.getenv("HOSTNAME")
        except:
            host = "NONE"
        # comment: I would like to generalize ugrizY to a sort but am not sure
        # of the logic for other filter names. so ducking for now
        header = "\n\n\n"
        header += f"SURVEY: {self.survey_name}   FILTERS: {self.survey_filters}\n"
        header += "USER: {0:}     HOST: {1}\n".format(user, host)
        header += f"NLIBID: {len(self.simlib_libentry_headers)}\n"
        if self.psf_unit is not None:
            header += f"PSF_UNIT: {self.psf_unit}\n"
        if self.skysig_unit is not None:
            header += f"SKYSIG_UNIT: {self.skysig_unit}\n"
        header += comments + "\n"
        header += "BEGIN LIBGEN\n"
        return header

    @staticmethod
    def LIBheader(
        obsdf,
        LIBID,
        ra,
        dec,
        *,
        pixsize=None,
        mwebv=0.0,
        groupID=None,
        field_label=None,
        peakmjd=None,
        nexpose=False,
    ):
        """Give the string of the header of a LIB entry.

        Parameters
        ----------
        LIBID : int
            The LIBID of the entry.
        ra : float
            RA [deg] coordinate of the entry
        dec : float
            Dec [deg] coordinate of the entry
        opsimdf : pandas.DataFrame
            LIB entry observations
        mwebv: float, optional
            MWEBV of the entry, default = 0.0
        groupID : int, optional
            GROUPID of the entry used to match with HOSTLIB hosts.

        Returns
        -------
        str
            The LIB entry header string.
        """
        nobs = len(obsdf)
        # String formatting
        s = "# --------------------------------------------" + "\n"
        s += "LIBID: {0:10d}".format(int(LIBID)) + "\n"
        tmp = "RA: {0:+10.6f} DEC: {1:+10.6f}   NOBS: {2:10d} MWEBV: {3:5.2f}"
        s += tmp.format(ra, dec, nobs, mwebv)
        if pixsize is not None:
            tmp += " PIXSIZE: {4:5.3f}"
        if field_label is not None:
            s += f" FIELD: {field_label}"
        if peakmjd is not None:
            s += f" PEAKMJD: {peakmjd:.4f}"
        if groupID is not None:
            s += f"\nHOSTLIB_GROUPID: {groupID}"
        if nexpose:
            IDlabel = "ID*NEXPOSE"
        else:
            IDlabel = "IDEXPT"

        s += "\n#                           CCD  CCD         PSF1 PSF2 PSF2/1" + "\n"
        s += (
            f"#     MJD      {IDlabel}  FLT GAIN NOISE SKYSIG (pixels)  RATIO  ZPTAVG ZPTERR  MAG"
            + "\n"
        )
        return s

    def LIBdata(self, obsdf):
        """Give the string of a LIB entry.

        Parameters
        ----------
        opsimdf : pandas.DataFrame
            LIB entry observations

        Returns
        -------
        str
            The str of the LIB entry.
        """
        lib = "\n".join(
            self.lib_dataline(
                obsdf["MJD"].values,
                obsdf["IDEXPT"].values,
                obsdf["FLT"].values,
                obsdf["CCDGAIN"].values,
                obsdf["CCDNOISE"].values,
                obsdf["SKYSIG"].values,
                obsdf["PSF1"].values,
                obsdf["PSF2"].values,
                obsdf["PSF12RATIO"].values,
                obsdf["ZPTAVG"].values,
                obsdf["ZPTERR"].values,
                obsdf["MAG"].values,
            )
        )
        return lib + "\n"

    @staticmethod
    def LIBfooter(LIBID):
        """Give the string of a LIB entry footer.

        Parameters
        ----------
        LIBID : int
            The LIBID of the entry

        Returns
        -------
        str
            The string of a LIB entry footer
        """
        footer = "END_LIBID: {0:10d}".format(int(LIBID))
        footer += "\n"
        return footer

    def get_SIMLIB_footer(self):
        """Give SIMLIB footer."""
        s = "END_OF_SIMLIB:    {0:10d} ENTRIES".format(
            len(self.simlib_libentry_headers)
        )
        return s

    def write_SIMLIB(self, out_path, write_batch_size=10, buffer_size=8192):
        """write the SIMLIB (and the HOSTLIB) file(s).

        Parameters
        ----------
        out_path: pathlib.Path
            path of SIMLIB
        write_batch_size : int
            Number of LIBID to write at the same time
        buffer_size : int
            buffering option for open() function
        """
        tstart = time.time()
        out_path = self._init_out_path(out_path)

        print(f"Writing SIMLIB in {out_path}")

        with open(out_path, "w", buffering=buffer_size) as simlib_file:
            simlib_file.write(self.get_SIMLIB_doc(out_path))
            simlib_file.write(self.get_SIMLIB_header())

            bcount = 0
            simlibstr = ""
            for (i, lib) in self.simlib_libentry_headers.iterrows():
                obsdf = pd.DataFrame(self.simlib_libentry_obs[lib['LIBID']])
                if self.survey_hosts is not None:
                    groupID = lib['LIBID']
                else:
                    groupID = None
                if "FIELD" in lib:
                    field_label = lib["FIELD"]
                else:
                    field_label = None
                if "PIXSIZE" in lib:
                    pixsize = lib["PIXSIZE"]
                else:
                    pixsize = None
                if "PEAKMJD" in lib:
                    peakmjd = lib["PEAKMJD"]
                else:
                    peakmjd = None
                simlibstr += self.LIBheader(
                    obsdf,
                    lib["LIBID"],
                    lib["RA"],
                    lib["DEC"],
                    pixsize=pixsize,
                    peakmjd=peakmjd,
                    groupID=groupID,
                    field_label=field_label,
                    nexpose=self.nexpose,
                )
                simlibstr += self.LIBdata(obsdf)
                simlibstr += self.LIBfooter(lib['LIBID'])

                if not bcount % write_batch_size:
                    simlib_file.write(simlibstr)
                    simlibstr = ""
                bcount += 1

            simlib_file.write(simlibstr)
            simlib_file.write(self.get_SIMLIB_footer())

        print(f"SIMLIB wrote in {time.time() - tstart:.2f} sec.\n")

        if self.survey_hosts is not None:
            tstart = time.time()
            print(
                f"Writting {len(self.survey_hosts)} hosts in {out_path.with_suffix('.HOSTLIB')}"
            )
            self.write_HOSTLIB(
                out_path,
                buffer_size=buffer_size,
            )
            print(f"HOSTLIB file wrote in {time.time() - tstart:.2f} sec.")

    def set_survey_hosts(self, host_file, matching_radius,
                         wgt_map_file=None,
                         nworkers=10,
                         host_file_kwargs={}):
        """MAtch host and survey obs coordinates.

        Parameters
        ----------
        host_file : str
            Hosts file (parquet)
        matching_radius : float
            Radius in which match the hosts (in deg)
        wgt_map_file : _type_, optional
            _description_, by default None
        nworkers : int, optional
            _description_, by default 10
        host_file_kwargs : dict, optional
            _description_, by default {}
        """

        hosts = self._read_host_file(host_file, wgt_map_file=wgt_map_file, **host_file_kwargs)

        # Cut in 2 circles that intersect edge limits (0, 2PI)
        _SPHERE_LIMIT_LOW_ = shp_geo.LineString([[0, -np.pi / 2], [0, np.pi / 2]])

        _SPHERE_LIMIT_HIGH_ = shp_geo.LineString(
            [[2 * np.pi, -np.pi / 2], [2 * np.pi, np.pi / 2]]
        )

        survey_fields = gpd.GeoDataFrame(
            index=self.simlib_libentry_headers.index,
            geometry=gpd.points_from_xy(
                np.deg2rad(self.simlib_libentry_headers['RA']),  np.deg2rad(self.simlib_libentry_headers['DEC'])
            ).buffer(np.deg2rad(matching_radius)),
        )

        # scale for dec dependance
        survey_fields = survey_fields.map(
            lambda x: shp_aff.scale(
                x, xfact=np.sqrt(2 / (1 + np.cos(2 * x.centroid.xy[1][0])))
            )
        )

        # mask for edge effect
        mask = survey_fields.intersects(_SPHERE_LIMIT_LOW_)
        survey_fields[mask] = survey_fields[mask].translate(2 * np.pi)
        mask |= survey_fields.intersects(_SPHERE_LIMIT_HIGH_)
        survey_fields.loc[mask, "geometry"] = gpd.GeoSeries(
            data=[ut.format_poly(p) for p in survey_fields[mask].geometry],
            index=survey_fields[mask].index,
        )

        host_joiner = partial(ut.host_joiner, survey_fields)

        sdfs = ut.df_subdiviser(hosts, Nsub=nworkers)

        with Pool(nworkers) as p:
            res = p.map(host_joiner, sdfs)

        self.survey_hosts = pd.concat(res)
        self.survey_hosts['GROUPID'] = self.simlib_libentry_headers.loc[
            self.survey_hosts['GROUPID'], 'LIBID'
            ].values
        self.survey_hosts.drop(columns=["ra", "dec"], inplace=True)

    def _read_host_file(
        self,
        host_file,
        col_ra="ra",
        col_dec="dec",
        ra_dec_unit="radians",
        wgt_map_file=None,
        add_SNMAGSHIFT=False,
    ):
        """Read a parquet file containing hosts.

        Parameters
        ----------
        host_file : str
            Path to the parquet file
        col_ra : str, optional
            Key of column containing RA, by default 'ra'
        col_dec : str, optional
            Key of column containing Dec, by default 'dec'
        ra_dec_unit : str, optional
            Unit of ra_dec (radians or degrees), by default 'radians'
        wgt_map_file : str,  optional
            Path to a SNANA wgt map to apply to subsample hosts
        add_SNMAGSHIFT :
            A MAGSHIFT to add in HOISTLIB file
        Returns
        -------
        pandas.DataFrame
            Dataframe of the hosts.
        """
        if host_file is None:
            print("No host file.")
            return None

        print("Reading host from {}".format(host_file))
        hostdf = pd.read_parquet(host_file)

        if wgt_map_file is not None:
            print(f"Reading and applying HOST WGT MAP from {wgt_map_file}")
            var_names, wgt_map = tls.read_WGTMAP(wgt_map_file)
            if len(var_names) > 1:
                raise NotImplementedError("HOST RESAMPLING ONLY WORK FOR 1 VARIABLES")
            keep_index = ut.host_resampler(
                wgt_map[var_names[0]],
                wgt_map["WGT"],
                hostdf.index.values,
                hostdf[var_names[0]].values,
            )

            hostdf = hostdf.loc[keep_index]

            if add_SNMAGSHIFT and "SNMAGSHIFT" in wgt_map:
                snmagshift = np.zeros(len(hostdf))
                for i in range(len(wgt_map["WGT"]) - 1):
                    mask = np.ones(len(hostdf), dtype=bool)
                    for v in var_names:
                        mask &= hostdf[v].between(wgt_map[v][i], wgt_map[v][i + 1])
                    snmagshift[mask] = wgt_map["SNMAGSHIFT"][i]
                hostdf["SNMAGSHIFT"] = snmagshift

        if ra_dec_unit == "degrees":
            hostdf[col_ra] += 360 * (hostdf[col_ra] < 0)
            hostdf.rename(columns={col_ra: "RA_GAL", col_dec: "DEC_GAL"}, inplace=True)
            hostdf["ra"] = np.radians(hostdf["RA_GAL"])
            hostdf["dec"] = np.radians(hostdf["DEC_GAL"])
        elif ra_dec_unit == "radians":
            hostdf[col_ra] += 2 * np.pi * (hostdf[col_ra] < 0)
            hostdf.rename(columns={col_ra: "ra", col_dec: "dec"}, inplace=True)
            hostdf["RA_GAL"] = np.degrees(hostdf["ra"])
            hostdf["DEC_GAL"] = np.degrees(hostdf["dec"])
        hostdf.attrs["file"] = host_file
        return hostdf

    @staticmethod
    def _lib_dataline(MJD,
                      IDEXPT,
                      FLT,
                      CCDGAIN,
                      CCDNOISE,
                      SKYSIG,
                      PSF1,
                      PSF2,
                      PSF12RATIO,
                      ZPTAVG,
                      ZPTERR,
                      MAG):
        """Write a SIMLIB data line

        Parameters
        ----------
        MJD : float
            Date of observation in MJD
        IDEXPT : int
            ID of the observation
        FLT : str
            Band used for the observation
        CCDGAIN : float
            CCd gain
        CCDNOISE : float
            CCD noise
        SKYSIG : float
            Sky noise error
        PSF1 : float
            Point spread function sigma1
        PSF2: float
            Point spread function sigma2
        PSF12RATIO : float
            Ratio between PSF1/PSF2
        ZPTAVG : float
            Zero point
        ZPTERR : float
            Zero point calibration error
        MAG : float
            Magnitude
        Returns
        -------
        str
            A SIMLIB LIB entry line
        """
        l = (
            "S: "
            f"{MJD:5.4f} "
            f"{IDEXPT} "
            f"{FLT} "
            f"{CCDGAIN:5.2f} "  # CCD Gain
            f"{CCDNOISE:5.2f} "  # CCD Noise
            f"{SKYSIG:6.2f} "  # SKYSIG
            f"{PSF1:4.2f} "  # PSF1
            f"{PSF2:4.2f} "  # PSF2
            f"{PSF12RATIO:4.3f} "  # PSFRatio
            f"{ZPTAVG:6.2f} "  # ZPTAVG
            f"{ZPTERR:6.3f} "  # ZPTNoise
            f"{MAG:+7.3f} "
        )
        return l

    def get_HOSTLIB_doc(self, out_path):
        """Give docstring for HOSTLIB file.

        Parameters
        ----------
        out_path: pathlib.Path
            path of associated SIMLIB

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
            f"    - ASSOCIATED SIMLIB : {out_path.absolute()}\n"
            f"    AUTHORS : {self.author_name}\n"
            "DOCUMENATION_END:\n"
            "# ========================\n"
        )
        return doc

    def get_HOSTLIB_header(self):
        """Give HOSTLIB header.
        Parameters
        ----------
        out_path: pathlib.Path
            path of associated SIMLIB

        Returns
        -------
        str
            Header of HSOTLIB file
        """
        header = (
            f"# Z_MIN={self.survey_hosts.ZTRUE_CMB.min()} Z_MAX={self.survey_hosts.ZTRUE_CMB.max()}\n\n"
            "VPECERR: 0\n\n"
        )
        return header

    def write_HOSTLIB(self, out_path, buffer_size=8192):
        """Write the HOSTLIB file. Called in write_SIMLIB.

        Parameters
        ----------
        out_path: pathlib.Path
            path of associated SIMLIB
        buffer_size : int
            buffering option for open() function
        """
        import csv

        VARNAMES = "VARNAMES:"
        for k in self.survey_hosts.columns:
            VARNAMES += f" {k}"

        with open(out_path.with_suffix(".HOSTLIB"), "w", buffering=buffer_size) as hostf:
            hostf.write(self.get_HOSTLIB_doc(out_path))
            hostf.write(self.get_HOSTLIB_header())
            columns = self.survey_hosts.columns.values
            columns = np.insert(columns, 0, "VARNAMES: ")
            self.survey_hosts["VARNAMES: "] = "GAL: "
            self.survey_hosts[columns].to_csv(
                hostf, sep=" ", index=False, quoting=csv.QUOTE_NONE, escapechar=" "
            )
