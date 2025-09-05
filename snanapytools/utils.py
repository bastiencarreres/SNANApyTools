import re
import numpy as np
import geopandas as gpd
from numba import njit, guvectorize
from shapely import geometry as shp_geo
from shapely import ops as shp_ops

_SPHERE_LIMIT_ = shp_geo.LineString([[2 * np.pi, -np.pi/2],
                                     [2 * np.pi,  np.pi/2]])

 

@njit
def R_base(theta, phi, vec):
    """Give rotation to RA := theta, DEC := phi.
    Parameters
    ----------
    theta : float
        RA amplitude of the rotation
    phi : float
        Dec amplitude of the rotation 
    vec : numpy.ndarray(float)
        Carthesian vector to rotate
    Returns
    -------
    numpy.ndarray(float)
        Rotated vector.
    """
    R = np.zeros((3, 3), dtype='float')
    R[0, 0] = np.cos(phi) * np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[0, 2] = -np.cos(theta) * np.sin(phi)
    R[1, 0] = np.cos(phi) * np.sin(theta)
    R[1, 1] = np.cos(theta)
    R[1, 2] = -np.sin(phi) * np.sin(theta)
    R[2, 0] = np.sin(phi)
    R[2, 1] = 0
    R[2, 2] = np.cos(phi)
    return R @ vec

def line_cleaner(line):
    if isinstance(line, bytes):
        line = line.decode()
    if line == '\n':
        return '\n'
    line = line.strip()
    line = re.sub(' +', ' ', line)
    return line

@guvectorize(["void(float64[:, :], float64[:, :], float64[:,:])"],
              "(m, n),(m, n)->(m, n)", nopython=True)
def new_coord_on_fields(ra_dec, ra_dec_frame, new_radec):
    """Compute new coordinates of an object in a list of fields frames.
    Parameters
    ----------
    ra_frame : numpy.ndarray(float)
        Field Right Ascension.
    dec_frame : numpy.ndarray(float)
        Field Declinaison.
    vec : numpy.ndarray(float, size = 3)
        The carthesian coordinates of the object.
    Returns
    -------
    numpy.ndarray(float, size = (2, ?))
        The new coordinates of the obect in each field frame.
    """
   
    for i in range(len(ra_dec_frame[0])):
        vec = np.array([np.cos(ra_dec[0][i]) * np.cos(ra_dec[1][i]),
                        np.sin(ra_dec[0][i]) * np.cos(ra_dec[1][i]),
                        np.sin(ra_dec[1][i])])
        x, y, z = R_base(ra_dec_frame[0][i], ra_dec_frame[1][i], vec)
        new_radec[0][i] = np.arctan2(y, x)
        if  new_radec[0][i] < 0: new_radec[0][i] +=  2 * np.pi
        new_radec[1][i] = np.arcsin(z)
        
        
def _format_corner(corner, RA):
    # -- Replace corners that cross sphere edges
    #    
    #     0 ---- 1
    #     |      |
    #     3 ---- 2
    # 
    #   conditions : 
    #       - RA_0 < RA_1
    #       - RA_3 < RA_2
    #       - RA_0 and RA_3 on the same side of the field center
    
    sign = (corner[3][0] - RA) * (corner[0][0] - RA) < 0
    comp = corner[0][0] < corner[3][0]

    corner[1][0][corner[1][0] < corner[0][0]] += 2 * np.pi
    corner[2][0][corner[2][0] < corner[3][0]] += 2 * np.pi


    corner[0][0][sign & comp] += 2 * np.pi
    corner[1][0][sign & comp] += 2 * np.pi

    corner[2][0][sign & ~comp] += 2 * np.pi
    corner[3][0][sign & ~comp] += 2 * np.pi
    return corner


def _compute_polygon(corners):
    """Create polygon on a sphere, check for edges conditions."""
    polygon = shp_geo.Polygon(corners)
    # -- Cut into 2 polygon if cross the edges
    if polygon.intersects(_SPHERE_LIMIT_):
        unioned = polygon.boundary.union(_SPHERE_LIMIT_)
        polygon = [p for p in shp_ops.polygonize(unioned)
                    if p.representative_point().within(polygon)]

        x0, y0 = polygon[0].boundary.xy
        x1, y1 = polygon[1].boundary.xy

        if x1 > x0: 
            x1 = np.array(x1) - 2 * np.pi
            polygon[1] = shp_geo.Polygon(np.array([x1, y1]).T)
        else:
            x0 = np.array(x0) - 2 * np.pi
            polygon[0] = shp_geo.Polygon(np.array([x0, y0]).T)
        polygon =  shp_geo.MultiPolygon(polygon)
    return polygon


def typer(val, dtype):
    if dtype == 'int':
        return int(val)
    if dtype == 'float':
        return float(val)
    if dtype == 'str':
        return str(val)


vstartswith = np.vectorize(lambda line, motif: line.startswith(motif), excluded=[1])
vcontains = np.vectorize(lambda line, motif: motif in line, excluded=[1])

@np.vectorize
def GalLine(*args, first="GAL: "):
    line = first + " {}" * len(args)
    return line.format(*args)


def create_hostlib(df, filename, key_mapper={}):
    _key_mapper_default = {'GALID': 'index', 'RA_GAL': 'ra', 'DEC_GAL': 'dec', 
                           'ZTRUE_CMB':'zcos', 'VPEC':'vpec_true', 'GROUPID':'groupid',
                           'a0_Sersic': 0.5, 'b0_Sersic': 0.5, 'a_rot': 0.0, 'n0_Sersic': 0.5}
    
    key_mapper = {**_key_mapper_default, **key_mapper}

    VARNAMES = "VARNAMES:"
    VALUES = []
    for k, v in key_mapper.items():
        VARNAMES += f" {k}"
        if isinstance(v, (int, float)):
            VALUES.append(np.ones(len(df)) * v)
        else:
            VALUES.append(getattr(df, v).values)       
    
    print('Writting {} hosts in HOSTLIB file'.format(len(df)))
    Header = ("DOCUMENTATION:\n"
              "PURPOSE: TEST\n"
              "DOCUMENATION_END:\n"
              "# ========================\n"
              f"# Z_MIN={df.zcos.min()} Z_MAX={df.zcos.max()}\n\n"
              "VPECERR: 0\n\n"
              f"{VARNAMES}\n\n")
    
    VAL_to_write = VALUES
    lines = [''] * len(VALUES[0])
    nloop = len(VAL_to_write) // 30 + 1
    for i in range(nloop):
        lines = np.char.add(lines, GalLine(*VALUES[i * 30:(i + 1) * 30]))

    file = Header + '\n\n' + "\n".join(lines)
    f = open(filename, "w")
    f.write(file)
    f.close()
   
 
def df_subdiviser(df, Nsub):
    """Divide a dataframe into a list of sub-dataframne

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas Dataframe to divide
    Nsub : int
        Number of subdivision

    Returns
    -------
    list(pandas.DataFrame)
        A list of dataframme subdivisions.
    """
    lensdf = len(df) // Nsub
    r = len(df) % Nsub
    subidx = np.arange(0, len(df), lensdf)
    subidx[:r] += np.arange(r)
    subidx[r:] += r
    sdfs = [df.iloc[i1:i2] for i1, i2 in zip(subidx[:-1], subidx[1:])]
    return sdfs


def host_joiner(survey_fields, host):
    """Use geopandas to match host in survey fields.

    Parameters
    ----------
    survey_fields : geopandas.GeoDataFrame
        Geodataframe describing survey fields
    host : pandas.DataFrame
        Dataframe that contains host informations

    Returns
    -------
    pandas.DataFrame
        Datafrane containing host that are in field with their correspind GROUPID
    """
    # Select host in circle
    host_pos = gpd.GeoDataFrame(
        index=host.index, geometry=gpd.points_from_xy(host.ra.values, host.dec.values)
    )
    grped_host = host_pos.sjoin(survey_fields, how="inner", predicate="intersects")

    # Create grp id
    grped_host = (
        grped_host.index_right.sort_values()
        .groupby(level=0)
        .apply(lambda x: np.random.choice(list(x)))
    )

    # Keep only hosts in fields
    survey_host = host.loc[grped_host.index]
    survey_host["GROUPID"] = grped_host
    return survey_host

def host_resampler(wgt_map_VAR, wgt_map_WGT, index, values, cdf_cut=0.95):
    """Resample host according to a WGTMAP

    Parameters
    ----------
    wgt_map_VAR : numpy.array(float)
        Variable of the WGTMAP
    wgt_map_WGT : numpy.array(float)
        Weight values of the WGTMAP
    index : numpy.array(int)
        Index of hosts
    values : numpy.array(float)
        Values of the Variable for host
    cdf_cut : float, optional
        A cut on the cdf to adjust , by default 0.95

    Returns
    -------
    _type_
        _description_
    """
    count, edges = np.histogram(values, bins="rice")
    medges = (edges[1:] + edges[:-1]) * 0.5

    prob_select = count * np.interp(medges, wgt_map_VAR, wgt_map_WGT)
    cdf = np.cumsum(prob_select)
    cdf /= cdf[-1]

    argmax = np.argmax(prob_select)
    count_max = count[argmax]
    prob_select_max = prob_select[argmax]

    N_to_draw = np.rint(prob_select / prob_select_max * count_max)
    cdf_mask = (cdf > 1.0 - cdf_cut) & (cdf < cdf_cut)
    correction_coeff = np.min(count[cdf_mask] / N_to_draw[cdf_mask])

    N_to_draw = np.rint(N_to_draw * correction_coeff).astype("int")

    wgt_values = np.interp(values, wgt_map_VAR, wgt_map_WGT)

    keep_idx = []
    for i, N in enumerate(N_to_draw):
        edmin, edmax = edges[i], edges[i + 1]
        mask = (values >= edmin) & (values < edmax)

        sdf_index = index[mask]

        if len(sdf_index) < N:
            keep_idx.extend(sdf_index)
        elif N > 0:
            wgt = wgt_values[mask]
            wgt /= np.sum(wgt)
            keep_idx.extend(np.random.choice(sdf_index, size=N, replace=False, p=wgt))
    return keep_idx


def format_poly(poly):
    """Fromat polygon that cross the 2 PI edge.

    Parameters
    ----------
    poly : shapely.polygons
        Polygon that represent fields

    Returns
    -------
    shapely.MultiPolygons
        Polygon that represent fields cutted on 2 PI edge.
    """
    _SPHERE_LIMIT_HIGH_ = shp_geo.LineString(
        [[2 * np.pi, -np.pi / 2], [2 * np.pi, np.pi / 2]]
    )
    polydiv = gpd.GeoSeries(
        [p for p in shp_ops.polygonize(poly.boundary.union(_SPHERE_LIMIT_HIGH_))]
    )
    transl_mask = polydiv.boundary.bounds["maxx"] > 2 * np.pi
    polydiv[transl_mask] = polydiv[transl_mask].translate(-2 * np.pi)
    return shp_geo.MultiPolygon(polydiv.values)