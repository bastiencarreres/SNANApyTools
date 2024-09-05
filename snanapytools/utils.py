import numpy as np
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