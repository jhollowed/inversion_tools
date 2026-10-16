# Joe Hollowed
# University of Michigan 2026
#
# Providing a set of utility functions for analyzing flux inversion datasets


# =========================================================================

import re
import os
import pdb
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from timeit import default_timer
import climate_utils.climate_toolbox as ctb

from IPython import get_ipython
from IPython.display import display, Pretty
_display_handle = None

# ---- imports from this package
from .constants import *
#from .data_io import *

# ---- location of this file, for relative paths
here = os.path.dirname(os.path.abspath(__file__))


# -------------------------------------------------------------------------

def is_yyyy_mm(s):
    '''
    Checks if the input string is a date of the form YYYY-MM
    '''
    return bool(re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", s))


# -------------------------------------------------------------------------


def arrays_equal(x, y):
    '''
    Checks for equality of xarray DataArrays
    '''
    sizes_equal  = x.size == y.size
    arrays_equal = np.array_equal(x.values, y.values, equal_nan=True)
    return (sizes_equal and arrays_equal)


# -------------------------------------------------------------------------


def tm5_lev_to_p(lev):
    '''
    Converts a TM5 level index to pressure in hPa. This does not come from the true
    vertical grid definition, but rather take the mean midlatitude pressure across time 
    for the underlying 3D pressure field (ERA5) at the specified level

    Parameters
    ----------
    lev : int
        level index

    Returns
    -------
    The level pressure, in hPa
    '''
    # get 3D field
    p = read_meteo('TM5', np.datetime64('2015-01-01'), np.datetime64('2015-12-30'), 'pressure', 
                   overwrite=False, cache='/work/noaa/co2/jhollo/TM5_CT2022_gridded_vars')
    # get average
    p = p.sel(lat=45, method='nearest').mean(('time', 'lon')).sel(lev=lev).values
    return p/100


# -------------------------------------------------------------------------


def gc_lev_to_p(lev):
    '''
    Converts a GEOS-Chem level index to pressure in hPa
    for the 47-level grid defined at
    https://geos-chem.readthedocs.io/en/latest/supplemental-guides/vertical-grids.html

    Parameters
    ----------
    lev : int
        level index

    Returns
    -------
    The level pressure, in hPa
    '''
    # keep only midpoint rows
    mid = GC_VERT[GC_VERT["eta_mid"].notna()].reset_index(drop=True)
    # get pressure
    return float(mid.loc[mid["L"] == lev, "p_hpa"].iloc[0])


# -------------------------------------------------------------------------


def gc_p_to_lev(p):
    '''
    Converts a pressure in hPa to the nearest GEOS-Chem level index 
    for the 47-level grid defined at
    https://geos-chem.readthedocs.io/en/latest/supplemental-guides/vertical-grids.html

    Parameters
    ----------
    p : float
        pressure in hPa

    Returns
    -------
    The level index
    '''
    # keep only midpoint rows
    mid = GC_VERT[GC_VERT["eta_mid"].notna()].reset_index(drop=True)
    # get pressure
    return float(mid.loc[np.isclose(mid["p_hpa"], p, rtol=0.013), "L"].iloc[0])


# -------------------------------------------------------------------------


def gc_lev_to_z(lev):
    '''
    Converts level index to altitude in km
    for the 47-level grid defined at
    https://geos-chem.readthedocs.io/en/latest/supplemental-guides/vertical-grids.html

    Parameters
    ----------
    lev : int
        level index

    Returns
    -------
    The level pressure, in km
    '''
    # keep only midpoint rows
    mid = GC_VERT[GC_VERT["eta_mid"].notna()].reset_index(drop=True)
    # get pressure
    return float(mid.loc[mid["L"] == lev, "alt_km"].iloc[0])


# -------------------------------------------------------------------------


def gc_lev_to_p_interfaces(lev):
    '''
    Converts level index to interface pressures in hPa
    for the 47-level grid defined at
    https://geos-chem.readthedocs.io/en/latest/supplemental-guides/vertical-grids.html

    Parameters
    ----------
    lev : int
        level index

    Returns
    -------
    The interface pressures as [p(k-1/2), p(k+1/2)], in hPa
    '''
    if(lev > 47 or lev < 1):
        raise RuntimeError('lev must be between 0 and 47')
    # add interface levels
    mask                   = GC_VERT["L"].isna()
    GC_VERT.loc[mask, "L"] = GC_VERT["L"].ffill().loc[mask] - 0.5
    GC_VERT.loc[0, "L"]    = GC_VERT.loc[1, "L"] + 0.5
    # get interface pressures
    edge = GC_VERT[GC_VERT["eta_edge"].notna()].reset_index(drop=True)
    # lookup functions
    get_p_edge = lambda lev: float(edge.loc[edge["L"] == lev, "p_hpa"].iloc[0])
    # return pressure
 

 # -------------------------------------------------------------------------


def gc_p_to_p_interfaces(p):
    '''
    Converts pressure in hPa to interface pressures in hPa
    for the 47-level grid defined at
    https://geos-chem.readthedocs.io/en/latest/supplemental-guides/vertical-grids.html

    Parameters
    ----------
    lev : int
        level index

    Returns
    -------
    The interface pressures as [p(k-1/2), p(k+1/2)], in hPa
    '''
    
    p *= 1000 # temporary, dunno why GC pressure data is in Pa/10

    # add interface levels
    mask                   = GC_VERT["L"].isna()
    GC_VERT.loc[mask, "L"] = GC_VERT["L"].ffill().loc[mask] - 0.5
    GC_VERT.loc[0, "L"]    = GC_VERT.loc[1, "L"] + 0.5
    # get interface pressures around input pressure
    edge = np.array(GC_VERT[GC_VERT["eta_edge"].notna()].reset_index(drop=True)['p_hpa'])
    if(p < edge[0] or p> edge[-1]):
        raise RuntimeError(f'p must be between {edge[0]} and {edge[-1]} hPa')
    idx  = np.searchsorted(edge, p)
    return [edge[idx-1], edge[idx]]
 

# -------------------------------------------------------------------------


def column_center_of_mass(mf, outfile=None, overwrite=False):
    '''
    Computes the pressure-weighted vertical center of mass (km)
    of a tracer defined on pressure levels, assuming hydrostatic balance

    Parameters
    ----------
    mf : xarray DataArray
        input concentration in kg/kg. Must have a vertical dimension 'lev' 
    outfile : str
        path to file at which to save the result. If file already exists, then read from 
        the file instead of computing the column average from scratch.
    overwrite : bool
        whether or not to overwrite the file specified at outfile, if exists

    Returns
    -------
    xarray DataArray of center-of-mass altitude (km)
    '''
    
    varname = f'{mf.name}_vertical_com'
    try:
        if overwrite:
            raise FileNotFoundError
        return xr.open_dataset(outfile)[varname]
    except FileNotFoundError:
        pass

    # --- pressure thickness (same as column_average)
    p  = mf.lev.values
    dp = np.array([np.diff(gc_p_to_p_interfaces(pi)) for pi in p])

    dim_idx = np.where(np.array(mf.dims) == 'lev')[0][0]
    shape          = np.ones(mf.ndim)
    shape[dim_idx] = mf.shape[dim_idx]
    dp = dp.reshape(shape.astype(int))

    # --- get altitude for each level
    z = np.array([lev_to_z(p_to_lev(pi*1000)) for pi in p])
    shape_z          = np.ones(mf.ndim)
    shape_z[dim_idx] = len(z)
    z = z.reshape(shape_z.astype(int))

    # --- compute center of mass
    z_com      = (mf * z * dp).sum('lev') / (mf * dp).sum('lev')
    z_com.name = varname

    if outfile is not None:
        z_com.to_netcdf(outfile)
    return z_com


# -------------------------------------------------------------------------


def iprint(text):
    '''
    overwrites the previous printed line with text. Works for either Jupyter
    notebooks or temrinal
    '''

    global _display_handle

    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            if _display_handle is None:
                _display_handle = display(Pretty(text), display_id=True)
            else:
                _display_handle.update(Pretty(text))
        else:
            print(f"\033[1A\033[2K{text}")
    except ImportError:
        print(f"\033[1A\033[2K{text}")
