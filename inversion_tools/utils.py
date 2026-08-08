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

# ---- imports from this package
from .constants import *

# ---- location of this file, for relative paths
here = os.path.dirname(os.path.abspath(__file__))


# -------------------------------------------------------------------------

def is_yyyy_mm(s):
    '''
    Checks if the input string is a date of the form YYYY-MM
    '''
    return bool(re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", s))

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
    df = pd.read_fwf(f'{here}/../data/levels47.csv',skiprows=3,
                     names=["L", "eta_edge", "eta_mid", "alt_km", "p_hpa"])
    # keep only midpoint rows
    mid = df[df["eta_mid"].notna()].reset_index(drop=True)
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
    df = pd.read_fwf(f'{here}/../data/levels47.csv',skiprows=3,
                     names=["L", "eta_edge", "eta_mid", "alt_km", "p_hpa"])
    # keep only midpoint rows
    mid = df[df["eta_mid"].notna()].reset_index(drop=True)
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
    df = pd.read_fwf(f'{here}/../data/levels47.csv',skiprows=3,
                     names=["L", "eta_edge", "eta_mid", "alt_km", "p_hpa"])
    # keep only midpoint rows
    mid = df[df["eta_mid"].notna()].reset_index(drop=True)
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
    df = pd.read_fwf(f'{here}/../data/levels47.csv',skiprows=3,
                     names=["L", "eta_edge", "eta_mid", "alt_km", "p_hpa"])
    # add interface levels
    mask              = df["L"].isna()
    df.loc[mask, "L"] = df["L"].ffill().loc[mask] - 0.5
    df.loc[0, "L"]    = df.loc[1, "L"] + 0.5
    # get interface pressures
    edge = df[df["eta_edge"].notna()].reset_index(drop=True)
    # lookup functions
    get_p_edge = lambda lev: float(edge.loc[edge["L"] == lev, "p_hpa"].iloc[0])
    # return pressure
    return [get_p_edge(lev-0.5), get_p_edge(lev+0.5)]


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

    df = pd.read_fwf(f'{here}/../data/levels47.csv',skiprows=3,
                     names=["L", "eta_edge", "eta_mid", "alt_km", "p_hpa"])
    # add interface levels
    mask              = df["L"].isna()
    df.loc[mask, "L"] = df["L"].ffill().loc[mask] - 0.5
    df.loc[0, "L"]    = df.loc[1, "L"] + 0.5
    # get interface pressures around input pressure
    edge = np.array(df[df["eta_edge"].notna()].reset_index(drop=True)['p_hpa'])
    if(p < edge[0] or p> edge[-1]):
        raise RuntimeError(f'p must be between {edge[0]} and {edge[-1]} hPa')
    idx  = np.searchsorted(edge, p)
    return [edge[idx-1], edge[idx]]


# -------------------------------------------------------------------------

def align_hybrid_datasets(x, xp, y, yp, p):
    '''
    Aligns the data x to the grid of data y by interpolating in the horizontal, then
    interpolates both datasets onto a common set of isobars.
    In the horizontal, a conservative regrid from the xESMF package is used. 
    In the vertical, a 1D logarithmic interpolation from MetPy is used to remap both
    datasets from a hybrid terrain-following coordinate to isobaric levels. 

    Parameters
    ----------
    x  : xarray DataArray
        A dataset to horizontally remap, with at least dimensions 'lat', 'lon', 
        and 'lev'
    xp : xarray DataArray
        The gridded pressure corresponding to the data x, with matching dimensions
    y  : xarray DataArray
        A dataset on the destination horizontal grid, with at least dimensions
        'lat, 'lon', and 'lev'
    p : array-like
        The desired pressure levels of the destination vertical grid
    '''

    # Verify dimensions, transpose to matching order
    if(set(x.dims) != set(y.dims) or set(xp.dims) != set(yp.dims) or \
       set(x.dims) != set(xp.dims)):
        raise RuntimeError('Datasets x, xp, y, and yp must all have the same dimensions')
    for dim in x.dims:
        if(x[dim] != y[dim] or x[dim] != xp[dim] or x[dim] != yp[dim]):
            raise RuntimeError(f'Dataset coordinates do not match in dimension {dim}')
    x  = x.transpose(*y.dims)
    yp = yp.transpose(*y.dims)
    xp = xp.transpose(*y.dims)

    # do horizontal interpolation x -> y
    x = ctb.align_datasets_horz(y, x)
    
    # do vertical interpolation
    x = ctb.align_datasets_vert(x, xp, p)
    y = ctb.align_datasets_vert(y, yp, p)
    
    return x, y

def align_datasets_horz(x, y):
    return ctb.align_datasets_horz(y, x)

def align_datasets_vert(x, xp, p):
    return ctb.log_interp_to_isobars(x, xp, p)

# -------------------------------------------------------------------------


def column_average(mf, outfile=None, overwrite=False):
    '''
    Computes the column-averaged dry-air mole fraction of a tracer

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
    The level pressure, in km
    '''
    varname = f'{mf.name}_column_average'
    try:
        if(overwrite or outfile is None):
            raise FileNotFoundError
        return xr.open_dataset(outfile)[varname]
    except FileNotFoundError:
        pass

    p  = mf.lev.values
    dp = np.array([np.diff(p_to_p_interfaces(pi)) for pi in p])
    
    dim_idx        = np.where(np.array(mf.dims) == 'lev')[0][0]
    shape          = np.ones(mf.ndim)
    shape[dim_idx] = mf.shape[dim_idx]
    dp             = dp.reshape(shape.astype(int))
    
    X      = (mf * dp).sum('lev') / np.sum(dp, axis=dim_idx)
    X.name = varname
    if(outfile is not None):
        X.to_netcdf(outfile)
    return X


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
    dp = np.array([np.diff(p_to_p_interfaces(pi)) for pi in p])

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
