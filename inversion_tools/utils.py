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


def lev_to_p(lev):
    '''
    Converts level index to pressure in hPa

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


def p_to_lev(p):
    '''
    Converts pressure in hPa to level index

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


def lev_to_z(lev):
    '''
    Converts level index to altitude in km

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


def lev_to_p_interfaces(lev):
    '''
    Converts level index to interface pressures in hPa

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


def p_to_p_interfaces(p):
    '''
    Converts level index to interface pressures in hPa

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
        if(overwrite):
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
