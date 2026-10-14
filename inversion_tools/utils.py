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


def interface_to_mid(x, idim='boundary'):
    '''
    Moves a variable x from interface levels to midpoint levels.
    Assumes that the input dat is defined on a vertical dimension idim. 
    Output is the linear average of each adjacent pair of values, on
    a new dimension called "lev".

    Parameters
    ----------
    x : xarray DataArray
        the input data, with at least on dimension idim
    idim : str, optional
        the vertical dimension name. Default is 'boundary'
    '''
    x = (x + x.shift({idim:-1})) / 2
    x = x.isel({idim:slice(0, -1)})
    x = x.rename({idim:'lev'})
    return x


# -------------------------------------------------------------------------


def hybrid_ps_to_p(ps, A, B, pref=None):
    '''
    Builds gridded pressure given a 2D surface pressure and A and B hybrid 
    coefficients defined on level interfaces. This uses the GEOS-Chem 
    formulation, which is (A + B*PS), rather than the common (A*P0 + B*PS)
    (i.e. A has units of pressure here)

    Parameters
    ----------
    ps : xarray DataArray
        surface pressure in Pa with dimensions of at least 'lat','lon'
    A  : xarray DataArray
        A coefficient (hPa) with dimension 'ilev'
    B : xarray DataArray
        B coefficient (dimensionless) with dimension 'ilev'
    pref : array-like
        optional reference pressure levels corresponding to the 
        midpoint model levels that result. If not supplied, then
        the pressure is returned with a 'lev' dimension with no
        coordinate data. If supplied, then these values are 
        added as the 'lev' coordinates.

    Returns
    -------
    The pressure on the input surface pressure dimensions, with an added vertical
    dimension 'lev', in Pa
    '''

    if(len(A) != len(B)):
        raise RuntimeError('length of A and B must be equal')

    p_edges = (A*100) + B * ps
    p = (p_edges + p_edges.shift(ilev=-1)) / 2
    p = p.isel(ilev=slice(0, -1))
    
    p = p.rename('pressure')
    p = p.rename({'ilev':'lev'})
    p.attrs['comment'] = 'air pressure at level midpoints '\
                         '(average of interface pressures, '\
                         'where each interface pressure is P = A + B*PS)'
    p.attrs.pop('long_name', None) 
    p.attrs.pop('standard_name', None)
    p.attrs.pop('gamap_category', None)

    if(pref is not None):
        p = p.assign_coords(lev=pref)

    return p
    

# -------------------------------------------------------------------------


def align_datasets(x, xv, y, yv, v, vertical='pressure'):
    '''
    Aligns the data x to the grid of data y by interpolating in the horizontal, then
    interpolates both datasets onto a common set of vertical levels.
    In the horizontal, a conservative regrid from the xESMF package is used. 
    In the vertical, a 1D interpolation from MetPy is used to remap both
    datasets from the input levels to constant surfaces of reference variable.
    For the case that the vertical coordinate is 'pressure', then a logarithmic 
    interpolation is used.
    For the case that the vertical coordinate is 'theta' (potential temperature), 
    then a linear interpolation is used. That is, it is assumed that the quantities 
    x and y vary approximately linearly across isentropes. If this assumption does
    not hold (e.g. x and y are a pressure variable), then this function is not 
    appropriate. See:
    https://doi.org/10.1175/1520-0450(1994)033<0694:ITIIEA>2.0.CO;2
    The time dimension, if present, is reduced to the intersection of the time 
    coordiantes of each dataset.

    Parameters
    ----------
    x  : xarray DataArray
        A dataset to horizontally remap, with at least dimensions 'lat', 'lon', 
        and 'lev'
    xv : xarray DataArray
        The gridded variable for the new vertical levels (either pressure or 
        potential temperature) corresponding to dataset x.
    y  : xarray DataArray
        A dataset on the destination horizontal grid, with at least dimensions
        'lat, 'lon', and 'lev'
    yv : xarray DataArray
        The gridded variable for the new vertical levels (either pressure or 
        potential temperature) corresponding to dataset y
    v : array-like
        The desired levels of the destination vertical grid, with units matching
        that of xv and yv
    vertical : str, optional
        The type of vertical coordinate. Currently supports either 'pressure' or
        'theta'. If 'pressure', then xv, yv, and v must be provided in Pa, and a 
        logarithmic interpolation is used. If 'theta', then xv, yv, and v must be
        provided in K, and a linear interpolation is used. That is, it is assumed
        that the quantities x and y vary approximately linearly across isentropes.
    '''

    # Enforce

    # Verify dimensions, transpose to matching order
    if(set(x.dims) != set(y.dims) or set(xv.dims) != set(yv.dims) or \
       set(x.dims) != set(xv.dims)):
        raise RuntimeError('Datasets x, xv, y, and yv must all have the same dimensions')
    x  = x.transpose(*y.dims)
    yv = yv.transpose(*y.dims)
    xv = xv.transpose(*y.dims)
    
    # Verify matching coordinates
    for dim in x.dims:
        if(not arrays_equal(x[dim], xv[dim])):
            raise RuntimeError(f'Coordinates of x and xv must match in dimension {dim}')
        if(not arrays_equal(y[dim], yv[dim])):
            raise RuntimeError(f'Coordinates of y and yv must match in dimension {dim}')
        if(not arrays_equal(x[dim], y[dim])):
            if(dim in ['lat', 'lon', 'lev']):
                continue
            elif(dim != 'time'):
                raise RuntimeError(f'Coordinates of x and y must match in dimension {dim}')
            else:
                warnings.warn('Datasets x and y do not have matching coordiantes in the time'\
                              'time dimension; reindexing each to their intersection.')
                time = x.time.to_index().intersection(y.time.to_index())
                x  = x.reindex(time=time)
                xv = xv.reindex(time=time)
                y  = y.reindex(time=time)
                yv = yv.reindex(time=time)

    # do horizontal interpolation x -> y, xv -> y
    print('doing horizontal regrid (1/2)...')
    x  = ctb.conservative_regrid(y, x)
    print('doing horizontal regrid (2/2)...')
    xv = ctb.conservative_regrid(y, xv)
    
    # do vertical interpolation
    if(vertical == 'pressure'):
        print('doing vertical remap (1/2)...')
        x = ctb.log_interp_to_isobars(x, xv, v)
        print('doing vertical remap (2/2)...')
        y = ctb.log_interp_to_isobars(y, yv, v)
    elif(vertical == 'theta'):
        print('doing vertical remap (1/2)...')
        x = ctb.linear_interp_to_isentropes(x, xv, v)
        print('doing vertical remap (2/2)...')
        y = ctb.linear_interp_to_isentropes(y, yv, v)

    return x, y


# -------------------------------------------------------------------------


def column_average(mf, outfile=None, overwrite=False):
    '''
    Computes the column-averaged dry-air mole fraction of a tracer

    Parameters
    ----------
    mf : xarray DataArray
        input concentration in kg/kg. Must have a vertical dimension 'lev' or 'plev' 
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
    
    try:
        p = mf.lev
        dim = 'lev'
    except AttributeError:
        p = mf.plev
        dim = 'plev'
    
    # approximate interface pressures as logarithmic average
    p_int = np.exp((np.log(p) + np.log(p.shift({dim:-1}))) / 2)
    dp    = p_int - p_int.shift({dim:1})
    
    #dim_idx        = np.where(np.array(mf.dims) == dim)[0][0]
    #shape          = np.ones(mf.ndim)
    #shape[dim_idx] = mf.shape[dim_idx]
    #dp             = dp.reshape(shape.astype(int))
    
    X      = (mf * dp).sum(dim) / dp.sum(dim)
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
