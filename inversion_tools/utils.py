# Joe Hollowed
# University of Michigan 2026
#
# Providing a set of utility functions for analyzing flux inversion datasets


# =========================================================================


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


def rechunk_dataset(dset, var, outfile, chunking={'time':24},
                    sel=None, isel=None):
    '''
    Rechunks a netCDF file for specified dimensions and chunk sizes, and writes out the
    result to a new netCDF file.

    Parameters
    ----------
    dset : xarray Dataset or DataArray
        the input dataset
    var : str
        variable to extract from the dataset before processing.
    outfile : str
        output file name
    chunking : dict
        dictionary providing the dimension name-chunk size pairs. Default is {'time':24}, 
        ehich will cause data to be rechunked in the time dimension with a chunk size of
        24. This will create daily chunks for hourly input data
    sel : dict
        argument to pass to the sel arg of xr.open_dataset. Default is None, in which
        case no argument is passed
    isel : dict
        argument to pass to the isel arg of xr.open_dataset. Default is None, in which
        case no argument is passed
    '''

    data = xr.open_dataset(dset)[var]
    if(sel is not None): data = data.sel(sel)
    if(isel is not None): data = data.isel(isel)

    encoding = {}
    chunks = []
    for dim in data.dims:
        if dim in list(chunking.keys()):
            chunks.append(min(chunking[dim], data.sizes[dim]))
        else:
            chunks.append(data.sizes[dim])
    encoding[data.name] = {'zlib':True, 'complevel':1, 'chunksizes':tuple(chunks)}
 
    fname = outfile.split('/')[-1]
    data.to_netcdf(outfile, format='NETCDF4', encoding=encoding)


# -------------------------------------------------------------------------


def find_split_containing_date(date, kind, debug=False):
    '''
    Finds the split which contains the input date

    Parameters
    ----------
    date : str
        the date, in 'YYYY-MM-DD' format
    kind : str
        the kind of dataset to use for searching the split information. Can be either
        'control, 'climatology', or 'residual'. The latter two refer to the non-control
        runs.
    debug : bool, optional
        flag to turn on verbose output for debugging

    Returns
    -------
    the split containing the input date, as an integer
    '''
    date = datetime.strptime(date, '%Y-%m-%d').date()
    mapping = _get_split_date_mapping(control=(kind =='control'))

    for i in range(len(mapping)):
        if(mapping[i][0] != kind): continue
        start = datetime.strptime(mapping[i][2], '%Y-%m-%d').date()
        end   = datetime.strptime(mapping[i][3], '%Y-%m-%d').date()
        if(debug): print(f'{start} <= {date} <= {end } = {start <= date <= end}')
        if start <= date < end:
            return int(mapping[i][1])

    start, end = mapping[0][2], mapping[-1][3]
    raise RuntimeError(f'No split found containing date {date} for kind={kind}! '\
                       f'Valid dates range from {start} to {end}')


# -------------------------------------------------------------------------


def _get_split_date_mapping(control=False, overwrite=False):
    '''
    Builds a mapping between split integer and left,right bounding dates, and saves
    the result to file.

    Parameters
    ----------
    control : bool, optional
        whether or not to build the mapping for the control runs. If False, then build the mapping
        for the climatological and residual non-control runs
    overwrite : bool, optional
        whether or not toverwrite existing mapping files. If False (the default), then the mapping
        on-file is read and returned if existing
    '''
    
    if control:
        top_dir   = gc_transport_control_dir
        mapping_file = f'{here}/../data/split_mapping_control.csv'
        kinds     = ['control']
    else:       
        top_dir  = gc_transport_dir 
        mapping_file = f'{here}/../data/split_mapping.csv'
        kinds    = ['climatology', 'residual']
        
    # read file if exists; generate if not
    try:
        if(overwrite): raise FileNotFoundError
        mapping = np.loadtxt(mapping_file, delimiter=',', skiprows=1, dtype=str)
    except FileNotFoundError:
        print('split-date mapping file not found; generating')
        # define table header
        mapping = [['kind', 'split', 'left_date', 'right_date']]
        for kind in kinds:

            # extract the splits from the name of each subdir at the data location
            split_dirs = sorted(glob.glob(f'{top_dir}/{kind}*part001**split[0-9][0-9]'))
            splits   = [int(sdir.split('_split')[-1]) for sdir in split_dirs]
            
            # take only the first instance of each split (assume that a unique split integer
            # corresponds to a unique time period, for this choice of 'kind')
            mask       = np.sort(np.unique(splits, return_index=True)[1])
            split_dirs = np.array(split_dirs)[mask]
            splits     = np.array(splits)[mask]

            for i,split in enumerate(splits):
                all_data = sorted(glob.glob(f'{split_dirs[i]}/OutputDir/GEOSChem.Species*.nc4'))
                left  = xr.open_dataset(all_data[0]).attrs['simulation_start_date_and_time']
                left  = left.split()[0].strip('-')
                right = xr.open_dataset(all_data[-1]).attrs['simulation_end_date_and_time']
                right = right.split()[0].strip('-')
                mapping.append([kind, int(split), left, right])

        np.savetxt(mapping_file, np.array(mapping), delimiter=',', fmt='%s')
        mapping = np.loadtxt(mapping_file, delimiter=',', skiprows=1, dtype=str)
        print(f'done; saved mapping file to {mapping_file}')
    return mapping
