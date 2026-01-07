# Joe Hollowed
# University of Michigan 2026
#
# Providing a set of utility functions for analyzing flux inversion datasets


# =========================================================================
    
import pdb
import numpy as np
import xarray as xr
from timeit import default_timer

# -------------------------------------------------------------------------

def lev_to_p(lev):
    '''
    Converts level index to pressure in hPa

    Parameters
    ----------
    lev : int
        level index
    '''
    df = pd.read_fwf("levels47.csv",skiprows=3,
                     names=["L", "eta_edge", "eta_mid", "alt_km", "p_hpa"])
    # keep only midpoint rows
    mid = df[df["eta_mid"].notna()].reset_index(drop=True)
    # get pressure
    return float(mid.loc[mid["L"] == lev, "p_hpa"].iloc[0])

# -------------------------------------------------------------------------

def lev_to_z(lev):
    '''
    Converts level index to altitude in km

    Parameters
    ----------
    lev : int
        level index
    '''
    df = pd.read_fwf("levels47.csv",skiprows=3,
                     names=["L", "eta_edge", "eta_mid", "alt_km", "p_hpa"])
    # keep only midpoint rows
    mid = df[df["eta_mid"].notna()].reset_index(drop=True)
    # get pressure
    return float(mid.loc[mid["L"] == lev, "alt_km"].iloc[0])

# -------------------------------------------------------------------------

def rechunk_dataset(dset, outfile, chunking={'time':24},
                    var=None, sel=None, isel=None, quiet=False):
    '''
    Rechunks a netCDF file for specified dimensions and chunk sizes, and writes out the
    result to a new netCDF file.

    Parameters
    ----------
    dset : xarray Dataset or DataArray
        the input dataset
    outfile : str
        output file name
    chunking : dict
        dictionary providing the dimension name-chunk size pairs. Default is {'time':24}, 
        ehich will cause data to be rechunked in the time dimension with a chunk size of
        24. This will create daily chunks for hourly input data
    var : str, optional
        variable(s) to extract from the dataset before processing. Default is None, 
        in which case the entire dataset will be processed.
    sel : dict
        argument to pass to the sel arg of xr.open_dataset. Default is None, in which
        case no argument is passed
    isel : dict
        argument to pass to the isel arg of xr.open_dataset. Default is None, in which
        case no argument is passed
    quiet : bool
        whether or not to suppress print statements from this function
    '''

    data = xr.open_dataset(dset, sel-sel, isel=sel)[var]

    encoding = {}
    for name, var in ds.data_vars.items():
        chunks = []
        for dim in var.dims:
            if dim in list(chunking.keys()):
                chunks.appens(chunking[dim])
            else:
                chunks.append(var.sizes[dim])
        encoding[name] = {'zlib':True, 'cmoplevel':1, chunksizes:tuple(chunks)}
  
    fname = outfile.split('/')[-1]
    print(f'writing out rechunked data to {fname}')
    data.to_netcdf(outfile, format='NETCDF4', encoding=encoding)
