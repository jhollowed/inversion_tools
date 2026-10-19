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
from .meteo import *
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


def arrays_equal(x, y):
    '''
    Checks for equality of xarray DataArrays
    '''
    sizes_equal  = x.size == y.size
    arrays_equal = np.array_equal(x.values, y.values, equal_nan=True)
    return (sizes_equal and arrays_equal)


# -------------------------------------------------------------------------
    

def check_inputs(model=None, obs=None, source=None, component=None, region=None, pft=None):
    '''
    Verifies that one or more input arguments is a valid choice for WOMBAT basis functions.
    '''

    if(model is not None and model not in valid_models):
        raise ValueError(f'model must be one of {valid_models}, not {model}')
    if(obs is not None and obs not in valid_obs):
        raise ValueError(f'obs must be one of {valid_obs}, not {obs}')
    if(source is not None and source not in valid_sources):
        raise ValueError(f'source must be one of {valid_sources}, not {source}')
    if(component is not None and component not in valid_components):
        raise ValueError(f'component must be one of {valid_components}, not {component}')
    if(region is not None and region not in valid_regions):
        raise ValueError(f'region must be one of {valid_regions}, not {region}')
    if(pft is not None and pft not in valid_pfts):
        raise ValueError(f'pft must be one of {valid_pfts}, not {pft}')


# -------------------------------------------------------------------------


def get_data_dir(model, source, component, region=None, emission_date=None, pft=None):
    '''
    Gets the data direcotry name for a basis function

    Parameters
    ----------
    model : str
        The transport model to search. Either 'GC' or 'TM5'
    source : str
        The emission source to search. Eiter 'ocean', 'gpp', or 'resp'
    component : str
        The component ot search. Must be one of:
        'intercept', 'trend', 'residual', 'sin12_1', 'sin12_2', 'sin12_3', 
        'cos12_1', 'cos12_2', 'cos12_3'
    region : int, optional
        The emission region to search. Defautls to None, which specifies that
        the control runs should be searched.
    emission_date : str, optional
        The emission date in YYYY-MM format, in the case that component='residual'
    pft : int, optional
        The PFT as an int between 1 and 15. 

    Returns
    -------
    str
        The data directory name
    '''

    # verify inputs
    check_inputs(model=model, source=source, component=component, region=region, pft=pft)
    if(component == 'residual'):
        if(emission_date is None):
            raise ValueError('emission_date must be supplied when component="residual"')
        if(not is_yyyy_mm(emission_date)):
            raise ValueError('emission_date must be supplied in YYYY-MM format')

    # no PFTs for ocean sources
    if(source == 'ocean' and pft is not None):
        raise ValueError('PFT must not be supplied when source=ocean')

    # build the data directory name
    comp_names = {True:control_shorthand_comps, False: shorthand_comps}[region is None]    
    source_str = comp_names[source]

    sfx = ''
    if(pft is not None):    sfx = f'_pft{pft:02d}'
    if(region is not None): sfx = f'{sfx}_regionRegion{region:02d}'
    if(component == 'residual'):
        sfx = f'{sfx}_month{emission_date}'
        if(region is None):
            raise RuntimeError('region must be supplied if component=residual')
    dirname = f'{source_str}_{component}{sfx}'
    
    # get the data location
    datadir = {'GC':gc_transport_dir, 'TM5':tm5_transport_dir}[model]

    # return
    return f'{datadir}/{dirname}'


# -------------------------------------------------------------------------


def get_available_pfts(model, source, component, region=None, emission_date=None):
    '''
    Returns a list of available pfts for a choice of model, source, component, and region.
    Parameters are identical to get_data_dir(); see docstrings therein.

    Returns

    '''

    # no PFTs for ocean emission sources
    if(source == 'ocean'):
        return []
    
    # get data directory name and location
    datapath  = get_data_dir(**locals(), pft=1) # use PFT 1 as a placeholder
    
    # find available pfts
    alldirs    = datapath.replace('_pft01', '_pft*')
    pfts_avail = sorted(glob.glob(alldirs))
    pfts_avail = [int(s.split('_pft')[-1].split('_')[0]) for s in pfts_avail]

    return pfts_avail


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
    p = read_tm5_meteo(np.datetime64('2015-01-01'), np.datetime64('2015-12-30'), 'pressure', overwrite=False)
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


def get_posterior_alpha(model, obs, sample, source, component, region, 
                        pft=None, emission_date=None):
    '''
    Extracts the WOMBAT posterior alpha scale factor for a choice of observations
    and basis function, and returns either a single sample, the mean, the mean and 
    standard deviation, or the full set of 1000 samples.

    Parameters
    ----------
    model : str
        Which transport model to read the posterior scale factor for.
        Must be either 'GC' or 'TM5'.
    obs : str
       Which observational optimization to read the posterior scale factor from.
       Must be either 'IS', 'LNLG', 'LNLGIS', or 'LNLGOGIS'.
    sample : int or str
        If an integer, specifies the sample to return between 1 and 1000.
        If a string, the options are:
        - 'mean': returns the mean scale factor
        - 'all' : returns the full set of 1000 samples as a list
    source : str
        The emission source associated with the basis function.
        Must be either "gpp", "resp", or "ocean"
    component : str
        The basis function component. Must be either
        "intercept", "trend", "residual", "sin12_1", "sin12_2", "sin12_3",
        "cos12_1", "cos12_2", or "cos12_3"
    region : int
        The TransCom region associated with the basis function. Must be an integer 
        between 1 and 22.
    pft : int, optional
        The PFT associated with the basis function, in the case that source = 'resp' or 'gpp'
    emission_date : str, optional
        The emission date in YYYY-MM format. Required if component = 'residual', ignored
        otherwise.

    Returns
    -------
    float or list of floats
        The scale factor sample, mean, or distribution, according to the arg "sample".
    '''
    
    # ------ check inputs
    check_inputs(model, obs, source, component, region, pft)

    if(component=='residual' and emission_date is None):
        raise ValueError('emission_date must be provided when component="residual"')
    if(component!='residual' and emission_date is not None):
        raise ValueError('emission_date should not be provided when component="residual"')

    if(type(sample) == int):
        if(sample < 1 or sample > 1000):
            raise ValueError('alpha sample must be between 1 and 1000')
    elif(type(sample) == str):
        if(sample not in ['mean', 'all']):
            raise ValueError('sample option not recognized; must be "mean" or "all" and an integer')

    # ------ get basis function name
    pft_str  = ''
    date_str = ''
    if(pft is not None):           pft_str = f'_pft{pft:02d}'
    if(emission_date is not None): date_str = f'_month{emission_date}'
    source_str     = shorthand_comps[source]
    basis_function = f'{source_str}_{component}{pft_str}_regionRegion{region:02d}{date_str}'

    # ------ get row from alpha table
    if(model == 'GC'): alpha_file  = f'{gc_alpha_dir}/state_gc_{obs}.csv'
    if(model == 'TM5'): alpha_file = f'{tm5_alpha_dir}/state_tm5_{obs}.csv'
    alpha_table = pd.read_csv(alpha_file)
    alpha_row   = alpha_table[alpha_table['basis_function'] == basis_function]

    # ------- extract requested sample and return
    if(type(sample) == int): sample_str = f'sample{sample:04d}'
    else:                    sample_str = sample
 
    if(type(sample) == int):
        return alpha_row[f'sample{sample:04d}'].iloc[0]
    elif(sample == 'mean'):
        return alpha_row[sample].iloc[0]
    elif(sample == 'all'):
        return xr.DataArray(np.array(alpha_row.iloc[0][2:], dtype='float'), 
                            coords={'sample':np.arange(1,1001)})
 

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
