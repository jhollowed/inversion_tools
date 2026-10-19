# Joe Hollowed
# University of Michigan 2026
#
# Providing a set of utility functions for reading and processing meteorological
# datasets associated with the GEOS-Chem and TM5 WOMBAT flux inversions


# =========================================================================

# ---- import dependencies
import os
import pdb
import glob
import timeit
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import urllib.request
from timeit import default_timer
import climate_utils.climate_toolbox as ctb

# ---- imports from this package
from .utils import *
from .constants import *

# =========================================================================


def read_tm5_meteo(start, end, var, overwrite=False, cache=tm5_meteo_cache):
    return read_meteo('TM5', start, end, var, overwrite, cache)

def read_gc_meteo(start, end, var, overwrite=False, cache=gc_meteo_cache):
    return read_meteo('GC', start, end, var, overwrite, cache)

def read_meteo(model, start, end, var, overwrite, cache):
    '''
    Temporary function for reading TM5 (CT2022) or GC (MERRA2) gridded 
    meteorological varaibles

    model : str
        either 'TM5 or 'GC'
    start : np.datetime64
        starting date (year, month, day)
    end : np.datetime64
        ending date (year, month, day)
    var : str
        name of variable to read. Options are:
        'pressure', 'theta'
    cache : str, optional
        location at which to cache intermediate computations
    overwrite : bool, optional
        whether or not to overwrite the existing cached data.
        Defaults to False, in which case the cached data is
        returned if it exists.
    '''

    # ---- check model and var inputs
    valid_models = ['TM5', 'GC']
    if(model not in valid_models):
        raise RuntimeError(f'model must be one of {valid_models}')
    
    tm5_valid_vars = ['pressure', 'temperature', 'theta', 'interface_pressure']
    gc_valid_vars  = ['pressure', 'temperature', 'theta', 'interface_pressure', 'surface_pressure']
    if(model=='GC' and var not in gc_valid_vars):
        raise RuntimeError(f'GC var must be one of {valid_vars}')
    if(model=='TM5' and var not in tm5_valid_vars):
        raise RuntimeError(f'TM5 var must be one of {valid_vars}')

    if(var == 'theta'):
        print(f'{model} potential temperature will be computed from pressure and temp...')
        p = read_meteo(model, start, end, 'pressure', overwrite, cache) / 100 # convert to hPa
        t = read_meteo(model, start, end, 'temperature', overwrite, cache)
        print(f'computing potential temperature...')
        theta = t*(P0/p)**KAPPA
        theta = theta.rename('potential_temperature')
        theta.attrs.pop("long_name", None)
        theta.attrs.pop("standard_name", None)
        theta.attrs['units']   = 'Kelvin'
        theta.attrs['comment'] = 'potential temperature at level midpoints'
        return theta

    print(f'reading {model} {var}...')
    dates     = np.arange(start, end + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    all_years = dates.astype('datetime64[Y]').astype(int) + 1970
    years     = np.unique(all_years)
    all_data  = [0]*len(years)
    
    # TM5 data (CT2022) needs dimensions renamed
    tm5_horz_coords      = {'latitude':'lat', 'longitude':'lon'}
    tm5_coords           = {'latitude':'lat', 'longitude':'lon', 'level':'lev'}
    tm5_interface_coords = {'latitude':'lat', 'longitude':'lon', 'boundary':'ilev'}

    # ------ First get data per requested year
    # For TM5, this means downloading files from the NOAA server. For GC, we read
    # from disc on Herules.
    # The strategy is to compute and cache the requested data for whole years at time,
    # and only trim to the requested date range (start, end) as a final step.
    # This allows us to cache variables in one-year intervals
    for i,year in enumerate(years):

        if(model == 'TM5'):
            cached = f'{cache}/TM5_CT2022_{year}_daily_{var}.nc'
        elif(model == 'GC'):
            cached = f'{cache}/GC_MERRA2_{year}_daily_{var}.nc'

        try:
            if(overwrite): raise FileNotFoundError
            year_data = xr.open_dataset(cached)[var]
            print(f'read {model} daily {var} data for {year}')

        except FileNotFoundError:
            print(f'daily {model} {var} file for {year} not found')
            year_data  = []
            year_dates = dates = np.arange(np.datetime64(f"{year}-01-01"),
                                           np.datetime64(f"{year + 1}-01-01"),
                                           np.timedelta64(1, "D"))

            # loop through all dates in this year; download/read, average, concat
            for j,date in enumerate(year_dates):
                prog   = f'({j+1}/{len(year_dates)})'
                iprint(f'{prog} processing date {date}')

                if(model == 'TM5'):
                    date_str   = str(date.astype('datetime64[D]'))
                    file       = f'CT2022.molefrac_glb3x2_{date_str}.nc'
                    downloaded = f'{cache}/daily_raw/{file}'
                    try:
                        xr.open_dataset(downloaded)
                    except FileNotFoundError:
                        # download from NOAA, trim variables, cache
                        iprint(f'{prog} downloading data from NOAA...')
                        urllib.request.urlretrieve(f'{tm5_meteo_url}/{file}', downloaded)
                        vars_to_save = ['temperature' if v == 'theta' else v for v in valid_vars]
                        iprint(f'{prog} caching select variables...')
                        downloaded_vars = xr.open_dataset(downloaded)[vars_to_save]
                        os.remove(downloaded)
                        downloaded_vars.to_netcdf(downloaded)
                
                elif(model == 'GC'):
                    _, month, day = str(date).split("-")
                    file = f'{gc_meteo_dir}/{year}/{month}/MERRA2.{year}{month}{day}.I3.2x25.nc4'

                # grab or compute the requested variable
                # ------ temperature ------
                if(var == 'temperature'):
                    iprint(f'{prog} reading temperature...')
                    
                    if(model == 'TM5'):
                        t = xr.open_dataset(downloaded)['temperature']
                        t = t.rename(tm5_coords)
                    
                    elif(model == 'GC'):
                        t = xr.open_dataset(file)['T']
                    
                    t = t.resample(time="1D").mean()
                    t = t.rename('temperature')
                    t.attrs['comment'] = 'temperature at level midpoints'
                    year_data.append(t)

                # ------ pressure ------
                elif(var=='pressure' or var=='interface_pressure' or var=='surface_pressure'):
                    
                    if(model == 'TM5'):
                        iprint(f'{prog} reading pressure...')
                        p = xr.open_dataset(downloaded)[var]
                        p = p.resample(time="1D").mean()
                        p = p.rename(tm5_interface_coords)
                        if(var != 'interface_pressure'):
                            # move pressure from interface to midpoint levels
                            p = ctb.level_interfaces_to_midpoints(p)

                    elif(model == 'GC'):
                        iprint(f'{prog} reading surface pressure...')
                        # first get daily average of surface presssure
                        ps = xr.open_dataset(file)['PS']
                        ps = ps.resample(time="1D").mean()
                        
                        # then convert to gridded pressure using the hybrid coefficients A,B
                        if(var != 'surface_pressure'):
                            iprint(f'{prog} building 3D pressure...')
                            if(var == 'interface_pressure'): move_p = False
                            else:                            move_p = True
                            p = ctb.compute_hybrid_pressure(ps, GC_HYAI, GC_HYBI, p0=P0*100,
                                                            move_to_midpoints=move_p)
                        # remove extraneous attributes
                        p.attrs.pop("standard_name", None)
                        p.attrs.pop('gamap_category', None)

                    year_data.append(p)

            # ------ write out this year to nc
            print('writing out...')
            year_data = xr.concat(year_data, dim='time')
            year_data.to_netcdf(cached)
        all_data[i] = year_data

    # ------ concatenate and trim all years of data
    print(f'concatenating and trimming data to '\
          f'({start.astype("datetime64[D]")}, {end.astype("datetime64[D]")})...')
    if(len(all_data)>1):
        all_data = xr.concat([d.sel(time=slice(start, end)) for d in all_data], dim='time')
    else:
        all_data = all_data[0].sel(time=slice(start, end))
    return all_data


