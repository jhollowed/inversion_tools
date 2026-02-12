# Joe Hollowed
# University of Michigan 2026
#
# Providing a set of utility functions for analyzing flux inversion datasets


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
from timeit import default_timer

# ---- imports from this package
from .utils import *
from .constants import *

# -------------------------------------------------------------------------

def read_transport_jacobians_control(source, start_date, end_date, 
                                     lev=None, lat=None, lon=None, pft=None,
                                     rechunk=True, processing_dir=None, substr=None, resample='1D', 
                                     return_flux=True, return_mf=True, quiet=False):
    '''
    Reads, resamples, and returns mole fraction and flux data for a chosen emission source, 
    either globally, for a single latitude, a single longitude, or a single lat/lon position.
    For all arguments not described here, see the docstring of _get_mf_and_flux_for_split()

    Parameters
    ----------
    source : str
        either 'ocean', 'gpp', or 'resp'
    start_date : str
        start of time period over which to read the data, in the format 'YYYY-MM-DD'
    end_date : str
        end of time period over which to read the data, in the format 'YYYY-MM-DD'
    
    Returns
    -------
    If both return_flux and return_mf, a tuple like (mole fraction data, flux data)
    Else, a single return of either the mole fraction or flux data
    In either case, data formats are dictionaries of xarray DataArrays, with the dictionary
    keys giving the WOMBAT components
    '''

    start_split = find_split_containing_date(start_date, 'control')
    end_split   = find_split_containing_date(end_date, 'control')
    splits = np.arange(start_split, end_split)
    print(f'reading data for splits {splits}')
    
    data = [0] * len(splits)
    for i,split in enumerate(splits):
        print(f'---------- split {split} ----------')
        data[i] = _get_mf_and_flux_for_split(source, split=split, lev=lev, lat=lat, lon=lon, pft=pft, 
                                             rechunk=rechunk, processing_dir=processing_dir, substr=substr, 
                                             resample=resample, return_flux=return_flux, return_mf=return_mf, 
                                             quiet=quiet)
    if return_flux and return_mf:
        return xr.concat(data.T[0], dim='time'), xr.concat(data.T[1], dim='time')
    else:
        return xr.concat(data, dim='time')


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


def read_transport_jacobians_climo(source, start_date, end_date, region,
                                   lev=None, lat=None, lon=None, pft=None,
                                   rechunk=True, processing_dir=None, substr=None, resample='1D', 
                                   return_flux=True, return_mf=True, quiet=False):
    '''
    Reads, temporally resamples, and returns mole fraction and/or flux data for a chosen emission source at 
    specified position, or globally. For all arguments not described below, see the docstring of 
    read_transport_jacobians_control()

    Parameters
    ----------
    region : int, optional
        an integer from 0 to 22 giving the emission region
    '''
    
    start_split = find_split_containing_date(start_date, 'climatology')
    end_split   = find_split_containing_date(end_date, 'climatology')
    splits = np.arange(start_split, end_split)
    print(f'reading data for splits {splits}')

    data = [0] * len(splits)
    for i,split in enumerate(splits):
        print(f'---------- split {split} ----------')
        data[i] = _get_mf_and_flux_for_split(source, split=split, region=region, 
                                          lev=lev, lat=lat, lon=lon, pft=pft, 
                                          rechunk=rechunk, processing_dir=processing_dir, substr=substr, 
                                          resample=resample, return_flux=return_flux, return_mf=return_mf, 
                                          quiet=quiet)
    if return_flux and return_mf:
        return xr.concat(data.T[0], dim='time'), xr.concat(data.T[1], dim='time')
    else:
        return xr.concat(data, dim='time')


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


def read_transport_jacobians_residual(source, start_date, end_date, region, month,
                                      lev=None, lat=None, lon=None, pft=None,
                                      rechunk=True, processing_dir=None, substr=None, resample='1D', 
                                      return_flux=True, return_mf=True, quiet=False):
    '''
    Reads, temporally resamples, and returns mole fraction and/or flux data for a chosen emission source at 
    specified position, or globally. For all arguments not described below, see the docstring of 
    read_transport_jacobians_climo()

    Parameters
    ----------
    month : string, optional
        a year-month pair in the format 'YYYY-MM', specifying a tracer species by emission time.
    '''
    
    start_split = find_split_containing_date(start_date, 'residual')
    end_split   = find_split_containing_date(end_date, 'residual')
    splits = np.arange(start_split, end_split)
    print(f'reading data for splits {splits}')

    data = [0] * len(splits)
    for i,split in enumerate(splits):
        print(f'---------- split {split} ----------')
        data[i] = _get_mf_and_flux_for_split(source, split=split, region=region, month=month, 
                                          lev=lev, lat=lat, lon=lon, pft=pft, 
                                          rechunk=rechunk, processing_dir=processing_dir, substr=substr, 
                                          resample=resample, return_flux=return_flux, return_mf=return_mf, 
                                          quiet=quiet)
    if return_flux and return_mf:
        return xr.concat(data.T[0], dim='time'), xr.concat(data.T[1], dim='time')
    else:
        return xr.concat(data, dim='time')


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


def _get_mf_and_flux_for_split(source, split, 
                              lev=None, lat=None, lon=None, pft=None, region=None, month=None, 
                              rechunk=True, processing_dir=None, substr=None, resample='1D', 
                              return_flux=True, return_mf=True, quiet=False):
    '''
    Reads, resamples, and returns mole fraction and flux data for a chosen emission source, 
    either globally, for a single latitude, a single longitude, or a single lat/lon position

    Parameters
    ----------
    source : str
        either 'ocean', 'gpp', or 'resp'
    split : int
        an integer giving the time split number
    lev : int, optional
        vertical level. If not provided, then all the full vertical domain is returned
    lat : float, optional
        latitude to extract data across time
    lon : float, optional
        longitude to extract data across time
    pft : int, optional
        an integer from 1 to 15 giving the plant functional type. If source is not gpp or resp, 
        this is ignored. If source is gpp or resp and pft is not supplied, then all pft's will
        be read and summed. Else, the specified pft is returned.
    region : int, optional
        an integer from 0 to 22 giving the emission region. If not supplied, then read from
        the control runs. If supplied, then either read from the climatology or residual files
        depending on the presence of the month arg.
    month : string, optional
        a year-month pair in the format 'YYYY-MM'. This gives the emission time of the tracers, 
        in the case that the region arg is supplied. If both region and month are supplied, then
        only the residual is returned.
    rechunk : bool
        whether or not to rechunk the original data files for more efficient reading.
        Defaults to True, in which case disk space will be used by intermediate processed
        data files (unless they already exist). Data is re-chunked in the time dimension 
        only, into 30-day chunks.
    processing_dir : str, optional
        path to location at which to write out processed (rechunked) data, if rechunk=True
    substr : str, optional
        optional substring to grep on when searching for data files. Any files not containing this 
        substring will be ignored
    resample : str, optional
        a valid time string to pass to xarray.resample. Default is '1D', which causes the returned data
        to be resampled to a daily frequency
    return_flux : bool, optional
        whether or not to read and return flux data
    return_mf : bool, optional
        whether or not to read and return mole fraction data
    quiet : bool, optional
        whether or not to silence progress print statements

    Returns
    -------
    If both return_flux and return_mf, a tuple like (mole fraction data, flux data)
    Else, a single return of either the mole fraction or flux data
    In either case, data formats are dictionaries of xarray DataArrays, with the dictionary
    keys giving the WOMBAT components
    '''

    # configure file naming conventions
    source_in = source
    if(source == 'gpp' or source == 'resp'):
        if(region is None):     pfx = 'sib4_'
        else:                   pfx = ''
        if(pft is not None):    sfx = f'_pft{pft:02d}'
        else:                   sfx = 'allpft'
        if(source == 'resp'):
            if(region is None): source = 'resp_tot'
            else:               source = 'bio_resp_tot'
        if(source == 'gpp'):
            if(region is None): source = 'gpp'
            else:               source = 'bio_gpp'
    elif(source == 'ocean'): 
        if(region is None and month is None): 
            source = 'ocean_lschulz'
        pfx, sfx = '', ''
    if(region is not None):
        sfx = f'{sfx}_regionRegion{region:02d}'
    if(month is not None):
        assert region is not None, 'region must be supplied if month is supplied'
        sfx = f'{sfx}_month{month}'
    if(region is None and month is None):
        pfx = f'control_{pfx}'

    # identify data directory
    if(region is None):
        top_dir = gc_transport_control_dir
    else:
        top_dir = gc_transport_dir

    # sanity check
    if(not return_flux and not return_mf):
        raise RuntimeError('At least one of return_flux and return_mf must be True')

    # if a pft was not specified, but the source is gpp or resp, then call this 
    # function recursively for each pft, summing the result for final return
    if('allpft' in sfx):
        for i in range(15):
            print(f'\n\n========== PFT {i+1}/15 ==========')
            args = {'source':source_in, 'split':split, 
                    'lev':lev, 'lat':lat, 'lon':lon, 'pft':i+1, 'region':region, 'month':month,
                    'rechunk':rechunk, 'processing_dir':processing_dir, 'substr':substr, 
                    'resample':resample, 'return_flux':return_flux, 'return_mf':return_mf, 
                    'quiet':quiet}
            if(i==0):
                comp_data = _get_mf_and_flux_for_split(**args)
                if(return_flux and return_mf): 
                    comp_mf_data, comp_flux_data = comp_data[0], comp_data[1]
            else:
                comp_datai= _get_mf_and_flux_for_split(**args)
                if(return_flux and return_mf):
                    comp_mf_data   += comp_datai[0]
                    comp_flux_data += comp_datai[1]
                else:
                    comp_data += comp_datai
            
        if(return_flux and return_mf): return comp_mf_data, comp_flux_data
        elif(return_flux):             return comp_data
        elif(rturn_mf):                return comp_data


    # get mapping file
    mapping = pd.read_csv(f'{top_dir}/mapping.csv')
    
    # lambda function for formatting file names per input component
    fname = lambda component: f'{pfx}{source}_{component}{sfx}'
    
    # make dict of component labels -> component names
    if('ocean' in source):
        comp_labels = ['intercept', 'trend', 'sin1', 'sin2', 'cos1', 'cos2', 'residual']
        comp_names  = dict(zip(comp_labels, ['intercept', 'trend', 'sin12_1', 'sin12_2', 
                                             'cos12_1', 'cos12_2', 'residual']))
    else:
        comp_labels = ['intercept', 'trend', 'sin1', 'sin2', 'sin3', 'cos1', 'cos2', 'cos3', 'residual']
        comp_names  = dict(zip(comp_labels, ['intercept', 'trend', 'sin12_1', 'sin12_2', 'sin12_3', 
                                             'cos12_1', 'cos12_2', 'cos12_3', 'residual']))
    if(region is not None):
        # if region was specified, then we should either look at the climatology data, or the 
        # residual data, depending on the value of the month arg
        if(month is None):
            comp_labels.remove('residual')
            comp_names.pop('residual')
        else:
            comp_labels, comp_names = ['residual'], {'residual':'residual'}
    
    # make dict of component labels -> maps
    comp_maps = dict(zip(comp_labels, [mapping[mapping['basis_function'] == fname(comp_names[comp])] 
                                       for comp in comp_labels]))
    comp_runs = dict(zip(comp_labels, [comp_maps[comp]['run'].iat[0] for comp in comp_labels]))
    comp_species = dict(zip(comp_labels, [comp_maps[comp]['species'].iat[0] for comp in comp_labels]))

    # prepare settings for reading either flux or mole fraction
    # TODO: this should obviously be a class...
    read_args = [source, split, lev, lat, lon, region, month, comp_labels, comp_runs, comp_species,\
                 rechunk, resample, substr, top_dir, quiet, processing_dir]
    if(return_mf):
        comp_mf_data   = _read_data('mole fraction', *read_args)
    if(return_flux):
        if(lev is not None):
            warnings.warn(f'flux data exists only at the surface;'\
                          f'ignoring specification of lev={lev} for flux data')
            read_args[0] = None
        comp_flux_data = _read_data('flux', *read_args)
    print('done')

    # ------ format return dicts as xr Datasets
    if(return_mf):   comp_mf_data   = xr.Dataset(comp_mf_data)
    if(return_flux): comp_flux_data = xr.Dataset(comp_flux_data)

    # ------ done; return
    if(return_flux and return_mf): return comp_mf_data, comp_flux_data
    elif(return_flux):             return comp_flux_data
    elif(return_mf):               return comp_mf_data


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------


def _read_data(data_class, source, split, lev, lat, lon, region, month, 
               comp_labels, comp_runs, comp_species, 
               rechunk, resample, substr, top_dir, quiet, processing_dir):
    '''
    Reads flux or mole fraction data for specified options. All arguments not detailed below are expected
    to be the same as those from get_mf_and_flux_for_split

    Parameters
    ----------
    data_class : str
        'flux' or 'mole fraction'
    '''

    if(data_class == 'flux'):
        file_glob = 'HEMCO_diagnostics*'
        var_pfx = 'Emis_'
        time_chunk = int(hours_per_month/2) # bi-monthly; 1 month per file
        scaling = lambda x: x * 1e3 * 86400 # convert to g m2/day from kg m2/s
        chunking = {'time':time_chunk}
        isel     = {'lev':0}
    elif(data_class == 'mole fraction'):
        file_glob = 'GEOSChem.SpeciesConcThreeHourly*'
        var_pfx = 'SpeciesConcVV_'
        time_chunk = 12 # 12 hours; 1 day per file
        scaling = lambda x: (x * 1e6) - 400 # convert to ppm and subtract 400 ppm
        if(lev is not None):
            chunking = {'time':time_chunk}
            isel     = {'lev':lev-1}
        else:
            chunking = {'time':time_chunk, 'lev':10}
            isel     = None
    
    # ----------- get data -----------
    # contatenate all files for this split
    # coarsen 1-hourly data to daily-mean (by default)
    if(lev is not None):
        print(f'reading {data_class} data for level {lev} = {lev_to_p(lev)} hPa = {lev_to_z(lev)} km...')
    else:
        if(data_class == 'flux'):
            print(f'reading {data_class} data at the surface...')
        elif(data_class == 'mole fraction'):
            print(f'reading {data_class} data for all vertical levels...')

    start_time = timeit.default_timer()
    
    comp_data = {}
    for i,comp in enumerate(comp_labels):
        data_files = sorted(glob.glob(f'{top_dir}/{comp_runs[comp]}_split{split:02d}'\
                                      f'/OutputDir/{file_glob}'))
        if(substr is not None):
            data_files = np.array(data_files)[[substr in v for v in data_files]]

        varname = f'{var_pfx}{comp_species[comp]}'
        
        redo = False
        N    = len(data_files)
        data = [0]*N
        for j in range(N):
            if(redo): j-=1
            if(not quiet):
                print(f'reading {source} {data_class} {comp} file {j+1}/{N}...'+''.join([' ']*30))

            if(rechunk):
                # read from re-chunked data if currently exists, or rechunk and write to disk 
                # if necessary
                rechunked_fname = data_files[j].split('/')[-1].split('.nc')[0] + '_' + varname
                if(lev is not None):
                    rechunked_fname += f'_lev{lev}'
                if(region is not None):
                    rechunked_fname += f'_region{region}'
                if(month is not None):
                    rechunked_fname += f'_month{month}'
                rechunked_path = processing_dir+'/rechunked_transport_files/'+rechunked_fname+'.nc'
                try:
                    try:
                        data[j] = xr.open_dataset(rechunked_path)[varname]
                        redo=False
                    except KeyError:
                        # if the file exists but the variable is not found, then that generally means
                        # that the previous run of this function crashed and left a corrupt file behind
                        # In that case, delete the file and then redo this iteration
                        print('file found but not tracer variable; likely corruption. Removing file and redoing...')
                        os.remove(rechunked_path)
                        redo=True
                        continue
                    if(not quiet): print('read rechunked data from file...')
                except FileNotFoundError:
                    if(not quiet): print(f'rechunking and writing out rechunked data to {rechunked_fname}...')
                    rechunk_dataset(data_files[j], var=varname, chunking=chunking, isel=isel, 
                                    outfile=rechunked_path)
                    data[j] = xr.open_dataset(rechunked_path)[varname]
            else:
                data[j] = xr.open_dataset(data_files[j])[varname]
                if(lev is not None):
                    data[j] = data[j].isel(lev=lev-1)

         
            # select for lat, lon if specified
            if(lat is not None):
                data[j] = data[j].sel(lat=lat, method='nearest')
            if(lon is not None):
                data[j] = data[j].sel(lon=lon, method='nearest')
            
            # resample, write result out to intermediate file
            resampled_fname = rechunked_fname.strip('.nc')+'_'+resample+'_resampled.nc'
            resampled_path  = processing_dir+f'{resample}_resampled_data/'+resampled_fname
            try:
                data[j] = xr.open_dataset(resampled_path)[varname]
                if(not quiet): print('read time-resampled data from file...')
            except FileNotFoundError:
                if(not quiet): print('Resampled data in time and writing out...')
                data[j] = data[j].resample(time=resample).mean()
                data[j].to_netcdf(resampled_path)
            
            # do scaling
            data[j] = scaling(data[j])

        # finally, concat in time
        data = xr.concat(data, dim='time')
        comp_data[comp] = data
    
    elapsed = timeit.default_timer() - start_time
    print(f'took {elapsed:.2f} s')
    return comp_data
