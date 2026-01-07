# Joe Hollowed
# University of Michigan 2026
#
# Providing a set of utility functions for analyzing flux inversion datasets


# =========================================================================

# ---- import dependencies
import pdb
import numpy as np
import xarray as xr
from timeit import default_timer

# ---- imports from this package
from .utils import *
from .constants import *

# -------------------------------------------------------------------------

data_dir = '/work/noaa/co2/aschuh/WOMBAT_stuff'
inverse_dir = f'{data_dir}/wombat-v3-inverse'
forward_dir = f'{data_dir}/wombat-v3-forward'
gc_transport_dir = f'{forward_dir}/3a_transport_gc/intermediates/runs-control'

# -------------------------------------------------------------------------

def get_mf_and_flux_for_split(source, lev, split, rechunk=True, datadir=None,
                              lat=None, lon=None, pft=None, substr=None, resample='1D', 
                              return_flux=True, return_mf=True, quiet=False):
    '''
    Reads, resamples, and returns mole fraction and flux data for a chosen emission source, 
    either globally, for a single latitude, a single longitude, or a single lat/lon position

    Parameters
    ----------
    source : str
        either 'ocean', 'gpp', or 'resp'
    lev : int
        vertical level
    split : int
        an integer giving the time split number
    rechunk : bool
        whether or not to rechunk the original data files for more efficient reading.
        Defaults to True, in which case disk space will be used by intermediate processed
        data files (unless they already exist). Data is re-chunked in the time dimension 
        only, into 30-day chunks.
    datadir : str, optional
        path to location at which to write out processed (rechunked) data, if rechunk=True
    lat : float, optional
        latitude to extract data across time
    lon : float, optional
        longitude to extract data across time
    pft : int, optional
        an integer from 1 to 15 giving the plant functional type. If source is not gpp or resp, 
        this is ignored. If source is gpp or resp and pft is not supplied, then all pft's will
        be read and summed. Else, the specified pft is returned.
    resample : str, optional
        a valid time string to pass to xarray.resample. Default is '1D', which causes the returned data
        to be resampled to a daily frequency
    '''

    if(source == 'gpp' or source == 'resp'):
        pfx = 'sib4_'
        if(pft is not None):
            sfx = f'_pft{pft:02d}'
        else:
            sfx = 'allpft'
        if(source == 'resp'):
            source = 'resp_tot'
    elif(source == 'ocean'): 
        source = 'ocean_lschulz'
        pfx, sfx = '', ''

    # get mapping file
    mapping = pd.read_csv(f'{gc_transport_dir}/mapping.csv')
    
    # lambda function for formatting file names per input component
    fname = lambda component: f'control_{pfx}{source}_{component}{sfx}'
    
    # make dict of component labels -> component names
    if(source == 'ocean_lschulz'):
        comp_labels = ['intercept', 'trend', 'sin1', 'sin2', 'cos1', 'cos2', 'residual']
        comp_names  = dict(zip(comp_labels, ['intercept', 'trend', 'sin12_1', 'sin12_2', 
                                             'cos12_1', 'cos12_2', 'residual']))
    else:
        comp_labels = ['intercept', 'trend', 'sin1', 'sin2', 'sin3', 'cos1', 'cos2', 'cos3', 'residual']
        comp_names  = dict(zip(comp_labels, ['intercept', 'trend', 'sin12_1', 'sin12_2', 'sin12_3', 
                                             'cos12_1', 'cos12_2', 'cos12_3', 'residual']))
    
    # make dict of component labels -> maps
    comp_maps = dict(zip(comp_labels, [mapping[mapping['basis_function'] == fname(comp_names[comp])] 
                                   for comp in comp_labels]))
    comp_runs = dict(zip(comp_labels, [comp_maps[comp]['run'].iat[0] for comp in comp_labels]))
    comp_species = dict(zip(comp_labels, [comp_maps[comp]['species'].iat[0] for comp in comp_labels]))

    # ----------- get flux field data -----------
    # contatenate all files for this split
    # coarsen 1-hourly data to daily-mean
    if(return_flux):
        print(f'reading flux data for level {lev} = {lev_to_p(lev)} hPa = {lev_to_z(lev)} km')
        start_time = timeit.default_timer()
        comp_flux_data = {}
        for i,comp in enumerate(comp_labels):
            data_files = sorted(glob.glob(f'{gc_transport_dir}/{comp_runs[comp]}_split{split:02d}'\
                                           '/OutputDir/HEMCO_diagnostics*'))
            if(substr is not None):
                data_files = np.array(data_files)[[substr in v for v in data_files]]
    
            varname = f'Emis_{comp_species[comp]}'
            
            N    = len(data_files)
            data = [0]*N
            for j in range(N):
                if(not quiet):
                    print(f'reading {source} flux {comp} file {j+1}/{N}...'+\
                          ''.join([' ']*30), end='\r')

                if(rechunk):
                    # read from re-chunked data if currently exists, or rechunk and write to disk 
                    # if necessary
                    rechunked_fname = data_files[j].split('/')[-1].split('.nc')[0] + \
                                      varname + f'lev{lev}' + '.nc'
                    rechunked_fname = datair + rechunked_fname
                    if(not os.path.exists(rechunked_fname)):
                        utils.rechunk_dataset(data_files[j], oufile=rechunked_fname, 
                                              isel={'lev':lev-1}, var=varname)
                    data_1hr = xr.open_dataset(rechunked_fname, {'time':hours_per_month}, 
                                               isel={'lev':lev-1})
                else:
                    data_1hr = xr.open_dataset(data_files[j])[varname]
                    data_1hr = data_1hr.isel(lev=lev-1)

             
                # select for lat, lon if specified
                if(lat is not None):
                    data_1hr = data_1hr.sel(lat=lat, method='nearest')
                if(lon is not None):
                    data_1hr = data_1hr.sel(lon=lon, method='nearest')
                # resample
                data[j] = data_1hr.resample(time=resample).mean()
                # convert to g m2/day from kg m2/s
                data[j] = (data[j]* 1e3 * 86400)

            # finally, concat in time
            data = xr.concat(data, dim='time')
            comp_flux_data[comp] = datadata_dir = '/work/noaa/co2/aschuh/WOMBAT_stuff'
        
        elapsed = timeit.default_timer() - start_time
        print(f'took {elapsed:.2f} s')

    
    # ----------- get mole fraction data -----------
    # contatenate all files for this split
    # coarsen 3-hourly data to daily-mean
    # scale to ppm and remove 400ppm IC
    if(return_mf):
        print(f'reading mole fraction data for level {lev} = {lev_to_p(lev)} hPa = {lev_to_z(lev)} km')
        start_time = timeit.default_timer()
        comp_mf_data = {}
        for i,comp in enumerate(comp_labels):
            data_files = sorted(glob.glob(f'{gc_transport_dir}/{comp_runs[comp]}_split{split:02d}'\
                                           '/OutputDir/GEOSChem.SpeciesConcThreeHourly*'))
            if(substr is not None):
                data_files = np.array(data_files)[[substr in v for v in data_files]]
                
            varname = f'SpeciesConcVV_{comp_species[comp]}'
                
            N    = len(data_files)
            data = [0]*N
            for j in range(N):
                if(not quiet):
                    print(f'reading {source} mole fraction {comp} file {j+1}/{N}...'+\
                          ''.join([' ']*30), end='\r')
                
                if(rechunk):
                    # read from re-chunked data if currently exists, or rechunk and write to disk 
                    # if necessary
                    rechunked_fname = data_files[j].split('/')[-1].split('.nc')[0] + \
                                      varname + f'lev{lev}' + '.nc'
                    rechunked_fname = datadir + rechunked_fname
                    if(not os.path.exists(rechunked_fname)):
                        utils.rechunk_dataset(data_files[j], oufile=rechunked_fname, 
                                              isel={'lev':lev-1}, var=varname)
                    data_3hr = xr.open_dataset(rechunked_fname, {'time':hours_per_month/3}, 
                                               isel={'lev':lev-1})
                else:
                    data_3hr = xr.open_dataset(data_files[j])[varname]
                    data_3hr = data_3hr.isel(lev=lev-1)
                
                # select for lat, lon if specified
                if(lat is not None):
                    data_3hr = data_3hr.sel(lat=lat, method='nearest')
                if(lon is not None):
                    data_3hr = data_3hr.sel(lon=lon, method='nearest')
                # resample
                data[j] = data_3hr.resample(time=resample).mean()
                # convert to ppm and subtract 400ppm
                data[j] = (data[j]*1e6) - 400
            
            # finally, concat in time
            data = xr.concat(data, dim='time')
            comp_mf_data[comp] = data
                
        elapsed = timeit.default_timer() - start_time
        print(f'took {elapsed:.2f} s')
            
    print('done')

    # ------ format return dicts as xr Datasets
    comp_mf_data   = xr.Dataset(comp_mf_data)
    comp_flux_data = xr.Dataset(comp_flux_data)

    # ------ done; return
    if(return_flux and return_mf): return comp_mf_data, comp_flux_data
    elif(return_flux):             return comp_flux_data
    elif(rturn_mf):                return comp_mf_data
