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
import urllib.request
from timeit import default_timer

# ---- imports from this package
from .utils import *
from .constants import *

def read_tm5_mf(*args, **kwargs):
    return read_tm5(*args, **kwargs, data_class='mf')
def read_tm5_flux(*args, **kwargs):
    return read_tm5(*args, **kwargs, data_class='flux')

def read_tm5(source, component, region, nmonths, start_year=None, et=None, pft=None, data_class=None):
    '''
    Temporary function for reading TM5 data

    source : str
        'ocean' or 'bio_resp_tot' or 'bio_gpp'
    component : str
        'residual' or 'trend' or 'intercept' or 'seasonal' or 
        'sin12_1' or 'sin12_2' or 'sin12_3' or 'cos12_1' or 'cos12_2' or 'cos12_3'
    region : int
        1-22
    start_year : int
        starting year to grab data from. Required when component != 'residual'
    nmonths : int
        number of months of data to grab.
    et : str
        emission time in YYY-MM. Required when component == 'residual'
    pft : int
        1-15
    data_class : str
        'flux' or 'mf'
    '''

    topdir  = '/work/noaa/co2/andy/Projects/WOMBAT/wombat-v3-forward/3b_transport_tm5/intermediates'
    rundir  = f'{topdir}/runs'
    mapping = f'{rundir}/tm5_mapping.csv'
    tmap    = pd.read_csv(mapping)

    if(source == 'ocean'):    
        pfts = ''
    else:                        
        pfts = f'_pft{pft:02d}'
    if(component == 'residual'):
        if(start_year is not None or et is None):
            raise RuntimeError('use et for residual, not start_year')
        ety = int(et.split("-")[0])
        etm = int(et.split("-")[1])
        ets  = f'_month{ety:04d}-{etm:02d}'
        start_year = ety
    else:                        
        if(start_year is None or et is not None):
            raise RuntimeError('use start_year when not residual, not et')
        ets  = ''
    name = f'{source}_{component}{pfts}_regionRegion{region:02d}{ets}'

    row              = tmap[tmap['name'] == name]
    run              = row['run'].values[0]
    flux_var, mf_var = row['flux_field_names'].values[0], row['species'].values[0]
    tm5_data         = None

    date_dirs = sorted(glob.glob(f'{rundir}/{run}/output/*'))
    i_start = None
    for i,ddir in enumerate(date_dirs):
        dirname = ddir.split('/')[-1]
        
        if(int(dirname[:4]) < start_year):
            continue
        else:
            if(i_start == None): i_start = i
            
        if((i-i_start)+1 > nmonths):
            continue
            
        year = int(dirname[:4])
        if(data_class == 'mf'):
            files = sorted(glob.glob(f'{ddir}/*mole*.nc'))
            print(f'reading {len(files)} {data_class} files for dir {dirname}...')
            data = [(xr.open_dataset(f)[mf_var]-400) for f in files]
        elif(data_class=='flux'):
            files = sorted(glob.glob(f'{ddir}/*flux1x1*.nc'))
            print(f'reading {len(files)} {data_class} files for dir {dirname}...')
            if(len(files)==0): 
                continue
            data = [(xr.open_dataset(f)[mf_var]) for f in files]
        
        print('concatenating...')
        if(tm5_data is None):
            tm5_data = xr.concat(data, dim='time')
        else:
            data     = xr.concat(data, dim='time')
            tm5_data = xr.concat([tm5_data, data], dim='time')

    # take daily average
    print('taking daily average...')
    tm5_data = tm5_data.resample(time="1D").mean()

    # rename dimensions to match GC
    tm5_data = tm5_data.rename({'longitude':'lon', 'latitude':'lat'})
    if(data_class == 'mf'):
        tm5_data = tm5_data.rename({'level':'lev'})

    if(data_class == 'flux'):
        # scale from mol/m2/s to g m2/day
        tm5_data *= 12.011 * 86400
    elif(data_class == 'mf'):
        # correct for erroneous CO2-to-C factor, per Mike
        tm5_data /= (44.009/12.011)

    return tm5_data

# -------------------------------------------------------------------------

def read_tm5_pressure(start, end):
    '''
    Temporary function for reading CT2022 daily gridded pressure data

    start : np.datetime64
        starting date (year, month, day)
    end : np.datetime64
        ending date (year, month, day)
    '''

    url   = 'https://gml.noaa.gov/aftp/products/carbontracker/co2/CT2022/molefractions/co2_total/'
    cache = '/work/noaa/co2/jhollo/CT2022_gridded_mfs' 

    dates    = np.arange(start, end + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    pressure = [0]*len(dates)

    for i in range(len(dates)):
        date    = str(dates[i].astype('datetime64[D]'))
        file    = f'CT2022.molefrac_glb3x2_{date}.nc'
        cached  = f'{cache}/{file}'
        cachedp = f'{cache}/{file.strip(".nc")}_pressure.nc'
        try:
            p = xr.open_dataset(cachedp)['pressure']
        except FileNotFoundError:
            # download from NOAA
            print(f'file for {date} not found; downloading...')
            urllib.request.urlretrieve(f'{url}/{file}', cached)
            # grab only the pressure data, then discard the downloaded file
            p = xr.open_dataset(cached)['pressure']
            os.remove(cached)
            # rename horizontal dims
            p = p.rename({'latitude':'lat', 'longitude':'lon'})
            # take linear average of interface pressures
            p = (p + p.shift(boundary=-1)) / 2
            p = p.isel(boundary=slice(0, -1))
            p = p.rename({'boundary':'lev'})
            # update meta data for clarity
            p.attrs['comment'] = 'air pressure at level midpoints'
            p.to_netcdf(cachedp)
        pressure[i] = p

    print('concatenating pressure data...')
    pressure = xr.concat(pressure, dim='time')

    print('taking daily average...')
    pressure = pressure.resample(time="1D").mean()

    return pressure

# -------------------------------------------------------------------------

def read_gc_pressure(start, end):
    '''
    Temporary function for reading GC gridded pressure data

    start : np.datetime64
        starting date (year, month, day)
    end : np.datetime64
        ending date (year, month, day)
    '''

    data_dir = '{forward_dir}/data/GEOS_Chem/ExtData/GEOS_2x2.5/MERRA2'
    cache    = '/work/noaa/co2/jhollo/GC_gridded_mfs' 

    dates     = np.arange(start, end + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    all_years = dates.astype('datetime64[Y]').astype(int) + 1970
    years     = np.unique(all_years)

    #pressure = [0]*len(dates)

    for i in range(len(years)):
        cached = f'{cache}/GC_MERRA2_{year}_daily_pressure.nc'
        try:
            p = xr.open_dataset(cached)['pressure']
        except FileNotFoundError:
            for j,date in enumerate(dates[all_years == years[i]]): 
                year, month, day = str(date).split("-")
                file   = f'{year}/{month}/MERRA2.{year}{month}{day}.I3.2x25.nc4'
                # grab 3-hourly from disc from NOAA
                print(f'daily file for {date} not found; processing...')
                urllib.request.urlretrieve(f'{url}/{file}', cached)
                # grab only the pressure data, then discard the downloaded file
                p = xr.open_dataset(cached)['pressure']
                os.remove(cached)
                # rename horizontal dims
                p = p.rename({'latitude':'lat', 'longitude':'lon'})
                # take linear average of interface pressures
                p = (p + p.shift(boundary=-1)) / 2
                p = p.isel(boundary=slice(0, -1))
                p = p.rename({'boundary':'lev'})
                # update meta data for clarity
                p.attrs['comment'] = 'air pressure at level midpoints'
                p.to_netcdf(cachedp)
        pressure[i] = p

    print('concatenating pressure data...')
    pressure = xr.concat(pressure, dim='time')

    print('taking daily average...')
    pressure = pressure.resample(time="1D").mean()

    return pressure

# -------------------------------------------------------------------------

class Reader:
    '''
    Interface for reading transport forward run mole fractions and fluxes

    Parameters
    ----------
    source : str
        Either 'ocean', 'gpp', or 'resp'.
    component : int
        Either 'residual', 'intercept', 'trend', 'seasonal', 
        'sin12_1', 'sin12_2', 'sin12_3', 'cos12_1', 'cos12_2', or 'cos12_3'. 
        If 'seasonal', then all six sin, cos terms will be read and summed.
    processing_dir : str, optional
        location for writing out intermediate analysis files. Defaults to None, 
        in which case no intermediate files are written out
    quiet : bool, optional
        whether or not to suppress print statements from the methods of this 
        object instance

    Methods
    -------
    read_mf()
        reads mole fraction data for specified parameters
    read_flux()
        reads flux data for specified parameters
    '''
    def __init__(self, source, component,
                 processing_dir=None, quiet=True):

        if(source not in valid_sources):
            raise RuntimeError(f'source must by one of {valid_sources}, not {source}')
        if(component not in valid_components):
            raise RuntimeError(f'component must by one of {valid_components}, not {component}')
        self.source    = source
        self.component = component
        self.pdir      = processing_dir
        self.quiet     = quiet
    
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    def read_mf(self, **kwargs):
        if('region' not in kwargs.keys()):
            raise RuntimeError('"region" must be supplied for read_mf()')
        if('data_class' in kwargs.keys()):
            raise RuntimeError('"data_class" not a valid arg for read_mf()')
        kwargs['data_class'] = 'mf'
        return self._read_transport_data(**kwargs)
    
    def read_flux(self, **kwargs):
        if('region' not in kwargs.keys()):
            raise RuntimeError('"region" must be supplied for read_flux()')
        if('data_class' in kwargs.keys()):
            raise RuntimeError('"data_class" not a valid arg for read_flux()')
        kwargs['data_class'] = 'flux'
        return self._read_transport_data(**kwargs)
    
    def read_mf_control(self, **kwargs):
        if('region' in kwargs.keys()):
            raise RuntimeError('"region" not a valid argument for read_mf_control()')
        if('data_class' in kwargs.keys()):
            raise RuntimeError('"data_class" not a valid arg for read_mf_control()')
        kwargs['data_class'] = 'mf'
        kwargs['region']     = None
        return self._read_transport_data(**kwargs)
    
    def read_flux_control(self, **kwargs):
        if('region' in kwargs.keys()):
            raise RuntimeError('"region" not a valid argument for read_flux_control()')
        if('data_class' in kwargs.keys()):
            raise RuntimeError('"data_class" not a valid arg for read_flux_control()')
        kwargs['data_class'] = 'flux'
        kwargs['region']     = None
        return self._read_transport_data(**kwargs)

    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    
    def _read_transport_data(self, region=None, start_date=None, emission_date=None, 
                             end_date=None, lev=None, lat=None, lon=None, pft=None,
                             data_class='mf'):
        '''
        Reads and returns mole fraction data for a chosen TransCom region, and 
        at specified position, or globally.

        Parameters
        ----------
        region : int, optional
            The TransCom region. Default is None, in which case the control runs are read.
        emission_date : str, optional
            The date of emission, in YYY-MM format.
            Required when component='residual', otherwise ignored.
        start_date : str, optional
            The starting date of the data to retrieve, in YYYY-MM format.
            Required when component!='residual'.
            If not supplied and component='residual', then start_date=emission_date is assumed.
        end_date : str, optional
            The ending date of the data to retrieve, in YYYY-MM format. This date is
            inclusive, meaning that e.g. start_date=2015-01, end-date=2015-02, then two
            months of data will be retruned.
            If component='residual' and data_class='flux', then the flux is set to zero for 
            all times outside of the emission month starting at emission_date.
        lev : int, optional
            Vertical level. If not provided, then all the full vertical domain is returned.
        lat : float, optional
            Latitude to select the data on.
        lon : float, optional
            Longitude to select the data on.
        pft : int, optional
            An integer from 1 to 15 giving the plant functional type. If source is not 
            'gpp' or 'resp', this is ignored. If source is 'gpp' or 'resp' and 'pft' is 
            not supplied, then all pft's will be read and summed. Else, the specified 
            pft is returned.
        data_class : str, optional
            Whether to return the mole fraction or flux data. Options are:
            'mf'
            'flux'
    
        Returns
        -------
        If both return_flux and return_mf, then returns a tuple (mole fraction data, flux data)
        with the data types (xarray DataArray, xarray DataArray)
        Else, a single return of either the mole fraction or flux data, as an xarray DataArray.
        '''

        # ------ check return type ------
        if(data_class not in ['mf', 'flux']):
            raise RuntimeError('data_class must be either "mf" or "flux"')

        # ------ check inputs ------
        if(self.component=='residual'):
            if(emission_date is None):
                raise RuntimeError('emission_date must be specified when component="residual"')
            if(not is_yyyy_mm(emission_date)):
                raise RuntimeError('emission_date must be of the form YYY-MM')
            if(start_date is None):
                start_date = emission_date
        else:
            if(start_date is None):
                raise RuntimeError('start_date must be specified when component is not "residual"')
            if(not is_yyyy_mm(start_date)):
                raise RuntimeError('start_date must be of the form YYY-MM')
        if(end_date is None):
            raise RuntimeError('end_date must be specified')
        if(not is_yyyy_mm(end_date)):
            raise RuntimeError('end_date must be of the form YYY-MM')
        if(data_class == 'flux' and lev is not None):
            raise RuntimeError('argument "lev" cannot be specified when data_class="flux"')
     
        # ------ build year list ------
        self.start_date    = np.datetime64(start_date+'-01')
        self.end_date      = np.datetime64(end_date+'-01')
        if(emission_date is not None):
            self.emission_date  = np.datetime64(emission_date+'-01')
            self.emission_year  = int(str(self.emission_date)[:10].split('-')[0])
            self.emission_month = int(str(self.emission_date)[:10].split('-')[1])
        else:
            self.emission_date  = None
            self.emission_year  = None
            self.emission_month = None

        self.start_year = int(start_date.split('-')[0])
        self.end_year   = int(end_date.split('-')[0])
        years           = np.arange(self.start_year, self.end_year+1)
        
        # ------ read data per year ------
        data = [0] * len(years)
        for i,year in enumerate(years):
            if(not self.quiet):
                print(f'---------- reading data for year {year} ----------')
            
            data[i] = self._get_mf_or_flux_for_year(year, region, lev, lat, lon, 
                                                    pft, data_class)
            # slice on time
            data[i] = data[i].sel(time=slice(start_date, end_date))
            if(data[i].time.size == 0):
                # for any data which has no entries within the specified time
                # window, set to zero to flag for removal
                data[i] = None

        # remove entries for data that could not be read
        data = [d for d in data if d is not None]

        # ------ concatenate and return ------
        if(not self.quiet): print(f'concatenating {len(data)} DataArrays')
        return xr.concat(data, dim='time')


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    def _get_mf_or_flux_for_year(self, year, region, lev, lat, lon, pft, 
                                 data_class, component=None, quiet=None):
        '''
        Reads and returns the mole fraction and/or flux data for the specified year.
        For all arguments not described here, see the docstring of _read_transport_data()

        Parameters
        ----------
        year : int
            Year in which to read the data.
        component : str, optional
            Optional component to override self.component. Defaults to None.
        quiet : bool, optional
            Optional flag to override self.quiet. Defaults to None.
        '''

        # ------ either take class-level params, or override from args
        if(component is None): component = self.component
        if(quiet is None):     quiet     = self.quiet
        
        # ------ if reading a residual flux, and the year is not the emission year,
        #        then we will instead read the mole fraction, set the values to zero,
        #        rename the variable to "flux", and return. This is done so that 
        #        there can be a continuous flux time series for the residual case.
        set_zero_fluxes = False
        if(component == 'residual' and data_class == 'flux'):
            if (year!=self.emission_year):
                mf = self._get_mf_or_flux_for_year(year, region, lev, lat, lon,
                                                   pft, 'mf', component, True)
                zero_flux = (mf.isel(lev=0)*0).drop_vars('lev').rename('flux')
                return zero_flux
            else:
                mf = self._get_mf_or_flux_for_year(year, region, lev, lat, lon, 
                                                   pft, 'mf', component, True)
                format_data = lambda x: x.reindex_like(mf.isel(lev=0).drop_vars('lev'),fill_value=0)
        else:
            format_data = lambda data: data

        # ------ set flag for reading control runs
        if(region is None): control = True
        else:               control = False

        # ------ configure component naming conventions
        pftstr = ''
        if(self.source == 'gpp'):
            if(control):         source = 'control_sib4_gpp'
            else:                source = 'bio_gpp'
        if(self.source == 'resp'):
            if(control):         source = 'control_sib4_resp_tot'
            else:                source = 'bio_resp_tot'
        if(self.source == 'gpp' or self.source == 'resp'):
            if(pft is not None): pftstr = f'_pft{pft:02d}'
            else:                pftstr = '_tmppft'
        if(self.source == 'ocean'): 
            if(control):         source = 'control_ocean_lschulz'
            else:                source = 'ocean'

        # ------ configure data directory names
        sfx = pftstr
        if(region is not None):
            sfx = f'{sfx}_regionRegion{region:02d}'
        if(self.emission_date is not None):
            sfx = f'{sfx}_month{self.emission_year:04d}-{self.emission_month:02d}'
            if(region is None):
                raise RuntimeError('region must be supplied if emission_time is supplied')
        dirname = f'{source}_{component}{sfx}'
        dirpath = f'{gc_transport_dir}/{dirname}'

        # ------ handle recursive PFT calls ------
        # if a pft was not specified, but the source is gpp or resp, then call this 
        # function recursively for each avaialble pft, summing the result for final return
        if(pft is None and 'ocean' not in source):
            pfts_avail = sorted(glob.glob(f'{gc_transport_dir}/{dirname.replace(pftstr, "_pft*")}'))
            pfts_avail = [int(s.split('_pft')[-1].split('_')[0]) for s in pfts_avail]
            if(not quiet): print(f'reading PFTs {pfts_avail}...')
            for i,pfti in enumerate(pfts_avail):
                if(not quiet):
                    print(f'========== PFT {pfti} ==========')
                kwargs = {'year':year, 'region':region, 'lev':lev, 'lat':lat, 'lon':lon, 'pft':pfti,
                          'data_class':data_class}
                if(i==0):
                    comp_data = self._get_mf_or_flux_for_year(**kwargs)
                else:
                    comp_data += self._get_mf_or_flux_for_year(**kwargs) 
            return format_data(comp_data)

        # ------ handle recursive seasonal calls ------
        # if self.component='seasonal', then call this function recursively for each 
        # sin,cos component, summing the result for final return
        if(component == 'seasonal'):
            if('ocean' in source):
                seasonal_comps = ['sin12_1', 'sin12_2', 'cos12_1', 'cos12_2']
            else:
                seasonal_comps = ['sin12_1', 'sin12_2', 'sin12_3', 'cos12_1', 'cos12_2', 'cos12_3']
            for i in range(len(seasonal_comps)):
                if(not quiet):
                    print(f'========== seasonal component {seasonal_comps[i]} ==========')
                kwargs = {'year':year, 'region':region, 'lev':lev, 'lat':lat, 'lon':lon, 'pft':pft,
                          'component':seasonal_comps[i], 'data_class':data_class}
                if(i==0):
                    comp_data = self._get_mf_or_flux_for_year(**kwargs)
                else:
                    comp_data += self._get_mf_or_flux_for_year(**kwargs)
            return format_data(comp_data)
                
        # ------ set parameters for data read ------
        if(data_class == 'mf'):
            file_glob = 'daily-mole-fraction*'
            var       = 'mole_fraction'
            scaling   = lambda x: (x * 1e6) # convert to ppm
        if(data_class == 'flux'):
            file_glob = 'daily-fluxes*'
            var = 'flux'
            scaling = lambda x: x * 1e3 * 86400 # convert to g m2/day from kg m2/s
        
        # ------- locate data file -------
        files = glob.glob(f'{dirpath}/{file_glob}*_{year}.nc4')
        if(len(files) > 1):
            raise RuntimeError('too many files found! Please debug.')
        if(len(files) < 1):
            if(len(glob.glob(f'{gc_transport_dir}/{dirname.replace(pftstr,"*")}')) > 1):
                if not(quiet):
                    print(f'pft {pft:02d} does not exist; skipping...')
            else:
                raise RuntimeError('no file matching these parameters! Double check that '\
                                   'specified flux source is nonzero in the specified region')


        # ------ read data ------
        if(not quiet):
            print(f'reading {files[0].split("/")[-1]}...')
        data = xr.open_dataset(files[0])[var]

        # ------ do slicing ------
        if(lat is not None):
            if(not quiet): print(f'slicing data on lat={lat}...')
            data = data.sel(lat=lat, method='nearest')
        if(lon is not None):
            if(not quiet): print(f'slicing data on lon={lon}...')
            data = data.sel(lon=lon, method='nearest')
        if(lev is not None):
            plev, zlev = lev_to_p(lev), lev_to_z(lev)
            if(not quiet): print(f'slicing data on lev={lev} ({plev} hPa // {zlev} km)...')
            data = data.isel(lev=lev)

        # ------ do scaling and return ------
        data = scaling(data)
        return format_data(data)
