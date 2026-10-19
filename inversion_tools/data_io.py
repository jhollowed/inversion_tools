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


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


class WOMBAT_Reader:
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
    def __init__(self, model, source, component,
                 processing_dir=None, quiet=True):
        
        # ------ check inputs
        check_inputs(source=source, component=component, model=model)

        self.model     = model
        self.source    = source
        self.component = component
        self.pdir      = processing_dir
        self.quiet     = quiet

        # ------ get model-specific settings
        self.data_dir   = {'GC':gc_transport_dir, 'TM5':tm5_transport_dir}[model]
        self.read_meteo = {'GC':read_gc_meteo, 'TM5':read_tm5_meteo}[model]
        
        # ------ set residual flag
        if(component == 'residual'):
            self.residual = True
        else:
            self.residual = False
    
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    def read_prior_mf(self, start_date, end_date, region=None,
                      pft=None, lev=None, lat=None, lon=None):
        kwargs = locals()
        del kwargs['self']
        return self._read_transport_data(**kwargs, data_class='mf')
    
    def read_prior_flux(self, start_date, end_date, region=None,
                        pft=None, lev=None, lat=None, lon=None):
        kwargs = locals()
        del kwargs['self']
        return self._read_transport_data(**kwargs, data_class='mf')
    
    def read_posterior_mf(self, start_date, end_date, sample, obs, 
                          region=None, pft=None, lev=None, lat=None, lon=None): 
        kwargs = locals()
        del kwargs['self']
        return self._read_transport_data(**kwargs, data_class='mf')
     
    def read_prior_mf_control(self, start_date, end_date, pft=None, 
                              lev=None, lat=None, lon=None):
        kwargs = locals()
        del kwargs['self']
        return self._read_transport_data(**kwargs, data_class='mf', region=None)
    
    def read_prior_flux_control(self, start_date, end_date, pft=None,
                                lev=None, lat=None, lon=None):
        kwargs = locals()
        del kwargs['self']
        return self._read_transport_data(**kwargs, data_class='flux', region=None)

    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    
    def _read_transport_data(self, start_date, end_date, 
                             sample=None, obs=None, region=None, pft=None,
                             lev=None, lat=None, lon=None, data_class='mf'):
        '''
        Reads and returns mole fraction data for a chosen TransCom region, and 
        at specified position, or globally.

        Parameters
        ----------
        start_date : str
            The starting date of the data to retrieve, in YYYY-MM format.
            Required when component!="residual".
            When component="residual", this specifies both the residual pulse emission date,
            as well as the start date of the retrieved data range.
        end_date : str
            The ending date of the data to retrieve, in YYYY-MM format. This date is
            inclusive, meaning that e.g. start_date=2015-01, end-date=2015-02, then two
            months of data will be retruned.
            If component="residual" and data_class="flux", then the flux is set to zero for 
            all times outside of the emission month.
        region : int, optional
            An integer 1 to 23 giving the TransCom region. Default is None, in which case 
            the control runs are read.
        pft : int, optional
            An integer from 1 to 15 giving the plant functional type. If source is not 
            "gpp" or "resp", this is ignored. If source is "gpp" or "resp" and "pft" is 
            not supplied, then all pft's will be read and summed. Else, the specified 
            pft is returned.
        sample : int or str, optional
            If provided, posterior is returned. If not provided, prior is returned.
            If an int, this arg specifies the posterior scale factor sample to use for
            the prior to posterior scaling, from 1 to 1000
            If a string, then the options are:
            - "mean": the mean scale factor is used for the prior to posterior scaling.
            - "stats": all scale factor samples 1 to 1000 are used to compute a mean and
                       standard deviation.
        obs : str, optional
            Which observational optimization to read from, in the case that the posterior
            is being read (the arg "sample" was passed). Options are
            'IS', 'LNLG', 'LNLGIS', 'LNLGOGIS'
        lev : int or str, optional
            Vertical level. If not provided, then all the full vertical domain is returned
            when data_class = "mf". When data_class = "flux", this arg is ignored.
            If an int, then this arg specifies the level index to select.
            If a str, options are:
            - "XCO2": the column-averaged mole fraction is returned
        lat : float, slice, list, or str, optional
            Latitude to select the data on.
            If a float, then the closest matching position in latitude is selected.
            If a slice, then the latitudes within the slice are selected.
            If a list, then the latitudes within the bounds are selected and averaged. Must be an 
            ascending list of length 2
            If a str, options are:
            - "mean" : the average of the full latitude range is taken.
        lon : float, optional
            Longitude to select the data on.
            If a float, then the closest matching position in longitude is selected.
            If a slice, then the longitudes within the slice are selected.
            If a list, then the longitudes within the bounds are selected and averaged. Must be an 
            ascending list of length 2
            If a str, options are:
            - "mean" : the average of the full latitude range is taken.
        data_class : str, optional
            Whether to return the mole fraction or flux data. Options are:
            "mf"
            "flux"
    
        Returns
        -------
        If both return_flux and return_mf, then returns a tuple (mole fraction data, flux data)
        with the data types (xarray DataArray, xarray DataArray)
        Else, a single return of either the mole fraction or flux data, as an xarray DataArray.
        '''
        
        # TMP
        if(sample is not None and data_class == 'flux'):
            raise ValueError('posterior sampling not yet implemented for fluxes :(')

        # ------ check return type ------
        if(data_class not in ['mf', 'flux']):
            raise ValueError('data_class must be either "mf" or "flux"')

        # ------ check inputs ------
        check_inputs(region=region, obs=obs)
        
        if(sample is not None):
            if(type(sample) == int):
                if(sample < 1 or sample > 1000):
                    raise ValueError('alpha sample must be between 1 and 1000')
            elif(type(sample) == str):
                if(sample not in ['mean', 'stats']):
                    raise ValueError('sample option not recognized; '\
                                     'must be "mean" or "all" and an integer')
 
        if(not is_yyyy_mm(start_date)):
            raise ValueError('start_date must be of the form YYYY-MM')
        if(not is_yyyy_mm(end_date)):
            raise ValueError('end_date must be of the form YYYY-MM')
        if(data_class == 'flux' and lev is not None):
            raise ValueError('argument "lev" cannot be specified when data_class="flux"')
 
        # ------ build year list ------
        self.start_year = int(start_date.split('-')[0])
        self.end_year   = int(end_date.split('-')[0])
        years           = np.arange(self.start_year, self.end_year+1)

        if(self.component == 'residual'):
            emission_date = start_date
        else:
            emission_date = None
        
        # ------ read data per year ------
        prior = [0] * len(years)
        for i,year in enumerate(years):
            if(not self.quiet):
                print(f'---------- reading data for year {year} ----------')
            
            prior[i] = self._get_mf_or_flux_for_year(year, region, pft, lev, lat, lon, 
                                                     data_class, emission_date=emission_date)
            # ------ slice on time
            prior[i] = prior[i].sel(time=slice(start_date, end_date))
            if(prior[i].time.size == 0):
                # for any data which has no entries within the specified time
                # window, set to zero to flag for removal
                prior[i] = None

        # remove entries for data that could not be read
        prior = [p for p in prior if p is not None]

        # ------ concatenate ------
        if(not self.quiet): print(f'concatenating {len(prior)} DataArrays')
        prior = xr.concat(prior, dim='time')
        if(sample is None):
            return prior
        
        # ------ scale to posterior if requested ------
        if(sample == 'stats'):
            if(not self.quiet): print(f'scaling prior data to obtain posterior mean and std')
            alpha = get_posterior_alpha(self.model, obs, 'all', self.source, 
                                        self.component, region, pft, emission_date)
            posterior_mean = (prior * (1 + alpha)).mean('sample')
            posterior_std  = (prior * (1 + alpha)).std('sample')
            return [posterior_mean, posterior_std]
        else:
            if(not self.quiet): 
                print(f'scaling prior data to obtain posterior for scale factor sample {sample}')
            alpha = get_posterior_alpha(self.model, obs, sample, self.source, 
                                        self.component, region, pft, emission_date)
            posterior = prior * (1 + alpha)
            return posterior

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _get_mf_or_flux_for_year(self, year, region, pft, lev, lat, lon, data_class, 
                                 emission_date=None, component=None, quiet=None):
        '''
        Reads and returns the mole fraction and/or flux data for the specified year.
        For all arguments not described here, see the docstring of _read_transport_data()

        Parameters
        ----------
        year : int
            Year in which to read the data.
        emission_date : str, optional
            The emission date in YYYY-MM format, for the case that component='residual'.
        component : str, optional
            Optional component to override self.component. Defaults to None.
        quiet : bool, optional
            Optional flag to override self.quiet. Defaults to None.
        '''

        # ------ either take class-level params, or override from args
        if(component is None): component = self.component
        if(quiet is None):     quiet     = self.quiet

        # ------ get emission year and month
        if(component == 'residual'):
            if(emission_date is None):
                raise ValueError('emission_date must be supplied when component=residual')
            else:
                emission_year  = emission_date.split('-')[0]
                emission_month = emission_date.split('-')[1]
        
        # ------ if reading a residual flux, and the year is not the emission year,
        #        then we will instead read the mole fraction, set the values to zero,
        #        rename the variable to "flux", and return. This is done so that 
        #        there can be a continuous flux time series for the residual case.
        set_zero_fluxes = False
        if(component == 'residual' and data_class == 'flux'):
            if (year!=emission_year):
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

        # ------ configure data directory names
        # if pft=None the source is not the ocean, then use pft=1 as a placeholder to later 
        # be replaced in the PFT looping procedure
        if(pft == None and self.source != 'ocean'): tmp_pft = 1 
        else:                                       tmp_pft = pft
        datapath = get_data_dir(self.model, self.source, component, 
                                region=region, emission_date=emission_date, pft=tmp_pft)
        dirpath, dirname = os.path.split(datapath)
        
        # ------ handle recursive seasonal calls ------
        # if self.component='seasonal', then call this function recursively for each 
        # sin,cos component, summing the result for final return
        if(component == 'seasonal'):
            if(self.source == 'ocean'):
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

        # ------ get available PFTs and verify
        pfts_avail = get_available_pfts(self.model, self.source, component, 
                                        region, emission_date=emission_date)
        if(pft is not None and pft not in pfts_avail):
            raise ValueError(f'pft={pft} is not available for '\
                             f'source={self.source}, component={component}, region={region};'\
                             f'\navailable pfts are {pfts_avail}') 

        # ------ handle recursive PFT calls ------
        # if a pft was not specified, but the source is gpp or resp, then call this 
        # function recursively for each avaialble pft, summing the result for final return
        if(pft is None and self.source != 'ocean'):
            if(not quiet): print(f'reading PFTs {pfts_avail}...')
            for i,pfti in enumerate(pfts_avail):
                if(not quiet):
                    print(f'========== PFT {pfti} ==========')
                kwargs = {'year':year, 'region':region, 'lev':lev, 'lat':lat, 'lon':lon, 
                          'pft':pfti, 'data_class':data_class}
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
        files = glob.glob(f'{datapath}/{file_glob}_{year}.nc4')
        if(len(files) > 1):
            raise RuntimeError('too many files found! Please debug.')
        # if no files were found, then this choice of region (and optionally pft) does 
        # not exist for the choice of source and component; notify user
        if(len(files) < 1):
            raise RuntimeError('no file matching these parameters! Double check that '\
                               'specified flux source is nonzero in the specified region;'\
                               f'\nsource={self.source}, component={component}, '\
                               f'region={region}, pft={pft}')

        # ------ read data ------
        if(not quiet):
            print(f'reading {files[0].split("/")[-1]}...')
        data = xr.open_dataset(files[0])[var]

        # ------ do latitude slicing ------
        if(lat is not None):
            if(type(lat) == float or type(lat) == int):
                if(not quiet): print(f'selecting data nearest lat={lat}...')
                data = data.sel(lat=lat, method='nearest')
            elif(type(lat) == slice):
                if(not quiet): print(f'slicing data on lat={lat}...')
                data = data.sel(lat=lat)
            elif(type(lat) == list):
                if(len(lat)!=2 or sorted(lat)!=lat):
                    raise RuntimeError('if type of lat is a list, then must be ascending and length-2')
                if(not quiet): print(f'averaging data over lat={lat}...')
                data = data.sel(lat=slice(lat[0], lat[1]))
                data = ctb.meridional_weighted_mean(data)
            elif(lat == 'mean'):
                if(not quiet): print(f'averaging data over all lat...')
                data = ctb.meridional_weighted_mean(data)
            else:
                raise RuntimeError('arg lat does not match expected input types; please check')
                
        # ------ do longitude slicing ------
        if(lon is not None):
            if(type(lon) == float or type(lon) == int):
                if(not quiet): print(f'selecting data nearest lon={lon}...')
                data = data.sel(lon=lon, method='nearest')
            elif(type(lon) == slice):
                if(not quiet): print(f'slicing data on lon={lon}...')
                data = data.sel(lon=lon)
            elif(type(lon) == list):
                if(len(lon)!=2 or sorted(lon)!=lon):
                    raise RuntimeError('if type of lon is a list, then must be ascending and length-2')
                if(not quiet): print(f'averaging data over lon={lon}...')
                data = data.sel(lon=slice(lon[0], lon[1])).mean('lon')
            elif(lon == 'mean'):
                if(not quiet): print(f'averaging data over all lon...')
                data = data.mean('lon')
            else:
                raise RuntimeError('arg lon does not match expected input types; please check')

        # ------ do vertical slicing ------
        if(lev is not None):
            if(type(lev) == float or type(lev) == int):
                #plev, zlev = gc_lev_to_p(lev), gc_lev_to_z(lev)
                #if(not quiet): print(f'slicing data on lev={lev} ({plev} hPa // {zlev} km)...')
                if(not quiet): print(f'selecting data nearest lev={lev}...')
                data = data.isel(lev=lev)
            elif(lev == 'XCO2'):
                if(not quiet): print(f'computing XCO2 using 3D pressure on interface levels...')
                pp   = self.read_meteo(data.time.values[0], data.time.values[-1], 'interface_pressure')
                data = ctb.pressure_weighted_column_mean(data, pp)
            else:
                raise RuntimeError('arg lev does not match expected input types; please check')

        # ------ do scaling and return ------
        data = scaling(data)
        return format_data(data)
