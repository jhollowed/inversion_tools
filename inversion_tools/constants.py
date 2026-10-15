# Joe Hollowed
# University of Michigan 2023

# Set of constants used in this package

import os
import pandas as pd
import metpy.constants as const

# ---- location of this file, for relative paths
here = os.path.dirname(os.path.abspath(__file__))

hours_per_month = 24*30
p0     = 1000 # in hPa
kappa  = const.Rd / const.Cp_d

valid_components = ['intercept', 'trend', 'residual', 'seasonal', 
                  'sin12_1', 'sin12_2', 'sin12_3', 'cos12_1', 'cos12_2', 'cos12_3']
valid_sources    = ['resp', 'gpp', 'ocean']

# ---- paths
data_dir    = '/work/noaa/co2/aschuh/WOMBAT_stuff'
inverse_dir = f'{data_dir}/wombat-v3-inverse'
forward_dir = f'{data_dir}/wombat-v3-forward'
alpha_dir   = '/work/noaa/co2/jhollo/wombat-v3-inverse/'
# ---- GC
gc_transport_dir = f'{forward_dir}/4a_postprocessing_gc/intermediates/daily'
gc_vert_grid_def = f'{here}/../data/GEOS-Chem_47level_grid_def.csv'
gc_alpha_dir     = f'{alpha_dir}/inversion_prior-correlated-tight-ocean_data-wombatv3_gc'
gc_meteo_dir     = f'{forward_dir}/data/GEOS_Chem/ExtData/GEOS_2x2.5/MERRA2'
# ---- TM5
tm5_alpha_dir = f'{alpha_dir}/inversion_prior-correlated-tight-ocean_data-wombatv3_tm5'
tm5_meteo_url = 'https://gml.noaa.gov/aftp/products/carbontracker/co2/CT2022/molefractions/co2_total/'
# ---- user-defined caches
tm5_meteo_cache = '/work/noaa/co2/jhollo/TM5_CT2022_gridded_vars'
gc_meteo_cache  = '/work/noaa/co2/jhollo/GC_MERRA2_gridded_vars'


# GC vertical grid definition
GC_VERT = pd.read_csv(gc_vert_grid_def, skiprows=1, sep=',',
                      names=['L', 'eta_edge', 'eta_mid', 'alt_km', 'p_hpa', 'Ap', 'Bp'])
