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

# paths
data_dir         = '/work/noaa/co2/aschuh/WOMBAT_stuff'
inverse_dir      = f'{data_dir}/wombat-v3-inverse'
forward_dir      = f'{data_dir}/wombat-v3-forward'
gc_transport_dir = f'{forward_dir}/4a_postprocessing_gc/intermediates/daily'
gc_vert_grid_def = f'{here}/../data/GEOS-Chem_47level_grid_def.csv'
CACHE            = '/work/noaa/co2/jhollo/GC_gridded_mfs'

# GC vertical grid definition
GC_VERT = pd.read_csv(gc_vert_grid_def, skiprows=1, sep=',',
                      names=['L', 'eta_edge', 'eta_mid', 'alt_km', 'p_hpa', 'Ap', 'Bp'])
