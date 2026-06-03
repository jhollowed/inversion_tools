# Joe Hollowed
# University of Michigan 2023

# Set of constants used in this package

hours_per_month = 24*30

valid_components = ['intercept', 'trend', 'residual', 'seasonal', 
                  'sin12_1', 'sin12_2', 'sin12_3', 'cos12_1', 'cos12_2', 'cos12_3']
valid_sources    = ['resp', 'gpp', 'ocean']

# paths
data_dir         = '/work/noaa/co2/aschuh/WOMBAT_stuff'
inverse_dir      = f'{data_dir}/wombat-v3-inverse'
forward_dir      = f'{data_dir}/wombat-v3-forward'
gc_transport_dir = f'{forward_dir}/4a_postprocessing_gc/intermediates/daily-mole-fraction'
