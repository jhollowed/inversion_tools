# Joe Hollowed
# University of Michigan 2023

# Set of constants used in this package

# ---- location of this file, for relative paths
here = os.path.dirname(os.path.abspath(__file__))

hours_per_month = 24*30

valid_components = ['intercept', 'trend', 'residual', 'seasonal', 
                  'sin12_1', 'sin12_2', 'sin12_3', 'cos12_1', 'cos12_2', 'cos12_3']
valid_sources    = ['resp', 'gpp', 'ocean']

# paths
data_dir         = '/work/noaa/co2/aschuh/WOMBAT_stuff'
inverse_dir      = f'{data_dir}/wombat-v3-inverse'
forward_dir      = f'{data_dir}/wombat-v3-forward'
gc_transport_dir = f'{forward_dir}/4a_postprocessing_gc/intermediates/daily'
gc_vert_grid_def = f'{here}/data/GEOS-Chem_47level_grid_def.csv'

# GC vertical grid definition
GC_VERT = pd.read_fwf(gc_vert_grid_def, skiprows=3,
                      names=['L', 'eta_edge', 'eta_mid', 'alt_km', 'p_hpa', 'Ap', 'Bp'])
