# inversion_tools

A set of tools for reading, slicing, and processing TM5 and GEOS-Chem flux and mole fractions from the WOMBAT global flux-inversion framework.

See docstrings at function definitions for arguments and usage.

## Installation:

1. Configure a Conda environment with the needed dependencies

   ```bash
   conda create -n climo-env -c conda-forge python=3.11 xesmf esmpy esmf netcdf4 metpy matplotlib numpy xarray
   conda activate climo_env
   ```

2. Follow the installation instructions for the required `climate_utils` package at https://github.com/jhollowed/climate_utils

3. Clone the repo

    ```bash
    git clone git@github.com:jhollowed/inversion_tools.git
    ```

4. Locally install the package

   ```bash
   cd climate_utils
   pip install -e .
   ```

   The `-e` flag to `pip install` installs the package with a symlink to the soruce code, so that later updates via git pull do not require re-installation.

5. Edit user-specific caching locations specified by `tm5_meteo_cache` and `gc_meteo_cache` in `inversion_tools/constants.py`. These should be writable locations with free storage space, which will be used for intermediate staging of meteorological variables obtained either from the internet, or on disk at MSU.

## Usage:

Import the functions as

```python
import inversion_tools as it
```

Define a `Reader` object as for e.g. the ocean's residual carbon flux:

```python
reader = it.Reader(source='ocean', component='residual', processing_dir='/your/preferred/location/', quiet=False)
```

and then read prior or posterior mole fractions for e.g. TransCom region 14 during the first 6 months of 2016, at the equator:

```python
state_prior     = reader.read_prior_mf(region=14, start_date='2016-01', end_date='2016-06', lat=0)
state_posterior = reader.read_posterior_mf(sample='mean', region=14, start_date='2016-01', end_date='2016-06', lat=0)
```

Currently working for GC only; work-in-progress to include TM5.
