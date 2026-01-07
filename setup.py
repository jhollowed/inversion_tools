# This is the main setup config file, which allows local installaton of the package via pip.
# From the top-level directory (location of this file), simply run `pip install .`
# For more info, see the guidelines for minimal Python package structure at
# https://python-packaging.readthedocs.io/en/latest/minimal.html

from setuptools import setup

setup(name='inversion_tools',
      version='0.1',
      description='Package providing convenience functions for anaylzing flux inversion datasets',
      url='https://github.com/jhollowed/inversion_tools',
      author='Joe Hollowed',
      author_email='hollowed@umich.edu',
      packages=['inversion_tools'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])
