from setuptools import setup

setup(name = 'dysts_data',
      packages=['dysts_data'],
      version='0.1',
      install_requires = [
          'numpy',
          'scipy',
          'pandas',
          'sdeint',
          'numba',
          'darts',
          'sktime',
          'tsfresh',
          'torch',
          'sympy'
      ],
      package_dir={'dysts_data': 'dysts_data'},
      package_data={'dysts_data': ['data/*', 'benchmarks/*']},
     )
