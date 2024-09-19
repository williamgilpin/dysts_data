This secondary data repository hosts precomputed time series datasets and benchmark results for our main repository, [dysts](https://github.com/williamgilpin/dysts)

```bash
    pip install dysts[data]
```

## Installation

The key dependencies are

+ Python 3+
+ numpy
+ scipy
+ pandas
+ sdeint (optional, but required for stochastic dynamics)
+ numba (optional, but speeds up generation of trajectories)
+ darts (used for forecasting benchmarks)
+ sktime (used for classification benchmarks)
+ tsfresh (used for statistical quantity extraction)
+ pytorch (used for neural network benchmarks)
+ sympy (used for equation analysis benchmarks)