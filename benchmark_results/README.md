
## A zero-shot benchmark for chaos forecasting

This repository contains code and benchmark outputs for the zero-shot chaos forecasting benchmark.

The scripts used to run the benchmarks may be found in the `scripts` directory. These include `.py` files for running the benchmarks, and `.sbatch` files for submitting jobs to a cluster. The folders `zero-shot` and `trained` contain `.npz` files with the benchmark outputs. These are raw forecasts, ƒrom which all error metrics are subsequently computed.

All time series metrics and trajectories are from the `dysts` library.

## Example usage

To run the Chronos benchmark for the Lorenz system, use the following command:

```bash
python scripts/chronos_benchmarks.py "Lorenz"
```

To run the baseline benchmarks (NBEATS, LSTM, etc) for the Lorenz system, use the following command:

```bash
python scripts/darts_benchmarks.py "Lorenz"
```
The outputs of running these benchmarks for all systems are stored in the `zero-shot` and `trained` directories, respectively. For example, the file `zero-shot/chronos_benchmarks_context_512_granularity_30/forecast_Lorenz_base_granularity30.npy` contains the zero-shot forecast for the Lorenz system with the `base`-size Chronos model at a granularity of 30 for 20 different initial conditions. The file `zero-shot/chronos_benchmarks_context_512_granularity_30/forecast_Aizawa_granularity30_true_chronos.npy` contains the corresponding true trajectories.

### Requirements

+ numpy
+ darts
+ torch
+ dysts
