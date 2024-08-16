import warnings 
import numpy as np

import dysts.flows

## Read system name from the command line
import argparse
# Function to process command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a command-line argument.')
    parser.add_argument('arg1', type=str, help='A string argument')
    args = parser.parse_args()
    return args
args = parse_arguments()
equation_name = args.arg1
print(equation_name, flush=True)

num_ic = 20
training_length = 512
context_length = 512 # Not used
forecast_length = 300 # Maximum horizon
pts_per_period = 30

dirname = f"darts_benchmarks_granularity_{pts_per_period}"
## Check if the directory exists, if not, create it
import os
if not os.path.exists(dirname):
    os.makedirs(dirname)

if training_length < context_length:
    warnings.warn("Training length has been increased to be greater than context length")
    training_length = context_length + 1

## Make a testing set of initial conditions  
eq = getattr(dysts.flows, equation_name)()
integrator_args = {
    "pts_per_period": pts_per_period,
    "atol": 1e-12,
    "rtol": 1e-12,
}
## Pick num_ic random initial conditions
ic_traj = eq.make_trajectory(1000, pts_per_period=10, atol=1e-12, rtol=1e-12)
np.random.seed(0)
sel_inds = np.random.choice(range(1000), size=num_ic, replace=False).astype(int)
# sel_inds = np.linspace(0, 1000, 20).astype(int)
ic_context_test = ic_traj[sel_inds, :]
traj_test = list()
for ic in ic_context_test:
    eq.ic = np.copy(ic)
    traj = eq.make_trajectory(training_length + forecast_length,
                              timescale="Lyapunov",
                              method="Radau", **integrator_args)
    traj_test.append(traj)
traj_test = np.array(traj_test)
traj_train = traj_test[:, :training_length] ## Training data for standard models                                     
traj_test_context = traj_test[:, training_length - context_length:training_length]
ic_test = traj_test_context[:, -1] # Last seen point                                                                 
traj_test_forecast = traj_test[:, training_length:]
save_str_reference = f"forecast_{eq.name}_granularity_{pts_per_period}_true_dysts"
save_str_reference = os.path.join(dirname, save_str_reference)
np.save(save_str_reference, traj_test_forecast, allow_pickle=True)


import torch
has_gpu = torch.cuda.is_available()
print("has gpu: ", torch.cuda.is_available(), flush=True)
n = torch.cuda.device_count()
print(f"{n} devices found.", flush=True)
if has_gpu:
    device = "cuda"
else:
    device = "cpu"
torch.set_float32_matmul_precision("high")


## Neural benchmarks
import darts
from darts import TimeSeries
import darts.models

if not has_gpu:
    warnings.warn("No GPU found.")
    gpu_params = {
        "accelerator": "cpu",                                                                                    
    }
else:
    warnings.warn("GPU working.")
    gpu_params = {
        "accelerator": "gpu",
        "devices": [0], # only one GPU
#         "devices": n,
         #    "gpus": [0],  # use "devices" instead of "gpus" for PyTorch Lightning >= 1.7.                                    #    "auto_select_gpu": True,                                                                                
    }
pl_trainer_kwargs = [gpu_params] # global
model_static_dict = {"pl_trainer_kwargs" : pl_trainer_kwargs}


def add_gpu_kwargs(kw):
    """Given a set of keyword arguments for a GPU run, update the arguments to exploit the GPU.
    Requires global arguments gpu_params and has_gpu"""
    if has_gpu:
        try:
            kw.update({"pl_trainer_kwargs": gpu_params})
        except Exception as e:
            print(f"Failed to initialize with GPU parameters: {e}", flush=True)
    return kw

def import_model_by_name(model_name, *args, **kwargs):
    if model_name == "NBEATS":
        from darts.models.forecasting.nbeats import NBEATSModel
        return NBEATSModel(*args, **add_gpu_kwargs(kwargs))
    elif model_name == "Transformer":
        from darts.models.forecasting.transformer_model import TransformerModel
        return TransformerModel(*args, **add_gpu_kwargs(kwargs))
    elif model_name == "LSTM":
        from darts.models.forecasting.rnn_model import RNNModel
        return RNNModel(*args, **add_gpu_kwargs(kwargs))
    elif model_name == "TiDE":
        from darts.models.forecasting.tide_model import TiDEModel
        return TiDEModel(*args, **add_gpu_kwargs(kwargs))
    elif model_name == "Linear":
        from darts.models.forecasting.linear_regression_model import LinearRegressionModel
        return LinearRegressionModel(*args, **kwargs)
    elif model_name == "NVAR":
        from esn import NVARForecast
        return NVARForecast(*args, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
def run_forecast_benchmark(model_name, traj_train, all_hyperparameters, forecast_length, gpu_params, equation_name, dirname=None):

    all_pred = list()
    
    for i in range(len(traj_train)):
        print(i, flush=True)
        train_data = np.copy(traj_train)[i]
        try:
            if model_name == "NVAR":
                model = import_model_by_name(model_name, traj_train.shape[-1], **all_hyperparameters)
                model.fit(np.copy(train_data))
                y_val_pred = model.predict(forecast_length).squeeze().copy()
            else:
                y_train_ts = TimeSeries.from_values(train_data.copy())
                model = import_model_by_name(model_name, **all_hyperparameters)
                model.fit(y_train_ts)
                y_val_pred = model.predict(forecast_length).values().squeeze()
        except Exception as e:
            print(f"Failed to predict {equation_name} {model_name}: {e}", flush=True)
            y_val_pred = np.array([None] * forecast_length)
        
        y_val_pred = np.array(y_val_pred.tolist())
        all_pred.append(np.copy(y_val_pred))
    
    all_pred = np.array(all_pred)
    save_str = f"forecast_{equation_name}_{model_name}_granularity{pts_per_period}"
    save_str = os.path.join(dirname, save_str)
    np.save(save_str, all_pred, allow_pickle=True)

    
def smape(y, yhat):
    """Symmetric mean absolute percentage error.

    Args:
        y (np.ndarray): An array containing the true values.
        yhat (np.ndarray): An array containing the predicted values.

    Returns:
        (float): the scalar sMAPE score
    """
    assert len(yhat) == len(y)
    n = len(y)
    err = np.abs(y - yhat) / (np.abs(y) + np.abs(yhat)) * 100
    return (2 / n) * np.sum(err)

import itertools
from darts import TimeSeries
# from sklearn.model_selection import TimeSeriesSplit
def cross_validate_hyperparameters(model_name, traj_train, hyperparams, hyperparameter_candidates, gpu_params, forecast_length=20, n_splits=5):
    all_scores = list()
    ## Create a stucture with all possible combinations of suggested hyperparameters
    all_combinations = list(itertools.product(*hyperparameter_candidates.values()))
    keys = list(hyperparameter_candidates.keys())

    for combination in all_combinations:
        ## Make the hyperparameters dictionary for a particular combination
        hyperparams_combination = dict(zip(keys, combination))
        
        split_index = int(0.85 * traj_train.shape[1])
        scores = list()
        for i in range(traj_train.shape[0]): # Iterate over all trajectories
            train_data, test_data = traj_train[i, :split_index].copy(), traj_train[i, split_index:].copy()
            print(train_data.shape, test_data.shape, traj_train.shape)
            
            pc = dict()
            pc.update(hyperparams)
            pc.update(hyperparams_combination)
            
            try:
                
                if model_name == "NVAR":
                    model = import_model_by_name(model_name, train_data.shape[-1], **pc)
                    model.fit(np.copy(train_data))
                    y_pred = model.predict(len(test_data)).squeeze().copy()
                else:
                    y_train_ts = TimeSeries.from_values(np.copy(train_data))
                    
                    model = import_model_by_name(model_name, **pc)
                    model.fit(y_train_ts)
                    y_pred = model.predict(len(test_data)).values().squeeze().copy()
                
                y_test_ts = TimeSeries.from_values(np.copy(test_data))
                y_true = y_test_ts.values().squeeze().copy()
                score = smape(y_pred, y_true)  # Example scoring metric: MSE
                scores.append(score)
                
            except Exception as e:
                print(f"Failed to cross-validate with parameters {hyperparams_combination}: {e}", flush=True)
                scores.append(np.nan)

        
        all_scores.append((np.nanmean(scores), hyperparams_combination))
       
    print(all_scores)
    best_score, best_params = min(all_scores, key=lambda x: x[0])
    best_params.update(hyperparams)
    print(f"Best hyperparameters for {model_name}: {best_params} with score: {best_score}", flush=True)
    
    return best_params


context_lengths = np.logspace(np.log10(5), np.log10(1024), 20).astype(int)
for context_length in context_lengths:
    print(context_length, flush=True)

    ## Reselect the training data
    traj_train = traj_test[:, :training_length] ## Training data for standard models                                     
    traj_test_context = traj_test[:, training_length - context_length:training_length]
    ic_test = traj_test_context[:, -1] # Last seen point                                                                 
    traj_test_forecast = traj_test[:, training_length:]

    model_name = "NVAR"
    print(model_name, flush=True)
    hyperparameters = {}
    hyperparameter_candidates = {"delay": [2, 5, 10, 15, 25]}
    hyperparameters = cross_validate_hyperparameters(model_name, traj_train, hyperparameters, hyperparameter_candidates, gpu_params)
    run_forecast_benchmark(
        model_name, traj_train, hyperparameters,
        forecast_length, gpu_params, equation_name, dirname=dirname
    )

    model_name = "NBEATS"
    print(model_name, flush=True)
    hyperparameters = {"output_chunk_length": 1}
    hyperparameter_candidates = {"input_chunk_length": [5, 25, 50, 75, 100]}
    hyperparameters = cross_validate_hyperparameters(model_name, traj_train, hyperparameters, hyperparameter_candidates, gpu_params)
    run_forecast_benchmark(
        model_name, traj_train, hyperparameters, 
        forecast_length, gpu_params, equation_name, dirname=dirname
    )

    model_name = "TiDE"
    print(model_name, flush=True)
    hyperparameters = {"output_chunk_length": 1}
    hyperparameter_candidates = {"input_chunk_length": [5, 25, 50, 75, 100]}
    hyperparameters = cross_validate_hyperparameters(model_name, traj_train, hyperparameters, hyperparameter_candidates, gpu_params)
    run_forecast_benchmark(
        model_name, traj_train, hyperparameters, 
        forecast_length, gpu_params, equation_name, dirname=dirname
    )

