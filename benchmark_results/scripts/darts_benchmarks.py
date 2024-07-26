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


num_ic = 20*2
training_length = 512
context_length = 512
# forecast_length = 400 # Maximum for these models
forecast_length = 300 # Maximum for these models
# n_average = 20
n_average = 1
# pts_per_period = 40
# pts_per_period = 30
pts_per_period = 25

dirname = f"darts_benchmarks_context_{context_length}_granularity_{pts_per_period}"
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
save_str_reference = f"forecast_{eq.name}_true_dysts"
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
pl_trainer_kwargs = [gpu_params]
model_static_dict = {"pl_trainer_kwargs" : pl_trainer_kwargs}


def import_model_by_name(model_name, *args):
    if model_name == "NBEATS":
        from darts.models.forecasting.nbeats import NBEATSModel
        return NBEATSModel
    elif model_name == "Transformer":
        from darts.models.forecasting.transformer_model import TransformerModel
        return TransformerModel
    elif model_name == "LSTM":
        from darts.models.forecasting.rnn_model import RNNModel
        return RNNModel
    elif model_name == "TiDE":
        from darts.models.forecasting.tide_model import TiDEModel
        return TiDEModel
    elif model_name == "Linear":
        from darts.models.forecasting.linear_regression_model import LinearRegressionModel
        return LinearRegressionModel
    # elif model_name == "NVAR":
    #     from esn import NVARForecast
    #     nvar_curry = lambda x: NVARForecast(x, delay=context_length)
    #     return NVARForecast
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
def run_forecast_benchmark(model_name, traj_train, all_hyperparameters, forecast_length, gpu_params, equation_name, dirname=None):
    ModelClass = import_model_by_name(model_name)
    all_pred = list()
    
    for i in range(len(traj_train)):
        print(i, flush=True)
        train_data = np.copy(traj_train)[i]
        y_train_ts = TimeSeries.from_values(train_data)
        
        try:
            pc = dict()
            pc.update(all_hyperparameters)
            pc.update({"pl_trainer_kwargs": gpu_params})
            model = ModelClass(**pc)
        except Exception as e:
            print(f"Failed to initialize with GPU parameters: {e}", flush=True)
            model = ModelClass(**all_hyperparameters)
        
        # Fit the forecasting model on the given data
        model.fit(y_train_ts)
        
        # Attempt to predict the validation data
        try:
            y_val_pred = model.predict(forecast_length).values().squeeze()
        except Exception as e:
            print(f"Failed to predict {equation_name} {model_name}: {e}", flush=True)
            y_val_pred = np.array([None] * forecast_length)
        
        y_val_pred = np.array(y_val_pred.tolist())
        all_pred.append(np.copy(y_val_pred))
    
    all_pred = np.array(all_pred)
    save_str = f"forecast_{equation_name}_{model_name}"
    save_str = os.path.join(dirname, save_str)
    np.save(save_str, all_pred, allow_pickle=True)

import itertools
from darts import TimeSeries
from sklearn.model_selection import TimeSeriesSplit
def cross_validate_hyperparameters(model_name, traj_train, hyperparams, hyperparameter_candidates, forecast_length=20, n_splits=5):
    ModelClass = import_model_by_name(model_name)
    all_scores = list()
    ## Create a stucture with all possible combinations of suggested hyperparameters
    all_combinations = list(itertools.product(*hyperparameter_candidates.values()))
    keys = list(hyperparameter_candidates.keys())

    for combination in all_combinations:
        ## Make the hyperparameters dictionary for a particular combination
        hyperparams_combination = dict(zip(keys, combination))
        scores = list()
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        ## Cross-validate the model
        for train_index, test_index in tscv.split(traj_train):
            train_data, test_data = traj_train[train_index], traj_train[test_index]
            y_train_ts = TimeSeries.from_values(train_data)
            y_test_ts = TimeSeries.from_values(test_data)
            
            pc = dict()
            pc.update(hyperparams)
            pc.update(hyperparams_combination)
            
            try:
                model = ModelClass(**pc)
                model.fit(y_train_ts)
                y_pred = model.predict(len(test_data)).values().squeeze()
                y_true = y_test_ts.values().squeeze()
                
                score = np.mean((y_pred - y_true) ** 2)  # Example scoring metric: MSE
                scores.append(score)
                
            except Exception as e:
                print(f"Failed to cross-validate with parameters {hyperparams_combination}: {e}", flush=True)
                scores.append(np.nan)
        
        mean_score = np.mean(scores)
        all_scores.append((mean_score, hyperparams_combination))
    
    best_score, best_params = min(all_scores, key=lambda x: x[0])
    best_params.update(hyperparams)
    print(f"Best hyperparameters: {best_params} with score: {best_score}", flush=True)
    
    return best_params
    

model_name = "Transformer"
print(model_name, flush=True)
context_length = 100
hyperparameters = {"input_chunk_length": context_length, "output_chunk_length": 1}
hyperparameter_candidates = {"input_chunk_length": [5, 25, 50, 75, 100]}
hyperparameters = cross_validate_hyperparameters(model_name, traj_train, hyperparameters, hyperparameter_candidates)
run_forecast_benchmark(
    model_name, traj_train, hyperparameters, context_length, 
    forecast_length, gpu_params, equation_name, dirname=dirname
)

model_name = "NBEATS"
print(model_name, flush=True)
context_length = 100
hyperparameters = {"input_chunk_length": context_length, "output_chunk_length": 1}
hyperparameter_candidates = {"input_chunk_length": [5, 25, 50, 75, 100]}
hyperparameters = cross_validate_hyperparameters(model_name, traj_train, hyperparameters, hyperparameter_candidates)
run_forecast_benchmark(
    model_name, traj_train, hyperparameters, context_length, 
    forecast_length, gpu_params, equation_name, dirname=dirname
)

model_name = "LSTM"
print(model_name, flush=True)
context_length = 100
hyperparameters = {"input_chunk_length": context_length, "output_chunk_length": 1}
hyperparameter_candidates = {"input_chunk_length": [5, 25, 50, 75, 100]}
hyperparameters = cross_validate_hyperparameters(model_name, traj_train, hyperparameters, hyperparameter_candidates)
run_forecast_benchmark(
    model_name, traj_train, hyperparameters, context_length, 
    forecast_length, gpu_params, equation_name, dirname=dirname
)

model_name = "TiDE"
print(model_name, flush=True)
context_length = 100
hyperparameters = {"input_chunk_length": context_length, "output_chunk_length": 1}
hyperparameter_candidates = {"input_chunk_length": [5, 25, 50, 75, 100]}
hyperparameters = cross_validate_hyperparameters(model_name, traj_train, hyperparameters, hyperparameter_candidates)
run_forecast_benchmark(
    model_name, traj_train, hyperparameters, context_length, 
    forecast_length, gpu_params, equation_name, dirname=dirname
)


model_name = "Linear"
print(model_name, flush=True)
context_length = 100
hyperparameters = {"lags": context_length}
hyperparameter_candidates = {"lags": [5, 25, 50, 75, 100]}
hyperparameters = cross_validate_hyperparameters(model_name, traj_train, hyperparameters, hyperparameter_candidates)
run_forecast_benchmark(
    model_name, traj_train, hyperparameters, context_length, 
    forecast_length, gpu_params, equation_name, dirname=dirname
)

from esn import NVARForecast
print("NVAR", flush=True)
# hyperparameters = {"delay": context_length}
hyperparameters = {"delay": 100}
hyperparameter_candidates = {"delay": [5, 25, 50, 75, 100]}
all_pred = list()
for i in range(len(traj_test_context)):
    print(i, flush=True)
    train_data = np.copy(traj_train)[i]
    try:
        model = NVARForecast(train_data.shape[-1], delay=context_length)
        model.fit(train_data)
        y_val_pred = model.predict(forecast_length)
    except AssertionError:
        print("Integration error encountered, skipping this entry for now")
        y_val_pred = np.array([None] * forecast_length)
    all_pred.append(np.copy(y_val_pred.squeeze()))
all_pred = np.array(all_pred)
save_str = f"forecast_{eq.name}_NVAR"
save_str = os.path.join(dirname, save_str)
np.save(save_str, all_pred, allow_pickle=True)

