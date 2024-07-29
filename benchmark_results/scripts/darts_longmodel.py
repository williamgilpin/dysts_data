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


training_length = 1000
forecast_length = 500
pts_per_period = 100


## Make a testing set of initial conditions  
eq = getattr(dysts.flows, equation_name)()
integrator_args = {
    "pts_per_period": pts_per_period,
    "atol": 1e-12,
    "rtol": 1e-12,
}
## Pick num_ic random initial conditions
traj_all = eq.make_trajectory(training_length + forecast_length, 
                             pts_per_period=pts_per_period, 
                             atol=1e-12, rtol=1e-12, 
                             timescale="Lyapunov", 
                             method="Radau")
traj_train = traj_all[:training_length]
traj_test = traj_all[training_length:]
save_str_reference = f"traj_{eq.name}_granularity{pts_per_period}_true"
save_str_reference = os.path.join(dirname, save_str_reference)
np.save(save_str_reference, traj_test, allow_pickle=True)



import torch
has_gpu = torch.cuda.is_available()
print("has gpu: ", has_gpu, flush=True)
n = torch.cuda.device_count()
print(f"{n} devices found.", flush=True)
torch.set_float32_matmul_precision("high")
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

## Neural benchmarks
import darts
from darts import TimeSeries
import darts.models

## Load the hyperparameter file
import json
hyperparameter_path = f"/hyperparameters_multivariate_train_multivariate__pts_per_period_{pts_per_period}__periods_12.json"
with open(hyperparameter_path, "r") as file:
    all_hyperparameters = json.load(file)


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
    # elif model_name == "TiDE":
    #     from darts.models.forecasting.tide_model import TiDEModel
    #     return TiDEModel
    # elif model_name == "Linear":
    #     from darts.models.forecasting.linear_regression_model import LinearRegressionModel
    #     return LinearRegressionModel
    # elif model_name == "NVAR":
    #     from esn import NVARForecast
    #     nvar_curry = lambda x: NVARForecast(x, delay=context_length)
    #     return NVARForecast
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
import os
from darts import TimeSeries
def run_forecast_benchmark(model_name, traj_train, all_hyperparameters, forecast_length, gpu_params, equation_name, dirname=""):
    ModelClass = import_model_by_name(model_name)
    all_pred = list()
    

    train_data = np.copy(traj_train)
    y_train_ts = TimeSeries.from_values(train_data)
    
    ## Update the hyperparameters with the GPU parameters, then initialize the model
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
    
    model.save(f"trained_{model_name}_{equation_name}_granularity{pts_per_period}.pt")

    all_pred = np.array(all_pred)
    save_str = f"forecast_{equation_name}_{model_name}"
    save_str = os.path.join(dirname, save_str)
    np.save(save_str, all_pred, allow_pickle=True)
    

for model_name in ["Transformer", "NBEATS", "LSTM"]:
    print(model_name, flush=True)
    hyperparameters = all_hyperparameters[equation_name][model_name]
    run_forecast_benchmark(
        model_name, traj_train, hyperparameters, 
        forecast_length, gpu_params, equation_name,
    )