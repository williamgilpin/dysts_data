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

# equation_name = "Lorenz"
# equation_name = "ArnoldWeb"

num_ic = 20*2
training_length = 512
context_length = 512
# forecast_length = 400 # Maximum for these models
forecast_length = 300 # Maximum for these models
# n_average = 20
n_average = 1
pts_per_period = 40
# pts_per_period = 30

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
#all_hyperparameters = {"input_chunk_length": context_length, "output_chunk_length": forecast_length}
all_hyperparameters = {"input_chunk_length": context_length, "output_chunk_length": 1}


from darts.models.forecasting.transformer_model import TransformerModel
print("Transformer", flush=True)
all_pred = list()
for i in range(len(traj_test_context)):
    print(i, flush=True)
    train_data = np.copy(traj_train)[i]
    y_train_ts = TimeSeries.from_values(train_data)
    try:
        pc = dict()
        pc.update(all_hyperparameters)
        pc.update({"pl_trainer_kwargs": gpu_params})
        model = TransformerModel(**pc)
    except:
        model = TransformerModel(**all_hyperparameters)
    
    ## Fit the forecasting model on the given data
    model.fit(y_train_ts)
    
    ## Attempt to predict the validation data. If it fails, then the model returns
    ## a None value for the prediction.
    try:
        y_val_pred = model.predict(forecast_length).values().squeeze()
    except:
        print(f"Failed to predict {equation_name} {model_name}", flush=True)
        y_val_pred = np.array([None] * forecast_length)
    y_val_pred = np.array(y_val_pred.tolist())
    all_pred.append(np.copy(y_val_pred))
all_pred = np.array(all_pred)
save_str = f"forecast_{eq.name}_Transformer"
np.save(save_str, all_pred, allow_pickle=True)


from darts.models.forecasting.nbeats import NBEATSModel
print("NBEATS", flush=True)
all_pred = list()
for i in range(len(traj_test_context)):
    print(i, flush=True)
    train_data = np.copy(traj_train)[i]
    y_train_ts = TimeSeries.from_values(train_data)
    try:
        pc = dict()
        pc.update(all_hyperparameters)
        pc.update({"pl_trainer_kwargs": gpu_params})
        model = NBEATSModel(**pc)
    except:
        model = NBEATSModel(**all_hyperparameters)
    
    ## Fit the forecasting model on the given data
    model.fit(y_train_ts)
    
    ## Attempt to predict the validation data. If it fails, then the model returns
    ## a None value for the prediction.
    try:
        y_val_pred = model.predict(forecast_length).values().squeeze()
    except:
        print(f"Failed to predict {equation_name} {model_name}", flush=True)
        y_val_pred = np.array([None] * forecast_length)
    y_val_pred = np.array(y_val_pred.tolist())
    all_pred.append(np.copy(y_val_pred))
all_pred = np.array(all_pred)
save_str = f"forecast_{eq.name}_NBEATS"
np.save(save_str, all_pred, allow_pickle=True)


from darts.models.forecasting.rnn_model import RNNModel
print("LSTM", flush=True)
# all_hyperparameters = {"input_chunk_length": context_length, "model": "LSTM", "training_length": training_length - 1}
# all_hyperparameters = {"input_chunk_length": int(0.1 * context_length), "model": "LSTM", "training_length": context_length}
all_hyperparameters = {"input_chunk_length": context_length, "model": "LSTM", "training_length": context_length + 1}
all_pred = list()
for i in range(len(traj_test_context)):
    print(i, flush=True)
    train_data = np.copy(traj_train)[i]
    y_train_ts = TimeSeries.from_values(train_data)
    try:
        pc = dict()
        pc.update(all_hyperparameters)
        pc.update({"pl_trainer_kwargs": gpu_params})
        model = RNNModel(**pc)
    except:
        model = RNNModel(**all_hyperparameters)
    
    ## Fit the forecasting model on the given data
    model.fit(y_train_ts)
    
    ## Attempt to predict the validation data. If it fails, then the model returns
    ## a None value for the prediction.
    try:
        y_val_pred = model.predict(forecast_length).values().squeeze()
    except:
        print(f"Failed to predict {equation_name} {model_name}", flush=True)
        y_val_pred = np.array([None] * forecast_length)
    y_val_pred = np.array(y_val_pred.tolist())
    all_pred.append(np.copy(y_val_pred))
all_pred = np.array(all_pred)
save_str = f"forecast_{eq.name}_LSTM"
np.save(save_str, all_pred, allow_pickle=True)


from darts.models.forecasting.tide_model import TiDEModel
print("TiDE", flush=True)
all_hyperparameters = {"input_chunk_length": context_length, "output_chunk_length": 1}
all_pred = list()
for i in range(len(traj_test_context)):
    print(i, flush=True)
    train_data = np.copy(traj_train)[i]
    y_train_ts = TimeSeries.from_values(train_data)
    try:
        pc = dict()
        pc.update(all_hyperparameters)
        pc.update({"pl_trainer_kwargs": gpu_params})
        model = TiDEModel(**pc)
    except:
        model = TiDEModel(**all_hyperparameters)
    
    ## Fit the forecasting model on the given data
    model.fit(y_train_ts)
    
    ## Attempt to predict the validation data. If it fails, then the model returns
    ## a None value for the prediction.
    try:
        y_val_pred = model.predict(forecast_length).values().squeeze()
    except:
        print(f"Failed to predict {equation_name} {model_name}", flush=True)
        y_val_pred = np.array([None] * forecast_length)
    y_val_pred = np.array(y_val_pred.tolist())
    all_pred.append(np.copy(y_val_pred))
all_pred = np.array(all_pred)
save_str = f"forecast_{eq.name}_TiDE"
np.save(save_str, all_pred, allow_pickle=True)



from esn import NVARForecast
print("NVAR", flush=True)
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
np.save(save_str, all_pred, allow_pickle=True)




from darts.models.forecasting.linear_regression_model import LinearRegressionModel
print("Linear", flush=True)
all_hyperparameters = {"lags": context_length}
all_pred = list()
for i in range(len(traj_test_context)):
    print(i, flush=True)
    train_data = np.copy(traj_train)[i]
    y_train_ts = TimeSeries.from_values(train_data)
    model = LinearRegressionModel(**all_hyperparameters)
    
    ## Fit the forecasting model on the given data
    model.fit(y_train_ts)
    
    ## Attempt to predict the validation data. If it fails, then the model returns
    ## a None value for the prediction.
    try:
        y_val_pred = model.predict(forecast_length).values().squeeze()
    except:
        print(f"Failed to predict {equation_name} {model_name}", flush=True)
        y_val_pred = np.array([None] * forecast_length)
    y_val_pred = np.array(y_val_pred.tolist())
    all_pred.append(np.copy(y_val_pred))
all_pred = np.array(all_pred)
save_str = f"forecast_{eq.name}_Linear"
np.save(save_str, all_pred, allow_pickle=True)




