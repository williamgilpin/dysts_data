import warnings 
import numpy as np

import dysts.flows

from utils import smape_rolling


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

# equation_name = "Rossler"

num_ic = 20
training_length = 1000
context_length = 100
forecast_length = 64 # Maximum for these models
n_average = 20

## Make a testing set of initial conditions  
eq = getattr(dysts.flows, equation_name)()
integrator_args = {
    "pts_per_period": 100,
    "atol": 1e-12,
    "rtol": 1e-12,
}
## Pick num_ic random initial conditions
ic_traj = eq.make_trajectory(1000, pts_per_period=10)
np.random.seed(0)
sel_inds = np.random.choice(range(1000), size=num_ic, replace=False).astype(int)
ic_context_test = ic_traj[sel_inds, :]
print("t ", ic_context_test[0], flush=True)
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
save_str_reference = f"forecast_{eq.name}_true_chronos"
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

# has_gpu = torch.cuda.is_available()
# print("has gpu: ", torch.cuda.is_available(), flush=True)
# n = torch.cuda.device_count()
# print(f"{n} devices found.", flush=True)

# if has_gpu:
#     device = "cuda"
# else:
#     device = "cpu"

# eq = getattr(dysts.flows, equation_name)()

# integrator_args = {
#     "pts_per_period": 100,
#     "atol": 1e-12,
#     "rtol": 1e-12,
# }

# ## Make a testing set of initial conditions
# np.random.seed(0)
# ic_context_test = sample_initial_conditions(eq, num_ic, pts_per_period=10)
# traj_test = list()
# for ic in ic_context_test:
#     eq.ic = np.copy(ic)
#     traj = eq.make_trajectory(training_length + forecast_length, 
#                               timescale="Lyapunov", 
#                               method="Radau", **integrator_args)
#     traj_test.append(traj)
# traj_test = np.array(traj_test)
# traj_train = traj_test[:, :training_length] ## Training data for standard models
# traj_test_context = traj_test[:, training_length - context_length:training_length]
# ic_test = traj_test_context[:, -1] # Last seen point
# traj_test_forecast = traj_test[:, training_length:]
# save_str_reference = f"forecast_{eq.name}_true"
# np.save(save_str_reference, traj_test_forecast, allow_pickle=True)


from models import ChronosModel

## Run chronos zero-shot benchmark
for model_size in ["tiny", "mini", "small", "base", "large"]:
# for model_size in ["large"]:
    print(model_size, flush=True)
    all_traj_forecasts = list()
    for itr, traj in enumerate(traj_test_context):
        print(itr, flush=True)

        ## Reload model for each time series to prevent in-context learning
        model = ChronosModel(model=model_size, context=context_length, n_samples=n_average, 
                                    prediction_length=forecast_length, device=device)
        forecast_multivariate = np.array(model.predict(traj.T)).squeeze()
        all_traj_forecasts.append(forecast_multivariate.copy())

    all_traj_forecasts = np.array(all_traj_forecasts)
    all_traj_forecasts = np.moveaxis(all_traj_forecasts, (1, 3, 2), (2, 1, 3))

    save_str = f"forecast_{eq.name}_{model.name}"
    np.save(save_str, all_traj_forecasts, allow_pickle=True)


