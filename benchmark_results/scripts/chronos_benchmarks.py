import warnings 
import numpy as np

import dysts.flows

## Read system name from the command line
import argparse
# Function to process command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a command-line argument.')
    parser.add_argument('arg1', type=str, help='The name of the dynamical system to be processed')
    parser.add_argument('arg2', type=int, nargs='?', const=30, default=30, help='An optional integer argument for pts_per_period')
    args = parser.parse_args()
    return args
args = parse_arguments()
equation_name = args.arg1
pts_per_period = int(args.arg2)
print(equation_name, flush=True)

# num_ic = 20
# training_length = 1000
# context_length = 100
# forecast_length = 64 # Maximum for these models
# n_average = 20

num_ic = 20
training_length = 512
context_length = 512
forecast_length = 300 # Maximum for these models
# forecast_length = 500 # Maximum for these models
# n_average = 20
n_average = 1
# pts_per_period = 40
# pts_per_period = 25
# pts_per_period = 30

dirname = f"chronos_benchmarks_context_{context_length}_granularity_{pts_per_period}"
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
save_str_reference = f"forecast_{eq.name}_granularity{pts_per_period}_true_chronos"
save_str_reference = os.path.join(dirname, save_str_reference)
np.save(save_str_reference, traj_test_forecast, allow_pickle=True)

from models import ChronosModel

## Run chronos zero-shot benchmark
#for model_size in ["tiny", "mini", "small", "base", "large"]:
for model_size in ["large"]:
    model = ChronosModel(model=model_size, context=context_length, n_samples=n_average, 
                                        prediction_length=forecast_length)
    print(model_size, flush=True)
    
    save_str = f"forecast_{eq.name}_{model_size}_granularity{pts_per_period}"
    save_str = os.path.join(dirname, save_str)
    if os.path.exists(save_str):
        print("Skipping, already found: " + save_str, flush=True)
        continue
    
    all_traj_forecasts = list()
    ## Loop over the replicate trajectories
    for itr, traj in enumerate(traj_test_context):
        print(itr, flush=True)
        forecast_multivariate = np.array(model.predict(traj.T))
        all_traj_forecasts.append(forecast_multivariate.copy())

        all_traj_forecasts2 = np.array(all_traj_forecasts)
        all_traj_forecasts2 = np.moveaxis(all_traj_forecasts2, (1, 3, 2), (2, 1, 3))

        np.save(save_str, all_traj_forecasts2, allow_pickle=True)
