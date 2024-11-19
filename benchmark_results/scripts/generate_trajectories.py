import warnings 
import numpy as np
import dysts.flows

num_ic = 20
training_length = 512
context_length = 512
forecast_length = 300 # Maximum for these models
n_average = 1
pts_per_period = 30

dirname = f"datasets_context_{context_length}_granularity_{pts_per_period}"
## Check if the directory exists, if not, create it
import os
if not os.path.exists(dirname):
    os.makedirs(dirname)

if training_length < context_length:
    warnings.warn("Training length has been increased to be greater than context length")
    training_length = context_length + 1

# from dysts.systems import get_attractor/_list
from dysts.base import get_attractor_list

attractor_list = get_attractor_list()
for equation_name in attractor_list:
    print(equation_name, flush=True)
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
    save_str_reference = f"forecast_{eq.name}_granularity{pts_per_period}_true_chronos_test"
    save_str_reference = os.path.join(dirname, save_str_reference)
    np.save(save_str_reference, traj_test_forecast, allow_pickle=True)

    save_str_reference2 = f"context_{eq.name}_granularity{pts_per_period}_true_chronos_context"
    save_str_reference2 = os.path.join(dirname, save_str_reference2)
    np.save(save_str_reference2, traj_test_context, allow_pickle=True)

