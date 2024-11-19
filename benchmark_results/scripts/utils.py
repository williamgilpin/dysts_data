import numpy as np
from dysts.metrics import smape

# def smape_rolling(ts1, ts2):
#     """Return the smape versus time for two time series."""
#     n = min(ts1.shape[0], ts2.shape[0])
#     all_smape = list()
#     for i in range(n):
#         smape_val = smape(ts1[:i], ts2[:i])
#         all_smape.append(smape_val)
#     return np.array(all_smape)

def smape_rolling(ts1, ts2):
    """Return the smape versus time for two time series."""
    n = min(ts1.shape[0], ts2.shape[0])
    all_smape = list()
    for i in range(1, n+1):
        smape_val = smape(ts1[:i], ts2[:i])
        all_smape.append(smape_val)
    return np.array(all_smape)

def vpt(arr, threshold=30):
    """
    Find the first time index at which an array exceeds a threshold.

    Args:
        arr (np.ndarray): The array to search. The first dimension should be a horizoned
            smape series.
        threshold (float): The threshold to search for.
    r
    Returns:
        int: The first time index at which the array exceeds the threshold.
    """
    all_exceed_times = list()
    for i in range(arr.shape[1]):
        exceed_times = np.where(arr[:, i] > threshold)[0]
        if len(exceed_times) == 0:
            tind = len(arr[:, i])
        else:
            tind = exceed_times[0]
        all_exceed_times.append(tind)
    return all_exceed_times

def nrmse(x, xhat):
    """
    Given a univariate forecast and ground truth, compute the NRMSE.
    
    Args:
        x (np.ndarray): The ground truth, a time series of shape (nt, d).
        xhat (np.ndarray): The forecast, a time series of shape (nt, d).

    Returns:
        float: The NRMSE.
    """
    denom = np.var(x, axis=1)
    numerator = np.linalg.norm(x - xhat, axis=1) ** 2
    return np.sqrt(np.mean(numerator / denom, axis=0))

def horizoned_nrmse(x, xhat):
    """Given a horizoned forecast and ground truth, compute the NRMSE as a function of time"""
    nt = min(x.shape[0], xhat.shape[0])
    nrmse_vals = list()
    for i in range(1, nt+1):
        nrmse_vals.append(nrmse(x[:i], xhat[:i]))
    nrmse_vals = np.array(nrmse_vals)
    return nrmse_vals

def horizoned_smape(x, xhat):
    """Given a horizoned forecast and ground truth, compute the SMAPE as a function of time"""
    nt = min(x.shape[0], xhat.shape[0])
    smape_vals = list()
    for i in range(1, nt+1):
        smape_vals.append(smape(x[:i], xhat[:i]))
    smape_vals = np.array(smape_vals)
    return smape_vals

def vpt_smape(x, xhat, threshold=30):
    """
    Find the first time index at which an array exceeds a threshold.

    Args:
        x (np.ndarray): The ground truth, a time series of shape (nt, 1).
        xhat (np.ndarray): The forecast, a time series of shape (nt, 1).
        threshold (float): The threshold to search for.

    Returns:
        int: The first time index at which the array exceeds the threshold.
    """
    arr = horizoned_smape(x, xhat)
    exceed_times = np.where(arr > threshold)[0]
    if len(exceed_times) == 0:
        tind = len(arr)
    else:
        tind = exceed_times[0]
    return tind

def vpt_nrmse(x, xhat, threshold=0.5):
    """
    Find the first time index at which an array exceeds a threshold.

    Args:
        arr (np.ndarray): The array to search. The first dimension should be a horizoned
            smape series.
        threshold (float): The threshold to search for.
    r
    Returns:
        int: The first time index at which the array exceeds the threshold.
    """
    arr = horizoned_nrmse(x, xhat)
    exceed_times = np.where(arr > threshold)[0]
    if len(exceed_times) == 0:
        tind = len(arr)
    else:
        tind = exceed_times[0]
    return tind