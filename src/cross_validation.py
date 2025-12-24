"""
Cross-validation: fit on inner radii, predict on outer radii.
"""
import numpy as np
from .mcmc_fitting import run_mcmc_vca, run_mcmc_halo, summarize_mcmc
from .models_extended import (
    baryonic_velocity_sq, proposed_model_velocity,
    nfw_velocity_sq, burkert_velocity_sq, einasto_velocity_sq,
    pseudo_isothermal_velocity_sq, mond_rar_velocity,
    total_velocity_halo
)

def split_data(r, v_obs, err_v, v_gas, v_disk, v_bul, train_fraction=0.7):
    """
    Split data into training (inner radii) and test (outer radii).
    Returns: (r_train, v_train, err_train, ...), (r_test, v_test, err_test, ...)
    """
    n_total = len(r)
    n_train = int(n_total * train_fraction)
    
    # Sort by radius
    idx_sorted = np.argsort(r)
    idx_train = idx_sorted[:n_train]
    idx_test = idx_sorted[n_train:]
    
    train_data = (
        r[idx_train], v_obs[idx_train], err_v[idx_train],
        v_gas[idx_train], v_disk[idx_train], v_bul[idx_train]
    )
    
    test_data = (
        r[idx_test], v_obs[idx_test], err_v[idx_test],
        v_gas[idx_test], v_disk[idx_test], v_bul[idx_test]
    )
    
    return train_data, test_data

def predict_vca(r_test, samples, v_gas_test, v_disk_test, v_bul_test, free_ml=False):
    """
    Predict velocities using VCA model posterior samples.
    Returns: v_pred (median prediction), v_pred_std (prediction uncertainty)
    """
    n_samples = len(samples)
    v_predictions = np.zeros((n_samples, len(r_test)))
    
    for i, theta in enumerate(samples):
        if free_ml:
            log_v_inf, log_r0, Y_d, Y_b = theta
            v_bar2 = baryonic_velocity_sq(v_gas_test, v_disk_test, v_bul_test, Y_d, Y_b)
        else:
            log_v_inf, log_r0 = theta
            v_bar2 = baryonic_velocity_sq(v_gas_test, v_disk_test, v_bul_test)
        
        v_predictions[i] = proposed_model_velocity(r_test, 10**log_v_inf, 10**log_r0, v_bar2)
    
    v_pred = np.median(v_predictions, axis=0)
    v_pred_std = np.std(v_predictions, axis=0)
    
    return v_pred, v_pred_std

def predict_halo(r_test, samples, v_gas_test, v_disk_test, v_bul_test, model_type):
    """Predict velocities using halo model posterior samples."""
    n_samples = len(samples)
    v_predictions = np.zeros((n_samples, len(r_test)))
    
    v_bar2 = baryonic_velocity_sq(v_gas_test, v_disk_test, v_bul_test)
    
    for i, theta in enumerate(samples):
        if model_type == 'NFW':
            v_predictions[i] = total_velocity_halo(r_test, nfw_velocity_sq, theta, v_bar2)
        elif model_type == 'Burkert':
            v_predictions[i] = total_velocity_halo(r_test, burkert_velocity_sq, theta, v_bar2)
        elif model_type == 'Einasto':
            v_predictions[i] = total_velocity_halo(r_test, einasto_velocity_sq, theta, v_bar2)
        elif model_type == 'PseudoIso':
            v_predictions[i] = total_velocity_halo(r_test, pseudo_isothermal_velocity_sq, theta, v_bar2)
        elif model_type == 'MOND':
            log_a0, = theta
            v_bar = np.sqrt(v_bar2)
            v_predictions[i] = mond_rar_velocity(r_test, v_bar, 10**log_a0)
    
    v_pred = np.median(v_predictions, axis=0)
    v_pred_std = np.std(v_predictions, axis=0)
    
    return v_pred, v_pred_std

def compute_prediction_metrics(v_obs, v_pred, err_v):
    """
    Compute prediction error metrics.
    Returns: RMSE, normalized RMSE
    """
    residuals = v_obs - v_pred
    rmse = np.sqrt(np.mean(residuals**2))
    rmse_normalized = np.sqrt(np.mean((residuals / err_v)**2))
    
    return rmse, rmse_normalized
