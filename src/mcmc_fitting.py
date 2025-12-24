"""
MCMC fitting infrastructure using emcee for Bayesian parameter estimation.
Updated with adaptive sampling and tuned moves.
"""
import numpy as np
import emcee
from multiprocessing import Pool
from .models_extended import (
    baryonic_velocity_sq, proposed_model_velocity,
    nfw_velocity_sq, burkert_velocity_sq, einasto_velocity_sq,
    pseudo_isothermal_velocity_sq, mond_rar_velocity,
    total_velocity_halo
)

def log_prior_vca_fixed_ml(theta):
    """
    Prior for VCA with fixed M/L: [log_v_inf, log_r0]
    """
    log_v_inf, log_r0 = theta
    if -1 < log_v_inf < 4 and -2 < log_r0 < 3:
        return 0.0
    return -np.inf

def log_prior_vca_free_ml(theta):
    """
    Prior for VCA with free M/L: [log_v_inf, log_r0, Upsilon_disk, Upsilon_bulge]
    """
    log_v_inf, log_r0, Y_disk, Y_bulge = theta
    
    if not (-1 < log_v_inf < 4 and -2 < log_r0 < 3):
        return -np.inf
    
    log_prior = 0.0
    
    if Y_disk > 0:
        log_prior += -0.5 * ((np.log(Y_disk) - np.log(0.5)) / 0.2)**2
    else:
        return -np.inf
    
    if Y_bulge > 0:
        log_prior += -0.5 * ((np.log(Y_bulge) - np.log(0.7)) / 0.2)**2
    else:
        return -np.inf
    
    return log_prior

def log_prior_halo(theta, model_type):
    """Generic prior for halo models."""
    if model_type in ['NFW', 'Burkert', 'PseudoIso']:
        log_rho0, log_rs = theta
        if 4 < log_rho0 < 10 and -1 < log_rs < 3:
            return 0.0
    elif model_type == 'Einasto':
        log_rhos, log_rs, alpha = theta
        if 4 < log_rhos < 10 and -1 < log_rs < 3 and 0.1 < alpha < 0.5:
            return 0.0
    elif model_type == 'MOND':
        log_a0, = theta
        if -28 < log_a0 < -25:
            return 0.0
    return -np.inf

def log_likelihood(theta, r, v_obs, sigma_eff, model_func):
    """Log-likelihood for Gaussian errors."""
    v_model = model_func(theta)
    if not np.all(np.isfinite(v_model)):
        return -np.inf
    chi2 = np.sum(((v_obs - v_model) / sigma_eff)**2)
    log_like = -0.5 * chi2
    return log_like

def log_probability(theta, r, v_obs, sigma_eff, model_func, log_prior_func):
    """Combined log-probability."""
    lp = log_prior_func(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, r, v_obs, sigma_eff, model_func)

def run_adaptive_mcmc_vca(r, v_obs, err_v, v_gas, v_disk, v_bul, 
                          sigma0=0.0, free_ml=False, 
                          nwalkers=32, burn_in=1000, 
                          min_steps=2000, max_steps=10000,
                          move_scale=2.0):
    """
    Run Adaptive MCMC for VCA model.
    Stops when N_steps > 50 * tau and tau fits stable.
    
    Parameters:
    -----------
    move_scale : float
        Scale factor 'a' for StretchMove. Increase to >2.0 to lower acceptance.
    """
    sigma_eff = np.sqrt(err_v**2 + sigma0**2)
    
    if free_ml:
        ndim = 4
        param_names = ['log_v_inf', 'log_r0', 'Y_disk', 'Y_bulge']
        def model_func(theta):
            log_v_inf, log_r0, Y_d, Y_b = theta
            v_bar2 = baryonic_velocity_sq(v_gas, v_disk, v_bul, Y_d, Y_b)
            return proposed_model_velocity(r, 10**log_v_inf, 10**log_r0, v_bar2)
        log_prior_func = log_prior_vca_free_ml
        p0 = np.array([2.0, 0.7, 0.5, 0.7]) + 0.1 * np.random.randn(nwalkers, ndim)
    else:
        ndim = 2
        param_names = ['log_v_inf', 'log_r0']
        v_bar2 = baryonic_velocity_sq(v_gas, v_disk, v_bul)
        def model_func(theta):
            log_v_inf, log_r0 = theta
            return proposed_model_velocity(r, 10**log_v_inf, 10**log_r0, v_bar2)
        log_prior_func = log_prior_vca_fixed_ml
        p0 = np.array([2.0, 0.7]) + 0.1 * np.random.randn(nwalkers, ndim)
    
    # Configure moves
    moves = [(emcee.moves.StretchMove(a=move_scale), 1.0)]
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(r, v_obs, sigma_eff, model_func, log_prior_func),
        moves=moves
    )
    
    # Run burn-in
    p0, _, _ = sampler.run_mcmc(p0, burn_in, progress=False)
    sampler.reset()
    
    # Run adaptive production
    old_tau = np.inf
    
    for sample in sampler.sample(p0, iterations=max_steps, progress=False):
        # Check convergence every 2000 steps
        if sampler.iteration % 2000 == 0:
            tau = sampler.get_autocorr_time(tol=0)
            tau_max = np.max(tau)
            
            # Check convergence
            converged = np.all(tau * 50 < sampler.iteration)
            tau_stable = np.abs(old_tau - tau_max) / tau_max < 0.1
            
            if converged and tau_stable:
                break
            
            old_tau = tau_max
            
    return sampler, param_names

def run_mcmc_vca(r, v_obs, err_v, v_gas, v_disk, v_bul, 
                 sigma0=0.0, free_ml=False, 
                 nwalkers=32, nsteps=2000, burn_in=500):
    """Legacy wrappers for backward compatibility."""
    return run_adaptive_mcmc_vca(
        r, v_obs, err_v, v_gas, v_disk, v_bul,
        sigma0=sigma0, free_ml=free_ml,
        nwalkers=nwalkers, burn_in=burn_in,
        min_steps=nsteps, max_steps=nsteps
    )
