"""
MCMC diagnostics and convergence assessment.
"""
import numpy as np
import matplotlib.pyplot as plt
import emcee

def compute_autocorr_time(chain, c=5.0, tol=50):
    """
    Compute integrated autocorrelation time using emcee's method.
    
    Parameters:
    -----------
    chain : array (nsteps, nwalkers, ndim)
        MCMC chain
    c : float
        Step size for autocorrelation window
    tol : float
        Minimum chain length / tau ratio
    
    Returns:
    --------
    tau : array (ndim,)
        Autocorrelation time per parameter
    """
    try:
        tau = emcee.autocorr.integrated_time(chain, c=c, tol=tol, quiet=True)
    except emcee.autocorr.AutocorrError:
        # Chain too short, return NaN
        tau = np.full(chain.shape[-1], np.nan)
    return tau

def compute_acceptance_fraction(sampler):
    """Compute mean acceptance fraction across walkers."""
    return np.mean(sampler.acceptance_fraction)

def compute_ess(nsamples, tau):
    """
    Compute effective sample size.
    ESS ~ N / (2 * tau)
    """
    if np.isnan(tau).any():
        return np.nan
    return nsamples / (2 * np.max(tau))

def check_convergence(chain, acceptance_frac, tau, min_length_factor=50):
    """
    Check if MCMC has converged.
    
    Criteria:
    1. Chain length > 50 * max(tau)
    2. Acceptance fraction in [0.15, 0.6]
    3. Stable posterior moments over last half
    """
    nsteps = chain.shape[0]
    
    # Criterion 1: Chain length
    if not np.isnan(tau).any():
        max_tau = np.max(tau)
        length_ok = nsteps > min_length_factor * max_tau
    else:
        length_ok = False
    
    # Criterion 2: Acceptance fraction
    accept_ok = 0.15 <= acceptance_frac <= 0.6
    
    # Criterion 3: Stable moments
    # Compare mean/std of first half vs second half
    mid = nsteps // 2
    first_half = chain[:mid].reshape(-1, chain.shape[-1])
    second_half = chain[mid:].reshape(-1, chain.shape[-1])
    
    mean_diff = np.abs(np.mean(first_half, axis=0) - np.mean(second_half, axis=0))
    std_pooled = np.sqrt((np.std(first_half, axis=0)**2 + np.std(second_half, axis=0)**2) / 2)
    
    # Require mean difference < 0.1 * pooled std
    stable_ok = np.all(mean_diff < 0.1 * std_pooled)
    
    converged = length_ok and accept_ok and stable_ok
    
    return converged, {
        'length_ok': length_ok,
        'accept_ok': accept_ok,
        'stable_ok': stable_ok
    }

def plot_trace(chain, param_names, save_path):
    """
    Plot trace plots for all parameters.
    
    Parameters:
    -----------
    chain : array (nsteps, nwalkers, ndim)
        MCMC chain
    param_names : list
        Parameter names
    save_path : str
        Output path
    """
    nsteps, nwalkers, ndim = chain.shape
    
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2*ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        # Plot all walkers
        for j in range(nwalkers):
            ax.plot(chain[:, j, i], alpha=0.3, lw=0.5)
        
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Step')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def compute_identifiability_metrics(samples, r_data, param_names):
    """
    Compute identifiability metrics.
    
    Returns:
    --------
    metrics : dict
        - posterior_widths: (p84 - p16) for each parameter
        - rmax_over_r0: maximum radius / median r0
        - constrained: bool flag
    """
    # Posterior widths (in log space for VCA)
    p16 = np.percentile(samples, 16, axis=0)
    p84 = np.percentile(samples, 84, axis=0)
    widths = p84 - p16
    
    # For VCA: samples are [log_v_inf, log_r0]
    # Convert r0 to linear for rmax comparison
    if 'log_r0' in param_names:
        idx_r0 = param_names.index('log_r0')
        r0_med = 10**np.median(samples[:, idx_r0])
    else:
        r0_med = np.nan
    
    rmax = np.max(r_data)
    rmax_over_r0 = rmax / r0_med if not np.isnan(r0_med) else np.nan
    
    # Constrained if:
    # - posterior widths < 0.3 dex for both params
    # - rmax / r0 > 2 (data extends beyond core)
    if len(widths) >= 2:
        w_v, w_r = widths[0], widths[1]
        constrained = (w_v < 0.3) and (w_r < 0.3) and (rmax_over_r0 > 2)
    else:
        constrained = False
    
    return {
        'posterior_widths': widths,
        'rmax_over_r0': rmax_over_r0,
        'constrained': constrained
    }

def compute_derived_parameters(samples, r_data, param_names):
    """
    Compute derived parameters from VCA posterior.
    
    Returns:
    --------
    derived : dict
        - v_eff: effective velocity at rmax
        - s: v_inf / r0 (velocity gradient scale)
    """
    if 'log_v_inf' not in param_names or 'log_r0' not in param_names:
        return {'v_eff': np.nan, 's': np.nan}
    
    idx_vinf = param_names.index('log_v_inf')
    idx_r0 = param_names.index('log_r0')
    
    v_inf_samples = 10**samples[:, idx_vinf]
    r0_samples = 10**samples[:, idx_r0]
    
    rmax = np.max(r_data)
    
    # v_eff = A(rmax) = v_inf * rmax / (rmax + r0)
    v_eff_samples = v_inf_samples * rmax / (rmax + r0_samples)
    
    # s = v_inf / r0
    s_samples = v_inf_samples / r0_samples
    
    return {
        'v_eff_samples': v_eff_samples,
        's_samples': s_samples,
        'v_eff_med': np.median(v_eff_samples),
        'v_eff_16': np.percentile(v_eff_samples, 16),
        'v_eff_84': np.percentile(v_eff_samples, 84),
        's_med': np.median(s_samples),
        's_16': np.percentile(s_samples, 16),
        's_84': np.percentile(s_samples, 84),
    }
