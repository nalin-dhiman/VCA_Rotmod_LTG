"""
Posterior predictive plotting and prior sensitivity analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from .models_extended import proposed_model_velocity, baryonic_velocity_sq

def plot_posterior_predictive(r, v_obs, err_v, v_gas, v_disk, v_bul,
                               samples, param_names, galaxy_name, save_path,
                               n_draws=200):
    """
    Plot posterior predictive distribution with credible bands.
    
    Parameters:
    -----------
    samples : array (n_samples, ndim)
        Posterior samples
    n_draws : int
        Number of posterior draws to plot
    """
    # Randomly select n_draws samples
    idx = np.random.choice(len(samples), size=min(n_draws, len(samples)), replace=False)
    selected_samples = samples[idx]
    
    # Compute v_model for each sample
    v_models = []
    for theta in selected_samples:
        if 'log_v_inf' in param_names and 'log_r0' in param_names:
            idx_vinf = param_names.index('log_v_inf')
            idx_r0 = param_names.index('log_r0')
            
            # Check if free M/L
            if len(theta) == 4:  # Free M/L
                Y_d, Y_b = theta[2], theta[3]
                v_bar2 = baryonic_velocity_sq(v_gas, v_disk, v_bul, Y_d, Y_b)
            else:  # Fixed M/L
                v_bar2 = baryonic_velocity_sq(v_gas, v_disk, v_bul)
            
            v_mod = proposed_model_velocity(r, 10**theta[idx_vinf], 10**theta[idx_r0], v_bar2)
            v_models.append(v_mod)
    
    v_models = np.array(v_models)
    
    # Compute percentiles
    v_med = np.median(v_models, axis=0)
    v_16 = np.percentile(v_models, 16, axis=0)
    v_84 = np.percentile(v_models, 84, axis=0)
    v_025 = np.percentile(v_models, 2.5, axis=0)
    v_975 = np.percentile(v_models, 97.5, axis=0)
    
    # Sort for plotting
    sort_idx = np.argsort(r)
    r_s = r[sort_idx]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Top panel: Data + posterior predictive
    ax1.errorbar(r, v_obs, yerr=err_v, fmt='ko', label='Data', capsize=3, alpha=0.6, zorder=10)
    
    # Plot credible bands
    ax1.fill_between(r_s, v_025[sort_idx], v_975[sort_idx], 
                     color='red', alpha=0.2, label='95% CI')
    ax1.fill_between(r_s, v_16[sort_idx], v_84[sort_idx],
                     color='red', alpha=0.4, label='68% CI')
    ax1.plot(r_s, v_med[sort_idx], 'r-', lw=2, label='Median', zorder=5)
    
    ax1.set_ylabel('Velocity [km/s]', fontsize=12)
    ax1.set_title(f'{galaxy_name} - Posterior Predictive', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Residuals
    resid = (v_obs - v_med) / err_v
    ax2.plot(r_s, resid[sort_idx], 'ko', alpha=0.6)
    ax2.axhline(0, color='r', lw=2)
    ax2.axhline(2, color='k', ls=':', alpha=0.5)
    ax2.axhline(-2, color='k', ls=':', alpha=0.5)
    ax2.set_ylabel(r'Residuals ($\sigma$)', fontsize=12)
    ax2.set_xlabel('Radius [kpc]', fontsize=12)
    ax2.set_ylim(-5, 5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(save_path, dpi=150)
    plt.close()

def run_prior_sensitivity_test(r, v_obs, err_v, v_gas, v_disk, v_bul,
                                sigma0=5.0, nwalkers=32, nsteps=1500, burn_in=500):
    """
    Run MCMC under two different priors to test sensitivity.
    
    Returns:
    --------
    results : dict
        - samples_broad: samples under broad prior
        - samples_physical: samples under physical prior
        - kl_divergence: approximate KL divergence between posteriors
    """
    from .mcmc_fitting import run_mcmc_vca
    
    # Prior A: Broad
    # Modify run_mcmc_vca to accept custom prior bounds
    # For now, we'll use the existing function and compare
    
    samples_broad, param_names = run_mcmc_vca(
        r, v_obs, err_v, v_gas, v_disk, v_bul,
        sigma0=sigma0, free_ml=False,
        nwalkers=nwalkers, nsteps=nsteps, burn_in=burn_in
    )
    
    # For Prior B, we'd need to modify the prior function
    # This is a placeholder - in practice, modify log_prior_vca_fixed_ml
    # to accept bounds as arguments
    
    # Simplified: just return broad prior results
    # Full implementation would require modifying mcmc_fitting.py
    
    return {
        'samples_broad': samples_broad,
        'param_names': param_names,
        'note': 'Full prior sensitivity requires modified prior bounds'
    }

def compare_prior_sensitivity(samples_broad, samples_narrow):
    """
    Compare two posterior distributions from different priors.
    
    Returns:
    --------
    metrics : dict
        - median_diff: difference in medians
        - width_ratio: ratio of posterior widths
        - overlap: fraction of samples that overlap
    """
    med_broad = np.median(samples_broad, axis=0)
    med_narrow = np.median(samples_narrow, axis=0)
    
    # Posterior widths
    width_broad = np.percentile(samples_broad, 84, axis=0) - np.percentile(samples_broad, 16, axis=0)
    width_narrow = np.percentile(samples_narrow, 84, axis=0) - np.percentile(samples_narrow, 16, axis=0)
    
    return {
        'median_diff': np.abs(med_broad - med_narrow),
        'width_ratio': width_broad / width_narrow,
        'prior_dominated': np.any(width_broad / width_narrow > 2)
    }
