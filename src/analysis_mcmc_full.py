"""
Comprehensive MCMC analysis with full diagnostics for all galaxies.
"""
import os
import glob
import numpy as np
import pandas as pd
import emcee
from .data_loader import load_galaxy
from .mcmc_fitting import run_mcmc_vca
from .mcmc_diagnostics import (
    compute_autocorr_time, compute_acceptance_fraction, compute_ess,
    check_convergence, plot_trace, compute_identifiability_metrics,
    compute_derived_parameters
)
from .posterior_analysis import plot_posterior_predictive

DATA_DIR = 'data'
RESULTS_DIR = 'results_mcmc_full'
TRACE_DIR = os.path.join(RESULTS_DIR, 'trace_plots')
POSTPRED_DIR = os.path.join(RESULTS_DIR, 'posterior_predictive')

def run_full_mcmc_analysis(sigma0=5.0, nwalkers=32, nsteps=2000, burn_in=500):
    """
    Run MCMC on ALL galaxies with comprehensive diagnostics.
    """
    # Create directories
    for d in [RESULTS_DIR, TRACE_DIR, POSTPRED_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    files = glob.glob(os.path.join(DATA_DIR, '*_rotmod.dat'))
    files.sort()
    
    diagnostics_list = []
    summary_list = []
    
    print(f"Running full MCMC analysis on {len(files)} galaxies...")
    print(f"Settings: {nwalkers} walkers, {nsteps} steps, {burn_in} burn-in")
    
    for i, fpath in enumerate(files):
        meta, df = load_galaxy(fpath)
        if df is None:
            continue
        
        galaxy_name = meta['Name']
        print(f"[{i+1}/{len(files)}] Processing {galaxy_name}...", end=' ')
        
        # Clean data
        df = df.dropna(subset=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk'])
        df = df[(df['Rad'] > 0) & (df['Vobs'] > 0)]
        if len(df) < 5:
            print("SKIP (too few points)")
            continue
        
        r = df['Rad'].values
        v_obs = df['Vobs'].values
        err_v = df['errV'].values
        err_v[err_v < 1.0] = 1.0
        v_gas = df['Vgas'].values
        v_disk = df['Vdisk'].values
        v_bul = df['Vbul'].values
        
        # Run MCMC - but we need the full sampler object for diagnostics
        # Modify to return sampler
        sigma_eff = np.sqrt(err_v**2 + sigma0**2)
        
        from .models_extended import baryonic_velocity_sq, proposed_model_velocity
        from .mcmc_fitting import log_prior_vca_fixed_ml, log_probability
        
        v_bar2 = baryonic_velocity_sq(v_gas, v_disk, v_bul)
        
        def model_func(theta):
            log_v_inf, log_r0 = theta
            return proposed_model_velocity(r, 10**log_v_inf, 10**log_r0, v_bar2)
        
        log_prior_func = log_prior_vca_fixed_ml
        param_names = ['log_v_inf', 'log_r0']
        ndim = 2
        
        # Initialize walkers
        p0 = np.array([2.0, 0.7]) + 0.1 * np.random.randn(nwalkers, ndim)
        
        # Create sampler
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(r, v_obs, sigma_eff, model_func, log_prior_func)
        )
        
        try:
            # Run MCMC
            sampler.run_mcmc(p0, nsteps, progress=False)
            
            # Get chain (nsteps, nwalkers, ndim)
            chain = sampler.get_chain()
            
            # Compute diagnostics
            tau = compute_autocorr_time(chain, tol=10)  # Relaxed tolerance
            accept_frac = compute_acceptance_fraction(sampler)
            ess = compute_ess(nsteps * nwalkers, tau)
            
            converged, conv_details = check_convergence(chain, accept_frac, tau)
            
            # Get samples after burn-in
            samples = sampler.get_chain(discard=burn_in, flat=True)
            
            # Compute identifiability
            id_metrics = compute_identifiability_metrics(samples, r, param_names)
            
            # Compute derived parameters
            derived = compute_derived_parameters(samples, r, param_names)
            
            # Posterior summaries
            medians = np.median(samples, axis=0)
            p16 = np.percentile(samples, 16, axis=0)
            p84 = np.percentile(samples, 84, axis=0)
            
            # Convert to linear space for reporting
            v_inf_med = 10**medians[0]
            v_inf_16 = 10**p16[0]
            v_inf_84 = 10**p84[0]
            r0_med = 10**medians[1]
            r0_16 = 10**p16[1]
            r0_84 = 10**p84[1]
            
            # Store diagnostics
            diag_entry = {
                'Name': galaxy_name,
                'N_data': len(r),
                'tau_v_inf': tau[0] if len(tau) > 0 else np.nan,
                'tau_r0': tau[1] if len(tau) > 1 else np.nan,
                'acceptance_frac': accept_frac,
                'ESS': ess,
                'converged': converged,
                'length_ok': conv_details['length_ok'],
                'accept_ok': conv_details['accept_ok'],
                'stable_ok': conv_details['stable_ok'],
            }
            diagnostics_list.append(diag_entry)
            
            # Store summary
            summary_entry = {
                'Name': galaxy_name,
                'v_inf_med': v_inf_med,
                'v_inf_16': v_inf_16,
                'v_inf_84': v_inf_84,
                'r0_med': r0_med,
                'r0_16': r0_16,
                'r0_84': r0_84,
                'v_eff_med': derived['v_eff_med'],
                'v_eff_16': derived['v_eff_16'],
                'v_eff_84': derived['v_eff_84'],
                's_med': derived['s_med'],
                's_16': derived['s_16'],
                's_84': derived['s_84'],
                'width_log_v_inf': id_metrics['posterior_widths'][0],
                'width_log_r0': id_metrics['posterior_widths'][1],
                'rmax_over_r0': id_metrics['rmax_over_r0'],
                'constrained': id_metrics['constrained'],
                'converged': converged,
            }
            summary_list.append(summary_entry)
            
            # Generate trace plot (for first 20 galaxies to save space)
            if i < 20:
                plot_trace(chain, param_names,
                          os.path.join(TRACE_DIR, f'{galaxy_name}_trace.png'))
            
            # Generate posterior predictive plot (for first 20)
            if i < 20:
                plot_posterior_predictive(
                    r, v_obs, err_v, v_gas, v_disk, v_bul,
                    samples, param_names, galaxy_name,
                    os.path.join(POSTPRED_DIR, f'{galaxy_name}_postpred.png')
                )
            
            status = "CONVERGED" if converged else "UNCONVERGED"
            flag = "CONSTRAINED" if id_metrics['constrained'] else "UNCONSTRAINED"
            print(f"{status}, {flag}")
            
        except Exception as e:
            print(f"FAILED: {e}")
            continue
    
    # Save results
    df_diag = pd.DataFrame(diagnostics_list)
    df_diag.to_csv(os.path.join(RESULTS_DIR, 'mcmc_diagnostics.csv'), index=False)
    
    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'mcmc_summary_all_galaxies.csv'), index=False)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete: {len(df_summary)} galaxies")
    print(f"Converged: {df_diag['converged'].sum()} / {len(df_diag)}")
    print(f"Constrained: {df_summary['constrained'].sum()} / {len(df_summary)}")
    print(f"{'='*60}")
    
    return df_diag, df_summary

if __name__ == "__main__":
    run_full_mcmc_analysis(sigma0=5.0, nwalkers=32, nsteps=2000, burn_in=500)
