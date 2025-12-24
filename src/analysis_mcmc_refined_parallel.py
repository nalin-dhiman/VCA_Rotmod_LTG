"""
Comprehensive MCMC analysis with adaptive sampling and PPC coverage.
PARALLELIZED VERSION.
"""
import os
import glob
import numpy as np
import pandas as pd
import emcee
from multiprocessing import Pool, cpu_count
from .data_loader import load_galaxy
from .mcmc_fitting import run_adaptive_mcmc_vca
from .mcmc_diagnostics import (
    compute_autocorr_time, compute_acceptance_fraction, compute_ess,
    plot_trace, compute_identifiability_metrics, compute_derived_parameters
)
from .posterior_analysis import plot_posterior_predictive
from .models_extended import baryonic_velocity_sq, proposed_model_velocity

DATA_DIR = 'data'
RESULTS_DIR = 'results_mcmc_refined'
TRACE_DIR = os.path.join(RESULTS_DIR, 'trace_plots')
POSTPRED_DIR = os.path.join(RESULTS_DIR, 'posterior_predictive')

# Global parameter for worker
SIGMA0 = 5.0
NWALKERS = 32

def check_ppc_coverage(r, v_obs, err_v, v_gas, v_disk, v_bul, samples, param_names):
    """Check coverage."""
    idx = np.random.choice(len(samples), size=min(500, len(samples)), replace=False)
    selected_samples = samples[idx]
    
    idx_vinf = param_names.index('log_v_inf')
    idx_r0 = param_names.index('log_r0')
    v_bar2 = baryonic_velocity_sq(v_gas, v_disk, v_bul)
        
    v_models = []
    for theta in selected_samples:
        v_mod = proposed_model_velocity(r, 10**theta[idx_vinf], 10**theta[idx_r0], v_bar2)
        v_models.append(v_mod)
    v_models = np.array(v_models)
    
    coverage_data = []
    for i in range(len(r)):
        # PPC with measurement noise
        sigma_eff_i = np.sqrt(err_v[i]**2 + SIGMA0**2)
        v_rep = v_models[:, i] + np.random.normal(0, sigma_eff_i, size=len(v_models))
        
        p16, p84 = np.percentile(v_rep, [16, 84])
        p025, p975 = np.percentile(v_rep, [2.5, 97.5])
        
        coverage_data.append({
            'radius': r[i],
            'v_obs': v_obs[i],
            'in_68': (v_obs[i] >= p16) and (v_obs[i] <= p84),
            'in_95': (v_obs[i] >= p025) and (v_obs[i] <= p975)
        })
    return coverage_data

def process_galaxy(fpath):
    """Worker function for single galaxy."""
    try:
        meta, df = load_galaxy(fpath)
        if df is None: return None
        
        galaxy_name = meta['Name']
        
        # Clean data
        df = df.dropna(subset=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk'])
        df = df[(df['Rad'] > 0) & (df['Vobs'] > 0)]
        if len(df) < 5: return None
        
        r = df['Rad'].values
        v_obs = df['Vobs'].values
        err_v = df['errV'].values
        err_v[err_v < 1.0] = 1.0
        v_gas = df['Vgas'].values
        v_disk = df['Vdisk'].values
        v_bul = df['Vbul'].values
        
        # Run Adaptive MCMC
        sampler, param_names = run_adaptive_mcmc_vca(
            r, v_obs, err_v, v_gas, v_disk, v_bul,
            sigma0=SIGMA0,
            nwalkers=NWALKERS,
            burn_in=1000,
            min_steps=2000,
            max_steps=10000,
            move_scale=3.0
        )
        
        # Metrics
        tau = sampler.get_autocorr_time(quiet=True)
        accept_frac = np.mean(sampler.acceptance_fraction)
        samples = sampler.get_chain(flat=True)
        n_samples = len(samples)
        
        converged = (sampler.iteration > 50 * np.max(tau)) if np.all(np.isfinite(tau)) else False
        ess = n_samples / (2 * np.max(tau)) if np.all(np.isfinite(tau)) else np.nan
        
        id_metrics = compute_identifiability_metrics(samples, r, param_names)
        derived = compute_derived_parameters(samples, r, param_names)
        
        medians = np.median(samples, axis=0)
        p16 = np.percentile(samples, 16, axis=0)
        p84 = np.percentile(samples, 84, axis=0)
        
        # PPC Coverage
        ppc_points = check_ppc_coverage(r, v_obs, err_v, v_gas, v_disk, v_bul, samples, param_names)
        for pt in ppc_points:
            pt['Name'] = galaxy_name
        
        # Plotting (only for first few alphabetically to avoid race conditions/IO spam)
        # Or just overwrite, it's fine
        chain = sampler.get_chain()
        plot_trace(chain, param_names, os.path.join(TRACE_DIR, f'{galaxy_name}_trace.png'))
        plot_posterior_predictive(r, v_obs, err_v, v_gas, v_disk, v_bul, samples, param_names, 
                                  galaxy_name, os.path.join(POSTPRED_DIR, f'{galaxy_name}_postpred.png'))
        
        return {
            'diag': {
                'Name': galaxy_name,
                'N_data': len(r),
                'tau_v_inf': tau[0] if len(tau)>0 else np.nan,
                'tau_r0': tau[1] if len(tau)>1 else np.nan,
                'acceptance_frac': accept_frac,
                'ESS': ess,
                'N_steps': sampler.iteration,
                'converged': converged
            },
            'summary': {
                'Name': galaxy_name,
                'v_inf_med': 10**medians[0],
                'v_inf_16': 10**p16[0],
                'v_inf_84': 10**p84[0],
                'r0_med': 10**medians[1],
                'r0_16': 10**p16[1],
                'r0_84': 10**p84[1],
                'v_eff_med': derived['v_eff_med'],
                's_med': derived['s_med'],
                'width_log_v': id_metrics['posterior_widths'][0],
                'width_log_r': id_metrics['posterior_widths'][1],
                'rmax_over_r0': id_metrics['rmax_over_r0'],
                'constrained': id_metrics['constrained']
            },
            'coverage': ppc_points
        }
    except Exception as e:
        print(f"FAILED {fpath}: {e}")
        return None

def run_parameter_estimation():
    for d in [RESULTS_DIR, TRACE_DIR, POSTPRED_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    files = glob.glob(os.path.join(DATA_DIR, '*_rotmod.dat'))
    files.sort()
    
    # files = files[:10] # Debug limit
    
    print(f"Starting parallel MCMCs on {len(files)} galaxies using {cpu_count()} cores.")
    
    with Pool(processes=cpu_count()) as pool:
        results = list(pool.imap_unordered(process_galaxy, files))
    
    # Aggregate
    diagnostics_list = []
    summary_list = []
    coverage_list = []
    
    for res in results:
        if res:
            diagnostics_list.append(res['diag'])
            summary_list.append(res['summary'])
            coverage_list.extend(res['coverage'])
            
            # Progress print
            # print(f"Finished {res['diag']['Name']}")
            
    # Save
    pd.DataFrame(diagnostics_list).to_csv(os.path.join(RESULTS_DIR, 'mcmc_diagnostics.csv'), index=False)
    pd.DataFrame(summary_list).to_csv(os.path.join(RESULTS_DIR, 'mcmc_summary_all_galaxies.csv'), index=False)
    pd.DataFrame(coverage_list).to_csv(os.path.join(RESULTS_DIR, 'coverage_results.csv'), index=False)
    
    print("All done.")

if __name__ == "__main__":
    run_parameter_estimation()
