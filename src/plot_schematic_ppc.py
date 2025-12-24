
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import emcee
from src.models_extended import proposed_model_velocity, baryonic_velocity_sq
from src.mcmc_fitting import run_mcmc_vca

def plot_schematic_ppc(galaxy_name='NGC2403', sigma0=5.0):
    """
    Generates a schematic Posterior Predictive Check plot for a single galaxy,
    distinguishing between the Latent Curve Uncertainty (Credible Interval)
    and the Observation-Level Posterior Predictive Interval.
    """
    print(f"Generating Schematic PPC for {galaxy_name}...")
    
    # 1. Load Data
    data_path = f'data/{galaxy_name}_rotmod.dat'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    data = pd.read_csv(data_path, sep='\s+', comment='#', names=['r', 'v_obs', 'err', 'v_gas', 'v_disk', 'v_bul'], usecols=[0,1,2,3,4,5])
    r = data['r'].values
    v_obs = data['v_obs'].values
    err = data['err'].values
    v_gas = data['v_gas'].values
    v_disk = data['v_disk'].values
    v_bul = data['v_bul'].values
    
    # 2. Run Fit (Quick MCMC)
    nwalkers = 32
    nsteps = 2000
    
    sampler, param_names = run_mcmc_vca(r, v_obs, err, v_gas, v_disk, v_bul, sigma0=sigma0, 
                                      nwalkers=nwalkers, nsteps=nsteps, burn_in=1000)
    
    flat_samples = sampler.get_chain(discard=1000, flat=True)
    
    # 3. Generate Bands
    n_draws = 500
    idx = np.random.choice(len(flat_samples), size=n_draws, replace=False)
    
    v_models = []
    v_predictive = []
    
    # Pre-compute baryonic
    v_bar2 = baryonic_velocity_sq(v_gas, v_disk, v_bul)
    
    for i in idx:
        theta = flat_samples[i]
        log_v_inf, log_r0 = theta
        
        # Latent Curve 
        v_mod = proposed_model_velocity(r, 10**log_v_inf, 10**log_r0, v_bar2)
        v_models.append(v_mod)
        
        # Predictive Realization (Model + Noise)
        # Noise = sqrt(sigma_obs^2 + sigma_0^2)
        total_sigma = np.sqrt(err**2 + sigma0**2)
        noise = np.random.normal(0, total_sigma)
        v_pred = v_mod + noise
        v_predictive.append(v_pred)
        
    v_models = np.array(v_models)
    v_predictive = np.array(v_predictive)
    
    # Calculate Percentiles
    # Latent
    lat_50 = np.median(v_models, axis=0)
    lat_16 = np.percentile(v_models, 16, axis=0)
    lat_84 = np.percentile(v_models, 84, axis=0)
    
    # Predictive
    pred_16 = np.percentile(v_predictive, 16, axis=0)
    pred_84 = np.percentile(v_predictive, 84, axis=0)
    pred_025 = np.percentile(v_predictive, 2.5, axis=0)
    pred_975 = np.percentile(v_predictive, 97.5, axis=0)
    
    # 4. Plotting
    plt.figure(figsize=(9, 7))
    
    # Sort by R for clean bands
    sort_idx = np.argsort(r)
    r_s = r[sort_idx]
    
    # Plot Predictive Interval (Observation Level) first (background)
    plt.fill_between(r_s, pred_025[sort_idx], pred_975[sort_idx], 
                     color='lightblue', alpha=0.3, label='Predictive 95% (Obs + Model Var)')
    plt.fill_between(r_s, pred_16[sort_idx], pred_84[sort_idx], 
                     color='skyblue', alpha=0.5, label='Predictive 68%')
    
    # Plot Latent Interval (Model Uncertainty)
    plt.fill_between(r_s, lat_16[sort_idx], lat_84[sort_idx], 
                     color='darkblue', alpha=0.6, label='Latent Curve 68% (Model Var Only)')
    
    # Plot Median
    plt.plot(r_s, lat_50[sort_idx], 'b-', lw=2, label='Median Model')
    
    # Plot Data
    plt.errorbar(r, v_obs, yerr=err, fmt='ko', capsize=3, label='Data', zorder=10)
    
    plt.xlabel('Radius [kpc]', fontsize=14)
    plt.ylabel('Velocity [km/s]', fontsize=14)
    plt.title(f'Posterior Predictive Check: {galaxy_name}', fontsize=16)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.2)
    
    # Annotate sigma0
    plt.text(0.05, 0.95, f'$\sigma_{{eff}} = {sigma0}$ km/s', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    out_dir = 'results_supplements'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    save_path = os.path.join(out_dir, 'schematic_ppc_NGC2403.pdf')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved schematic plot to {save_path}")

if __name__ == "__main__":
    plot_schematic_ppc()
