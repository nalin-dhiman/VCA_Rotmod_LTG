
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from src.analysis_model_comparison import chi2_reduced_vca

def unit_test_reduced_vca(galaxy_name='UGC06787'):
    print(f"--- Unit Test: {galaxy_name} ---")
    
    # 1. Load Data
    data_path = f'data/{galaxy_name}_rotmod.dat'
    if not os.path.exists(data_path):
        print("Data file not found.")
        return
        
    data = pd.read_csv(data_path, sep='\s+', comment='#', names=['r', 'v_obs', 'err', 'v_gas', 'v_disk', 'v_bul'], usecols=[0,1,2,3,4,5])
    r = data['r'].values
    v_obs = data['v_obs'].values
    err = data['err'].values
    v_gas = data['v_gas'].values
    v_disk = data['v_disk'].values
    v_bul = data['v_bul'].values
    
    # Compute V_bar
    v_bar_sq = np.maximum(v_gas*np.abs(v_gas) + v_disk**2 + v_bul**2, 0)
    v_bar = np.sqrt(v_bar_sq)
    
    print("Radius range:", r.min(), r.max())
    print("V_obs range:", v_obs.min(), v_obs.max())
    print("V_bar range:", v_bar.min(), v_bar.max())
    
    # Check for negative DM room
    diff = v_obs - v_bar
    print("Mean (V_obs - V_bar):", np.mean(diff))
    print("Min (V_obs - V_bar):", np.min(diff))
    
    if np.min(diff) < -10:
        print("WARNING: V_obs is significantly below V_bar. VCA (and any DM-only halo) cannot fit this.")
    
    # 2. Reduced VCA Setup
    v_max_obs = np.max(v_obs)
    print(f"V_max (observed) fixed to: {v_max_obs}")
    
    # 3. Manual Chi2 Scan
    log_r0_grid = np.linspace(-2, 3, 50)
    chi2_list = []
    
    for lr0 in log_r0_grid:
        chi2 = chi2_reduced_vca([lr0], r, v_obs, err, v_bar_sq, v_max_obs)
        chi2_list.append(chi2)
        
    min_chi2 = min(chi2_list)
    best_lr0 = log_r0_grid[np.argmin(chi2_list)]
    print(f"Grid Scan Best Chi2: {min_chi2:.2f} at log_r0 = {best_lr0:.2f}")
    
    # 4. Optimization
    res = minimize(chi2_reduced_vca, [0.0], args=(r, v_obs, err, v_bar_sq, v_max_obs), method='Nelder-Mead')
    print(f"Optimization Result: Success={res.success}, Chi2={res.fun:.2f}, x={res.x}")
    
    # 5. Full VCA comparison (approx)
    # Why is Full VCA good?
    # Maybe Full VCA lowers v_inf? 
    # Or maybe Full VCA fits a v_inf essentially 0? 
    # Let's assume MCMC result for this galaxy from summary
    
    try:
        summary = pd.read_csv('results_mcmc_refined/mcmc_summary_all_galaxies.csv')
        row = summary[summary['Name'] == galaxy_name].iloc[0]
        print(f"Full VCA MCMC: v_inf={row['v_inf_med']:.2f}, r0={row['r0_med']:.2f}")
        
        # Check Full VCA Chi2
        v_inf_full = row['v_inf_med']
        r0_full = row['r0_med']
        v_dm_sq_full = (v_inf_full**2) * (1 - np.exp(-r/r0_full))
        v_tot_full = np.sqrt(v_bar_sq + v_dm_sq_full)
        chi2_full = np.sum(((v_obs - v_tot_full)/err)**2)
        print(f"Full VCA Chi2 (approx): {chi2_full:.2f}")
        
    except Exception as e:
        print("Could not check Full VCA summary", e)

if __name__ == "__main__":
    unit_test_reduced_vca('UGC06787')
