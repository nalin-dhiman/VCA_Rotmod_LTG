
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.data_loader import load_galaxy

def analyze_vca_preference():
    print("Loading fit results...")
    # Use sigma0=5.0 as fiducial
    try:
        df = pd.read_csv('fit_results_all_models.csv')
    except FileNotFoundError:
        print("Error: fit_results_all_models.csv not found.")
        return
        
    df_fid = df[df['sigma0'] == 5.0].copy()
    if len(df_fid) == 0:
        print("No results for sigma0=5.0")
        return

    # Calculate dAIC
    # Negative dAIC means VCA is better
    df_fid['dAIC_VCA_NFW'] = df_fid['aic_Proposed'] - df_fid['aic_NFW']
    df_fid['dAIC_VCA_Burkert'] = df_fid['aic_Proposed'] - df_fid['aic_Burkert']
    
    # Load extra metadata (R_eff, etc) from individual files if needed
    # Ideally fit_results has some, but maybe not R_eff or L_3.6.
    # We'll reload some meta.
    
    r_effs = []
    lume = []
    
    print("Loading detailed metadata for correlator plots...")
    data_dir = 'data'
    name_map = dict(zip(df_fid['Name'], df_fid.index))
    
    df_fid['R_eff'] = np.nan
    df_fid['L_36'] = np.nan
    df_fid['Rmax_kpc'] = np.nan # fit_results doesn't have Rmax, let's get it.
    
    # Actually fit_results has 'V_max'.
    
    import glob
    files = glob.glob(os.path.join(data_dir, '*_rotmod.dat'))
    
    for fpath in files:
        meta, glx_data = load_galaxy(fpath)
        name = meta['Name']
        if name in name_map:
            idx = name_map[name]
            # Assumes meta loader extracts Reff, L3.6 if strictly parsed.
            # Our data_loader usually creates a dict.
            # Let's hope commonly used keys exist.
            # Standard SPARC keys: 'Reff', 'L3.6', 'D'
            
            # Check what keys are available
            # If not in meta dict, we might need to parse header manually or look at data ranges
            
            # Approximations if Keys missing:
            # R_eff from data? No.
            # We can calculate Rmax from data
            if glx_data is not None:
                rmax = glx_data['Rad'].max()
                df_fid.loc[idx, 'Rmax_kpc'] = rmax
                
            # Try to get Reff/L from meta if available
            # Standard SPARC headers usually have: Include 'Reff', 'L3.6'
            if 'Reff' in meta:
               df_fid.loc[idx, 'R_eff'] = float(meta['Reff'])
            if 'L3.6' in meta:
               df_fid.loc[idx, 'L_36'] = float(meta['L3.6'])

    # Diagnostics Plots
    output_dir = 'results_comparison'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Variables to try correlating with
    vars_to_plot = [
        ('V_max', 'Max Velocity [km/s]', 'log'), 
        ('Rmax_kpc', 'Max Radius [kpc]', 'log'),
        ('Distance', 'Distance [Mpc]', 'linear')
    ]
    
    for i, (col, label, scale) in enumerate(vars_to_plot):
        # Filter valid data for this column
        valid = df_fid[df_fid[col] > 0]
        if len(valid) == 0: continue
        # Top Row: VCA vs NFW
        ax = axes[0, i]
        sc = ax.scatter(valid[col], valid['dAIC_VCA_NFW'], c='k', alpha=0.6, s=15)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel(label)
        ax.set_ylabel(r'$\Delta$AIC (VCA - NFW)')
        if scale == 'log':
            ax.set_xscale('log')
        ax.set_title(f"VCA-NFW vs {col}")
        
        # Bottom Row: VCA vs Burkert
        ax = axes[1, i]
        sc = ax.scatter(valid[col], valid['dAIC_VCA_Burkert'], c='b', alpha=0.6, s=15)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel(label)
        ax.set_ylabel(r'$\Delta$AIC (VCA - Burkert)')
        if scale == 'log':
            ax.set_xscale('log')
        ax.set_title(f"VCA-Burkert vs {col}")
        
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dAIC_correlations.png'), dpi=300)
    print("Saved dAIC_correlations.png")

if __name__ == "__main__":
    analyze_vca_preference()
