"""
Main MCMC analysis driver for submission-grade study.
Runs on subset of galaxies for computational efficiency.
"""
import os
import glob
import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt
from .data_loader import load_galaxy
from .mcmc_fitting import run_mcmc_vca, run_mcmc_halo, summarize_mcmc
from .cross_validation import split_data, predict_vca, predict_halo, compute_prediction_metrics
from .models_extended import baryonic_velocity_sq

DATA_DIR = 'data'
RESULTS_DIR = 'results_mcmc'
CORNER_DIR = os.path.join(RESULTS_DIR, 'corner_plots')

def select_representative_galaxies(fit_results_csv, n_per_category=2):
    """
    Select representative galaxies based on previous AIC analysis.
    Categories: Strong VCA win, Tie, Strong VCA loss (vs NFW and Burkert)
    """
    df = pd.read_csv(fit_results_csv)
    df_primary = df[df['sigma0'] == 5.0].copy()
    
    # Compute ΔAIC
    df_primary['dAIC_NFW'] = df_primary['aic_NFW'] - df_primary['aic_Proposed']
    df_primary['dAIC_Bur'] = df_primary['aic_Burkert'] - df_primary['aic_Proposed']
    
    selected = []
    
    # Strong wins vs NFW
    wins = df_primary[df_primary['dAIC_NFW'] >= 10].nlargest(n_per_category, 'dAIC_NFW')
    selected.extend(wins['Name'].tolist())
    
    # Ties
    ties = df_primary[df_primary['dAIC_NFW'].abs() <= 2].sample(min(n_per_category, len(df_primary)))
    selected.extend(ties['Name'].tolist())
    
    # Losses
    losses = df_primary[df_primary['dAIC_NFW'] <= -10].nsmallest(n_per_category, 'dAIC_NFW')
    selected.extend(losses['Name'].tolist())
    
    # Add some diverse galaxies (LSB, HSB, etc.)
    # For now, just add a few well-known ones
    diverse = ['NGC3198', 'NGC6503', 'DDO154', 'UGC02885']
    for name in diverse:
        if name not in selected and name in df_primary['Name'].values:
            selected.append(name)
    
    return list(set(selected))[:30]  # Cap at 30 for reasonable compute time

def run_mcmc_analysis_subset(galaxy_list=None, sigma0=5.0):
    """
    Run MCMC analysis on a subset of galaxies.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    if not os.path.exists(CORNER_DIR):
        os.makedirs(CORNER_DIR)
    
    files = glob.glob(os.path.join(DATA_DIR, '*_rotmod.dat'))
    files.sort()
    
    results = []
    cv_results = []
    
    print(f"Running MCMC on {len(galaxy_list) if galaxy_list else 'all'} galaxies...")
    
    for fpath in files:
        meta, df = load_galaxy(fpath)
        if df is None:
            continue
        
        galaxy_name = meta['Name']
        
        # Filter to selected galaxies if specified
        if galaxy_list and galaxy_name not in galaxy_list:
            continue
        
        print(f"Processing {galaxy_name}...")
        
        # Clean data
        df = df.dropna(subset=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk'])
        df = df[(df['Rad'] > 0) & (df['Vobs'] > 0)]
        if len(df) < 10:  # Need enough points for CV
            continue
        
        r = df['Rad'].values
        v_obs = df['Vobs'].values
        err_v = df['errV'].values
        err_v[err_v < 1.0] = 1.0
        v_gas = df['Vgas'].values
        v_disk = df['Vdisk'].values
        v_bul = df['Vbul'].values
        
        # Split for cross-validation
        train_data, test_data = split_data(r, v_obs, err_v, v_gas, v_disk, v_bul)
        r_train, v_train, err_train, vg_train, vd_train, vb_train = train_data
        r_test, v_test, err_test, vg_test, vd_test, vb_test = test_data
        
        # --- VCA Model (fixed M/L) ---
        try:
            samples_vca, param_names_vca = run_mcmc_vca(
                r_train, v_train, err_train, vg_train, vd_train, vb_train,
                sigma0=sigma0, free_ml=False, nwalkers=32, nsteps=1500, burn_in=500
            )
            
            medians_vca, lower_vca, upper_vca = summarize_mcmc(samples_vca)
            
            # Predict on test set
            v_pred_vca, v_pred_std_vca = predict_vca(
                r_test, samples_vca, vg_test, vd_test, vb_test, free_ml=False
            )
            rmse_vca, rmse_norm_vca = compute_prediction_metrics(v_test, v_pred_vca, err_test)
            
            # Corner plot for first 6 galaxies
            if len(results) < 6:
                fig = corner.corner(samples_vca, labels=param_names_vca,
                                    quantiles=[0.16, 0.5, 0.84], show_titles=True)
                fig.savefig(os.path.join(CORNER_DIR, f'{galaxy_name}_VCA_corner.png'))
                plt.close(fig)
            
        except Exception as e:
            print(f"  VCA MCMC failed: {e}")
            medians_vca = [np.nan, np.nan]
            lower_vca = [np.nan, np.nan]
            upper_vca = [np.nan, np.nan]
            rmse_vca = np.nan
            rmse_norm_vca = np.nan
        
        # --- NFW Model ---
        try:
            samples_nfw, param_names_nfw = run_mcmc_halo(
                r_train, v_train, err_train, vg_train, vd_train, vb_train,
                model_type='NFW', sigma0=sigma0, nwalkers=32, nsteps=1500, burn_in=500
            )
            
            medians_nfw, lower_nfw, upper_nfw = summarize_mcmc(samples_nfw)
            v_pred_nfw, _ = predict_halo(r_test, samples_nfw, vg_test, vd_test, vb_test, 'NFW')
            rmse_nfw, rmse_norm_nfw = compute_prediction_metrics(v_test, v_pred_nfw, err_test)
            
        except Exception as e:
            print(f"  NFW MCMC failed: {e}")
            medians_nfw = [np.nan, np.nan]
            rmse_nfw = np.nan
            rmse_norm_nfw = np.nan
        
        # --- Burkert Model ---
        try:
            samples_bur, _ = run_mcmc_halo(
                r_train, v_train, err_train, vg_train, vd_train, vb_train,
                model_type='Burkert', sigma0=sigma0, nwalkers=32, nsteps=1500, burn_in=500
            )
            
            medians_bur, _, _ = summarize_mcmc(samples_bur)
            v_pred_bur, _ = predict_halo(r_test, samples_bur, vg_test, vd_test, vb_test, 'Burkert')
            rmse_bur, rmse_norm_bur = compute_prediction_metrics(v_test, v_pred_bur, err_test)
            
        except Exception as e:
            print(f"  Burkert MCMC failed: {e}")
            medians_bur = [np.nan, np.nan]
            rmse_bur = np.nan
            rmse_norm_bur = np.nan
        
        # Store results
        entry = {
            'Name': galaxy_name,
            'sigma0': sigma0,
            'N_train': len(r_train),
            'N_test': len(r_test),
            # VCA
            'v_inf_med': 10**medians_vca[0] if len(medians_vca) > 0 else np.nan,
            'v_inf_lower': 10**lower_vca[0] if len(lower_vca) > 0 else np.nan,
            'v_inf_upper': 10**upper_vca[0] if len(upper_vca) > 0 else np.nan,
            'r0_med': 10**medians_vca[1] if len(medians_vca) > 1 else np.nan,
            'r0_lower': 10**lower_vca[1] if len(lower_vca) > 1 else np.nan,
            'r0_upper': 10**upper_vca[1] if len(upper_vca) > 1 else np.nan,
            'RMSE_pred_VCA': rmse_vca,
            'RMSE_norm_VCA': rmse_norm_vca,
            # NFW
            'rho0_NFW_med': 10**medians_nfw[0] if len(medians_nfw) > 0 else np.nan,
            'rs_NFW_med': 10**medians_nfw[1] if len(medians_nfw) > 1 else np.nan,
            'RMSE_pred_NFW': rmse_nfw,
            'RMSE_norm_NFW': rmse_norm_nfw,
            # Burkert
            'rho0_Bur_med': 10**medians_bur[0] if len(medians_bur) > 0 else np.nan,
            'rb_Bur_med': 10**medians_bur[1] if len(medians_bur) > 1 else np.nan,
            'RMSE_pred_Bur': rmse_bur,
            'RMSE_norm_Bur': rmse_norm_bur,
        }
        
        results.append(entry)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(RESULTS_DIR, 'mcmc_results.csv'), index=False)
    print(f"Saved MCMC results for {len(df_results)} galaxies")
    
    return df_results

if __name__ == "__main__":
    # Select representative subset
    if os.path.exists('fit_results_all_models.csv'):
        galaxy_list = select_representative_galaxies('fit_results_all_models.csv')
        print(f"Selected {len(galaxy_list)} representative galaxies")
    else:
        # Run on first 20 galaxies as default
        files = glob.glob(os.path.join(DATA_DIR, '*_rotmod.dat'))
        galaxy_list = [os.path.basename(f).replace('_rotmod.dat', '') for f in sorted(files)[:20]]
    
    run_mcmc_analysis_subset(galaxy_list=galaxy_list, sigma0=5.0)
