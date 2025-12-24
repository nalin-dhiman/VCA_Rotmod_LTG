
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from src.data_loader import load_galaxy
from src.fitting import fit_galaxy_all_models

def run_cross_validation_ls():
    print("Running Radial Split Cross-Validation (70/30 split)...")
    
    data_dir = 'data'
    files = glob.glob(os.path.join(data_dir, '*_rotmod.dat'))
    files.sort()
    
    results = []
    
    # Sigma floor for likelihood calculation
    sigma0 = 5.0
    
    for fpath in files:
        meta, df = load_galaxy(fpath)
        if df is None: continue
        
        df = df.dropna(subset=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk'])
        df = df[(df['Rad'] > 0) & (df['Vobs'] > 0)]
        if len(df) < 8: continue # Need enough points to split
        
        r = df['Rad'].values
        v_obs = df['Vobs'].values
        err_v = df['errV'].values
        sigma_eff = np.sqrt(err_v**2 + sigma0**2)
        
        v_gas = df['Vgas'].values
        v_disk = df['Vdisk'].values
        v_bul = df['Vbul'].values
        
        # Split
        n_points = len(r)
        n_train = int(0.7 * n_points)
        if n_train < 5: continue
        
        # Training Data
        r_tr = r[:n_train]
        v_tr = v_obs[:n_train]
        e_tr = err_v[:n_train]
        vg_tr = v_gas[:n_train]
        vd_tr = v_disk[:n_train]
        vb_tr = v_bul[:n_train]
        
        # Testing Data
        r_te = r[n_train:]
        v_te = v_obs[n_train:]
        e_te = err_v[n_train:]
        vg_te = v_gas[n_train:]
        vd_te = v_disk[n_train:]
        vb_te = v_bul[n_train:]
        sig_te = sigma_eff[n_train:]
        
        # Fit models to Training Data
        fits, _ = fit_galaxy_all_models(
            r_tr, v_tr, e_tr, vg_tr, vd_tr, vb_tr, sigma0=sigma0
        )
        
        # Evaluate on Test Data
        res = {'Name': meta['Name']}
        
        for mname, fit_res in fits.items():
            if fit_res is None:
                res[f'rmse_te_{mname}'] = np.nan
                res[f'chi2_te_{mname}'] = np.nan
                continue
            
            # Predict
            # Need to call model function again.
            # fit_galaxy_all_models code is local, uses closure.
            # Can I reuse the model definitions?
            # They are in src.model.
            
            popt = fit_res['popt']
            
            # Reconstruct model prediction
            # Need to import model functions again to call them with parameters
            # Or make fit_galaxy_all_models return the model function? No.
            # I will use src.model functions directly.
            
            from src.model import (
                proposed_model_velocity_sq_term, 
                nfw_velocity_sq_physical, 
                burkert_velocity_sq_physical,
                total_velocity_halo_model,
                baryonic_velocity_sq
            )
            
            v_bar_sq_te = baryonic_velocity_sq(vg_te, vd_te, vb_te)
            v_mod_te = None
            
            if mname == 'Baryons':
                v_mod_te = np.sqrt(v_bar_sq_te)
                
            elif mname == 'Proposed':
                v_inf, r0 = popt
                v_mod_te = proposed_model_velocity_sq_term(r_te, v_inf, r0, v_bar_sq_te)
                
            elif mname == 'NFW':
                 # popt is linear, func takes log. Wait. fits returned linear popt?
                 # src/fitting.py: results['NFW']['popt'] = [10**popt[0], 10**popt[1]]
                 # But total_velocity_halo_model expects log params if we call func_nfw_log.
                 # Actually src.model functions (nfw_velocity_sq_physical) take LOG params.
                 # So we need to convert back to log for the function call.
                 
                 params_log = np.log10(popt)
                 v_sq = total_velocity_halo_model(r_te, nfw_velocity_sq_physical, params_log, v_bar_sq_te)
                 v_mod_te = v_sq # wait function returns v not v_sq?
                 # No, total_velocity_halo_model returns v (sqrt).
                 v_mod_te = v_sq
                 
            elif mname == 'Burkert':
                 params_log = np.log10(popt)
                 v_mod_te = total_velocity_halo_model(r_te, burkert_velocity_sq_physical, params_log, v_bar_sq_te)

            # Compute Metric
            # RMSE
            rmse = np.sqrt(np.mean((v_te - v_mod_te)**2))
            # Chi2
            chi2 = np.sum(((v_te - v_mod_te)/sig_te)**2)
            
            res[f'rmse_te_{mname}'] = rmse
            res[f'chi2_te_{mname}'] = chi2
            
        results.append(res)
        
    df_res = pd.DataFrame(results)
    df_res.to_csv('results_comparison/cv_results_ls.csv', index=False)
    
    print("Cross validation complete.")
    print("Median RMSE Test:")
    print(df_res[['rmse_te_Baryons', 'rmse_te_Proposed', 'rmse_te_NFW', 'rmse_te_Burkert']].median())

    # Generate Plot
    # Comparison of RMSE
    
    plt.figure(figsize=(8,6))
    data = [
        df_res['rmse_te_Proposed'].dropna(),
        df_res['rmse_te_NFW'].dropna(),
        df_res['rmse_te_Burkert'].dropna()
    ]
    plt.boxplot(data, labels=['VCA', 'NFW', 'Burkert'])
    plt.ylabel('RMSE (Test Data) [km/s]')
    plt.title('Radial Cross-Validation (Outer 30%)')
    plt.grid(True, alpha=0.3)
    plt.savefig('results_comparison/cv_rmse_boxplot.pdf')
    print("Saved cv_rmse_boxplot.pdf")
    
if __name__ == "__main__":
    run_cross_validation_ls()
