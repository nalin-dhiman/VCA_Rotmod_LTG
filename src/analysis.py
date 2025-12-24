
import os
import glob
import numpy as np
import pandas as pd
from .data_loader import load_galaxy
from .fitting import fit_galaxy_all_models
from .plotting import plot_fit_all, plot_histograms_upgraded

DATA_DIR = 'data'
FIGURES_DIR = 'figures'
GALLERY_DIR = os.path.join(FIGURES_DIR, 'gallery')

def run_analysis_upgraded():
    files = glob.glob(os.path.join(DATA_DIR, '*_rotmod.dat'))
    files.sort()
    
    if not os.path.exists(GALLERY_DIR):
        os.makedirs(GALLERY_DIR)
        
    sigma0_vals = [0.0, 3.0, 5.0, 8.0]
    
    all_results = []
    
    print(f"Starting analysis on {len(files)} galaxies with sigma0 grid: {sigma0_vals}")
    
    for sigma0 in sigma0_vals:
        print(f"Processing sigma0 = {sigma0} km/s...")
        
        for fpath in files:
            meta, df = load_galaxy(fpath)
            if df is None: continue
            
            # Cleaning
            df = df.dropna(subset=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk'])
            df = df[(df['Rad'] > 0) & (df['Vobs'] > 0)]
            if len(df) < 5: continue
            
            r = df['Rad'].values
            v_obs = df['Vobs'].values
            err_v = df['errV'].values
            err_v[err_v < 1.0] = 1.0
            
            v_gas = df['Vgas'].values
            v_disk = df['Vdisk'].values
            v_bul = df['Vbul'].values
            
            # Fit all models
            fits, sigma_eff = fit_galaxy_all_models(
                r, v_obs, err_v, v_gas, v_disk, v_bul, sigma0=sigma0
            )
            
            # Store results
            entry = {
                'Name': meta['Name'],
                'sigma0': sigma0,
                'Distance': meta.get('Distance_Mpc', np.nan),
                'V_max': np.max(v_obs),
            }
            
            # Extract metrics
            for mname, res in fits.items():
                if res is None:
                    # Failed fit
                    entry[f'chi2_{mname}'] = np.nan
                    entry[f'aic_{mname}'] = np.nan
                    entry[f'bic_{mname}'] = np.nan
                    entry[f'hit_{mname}'] = True
                else:
                    entry[f'chi2_{mname}'] = res['chi2']
                    entry[f'red_chi2_{mname}'] = res['red_chi2']
                    entry[f'aic_{mname}'] = res['aic']
                    entry[f'bic_{mname}'] = res['bic']
                    entry[f'hit_{mname}'] = res['bound_hit']
                    
                    if mname == 'Proposed':
                        entry['v_inf'] = res['popt'][0]
                        entry['r0'] = res['popt'][1]
                    elif mname == 'NFW':
                        entry['rho0_NFW'] = res['popt'][0]
                        entry['rs_NFW'] = res['popt'][1]
                    elif mname == 'Burkert':
                        entry['rho0_Bur'] = res['popt'][0]
                        entry['rb_Bur'] = res['popt'][1]

            all_results.append(entry)
            
            # Plotting (only for sigma0 = 5.0 as the primary visualization set)
            if sigma0 == 5.0:
                models_vectors = {k: (v['v_model'] if v else None) for k, v in fits.items()}
                plot_fit_all(
                    meta['Name'], r, v_obs, sigma_eff, models_vectors,
                    os.path.join(GALLERY_DIR, f"{meta['Name']}_fit_resid.png")
                )
    
    # Save Results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('fit_results_all_models.csv', index=False)
    print("Saved fit_results_all_models.csv")
    
    # Generate Summary Plots for sigma0=5
    df_primary = df_results[df_results['sigma0'] == 5.0]
    if len(df_primary) > 0:
        # Uppercase column names matching plotting script expectation
        # wait, plotting script expects 'AIC_NFW' etc.
        # My keys above are 'aic_NFW'. I should fix plotting or rename here.
        # Let's rename columns for compatibility.
        rename_map = {
            'aic_Proposed': 'AIC_Prop', 'aic_NFW': 'AIC_NFW', 'aic_Burkert': 'AIC_Burkert',
            'aic_Baryons': 'AIC_Bar'
        }
        df_plot = df_primary.rename(columns=rename_map)
        plot_histograms_upgraded(df_plot, FIGURES_DIR, prefix='sigma5_')
        
    return df_results

if __name__ == "__main__":
    run_analysis_upgraded()
