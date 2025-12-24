
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.optimize import minimize

def chi2_reduced_vca(theta, r, v_obs, err, v_bar_sq, v_max_obs):
    """
    Chi2 for Reduced VCA: v_inf fixed to v_max_obs.
    Parameter: log_r0
    """
    log_r0 = theta[0]
    r0 = 10**log_r0
    
    # Model velocity
    # V_tot^2 = V_bar^2 + V_dm^2
    # V_dm^2 = V_inf^2 * (1 - exp(-r/r0))
    # V_inf = V_max_obs
    
    v_dm_sq = (v_max_obs**2) * (1 - np.exp(-r/r0))
    v_tot_sq = v_bar_sq + v_dm_sq
    
    # Avoid negative sqrt
    v_tot = np.sqrt(np.maximum(v_tot_sq, 0))
    
    chi2 = np.sum(((v_obs - v_tot) / err)**2)
    return chi2


def calculate_chi2_vca(v_inf, r0, r, v_obs, err, v_bar_sq):
    if r0 <= 0: return np.inf
    v_dm_sq = (v_inf**2) * (1 - np.exp(-r/r0))
    v_tot_sq = v_bar_sq + v_dm_sq
    v_tot = np.sqrt(np.maximum(v_tot_sq, 0))
    chi2 = np.sum(((v_obs - v_tot) / err)**2)
    return chi2

def analyze_model_comparison(
        fit_results_file='fit_results_all_models.csv',
        identifiability_file='results_identifiability/identifiability_classification.csv',
        summary_file='results_mcmc_refined/mcmc_summary_all_galaxies.csv',
        output_dir='results_comparison'
    ):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Loading MCMC summary...")
    if not os.path.exists(summary_file):
        print(f"Error: {summary_file} not found")
        return
        
    df_summary = pd.read_csv(summary_file)
    
    # We will compute AIC_VCA and AIC_Reduced
    # We will try to get AIC_Burkert from fit_results if available
    
    aic_vca_list = []
    aic_reduced_list = []
    aic_burkert_list = []
    
    # Load fit results just for Burkert if possible
    map_burkert = {}
    if os.path.exists(fit_results_file):
        df_fits = pd.read_csv(fit_results_file)
        if 'Name' in df_fits.columns and 'aic_Burkert' in df_fits.columns:
            # Handle duplicates
            df_fits = df_fits.drop_duplicates(subset=['Name'], keep='last')
            map_burkert = dict(zip(df_fits['Name'], df_fits['aic_Burkert']))
    
    data_files = glob.glob('data/*_rotmod.dat')
    name_to_file = {os.path.basename(f).replace('_rotmod.dat', ''): f for f in data_files}
    
    print(f"Processing {len(df_summary)} galaxies...")
    
    for index, row in df_summary.iterrows():
        name = row['Name']
        if name not in name_to_file:
            aic_vca_list.append(np.nan)
            aic_reduced_list.append(np.nan)
            aic_burkert_list.append(np.nan)
            continue
            
        # Get MCMC params
        v_inf = row['v_inf_med']
        r0 = row['r0_med']
        
        # Read data (CORRECTLY)
        try:
            data = pd.read_csv(name_to_file[name], sep='\s+', comment='#', 
                               names=['r', 'v_obs', 'err', 'v_gas', 'v_disk', 'v_bul'], usecols=[0,1,2,3,4,5])
        except:
            aic_vca_list.append(np.nan)
            aic_reduced_list.append(np.nan)
            aic_burkert_list.append(np.nan)
            continue
            
        r = data['r'].values
        v_obs = data['v_obs'].values
        err = data['err'].values # This is usually statistical error. MCMC used sigma0=5.0
        
        # Effective error for Chi2
        sigma0 = 5.0
        err_eff = np.sqrt(err**2 + sigma0**2)
        
        v_bar_sq = np.maximum(data['v_gas']*np.abs(data['v_gas']) + data['v_disk']**2 + data['v_bul']**2, 0)
        v_max_obs = np.max(v_obs)
        
        # 1. Full VCA AIC
        # k = 2 (v_inf, r0)
        chi2_vca = calculate_chi2_vca(v_inf, r0, r, v_obs, err_eff, v_bar_sq)
        aic_vca = chi2_vca + 2*2
        aic_vca_list.append(aic_vca)
        
        # 2. Reduced VCA AIC
        # k = 1 (r0), v_inf fixed to v_max_obs
        # Optimize log_r0
        res = minimize(chi2_reduced_vca, [np.log10(r0)], args=(r, v_obs, err_eff, v_bar_sq, v_max_obs), method='Nelder-Mead')
        aic_red = res.fun + 2*1
        aic_reduced_list.append(aic_red)
        
        # 3. Burkert AIC
        aic_burkert_list.append(map_burkert.get(name, np.nan))
        
    # Create results DF
    df_results = df_summary[['Name', 'constrained']].copy()
    df_results['AIC_VCA'] = aic_vca_list
    df_results['AIC_Reduced'] = aic_reduced_list
    df_results['AIC_Burkert'] = aic_burkert_list
    
    # Analysis
    # Wins/Losses
    df_results['dAIC_VCA_Burkert'] = df_results['AIC_VCA'] - df_results['AIC_Burkert']
    df_results['dAIC_Red_VCA'] = df_results['AIC_Reduced'] - df_results['AIC_VCA']
    
    output_df = df_results

    def classify(daic):
        if np.isnan(daic): return 'N/A'
        if daic < -10: return 'Strong Win'
        elif daic < -2: return 'Win'
        elif daic > 10: return 'Strong Loss'
        elif daic > 2: return 'Loss'
        else: return 'Tie'
        
    output_df['Result_VCA_vs_Burkert'] = output_df['dAIC_VCA_Burkert'].apply(classify)
    
    # Save results
    output_df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)
    
    # Stats
    print("\n--- VCA vs Burkert ---")
    print(output_df['Result_VCA_vs_Burkert'].value_counts())
    
    print("\nConstrained Subset:")
    print(output_df[output_df['constrained'] == True]['Result_VCA_vs_Burkert'].value_counts())
    
    # Plot Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(output_df['dAIC_VCA_Burkert'].dropna(), bins=30, range=(-50, 50), color='purple', alpha=0.7)
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel(r'$\Delta AIC (VCA - Burkert)$')
    plt.title('VCA vs Burkert Model Comparison')
    plt.ylabel('N Galaxies')
    plt.savefig(os.path.join(output_dir, 'hist_dAIC_VCA_Burkert.pdf'))
    plt.close()
    
    # Plot Reduced vs Full VCA
    plt.figure(figsize=(8, 6))
    plt.hist(output_df['dAIC_Red_VCA'].dropna(), bins=30, range=(-20, 50), color='green', alpha=0.7)
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel(r'$\Delta AIC (Reduced - Full)$')
    plt.title('Reduced VCA (1-param) vs Full VCA (2-param)')
    plt.savefig(os.path.join(output_dir, 'hist_dAIC_Reduced_Full.pdf'))
    plt.close()

if __name__ == "__main__":
    analyze_model_comparison()
