
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from scipy.interpolate import interp1d
from src.models_extended import nfw_velocity_sq, burkert_velocity_sq, proposed_model_velocity

def analyze_rar(summary_file='results_mcmc_refined/mcmc_summary_all_galaxies.csv',
                fit_results_file='fit_results_all_models.csv',
                output_dir='results_rar'):
    """
    Analyzes the Radial Acceleration Relation (RAR) for SPARC galaxies
    and overlays VCA model predictions to check for r-dependence.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Loading summary...")
    df_summary = pd.read_csv(summary_file)
    
    # Load fit results for NFW/Burkert
    map_fits = {}
    if os.path.exists(fit_results_file):
        df_fits = pd.read_csv(fit_results_file)
        # Drop duplicates
        if 'Name' in df_fits.columns:
            df_fits = df_fits.drop_duplicates(subset=['Name'], keep='last')
        map_fits = df_fits.set_index('Name').to_dict('index')
    
    # Store all points
    g_bar_all = []
    g_obs_all = []
    g_vca_all = []
    r_all = []
    names_all = []
    
    # Residuals for scatter
    res_vca = []
    res_nfw = []
    res_bur = []
    
    # Unit conversion: Velocity in km/s, Radius in kpc.
    # Acceleration g = V^2 / R.
    # To get m/s^2, we need factor.
    # (km/s)^2 / kpc = (10^3 m/s)^2 / (3.086e19 m) = 1e6 / 3.086e19 m/s^2 ~ 3.2e-14 m/s^2
    # Typically RAR is plotted in units of 10^-10 m/s^2 (Angstrom/s^2?).
    # Standard: g_obs [m/s^2].
    # Conversion: 
    # 1 (km/s)^2 / kpc = 10^6 (m/s)^2 / (3.086e16 km * 1000 m/km?? No, kpc to m)
    # 1 kpc = 3.086e19 m.
    # Factor = 1e6 / 3.086e19 = 3.24e-14.
    # Let's work in log10(g) in units of m/s^2.
    factor = 3.24077929e-14
    
    data_files = glob.glob('data/*_rotmod.dat')
    name_to_file = {os.path.basename(f).replace('_rotmod.dat', ''): f for f in data_files}
    
    print(f"Processing {len(df_summary)} galaxies...")
    
    for index, row in df_summary.iterrows():
        name = row['Name']
        if name not in name_to_file:
            continue
            
        # Get parameters
        v_inf = row['v_inf_med']
        r0 = row['r0_med']
        
        # Read data
        try:
            data = pd.read_csv(name_to_file[name], sep='\s+', comment='#', 
                               names=['r', 'v_obs', 'err', 'v_gas', 'v_disk', 'v_bul'], usecols=[0,1,2,3,4,5])
        except Exception:
            continue
            
        r = data['r'].values
        v_obs = data['v_obs'].values
        v_gas = data['v_gas'].values
        v_disk = data['v_disk'].values
        v_bul = data['v_bul'].values
        
        # Compute V_bar
        # V_bar^2 = V_gas|V_gas| + V_disk^2 + V_bul^2
        # Note: V_gas can be negative? In SPARC files they usually are positive or standard.
        # But standard def: V_bar^2 = V_gas_sq + V_disk_sq + V_bulge_sq
        v_bar_sq = np.abs(v_gas)*v_gas + np.abs(v_disk)*v_disk + np.abs(v_bul)*v_bul
        # Ensure non-negative for log
        v_bar_sq = np.maximum(v_bar_sq, 0.0)
        
        # Compute V_VCA^2
        # V_VCA_tot^2 = V_bar^2 + V_dm^2
        # V_dm^2 = v_inf^2 * (1 - exp(-r/r0))
        if r0 > 0:
            v_dm_sq = v_inf**2 * (1 - np.exp(-r/r0))
        else:
            v_dm_sq = np.zeros_like(r)
            
        v_vca_sq = v_bar_sq + v_dm_sq
        
        # Compute NFW/Burkert
        v_nfw_sq = np.zeros_like(r) * np.nan
        v_bur_sq = np.zeros_like(r) * np.nan
        
        if name in map_fits:
            fit = map_fits[name]
            # NFW
            if 'rho0_NFW' in fit and 'rs_NFW' in fit:
                try:
                    vnfw = nfw_velocity_sq(r, np.log10(fit['rho0_NFW']), np.log10(fit['rs_NFW']))
                    v_nfw_sq = v_bar_sq + vnfw
                except: pass
            # Burkert
            if 'rho0_Bur' in fit and 'rb_Bur' in fit:
                try:
                    vbur = burkert_velocity_sq(r, np.log10(fit['rho0_Bur']), np.log10(fit['rb_Bur']))
                    v_bur_sq = v_bar_sq + vbur
                except: pass
        
        # Accelerations
        # Avoid division by zero
        mask = r > 0
        r = r[mask]
        v_obs = v_obs[mask]
        v_bar_sq = v_bar_sq[mask]
        v_vca_sq = v_vca_sq[mask]
        v_nfw_sq = v_nfw_sq[mask]
        v_bur_sq = v_bur_sq[mask]
        
        g_obs = (v_obs**2 / r) * factor
        g_bar = (v_bar_sq / r) * factor
        g_vca = (v_vca_sq / r) * factor
        g_nfw = (v_nfw_sq / r) * factor
        g_bur = (v_bur_sq / r) * factor
        
        g_bar_all.extend(g_bar)
        g_obs_all.extend(g_obs)
        g_vca_all.extend(g_vca)
        r_all.extend(r)
        names_all.extend([name]*len(r))
        
        # Collect log residuals
        # Log10(g_obs) - Log10(g_model)
        # Avoid zeros
        
        def safe_log(arr):
            return np.log10(np.maximum(arr, 1e-15))
            
        l_obs = safe_log(g_obs)
        
        if len(g_vca) > 0: 
            res_vca.extend(l_obs - safe_log(g_vca))
        else:
            res_vca.extend(np.full(len(l_obs), np.nan))
            
        if len(g_nfw) > 0 and not np.isnan(g_nfw).all(): 
            res_nfw.extend(l_obs - safe_log(g_nfw))
        else:
            res_nfw.extend(np.full(len(l_obs), np.nan))
            
        if len(g_bur) > 0 and not np.isnan(g_bur).all(): 
            res_bur.extend(l_obs - safe_log(g_bur))
        else:
            res_bur.extend(np.full(len(l_obs), np.nan))
        
    # Create DataFrame first to keep alignment
    df_resid = pd.DataFrame({
        'r': r_all,
        'g_bar': g_bar_all,
        'g_obs': g_obs_all,
        'g_vca': g_vca_all,
        'res_log_nfw': res_nfw, # This is log(g_obs) - log(g_nfw)
        'res_log_bur': res_bur,
        'Name': names_all
    })
    
    # Filter valid
    mask = (df_resid['g_bar'] > 0) & (df_resid['g_obs'] > 0) & (df_resid['g_vca'] > 0)
    df_resid = df_resid[mask].copy()
    
    # Extract arrays for plotting
    g_bar_all = df_resid['g_bar'].values
    g_obs_all = df_resid['g_obs'].values
    g_vca_all = df_resid['g_vca'].values
    r_all = df_resid['r'].values
    names_all = df_resid['Name'].values
    
    print(f"Total points: {len(df_resid)}")
    
    # PLOT
    plt.figure(figsize=(8, 7))
    
    # 2D Histogram of Data
    # Log scale
    x = np.log10(g_bar_all)
    y = np.log10(g_obs_all)
    
    plt.hexbin(x, y, gridsize=50, cmap='Greys', mincnt=1, bins='log', alpha=0.6, label='Data Density')
    
    # Overlay VCA (Contour or Scatter?)
    # Scatter is too messy. Contour of density?
    # Or just mean relation?
    # Let's plot VCA density as coloured contours
    import seaborn as sns
    
    # Create DataFrame for plotting
    df_plot = pd.DataFrame({'log_g_bar': np.log10(g_bar_all), 
                            'log_g_vca': np.log10(g_vca_all)})
    
    sns.kdeplot(data=df_plot, x='log_g_bar', y='log_g_vca', 
                levels=5, color='red', linewidths=1.5, label='VCA Model Density')
    
    # 1:1 Line
    lims = [-12, -8]
    plt.plot(lims, lims, 'k--', lw=2, label='1:1 (No DM)')
    
    # RAR Function (MOND) for reference
    # g_obs = g_bar / (1 - exp(-sqrt(g_bar/g0)))
    # g0 ~ 1.2e-10
    g0 = 1.2e-10
    g_bar_grid = np.logspace(-12, -8, 100)
    g_mond = g_bar_grid / (1 - np.exp(-np.sqrt(g_bar_grid/g0)))
    plt.plot(np.log10(g_bar_grid), np.log10(g_mond), 'b-.', lw=2, label='Standard RAR (MOND)')
    
    plt.xlabel(r'$\log(g_{bar} \, [m/s^2])$', fontsize=14)
    plt.ylabel(r'$\log(g_{obs} \, [m/s^2])$', fontsize=14)
    plt.title('Radial Acceleration Relation: Data vs VCA', fontsize=16)
    plt.xlim(-12, -8)
    plt.ylim(-12, -8)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add colorbar for hexbin
    # cb = plt.colorbar()
    # cb.set_label('log(N)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'RAR_obs_vs_VCA.pdf'))
    plt.close()
    print("Generated RAR_obs_vs_VCA.pdf")
    
    # Plot residuals vs Radius (to check r-dependence)
    # Residual = log(g_obs) - log(g_RAR_MOND) ?? 
    # Or Residual = log(g_obs) - log(g_VCA)
    # Referee asks: "VCA has explicit r dependence".
    # Show log(g_VCA) - log(g_MOND_prediction) vs r?
    # Or log(g_VCA) / g_bar vs r?
    
    # Let's plot Observed Residual from MOND vs Radius
    # And VCA Residual from MOND vs Radius
    
    # Calculate MOND prediction for every point
    g_mond_pred = g_bar_all / (1 - np.exp(-np.sqrt(g_bar_all/g0)))
    resid_obs = np.log10(g_obs_all) - np.log10(g_mond_pred)
    resid_vca = np.log10(g_vca_all) - np.log10(g_mond_pred)
    
    # Ensure res_nfw and res_bur are arrays and filtered correctly
    # The global res_nfw and res_bur are log(g_obs) - log(g_model)
    # We need them relative to MOND prediction for consistency with resid_obs and resid_vca
    # First, filter res_nfw and res_bur based on the same mask as g_obs_all, etc.
    res_nfw_filtered = np.array(res_nfw)[mask]
    res_bur_filtered = np.array(res_bur)[mask]

    # Now, convert these to residuals relative to MOND prediction
    # res_nfw_filtered is log(g_obs) - log(g_nfw)
    # We want log(g_nfw) - log(g_mond_pred)
    # Correct residuals relative to DATA
    resid_nfw = res_nfw_filtered
    resid_bur = res_bur_filtered
    
    plt.figure(figsize=(8, 6))
    bins_r = np.linspace(0, 30, 30) # 0 to 30 kpc
    
    # Bin the residuals by radius
    # Bin the residuals by radius
    # df_resid already exists and is filtered/aligned
    # Add calculated residuals to it
    
    df_resid['resid_obs'] = resid_obs
    df_resid['resid_vca'] = resid_vca
    
    # NFW residual relative to data (log_obs - log_nfw) is already in 'res_log_nfw'
    # User wanted "residuals log g_obs - log g_model"
    # So 'res_log_nfw' IS the correct quantity for scatter calculation.
    # But for plotting relative to MOND? 
    # resid_obs = log_obs - log_mond
    # resid_nfw = log_nfw - log_mond = resid_obs - res_log_nfw
    
    df_resid['resid_nfw'] = resid_obs - df_resid['res_log_nfw']
    df_resid['resid_bur'] = resid_obs - df_resid['res_log_bur']
    df_resid['r_bin'] = pd.cut(df_resid['r'], bins=bins_r, labels=bins_r[:-1])
    
    mean_obs = df_resid.groupby('r_bin')['resid_obs'].mean()
    std_obs = df_resid.groupby('r_bin')['resid_obs'].std()
    
    mean_vca = df_resid.groupby('r_bin')['resid_vca'].mean()
    std_vca = df_resid.groupby('r_bin')['resid_vca'].std()
    
    bin_centers = bins_r[:-1]
    
    plt.errorbar(bin_centers, mean_obs, yerr=std_obs, fmt='ko', alpha=0.5, label='Data Residual from MOND')
    plt.plot(bin_centers, mean_vca, 'r-', lw=3, label='VCA Residual from MOND')
    # plt.fill_between(bin_centers, mean_vca-std_vca, mean_vca+std_vca, color='red', alpha=0.2)
    
    plt.axhline(0, color='b', ls='-.')
    plt.xlabel('Radius [kpc]', fontsize=14)
    plt.ylabel(r'$\Delta \log g$ (Relative to RAR)', fontsize=14)
    plt.title('Explicit Radius Dependence Check', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'RAR_residuals_vs_r.pdf'))
    plt.close()
    print("Generated RAR_residuals_vs_r.pdf")
    
    # RMS Scatter Table
    res_vca = np.array(res_vca)
    res_nfw = np.array(res_nfw)
    res_bur = np.array(res_bur)
    
    # Remove NaNs
    # --- RAR Scatter Analysis (RMS dex) ---
    def compute_rms(resid):
        # Filter NaNs before computing RMS
        resid_filtered = resid[np.isfinite(resid)]
        if len(resid_filtered) == 0:
            return np.nan
        return np.sqrt(np.mean(resid_filtered**2))
        
    rms_vca = compute_rms(df_resid['resid_vca'])
    rms_nfw = compute_rms(df_resid['resid_nfw'])
    rms_bur = compute_rms(df_resid['resid_bur'])
    
    print(f"\n--- RAR Scatter Analysis (RMS dex) ---")
    print(f"VCA:     {rms_vca:.3f} dex (N={len(df_resid['resid_vca'][np.isfinite(df_resid['resid_vca'])])})")
    print(f"NFW:     {rms_nfw:.3f} dex (N={len(df_resid['resid_nfw'][np.isfinite(df_resid['resid_nfw'])])})")
    print(f"Burkert: {rms_bur:.3f} dex (N={len(df_resid['resid_bur'][np.isfinite(df_resid['resid_bur'])])})")

    # Tier-A Analysis
    try:
        ident_df = pd.read_csv('results_identifiability/identifiability_classification.csv')
        # Merge
        df_merged = df_resid.merge(ident_df[['Name', 'ident_tier']], on='Name', how='left')
        
        subset_a = df_merged[df_merged['ident_tier'] == 'A']
        
        rms_vca_a = compute_rms(subset_a['resid_vca'])
        rms_nfw_a = compute_rms(subset_a['resid_nfw'])
        rms_bur_a = compute_rms(subset_a['resid_bur'])
        
        print(f"\n--- Tier-A Only (N={len(subset_a)}) ---")
        print(f"VCA:     {rms_vca_a:.3f} dex")
        print(f"NFW:     {rms_nfw_a:.3f} dex")
        print(f"Burkert: {rms_bur_a:.3f} dex")
        
        # Add to table if needed
        # We can just print for now, or append to tex.
        # Let's add a second table or rows.
        
    except FileNotFoundError:
        print("Identifiability classification not found. Skipping Tier A separation.")
    
    with open(os.path.join(output_dir, 'rar_scatter_table.tex'), 'w') as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\hline\n")
        f.write("Model & RMS Scatter (dex) \\\\\n")
        f.write("\\hline\n")
        f.write(f"VCA (Proposed) & {rms_vca:.3f} \\\\\n")
        f.write(f"NFW & {rms_nfw:.3f} \\\\\n")
        f.write(f"Burkert & {rms_bur:.3f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Global scatter of the Radial Acceleration Relation residuals relative to observed acceleration.}\n")
        f.write("\\label{tab:rar_scatter}\n")
        f.write("\\end{table}\n")

if __name__ == "__main__":
    analyze_rar()
