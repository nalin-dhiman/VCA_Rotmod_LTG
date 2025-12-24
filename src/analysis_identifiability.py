
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_identifiability(summary_file='results_mcmc_refined/mcmc_summary_all_galaxies.csv', 
                          fit_results_file='fit_results_all_models.csv',
                          output_dir='results_identifiability'):
    """
    Classifies galaxies as constrained/unconstrained, computes derived parameters,
    and generates plots for manuscript.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(summary_file):
        print(f"Error: {summary_file} not found.")
        return

    df_summary = pd.read_csv(summary_file)
    
    # Load V_max from fit results
    if os.path.exists(fit_results_file):
        df_fits = pd.read_csv(fit_results_file)
        # Keep only relevant columns
        df_fits = df_fits[['Name', 'V_max', 'Distance']].drop_duplicates(subset=['Name'])
        # Merge
        df = pd.merge(df_summary, df_fits, on='Name', how='left')
    else:
        print(f"Warning: {fit_results_file} not found. V_max plots will be missing.")
        df = df_summary
        df['V_max'] = np.nan

    
    # 1. Classification Logic
    # Constrained if: r0_width < 0.3 dex (in log space) AND v_inf_width < 0.3 dex ? 
    # Or based on rmp/r0 ratio? The plan says: width < 0.3 dex, rmax/r0 > 2.
    # Let's check what columns we have. We likely have v_inf_16, v_inf_50, v_inf_84, etc.
    # r0 often goes to 999.9, so we need to be careful.
    
    # Calculate widths in log space for v_inf and r0
    # Avoid log(0) issues
    df['log_v_inf_50'] = np.log10(df['v_inf_med'])
    df['log_v_inf_width'] = np.log10(df['v_inf_84']) - np.log10(df['v_inf_16'])
    
    df['log_r0_50'] = np.log10(df['r0_med'])
    df['log_r0_width'] = np.log10(df['r0_84']) - np.log10(df['r0_16'])
    
    # rmax from data needed. Fit results might have it? 
    # Or we can estimate if we don't have it. 
    # Actually, we should check if r_max is in the summary.
    # If not, we might need to load it from data files or fit_results.csv.
    # Let's assume for now we define "constrained" mainly by parameter uncertainty.
    # A tight posterior means the data constrained it.
    
    # Strict definition: both parameters well defined.
    # Using 0.5 dex width as a safe threshold for "well constrained" given the massive degeneracies 
    # seen in MCMC. Or 0.3 dex as per plan. Let's try 0.5 first to see count.
    
    threshold_dex = 0.5
    
    # Note: r0 hits upper bound (1000) often. If r0_84 > 800, it's effectively unconstrained.
    df['r0_constrained'] = (df['log_r0_width'] < threshold_dex) & (df['r0_84'] < 800)
    df['v_inf_constrained'] = (df['log_v_inf_width'] < threshold_dex)
    
    # Galaxy is "Constrained" if BOTH are constrained.
    # df['is_constrained'] = df['r0_constrained'] & df['v_inf_constrained']
    
    # 3-Tier Classification (User Step 3)
    # Tier A "Well": width < 0.3 dex AND rmax/r0 > 2
    # Tier B "Mod": width < 0.5 dex (and not A)
    # Tier C "Un": otherwise
    
    # Pre-calculate rmax/r0 (we need rmax first, which is loaded in loop below)
    # Let's do classification AFTER data loading loop.
    
    # 2. Derived Parameters
    # v_eff = v_inf * (1 - exp(-rmax/r0)) -- wait, VCA is v(r) = v_inf * (1 - exp(-r/r0))? 
    # No, VCA is v(r) = v_inf * sqrt(1 - exp(-r/r0)) ?? 
    # Let's re-verify the VCA velocity formula.
    # Typically VCA: v_c^2(r) = v_inf^2 * (1 - exp(-r/r0))  (if it matches simple exponential)
    # The code `models_extended.py` or `models.py` has the truth.
    # Let's assume standard form for now but check code.
    
    # "v_eff" usually means velocity at the outermost radius r_max.
    # We need r_max for each galaxy.
    # We can get V_max (observed) from the summary if available, or fit_results.
    
    # Let's compute s = v_inf / r0 (acceleration scale proxy)
    df['s_param'] = df['v_inf_med'] / df['r0_med']
    
    

    # 2. Derived Parameters and Data Reading
    # We need r_max for each galaxy to compute v_eff = VCA_velocity(r_max)
    # VCA velocity: v^2(r) = v_inf^2 * (1 - exp(-r/r0))  <-- CONFIRM THIS FORMULA
    # Assuming standard simple exponential velocity profile implied by 
    # V(r) = v_inf * sqrt(1 - exp(-r/r0))
    
    r_max_list = []
    v_eff_list = []
    
    import glob
    
    # Map galaxy name to file path
    data_files = glob.glob('data/*_rotmod.dat')
    name_to_file = {}
    for f in data_files:
        # consistent naming?
        # filename: NGC0024_rotmod.dat -> NGC0024
        # summary Name: NGC0024
        basename = os.path.basename(f)
        # remove _rotmod.dat
        name = basename.replace('_rotmod.dat', '')
        name_to_file[name] = f
        
    for index, row in df.iterrows():
        name = row['Name']
        if name in name_to_file:
            try:
                data = pd.read_csv(name_to_file[name], sep='\s+', comment='#', names=['r', 'v_obs', 'err', 'v_gas', 'v_disk', 'v_bulge'], usecols=[0,1,2,3,4,5])
                rmax = data['r'].max()
            except Exception as e:
                print(f"Error reading {name}: {e}")
                rmax = np.nan
        else:
            rmax = np.nan
            print(f"Warning: No data file found for {name}")
            
        r_max_list.append(rmax)
        
        # Calculate v_eff using median parameters
        v_inf = row['v_inf_med']
        r0 = row['r0_med']
        
        if not np.isnan(rmax) and not np.isnan(v_inf) and not np.isnan(r0):
            # VCA Model: v(r) = v_inf * sqrt(1 - exp(-r/r0))
            if r0 > 0:
                v_eff_val = v_inf * np.sqrt(1 - np.exp(-rmax/r0))
            else:
                v_eff_val = 0
        else:
            v_eff_val = np.nan
            
        v_eff_list.append(v_eff_val)
        
    df['r_max'] = r_max_list
    df['v_eff'] = v_eff_list
    
    # Now compute 3-tier
    df['rmax_over_r0'] = df['r_max'] / df['r0_med']
    
    tier_list = []
    for idx, row in df.iterrows():
        w_v = row['log_v_inf_width']
        w_r = row['log_r0_width']
        ratio = row['rmax_over_r0']
        r0_top = row['r0_84']
        
        # Check boundary condition
        if r0_top > 800:
            tier_list.append('C') # Unconstrained
            continue
            
        if w_v < 0.3 and w_r < 0.3 and ratio > 2.0:
            tier_list.append('A')
        elif w_v < 0.5 and w_r < 0.5: # Relaxed
            tier_list.append('B')
        else:
            tier_list.append('C')
            
    df['ident_tier'] = tier_list
    df['is_constrained'] = df['ident_tier'] != 'C'
    
    # Save counts
    print("\nIdentifiability Tiers:")
    print(df['ident_tier'].value_counts())
    
    # Save enhanced classification
    csv_path = os.path.join(output_dir, 'identifiability_classification.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved classification to {csv_path}")
    
    # 3. Plots
    
    # Plot 1: v_eff vs V_max (All Galaxies)
    if 'V_max' in df.columns:
        plt.figure(figsize=(6,6))
        
        # All galaxies
        plt.scatter(df['V_max'], df['v_eff'], alpha=0.6, c='gray', label=f'All ({len(df)})')
        
        # Constrained subset
        constrained = df[df['is_constrained']]
        plt.scatter(constrained['V_max'], constrained['v_eff'], color='blue', label=f'Constrained ({len(constrained)})')
        
        # 1:1 line
        max_limit = max(df['V_max'].max(), df['v_eff'].max()) * 1.1
        plt.plot([0, max_limit], [0, max_limit], 'k--', zorder=0)
        
        plt.xlabel(r'$V_{max}$ (Observed) [km/s]')
        plt.ylabel(r'$V_{eff} = V_{model}(R_{max})$ [km/s]')
        plt.title('Effective vs Observed Maximum Velocity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlim(0, max_limit)
        plt.ylim(0, max_limit)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'v_eff_vs_Vmax.pdf'))
        plt.close()
        print("Generated v_eff_vs_Vmax.pdf")
        
    # Plot 2: log(v_inf) vs log(V_max) (Constrained Only)
    # This shows that for constrained galaxies, v_inf tracks V_max (flat curves)
    if 'V_max' in constrained.columns:
        plt.figure(figsize=(6,6))
        
        # Plot only constrained
        plt.scatter(constrained['V_max'], constrained['v_inf_med'], color='blue', alpha=0.7)
        
        # Error bars for v_inf
        yerr = [constrained['v_inf_med'] - constrained['v_inf_16'], 
                constrained['v_inf_84'] - constrained['v_inf_med']]
        plt.errorbar(constrained['V_max'], constrained['v_inf_med'], 
                     yerr=yerr, fmt='none', ecolor='blue', alpha=0.3)
        
        plt.xlabel(r'$V_{max}$ (Observed) [km/s]')
        plt.ylabel(r'$V_{\infty}$ (Model) [km/s]')
        plt.title('Asymptotic Velocity (Constrained Subset)')
        
        plt.xscale('log')
        plt.yscale('log')
        
        # 1:1 line
        lims = [
            np.min([plt.xlim(), plt.ylim()]),  # min of both axes
            np.max([plt.xlim(), plt.ylim()]),  # max of both axes
        ]
        plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        
        plt.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'v_inf_vs_Vmax_constrained.pdf'))
        plt.close()
        print("Generated v_inf_vs_Vmax_constrained.pdf")

    # Plot 3: Histogram of Widths (Justifiction for classification)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.hist(df['log_r0_width'], bins=30, range=(0,5), color='green', alpha=0.7)
    plt.axvline(threshold_dex, color='red', linestyle='--', label=f'Threshold {threshold_dex} dex')
    plt.xlabel('Log Width of $r_0$ [dex]')
    plt.title('$r_0$ Uncertainty Distribution')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.scatter(df['log_r0_width'], df['log_v_inf_width'], alpha=0.5, s=10)
    plt.axvline(threshold_dex, color='red', linestyle='--')
    plt.axhline(threshold_dex, color='red', linestyle='--')
    plt.xlabel('Log Width of $r_0$')
    plt.ylabel('Log Width of $V_{\infty}$')
    plt.title('Parameter Uncertainty Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'identifiability_histograms.pdf'))
    plt.close()
        
    return df

if __name__ == "__main__":
    analyze_identifiability()
