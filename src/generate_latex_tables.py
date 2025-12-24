"""
Generate LaTeX tables for the paper.
"""
import pandas as pd
import numpy as np

def generate_diagnostics_table(diag_csv, summary_csv):
    """
    Generate diagnostics table snippet.
    """
    if not os.path.exists(diag_csv) or not os.path.exists(summary_csv):
        print("CSV files not found.")
        return

    df_diag = pd.read_csv(diag_csv)
    df_sum = pd.read_csv(summary_csv)
    
    # Merge
    df = pd.merge(df_diag, df_sum, on='Name')
    
    # Select a few representative galaxies
    # 2 constrained + 1 unconstrained as requested
    # Let's pick: NGC2841 (constrained), NGC6503 (constrained), DDO154 (unconstrained)
    
    targets = ['NGC2841', 'NGC6503', 'DDO154']
    subset = df[df['Name'].isin(targets)].copy()
    
    print("\\begin{table*}")
    print("\\centering")
    print("\\caption{MCMC Diagnostics and Posterior Constraints for Example Galaxies}")
    print("\\label{tab:mcmc_diag}")
    print("\\begin{tabular}{l c c c c c c c c}")
    print("\\toprule")
    print("Galaxy & $f_{acc}$ & $\\tau_{v}$ & $\\tau_{r}$ & ESS & $N_{data}$ & $r_{max}/r_0$ & Constrained? & $v_{eff}$ (km/s) \\\\")
    print("\\midrule")
    
    for _, row in subset.iterrows():
        f_acc = row['acceptance_frac']
        tau_v = row['tau_v_inf']
        tau_r = row['tau_r0']
        ess = row['ESS']
        n_data = row['N_data']
        rmax_r0 = row['rmax_over_r0']
        constr = "Yes" if row['constrained'] else "No"
        v_eff = row['v_eff_med']
        v_eff_err = (row['v_eff_84'] - row['v_eff_16'])/2
        
        print(f"{row['Name']} & {f_acc:.2f} & {tau_v:.1f} & {tau_r:.1f} & {int(ess)} & {int(n_data)} & {rmax_r0:.1f} & {constr} & ${v_eff:.1f} \\pm {v_eff_err:.1f}$ \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")

import os
if __name__ == "__main__":
    RES_DIR = 'results_mcmc_refined'
    generate_diagnostics_table(
        os.path.join(RES_DIR, 'mcmc_diagnostics.csv'),
        os.path.join(RES_DIR, 'mcmc_summary_all_galaxies.csv')
    )
