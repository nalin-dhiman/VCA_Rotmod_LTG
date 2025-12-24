"""
Generate LaTeX tables for the paper.
Fixed to use available columns.
"""
import pandas as pd
import numpy as np
import os

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
    
    # Select representative galaxies
    # Constrained vs Unconstrained examples
    targets = ['NGC2841', 'NGC6503', 'DDO154']
    subset = df[df['Name'].isin(targets)].copy()
    
    print("% Table: Diagnostics and Parameters")
    print("\\begin{table*}")
    print("\\centering")
    print("\\caption{MCMC Diagnostics and Posterior Constraints for Example Galaxies}")
    print("\\label{tab:mcmc_diag}")
    print("\\begin{tabular}{l c c c c c c c c}")
    print("\\toprule")
    print("Galaxy & $f_{acc}$ & $\\tau_{v}$ & $\\tau_{r}$ & ESS & $N_{steps}$ & Converged? & Constrained? & $v_{\\infty}$ (km/s) \\\\")
    print("\\midrule")
    
    for _, row in subset.iterrows():
        f_acc = row['acceptance_frac']
        tau_v = row['tau_v_inf']
        tau_r = row['tau_r0']
        ess = row['ESS']
        n_steps = row['N_steps']
        conv = "Yes" if row['converged'] else "No"
        constr = "Yes" if row['constrained'] else "No"
        
        v_inf = row['v_inf_med']
        v_inf_up = row['v_inf_84'] - row['v_inf_med']
        v_inf_lo = row['v_inf_med'] - row['v_inf_16']
        
        # Format errors
        try:
            val_str = f"${v_inf:.1f}_{{-{v_inf_lo:.1f}}}^{{+{v_inf_up:.1f}}}$"
        except:
            val_str = "N/A"
            
        print(f"{row['Name']} & {f_acc:.2f} & {tau_v:.1f} & {tau_r:.1f} & {int(ess)} & {int(n_steps)} & {conv} & {constr} & {val_str} \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")

if __name__ == "__main__":
    RES_DIR = 'results_mcmc_refined'
    generate_diagnostics_table(
        os.path.join(RES_DIR, 'mcmc_diagnostics.csv'),
        os.path.join(RES_DIR, 'mcmc_summary_all_galaxies.csv')
    )
