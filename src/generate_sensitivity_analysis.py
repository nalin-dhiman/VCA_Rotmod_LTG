
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_sensitivity_artifacts():
    df = pd.read_csv('fit_results_all_models.csv')
    output_dir = 'results_comparison'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sigma_vals = sorted(df['sigma0'].unique())
    print(f"Analyzing sensitivity for sigma0: {sigma_vals}")
    
    # 1. Summary Table
    # Rows: sigma0
    # Cols: Med RedChi2 (Bar, VCA, NFW, Burkert), Win% (VCA vs NFW, VCA vs Burkert)
    
    table_lines = []
    table_lines.append("\\begin{table}")
    table_lines.append("\\centering")
    table_lines.append("\\begin{tabular}{l|cccc|cc}")
    table_lines.append("\\hline")
    table_lines.append("$\\sigma_0$ [km/s] & \\multicolumn{4}{c|}{Median $\\chi^2_\\nu$} & \\multicolumn{2}{c}{VCA AIC Win \\% vs} \\\\")
    table_lines.append(" & Bar & VCA & NFW & Bur & NFW & Bur \\\\")
    table_lines.append("\\hline")
    
    for s in sigma_vals:
        sub = df[df['sigma0'] == s].copy()
        n = len(sub)
        if n == 0: continue
        
        # Medians
        med_bar = sub['red_chi2_Baryons'].median()
        med_vca = sub['red_chi2_Proposed'].median()
        med_nfw = sub['red_chi2_NFW'].median()
        med_bur = sub['red_chi2_Burkert'].median()
        
        # Wins (AIC < AIC_other)
        # Note: AIC_Proposed vs AIC_NFW
        vca_beat_nfw = (sub['aic_Proposed'] < sub['aic_NFW']).sum() / n * 100
        vca_beat_bur = (sub['aic_Proposed'] < sub['aic_Burkert']).sum() / n * 100
        
        line = f"{s:.0f} & {med_bar:.2f} & {med_vca:.2f} & {med_nfw:.2f} & {med_bur:.2f} & {vca_beat_nfw:.1f}\\% & {vca_beat_bur:.1f}\\% \\\\"
        table_lines.append(line)
        
    table_lines.append("\\hline")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\caption{Median reduced $\\chi^2$ and AIC model preference fractions as a function of the systematic error floor $\\sigma_0$.}")
    table_lines.append("\\label{tab:sensitivity}")
    table_lines.append("\\end{table}")
    
    with open(os.path.join(output_dir, 'sensitivity_table.tex'), 'w') as f:
        f.write("\n".join(table_lines))
    print("Saved sensitivity_table.tex")
    
    # 2. Histograms for sigma0=0 and sigma0=5
    # dAIC = AIC_VCA - AIC_Halo
    # Negative dAIC means VCA is better.
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    
    for i, s in enumerate([0.0, 5.0]):
        sub = df[df['sigma0'] == s]
        
        # VCA vs NFW
        daic_nfw = sub['aic_Proposed'] - sub['aic_NFW']
        axes[i, 0].hist(daic_nfw, bins=30, range=(-50, 50), color='skyblue', alpha=0.7)
        axes[i, 0].axvline(0, color='k', ls='--')
        axes[i, 0].set_title(f"$\\sigma_0={s}$ km/s: VCA vs NFW")
        axes[i, 0].set_ylabel("N Galaxies")
        axes[i, 0].text(0.05, 0.9, f"VCA Wins: {(daic_nfw < 0).sum()/len(sub)*100:.1f}%", transform=axes[i,0].transAxes)
        
        # VCA vs Burkert
        daic_bur = sub['aic_Proposed'] - sub['aic_Burkert']
        axes[i, 1].hist(daic_bur, bins=30, range=(-50, 50), color='salmon', alpha=0.7)
        axes[i, 1].axvline(0, color='k', ls='--')
        axes[i, 1].set_title(f"$\\sigma_0={s}$ km/s: VCA vs Burkert")
        axes[i, 1].text(0.05, 0.9, f"VCA Wins: {(daic_bur < 0).sum()/len(sub)*100:.1f}%", transform=axes[i,1].transAxes)
        
    axes[1, 0].set_xlabel(r"$\Delta AIC (VCA - NFW)$")
    axes[1, 1].set_xlabel(r"$\Delta AIC (VCA - Burkert)$")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_histograms.pdf'))
    print("Saved sensitivity_histograms.pdf")

if __name__ == "__main__":
    generate_sensitivity_artifacts()
