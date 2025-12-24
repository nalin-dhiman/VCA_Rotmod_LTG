
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_diagnostics_plot():
    df = pd.read_csv('results_mcmc_refined/mcmc_diagnostics.csv')
    
    plt.figure(figsize=(10, 8))
    
    # 1. Autocorrelation Time Histogram
    plt.subplot(2, 2, 1)
    
    # Filter valid tau
    tau = df['tau_v_inf'].dropna()
    plt.hist(tau, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel(r'Autocorrelation Time $\tau$ (steps)')
    plt.ylabel('N Galaxies')
    plt.title('Convergence Speed')
    
    # 2. ESS vs N_steps
    plt.subplot(2, 2, 2)
    plt.plot(df['N_steps'], df['ESS'], 'ko', alpha=0.5)
    plt.xlabel('Total Steps')
    plt.ylabel('Effective Sample Size (ESS)')
    plt.title('Sampling Efficiency')
    plt.grid(True, alpha=0.3)
    
    # 3. Acceptance Fraction
    plt.subplot(2, 2, 3)
    plt.hist(df['acceptance_frac'], bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel('Acceptance Fraction')
    plt.axvline(0.2, color='r', ls='--')
    plt.axvline(0.5, color='r', ls='--')
    plt.title('Sampler Health')
    
    # 4. Convergence Status
    plt.subplot(2, 2, 4)
    counts = df['converged'].value_counts()
    plt.bar(counts.index.astype(str), counts.values, color=['green', 'red'])
    plt.title(f'Convergence (Total={len(df)})')
    
    plt.tight_layout()
    plt.savefig('results_mcmc_refined/diagnostics_summary.pdf')
    print("Saved diagnostics_summary.pdf")

def generate_coverage_plot():
    df = pd.read_csv('results_mcmc_refined/coverage_results.csv')
    
    # Calculate global coverage
    cov_68 = df['in_68'].mean()
    cov_95 = df['in_95'].mean()
    
    # Observed fractions vs Radius
    # Bin by radius
    bins = np.linspace(0, 30, 10)
    df['r_bin'] = pd.cut(df['radius'], bins=bins)
    grouped = df.groupby('r_bin', observed=True)
    
    mean_r = grouped['radius'].mean()
    frac_68 = grouped['in_68'].mean()
    frac_95 = grouped['in_95'].mean()
    
    plt.figure(figsize=(8, 6))
    
    plt.plot(mean_r, frac_68, 'o-', label='68% Interval')
    plt.axhline(0.68, color='k', linestyle='--')
    
    plt.plot(mean_r, frac_95, 's-', label='95% Interval')
    plt.axhline(0.95, color='k', linestyle='--')
    
    plt.xlabel('Radius [kpc]')
    plt.ylabel('Coverage Fraction')
    plt.title(f'Global PPC Coverage\n(Target: 0.68 -> {cov_68:.2f}, 0.95 -> {cov_95:.2f})')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('results_mcmc_refined/coverage_summary.pdf')
    print("Saved coverage_summary.pdf")

def generate_identifiability_table():
    df = pd.read_csv('results_identifiability/identifiability_classification.csv')
    
    counts = df['ident_tier'].value_counts().sort_index()
    total = len(df)
    
    with open('results_identifiability/identifiability_table.tex', 'w') as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("Tier & Definition & Count (\\%) \\\\\n")
        f.write("\\hline\n")
        
        # A
        n_a = counts.get('A', 0)
        p_a = n_a / total * 100
        f.write(f"A (Well-Constrained) & $\\Delta < 0.3$ dex, $R_{{max}}/r_0 > 2$ & {n_a} ({p_a:.1f}\\%) \\\\\n")
        
        # B
        n_b = counts.get('B', 0)
        p_b = n_b / total * 100
        f.write(f"B (Moderately) & $\\Delta < 0.5$ dex & {n_b} ({p_b:.1f}\\%) \\\\\n")
        
        # C
        n_c = counts.get('C', 0)
        p_c = n_c / total * 100
        f.write(f"C (Unconstrained) & Loose or Bound Hit & {n_c} ({p_c:.1f}\\%) \\\\\n")
        
        f.write("\\hline\n")
        f.write(f"Total & & {total} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Classification of galaxy parameter identifiability.}\n")
        f.write("\\label{tab:identifiability}\n")
        f.write("\\end{table}\n")
    print("Saved identifiability_table.tex")

if __name__ == "__main__":
    generate_diagnostics_plot()
    generate_coverage_plot()
    generate_identifiability_table()
