"""
Refined plotting for MCMC diagnostics and results.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mcmc_diagnostics_summary(summary_csv, output_path):
    """
    Plot acceptance fraction, tau, and ESS distributions.
    """
    df = pd.read_csv(summary_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Acceptance Fraction
    sns.histplot(df['acceptance_frac'], bins=20, ax=axes[0,0], kde=True)
    axes[0,0].set_title('Acceptance Fraction')
    axes[0,0].axvline(0.2, color='r', ls='--')
    axes[0,0].axvline(0.5, color='r', ls='--')
    
    # Tau
    if 'tau_v_inf' in df.columns:
        sns.histplot(df['tau_v_inf'].dropna(), bins=20, ax=axes[0,1], label='tau_v', kde=True, color='blue')
        sns.histplot(df['tau_r0'].dropna(), bins=20, ax=axes[0,1], label='tau_r', kde=True, color='orange', alpha=0.5)
        axes[0,1].set_title('Autocorrelation Time (tau)')
        axes[0,1].legend()
    
    # ESS
    sns.histplot(df['ESS'].dropna(), bins=20, ax=axes[1,0], kde=True)
    axes[1,0].set_title('Effective Sample Size')
    
    # Tau vs N_data
    axes[1,1].scatter(df['N_data'], df['tau_v_inf'], alpha=0.5, label='v_inf')
    axes[1,1].scatter(df['N_data'], df['tau_r0'], alpha=0.5, label='r0')
    axes[1,1].set_xlabel('N_data points')
    axes[1,1].set_ylabel('Tau')
    axes[1,1].set_title('Tau vs N_data')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_ppc_coverage_summary(coverage_csv, output_path):
    """
    Plot global PPC coverage summary.
    """
    df = pd.read_csv(coverage_csv)
    
    # Compute global coverage
    cov68 = df['in_68'].mean() * 100
    cov95 = df['in_95'].mean() * 100
    
    # Per-galaxy coverage
    galaxy_cov = df.groupby('Name')[['in_68', 'in_95']].mean() * 100
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of per-galaxy coverage
    sns.histplot(galaxy_cov['in_68'], bins=15, ax=ax[0], kde=True)
    ax[0].axvline(68, color='k', ls='--', label='Expected (68%)')
    ax[0].axvline(cov68, color='r', ls='-', label=f'Global ({cov68:.1f}%)')
    ax[0].set_title('Galaxy-level 68% Coverage')
    ax[0].set_xlabel('Percentage of points in 68% CI')
    ax[0].legend()
    
    sns.histplot(galaxy_cov['in_95'], bins=15, ax=ax[1], kde=True)
    ax[1].axvline(95, color='k', ls='--', label='Expected (95%)')
    ax[1].axvline(cov95, color='r', ls='-', label=f'Global ({cov95:.1f}%)')
    ax[1].set_title('Galaxy-level 95% Coverage')
    ax[1].set_xlabel('Percentage of points in 95% CI')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    RES_DIR = 'results_mcmc_refined'
    if os.path.exists(os.path.join(RES_DIR, 'mcmc_diagnostics.csv')):
        plot_mcmc_diagnostics_summary(
            os.path.join(RES_DIR, 'mcmc_diagnostics.csv'),
            os.path.join(RES_DIR, 'diagnostics_summary.pdf')
        )
    
    if os.path.exists(os.path.join(RES_DIR, 'coverage_results.csv')):
        plot_ppc_coverage_summary(
            os.path.join(RES_DIR, 'coverage_results.csv'),
            os.path.join(RES_DIR, 'coverage_summary.pdf')
        )
