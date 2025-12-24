
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_fit_all(galaxy_name, r, v_obs, sigma_eff, models_dict, save_path):
    """
    Plots Observed vs Models + Residuals.
    models_dict: { 'Proposed': v_array, 'NFW': v_array, ... }
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Top Panel: Rotation Curves
    ax1.errorbar(r, v_obs, yerr=sigma_eff, fmt='ko', label='Observed', capsize=3, alpha=0.5)
    
    colors = {'Baryons': 'gray', 'Proposed': 'red', 'NFW': 'blue', 'Burkert': 'green'}
    styles = {'Baryons': ':', 'Proposed': '-', 'NFW': '--', 'Burkert': '-.'}
    
    sort_idx = np.argsort(r)
    r_s = r[sort_idx]
    
    for name, v_mod in models_dict.items():
        if v_mod is not None:
            ax1.plot(r_s, v_mod[sort_idx], color=colors.get(name,'k'), ls=styles.get(name,'-'), label=name, lw=2)
            
            # Bottom Panel: Residuals (normalized)
            resid = (v_obs - v_mod) / sigma_eff
            ax2.plot(r_s, resid[sort_idx], color=colors.get(name,'k'), ls=styles.get(name,'-'), lw=1.5)
    
    ax1.set_ylabel('Velocity [km/s]')
    ax1.set_title(f'{galaxy_name}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel(r'Residuals ($\sigma$)')
    ax2.set_xlabel('Radius [kpc]')
    ax2.axhline(0, color='k', lw=1)
    ax2.axhline(2, color='k', ls=':', alpha=0.5)
    ax2.axhline(-2, color='k', ls=':', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 5) # truncate extreme outliers in plot
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(save_path)
    plt.close()

def plot_histograms_upgraded(results_df, save_dir, prefix=''):
    # Delta AIC
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    dble = results_df['AIC_NFW'] - results_df['AIC_Prop']
    plt.hist(dble, bins=20, color='purple', alpha=0.7, label='NFW - Prop')
    plt.xlabel(r'$\Delta$AIC (NFW - Prop)')
    plt.title('Prop vs NFW')
    plt.axvline(0, color='k', ls='--')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    dble2 = results_df['AIC_Burkert'] - results_df['AIC_Prop']
    plt.hist(dble2, bins=20, color='green', alpha=0.7, label='Burkert - Prop')
    plt.xlabel(r'$\Delta$AIC (Burkert - Prop)')
    plt.title('Prop vs Burkert')
    plt.axvline(0, color='k', ls='--')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}aic_distribution.png'))
    plt.close()
    
    # Parameters
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(np.log10(results_df['v_inf']), bins=15, color='skyblue')
    plt.xlabel(r'$\log_{10}(v_\infty)$')
    
    plt.subplot(2, 2, 2)
    plt.hist(np.log10(results_df['r0']), bins=15, color='salmon')
    plt.xlabel(r'$\log_{10}(r_0)$')
    
    plt.subplot(2, 2, 3)
    plt.scatter(np.log10(results_df['v_inf']), np.log10(results_df['r0']), alpha=0.5)
    plt.xlabel(r'$\log_{10}(v_\infty)$')
    plt.ylabel(r'$\log_{10}(r_0)$')
    
    plt.subplot(2, 2, 4)
    # Correlation Vmax vs v_inf
    plt.scatter(results_df['V_max'], results_df['v_inf'], alpha=0.5)
    plt.xlabel('V_max [km/s]')
    plt.ylabel(r'$v_\infty$')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}param_correlations.png'))
    plt.close()
