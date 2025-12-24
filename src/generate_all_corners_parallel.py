"""
Generate corner plots for ALL 171 galaxies using parallelized MCMC.
"""
import os
import glob
import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from .data_loader import load_galaxy
from .mcmc_fitting import run_adaptive_mcmc_vca
from .models_extended import baryonic_velocity_sq

DATA_DIR = 'data'
OUTPUT_DIR = 'results_mcmc_refined/corner_plots'

# Global parameters
SIGMA0 = 5.0
NWALKERS = 32

def process_galaxy_corner(fpath):
    """Run MCMC and plot corner for a single galaxy."""
    try:
        meta, df = load_galaxy(fpath)
        if df is None: return None
        galaxy_name = meta['Name']
        
        # Output check
        out_path = os.path.join(OUTPUT_DIR, f'{galaxy_name}_corner.png')
        if os.path.exists(out_path):
            # Removing the existence check to force regeneration with fixed titles
            # return f"Skipped {galaxy_name} (exists)"
            pass

        # Clean data
        df = df.dropna(subset=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk'])
        df = df[(df['Rad'] > 0) & (df['Vobs'] > 0)]
        if len(df) < 5: return None
        
        r = df['Rad'].values
        v_obs = df['Vobs'].values
        err_v = df['errV'].values
        err_v[err_v < 1.0] = 1.0
        v_gas = df['Vgas'].values
        v_disk = df['Vdisk'].values
        v_bul = df['Vbul'].values
        
        # Run Adaptive MCMC
        # We can use slightly shorter parameters since we just need a good posterior for plotting
        # But let's stick to the robust adaptive settings to be safe
        sampler, param_names = run_adaptive_mcmc_vca(
            r, v_obs, err_v, v_gas, v_disk, v_bul,
            sigma0=SIGMA0,
            nwalkers=NWALKERS,
            burn_in=500,        # Slightly reduced for speed
            min_steps=1500,     # Slightly reduced
            max_steps=5000,
            move_scale=2.0      # Standard scale is fine for visualization
        )
        
        # Get samples
        samples = sampler.get_chain(flat=True, discard=500)
        
        # Plot Corner
        # Transform log10 to linear for nicer plots? 
        # Usually physics params are better in linear if they are constrained, 
        # but log if unconstrained. Let's stick to log params as fitted for consistency,
        # or maybe dual? Let's do Log parameters as they are the sampling parameters.
        # Labelling:
        labels = [r"$\log_{10}(v_\infty)$", r"$\log_{10}(r_0)$"]
        
        fig = corner.corner(
            samples, 
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 14}
        )
        fig.suptitle(f"{galaxy_name} MCMC Posteriors", fontsize=16, y=1.05)
        
        # Save
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        
        return f"Processed {galaxy_name}"
        
    except Exception as e:
        return f"FAILED {galaxy_name}: {e}"

def generate_all_corners():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    files = glob.glob(os.path.join(DATA_DIR, '*_rotmod.dat'))
    files.sort()
    
    print(f"Generating corner plots for {len(files)} galaxies using {cpu_count()} cores...")
    
    with Pool(processes=cpu_count()) as pool:
        for res in pool.imap_unordered(process_galaxy_corner, files):
            if res:
                pass # print(res) reduces clutter
                
    print("Done generating corner plots.")

if __name__ == "__main__":
    generate_all_corners()
