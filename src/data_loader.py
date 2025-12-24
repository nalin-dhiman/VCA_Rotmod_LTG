
import os
import pandas as pd
import numpy as np

def load_galaxy(filepath):
    """
    Parses a SPARC .dat file.
    
    Expected format:
    # Distance = ...
    # Rad   Vobs    errV    Vgas    Vdisk   Vbul ...
    # kpc   km/s    km/s    km/s    km/s    km/s ...
    
    Returns:
        meta (dict): metadata like distance, hierarchy (e.g. NGC5055)
        df (pd.DataFrame): columns [Rad, Vobs, errV, Vgas, Vdisk, Vbul]
    """
    
    # Read metadata lines manually
    meta = {}
    with open(filepath, 'r') as f:
        first_line = f.readline()
        if 'Distance' in first_line:
            try:
                # Example: # Distance = 9.9 Mpc
                dist_str = first_line.split('=')[1].strip().split()[0]
                meta['Distance_Mpc'] = float(dist_str)
            except:
                meta['Distance_Mpc'] = np.nan
    
    # Read table
    # Skip rows with '#' except the one with column names (which is usually row 1 or 2)
    # Actually, pandas read_csv with comment='#' simplifies this, but we lose the column names if they are commented.
    # The SPARC files usually have a header line like `# Rad Vobs ...`
    
    # Let's try to infer clean mapping.
    # Columns usually: Rad, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul
    names = ['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
    
    try:
        df = pd.read_csv(filepath, sep=r'\s+', names=names, comment='#')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

    # Determine galaxy name from filename
    basename = os.path.basename(filepath)
    galaxy_name = basename.replace('_rotmod.dat', '')
    meta['Name'] = galaxy_name
    
    # Basic validation
    required = ['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul']
    for col in required:
        if col not in df.columns:
            # Maybe the file has fewer columns (e.g. no bulge)
            # If Vbul is missing, fill with 0
            if col == 'Vbul':
                df['Vbul'] = 0.0
            else:
                # If Rad etc are missing, huge problem
                print(f"Missing column {col} in {galaxy_name}")
                return None, None
                
    return meta, df

if __name__ == "__main__":
    # Test on one file
    import sys
    if len(sys.argv) > 1:
        m, d = load_galaxy(sys.argv[1])
        print("Meta:", m)
        print("Head:", d.head())
