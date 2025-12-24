
import pandas as pd
import numpy as np

def find_outliers():
    try:
        df = pd.read_csv('results_comparison/model_comparison_results.csv')
        df_ident = pd.read_csv('results_identifiability/identifiability_classification.csv')
        
        # Check dAIC_Red_VCA = AIC_Reduced - AIC_VCA
        # If Reduced model (1 param) is strictly nested in VCA (2 param), 
        # AIC_Reduced should ideally be similar or slightly better if r0 is constrained,
        # or worse if v_inf was needed.
        # But if it's WAY off (e.g. +1000 or -1000), something is wrong.
        
        print(f"Loaded {len(df)} rows.")
        
        df['dAIC_Red_VCA'] = df['AIC_Reduced'] - df['AIC_VCA']
        
        print("\n--- Top 5 Worst Fits for Reduced Model (dAIC > 0, large) ---")
        print(df.sort_values('dAIC_Red_VCA', ascending=False)[['Name', 'AIC_VCA', 'AIC_Reduced', 'dAIC_Red_VCA']].head(5))
        
        print("\n--- Top 5 'Too Good' Fits for Reduced Model (dAIC < 0, large negative?) ---")
        print("Note: AIC_Red < AIC_VCA is normal if fit is good and k is smaller. But extreme differences are suspicious.")
        print(df.sort_values('dAIC_Red_VCA', ascending=True)[['Name', 'AIC_VCA', 'AIC_Reduced', 'dAIC_Red_VCA']].head(5))
        
        # Check V_max values
        print("\n--- Checking V_max column statistics ---")
        print(df['V_max'].describe())
        
        # Merge with ident to check if V_max matches
        df_merged = pd.merge(df, df_ident[['Name', 'V_max']], on='Name', suffixes=('', '_ident'))
        pass
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    find_outliers()
