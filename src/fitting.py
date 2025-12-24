
import numpy as np
import scipy.optimize as opt
from .model import (
    proposed_model_velocity_sq_term, 
    nfw_velocity_sq_physical, 
    burkert_velocity_sq_physical,
    total_velocity_halo_model,
    baryonic_velocity_sq
)

def check_bound_hit(popt, bounds, rtol=0.01):
    """
    Returns True if any parameter is close to the bounds.
    bounds is a tuple of (lower_array, upper_array).
    """
    lower, upper = bounds
    # Check lower
    hit_lower = np.any(np.isclose(popt, lower, rtol=rtol))
    hit_upper = np.any(np.isclose(popt, upper, rtol=rtol))
    return hit_lower or hit_upper

def calculate_stats(v_obs, v_model, err_v, n_params):
    chi2 = np.sum(((v_obs - v_model)/err_v)**2)
    n_data = len(v_obs)
    dof = n_data - n_params
    red_chi2 = chi2 / dof if dof > 0 else np.nan
    aic = 2 * n_params + chi2
    bic = n_params * np.log(n_data) + chi2
    return chi2, red_chi2, aic, bic

def fit_generic(model_func, p0, bounds, r, v_obs, err_v, sigma0=0.0, **kwargs):
    """
    Generic fitter.
    model_func(r, *params) -> v_model
    sigma_eff = sqrt(err_v^2 + sigma0^2)
    """
    sigma_eff = np.sqrt(err_v**2 + sigma0**2)
    
    try:
        # curve_fit seeks to minimize chisq
        popt, pcov = opt.curve_fit(
            model_func, r, v_obs, p0=p0, sigma=sigma_eff, 
            absolute_sigma=True, bounds=bounds, maxfev=10000
        )
        bound_hit = check_bound_hit(popt, bounds)
        return popt, pcov, bound_hit, sigma_eff
    except (RuntimeError, ValueError):
        return None, None, True, sigma_eff

def get_fit_bounds(model_type):
    if model_type == 'Proposed':
        # log10(v_inf), log10(r0)
        # v_inf: 0.1 to 5000 -> -1 to 3.7
        # r0: 0.01 to 500 -> -2 to 2.7
        return ([-1.0, -2.0], [4.0, 3.0])
    elif model_type == 'NFW':
        # log10(rho0), log10(rs)
        # rho0: 1e4 to 1e10 -> 4 to 10 ??
        # Let's check typical NFW densities. 
        # Critical density ~ 1e-7 Msun/pc^3 ~ 100 Msun/kpc^3 * 1e9? No.
        # rho_crit ~ 140 Msun/kpc^3. 
        # Halo densities can be 1e5 - 1e8 Msun/kpc^3.
        # rs: 0.1 to 200 kpc. -> -1 to 2.3
        return ([4.0, -1.0], [10.0, 3.0]) # Broad bounds
    elif model_type == 'Burkert':
         # log10(rho0), log10(rb)
         # Similar to NFW
        return ([4.0, -1.0], [10.0, 3.0])
    return None

def fit_galaxy_all_models(r, v_obs, err_v, v_gas, v_disk, v_bul, sigma0=0.0, Y_disk=0.5, Y_bul=0.7):
    """
    Fits all models to a galaxy.
    Returns a dict of results.
    """
    v_bar2 = baryonic_velocity_sq(v_gas, v_disk, v_bul, Y_disk, Y_bul)
    v_bar = np.sqrt(v_bar2)
    
    results = {}
    
    # --- Baryons Only ---
    # 0 parameters
    # But for stats we just compute chi2
    sigma_eff = np.sqrt(err_v**2 + sigma0**2)
    chi2_bar, red_chi2_bar, aic_bar, bic_bar = calculate_stats(v_obs, v_bar, sigma_eff, 0)
    results['Baryons'] = {
        'chi2': chi2_bar, 'red_chi2': red_chi2_bar, 'aic': aic_bar, 'bic': bic_bar,
        'v_model': v_bar, 'bound_hit': False, 'popt': []
    }
    
    # --- Proposed Model ---
    # fit log10 params
    def func_prop_log(r_pts, log_v_inf, log_r0):
        return proposed_model_velocity_sq_term(r_pts, 10**log_v_inf, 10**log_r0, v_bar2)
    
    bounds_prop = get_fit_bounds('Proposed')
    p0_prop = [2.0, 0.7] # v_inf=100, r0=5
    popt, pcov, bh, se = fit_generic(func_prop_log, p0_prop, bounds_prop, r, v_obs, err_v, sigma0)
    
    if popt is not None:
        v_mod = func_prop_log(r, *popt)
        chi2, red, aic, bic = calculate_stats(v_obs, v_mod, se, 2)
        results['Proposed'] = {
            'chi2': chi2, 'red_chi2': red, 'aic': aic, 'bic': bic,
            'v_model': v_mod, 'bound_hit': bh, 
            'popt': [10**popt[0], 10**popt[1]] # convert back to linear
        }
    else:
        results['Proposed'] = None

    # --- NFW Model ---
    def func_nfw_log(r_pts, log_rho0, log_rs):
        return total_velocity_halo_model(r_pts, nfw_velocity_sq_physical, [log_rho0, log_rs], v_bar2)
        
    bounds_nfw = get_fit_bounds('NFW')
    p0_nfw = [7.0, 1.0] # rho0=1e7, rs=10
    popt, pcov, bh, se = fit_generic(func_nfw_log, p0_nfw, bounds_nfw, r, v_obs, err_v, sigma0)
    
    if popt is not None:
        v_mod = func_nfw_log(r, *popt)
        chi2, red, aic, bic = calculate_stats(v_obs, v_mod, se, 2)
        results['NFW'] = {
            'chi2': chi2, 'red_chi2': red, 'aic': aic, 'bic': bic,
            'v_model': v_mod, 'bound_hit': bh, 
            'popt': [10**popt[0], 10**popt[1]]
        }
    else:
        results['NFW'] = None

    # --- Burkert Model ---
    def func_burkert_log(r_pts, log_rho0, log_rb):
        return total_velocity_halo_model(r_pts, burkert_velocity_sq_physical, [log_rho0, log_rb], v_bar2)
        
    bounds_bur = get_fit_bounds('Burkert')
    p0_bur = [7.0, 1.0]
    popt, pcov, bh, se = fit_generic(func_burkert_log, p0_bur, bounds_bur, r, v_obs, err_v, sigma0)
    
    if popt is not None:
        v_mod = func_burkert_log(r, *popt)
        chi2, red, aic, bic = calculate_stats(v_obs, v_mod, se, 2)
        results['Burkert'] = {
            'chi2': chi2, 'red_chi2': red, 'aic': aic, 'bic': bic,
            'v_model': v_mod, 'bound_hit': bh, 
            'popt': [10**popt[0], 10**popt[1]]
        }
    else:
        results['Burkert'] = None
        
    return results, sigma_eff
