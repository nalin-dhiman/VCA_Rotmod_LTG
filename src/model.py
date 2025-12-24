
import numpy as np

# Gravitational constant in units of (km/s)^2 kpc / M_sun
 # G = 4.30e-6 kpc km^2/s^2 Msun^-1
G = 4.30091e-6

def baryonic_velocity_sq(v_gas, v_disk, v_bul, Y_disk=0.5, Y_bul=0.7):
    """
    Computes squared baryonic velocity contribution.
    Y_disk, Y_bul are Mass-to-Light ratios.
    """
    return v_gas**2 + Y_disk * v_disk**2 + Y_bul * v_bul**2

def proposed_model_velocity_sq_term(r, v_inf, r0, v_bar_sq_arr):
    """
    Computes model velocity v(r) for Proposed Model.
    """
    # Avoid division by zero
    r_safe = np.where(r==0, 1e-9, r)
    
    A = v_inf * r_safe / (r_safe + r0)
    
    # Quadratic solution for positive branch: v^2 - A*v - v_bar^2 = 0
    # v = 0.5 * (A + sqrt(A^2 + 4 * v_bar_sq))
    term = np.sqrt(A**2 + 4 * v_bar_sq_arr)
    v_model = 0.5 * (A + term)
    return v_model

def nfw_velocity_sq_physical(r, log_rho0, log_rs):
    """
    NFW Halo velocity squared using physical density parameters.
    rho0 in M_sun/kpc^3
    rs in kpc
    
    M(r) = 4 pi rho0 rs^3 [ln(1+x) - x/(1+x)]
    v^2 = G M(r) / r
    """
    rho0 = 10**log_rho0
    rs = 10**log_rs
    
    r_safe = np.where(r==0, 1e-9, r)
    x = r_safe / rs
    
    mass_term = np.log(1 + x) - x / (1 + x)
    M_enclosed = 4 * np.pi * rho0 * rs**3 * mass_term
    
    v_sq = G * M_enclosed / r_safe
    return v_sq

def burkert_velocity_sq_physical(r, log_rho0, log_rb):
    """
    Burkert Halo velocity squared.
    rho0 in M_sun/kpc^3
    rb in kpc
    
    M(r) = pi rho0 rb^3 [2 ln(1+x) + ln(1+x^2) - 2 arctan(x)]
    v^2 = G M(r) / r
    """
    rho0 = 10**log_rho0
    rb = 10**log_rb
    
    r_safe = np.where(r==0, 1e-9, r)
    x = r_safe / rb
    
    term = 2 * np.log(1 + x) + np.log(1 + x**2) - 2 * np.arctan(x)
    M_enclosed = np.pi * rho0 * rb**3 * term
    
    v_sq = G * M_enclosed / r_safe
    return v_sq

def total_velocity_halo_model(r, halo_func, params, v_bar_sq_arr):
    """
    Generic total velocity for halo models.
    v_tot = sqrt(v_bar^2 + v_halo^2)
    """
    v_halo2 = halo_func(r, *params)
    return np.sqrt(v_bar_sq_arr + v_halo2)
