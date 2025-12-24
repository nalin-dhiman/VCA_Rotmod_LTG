"""
Extended model definitions including Einasto, pseudo-isothermal, and MOND.
"""
import numpy as np
from scipy.special import gammainc, gamma as gamma_func

# Gravitational constant in units of (km/s)^2 kpc / M_sun
G = 4.30091e-6

def baryonic_velocity_sq(v_gas, v_disk, v_bul, Y_disk=0.5, Y_bul=0.7):
    """Baryonic velocity squared."""
    return v_gas**2 + Y_disk * v_disk**2 + Y_bul * v_bul**2

def proposed_model_velocity(r, v_inf, r0, v_bar_sq_arr):
    """
    VCA model velocity: Quadratic Solution.
    v^2 = v_bar^2 + gamma(r)*v*r
    gamma(r) = v_inf / (r + r0)
    
    A = v_inf * r / (r + r0)
    v = 0.5 * (A + sqrt(A^2 + 4 * v_bar^2))
    """
    r_safe = np.where(r==0, 1e-9, r)
    A = v_inf * r_safe / (r_safe + r0)
    term = np.sqrt(A**2 + 4 * v_bar_sq_arr)
    return 0.5 * (A + term)

def nfw_velocity_sq(r, log_rho0, log_rs):
    """NFW halo velocity squared."""
    rho0 = 10**log_rho0
    rs = 10**log_rs
    r_safe = np.where(r==0, 1e-9, r)
    x = r_safe / rs
    mass_term = np.log(1 + x) - x / (1 + x)
    M_enclosed = 4 * np.pi * rho0 * rs**3 * mass_term
    return G * M_enclosed / r_safe

def burkert_velocity_sq(r, log_rho0, log_rb):
    """Burkert halo velocity squared."""
    rho0 = 10**log_rho0
    rb = 10**log_rb
    r_safe = np.where(r==0, 1e-9, r)
    x = r_safe / rb
    term = 2 * np.log(1 + x) + np.log(1 + x**2) - 2 * np.arctan(x)
    M_enclosed = np.pi * rho0 * rb**3 * term
    return G * M_enclosed / r_safe

def einasto_velocity_sq(r, log_rhos, log_rs, alpha):
    """
    Einasto profile: rho(r) = rho_s exp(-2/alpha * [(r/r_s)^alpha - 1])
    Mass integral uses incomplete gamma function.
    """
    rho_s = 10**log_rhos
    r_s = 10**log_rs
    r_safe = np.where(r==0, 1e-9, r)
    
    # Dimensionless radius
    x = r_safe / r_s
    
    # Mass enclosed (using incomplete gamma function)
    # M(r) = 4π ρ_s r_s^3 * (α/2)^(-3/α) * Γ(3/α) * P(3/α, 2x^α/α)
    # where P is regularized lower incomplete gamma
    n = 3.0 / alpha
    arg = 2.0 * x**alpha / alpha
    
    # Avoid overflow for large arguments
    arg_safe = np.clip(arg, 0, 100)
    gamma_lower = gammainc(n, arg_safe) * gamma_func(n)
    
    M_enclosed = 4 * np.pi * rho_s * r_s**3 * (alpha / 2.0)**(n) * gamma_lower
    
    return G * M_enclosed / r_safe

def pseudo_isothermal_velocity_sq(r, log_rho0, log_rc):
    """
    Pseudo-isothermal sphere: rho(r) = rho_0 / (1 + (r/r_c)^2)
    M(r) = 4π ρ_0 r_c^3 [r/r_c - arctan(r/r_c)]
    """
    rho0 = 10**log_rho0
    rc = 10**log_rc
    r_safe = np.where(r==0, 1e-9, r)
    x = r_safe / rc
    
    M_enclosed = 4 * np.pi * rho0 * rc**3 * (x - np.arctan(x))
    return G * M_enclosed / r_safe

def mond_rar_velocity(r, v_bar, a0_kms2_kpc):
    """
    MOND/RAR empirical fit: g_obs = ν(g_bar/a_0) * g_bar
    where ν(x) = x/(1 + x) (simple interpolating function)
    
    g_bar = v_bar^2 / r
    g_obs = v_obs^2 / r
    
    Returns v_obs given v_bar and a_0.
    """
    r_safe = np.where(r==0, 1e-9, r)
    g_bar = v_bar**2 / r_safe
    
    # Interpolating function
    x = g_bar / a0_kms2_kpc
    nu = x / (1 + x)
    
    g_obs = nu * g_bar
    v_obs = np.sqrt(g_obs * r_safe)
    return v_obs

def total_velocity_halo(r, halo_func, params, v_bar_sq):
    """Generic total velocity for halo models."""
    v_halo2 = halo_func(r, *params)
    return np.sqrt(v_bar_sq + v_halo2)
