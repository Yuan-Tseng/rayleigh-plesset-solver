import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d 
from scipy.optimize import minimize

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================
# Physics Constants
RHO = 1000.0              # [kg/m^3]
MU = 1e-3                 # [Pa.s]
GAMMA = 0.072             # [N/m]
P_INF_STATIC = 100000.0   # [Pa]
K_POLY = 1.4              # Polytropic index (Adiabatic recommended)

# Experimental Settings
HYDROPHONE_SENSITIVITY = 39.0 # [mV/MPa]
CAMERA_OFFSET = 55.75e-6      # [s] Time alignment offset
FIT_WINDOW = (55e-6, 70e-6)   # [s] Optimization window

# File Paths
FILE_PRESSURE = 'Data/cleaned_F1--1.5mhz-100mV-4.csv'
FILE_RADIUS   = 'Data/AVI24/Camera_15_02_48/Camera_15_02_48_radius.csv'

# Optimization Scaling
SCALE_KS = 1e9  # Scale Ks to ~1.0

# ==============================================================================
# 2. DATA LOADING FUNCTIONS
# ==============================================================================
def load_data():
    """Loads and aligns pressure and radius data."""
    try:
        df_p = pd.read_csv(FILE_PRESSURE)
        t_p = df_p['Time_s'].values
        # Convert voltage to raw acoustic pressure [Pa]
        p_raw_acoustic = (df_p['Voltage_V'].values * 1e3 / HYDROPHONE_SENSITIVITY) * 1e6
        print(f"✅ Pressure Data Loaded: {len(t_p)} points")
    except Exception as e:
        raise ValueError(f"Error loading pressure file: {e}")

    try:
        df_r = pd.read_csv(FILE_RADIUS)
        t_r = df_r['Time_s'].values + CAMERA_OFFSET
        r_r = df_r['Radius_um'].values
        R0 = r_r[0] * 1e-6 # Initial radius [m]
        print(f"✅ Radius Data Loaded: {len(t_r)} points, R0 = {R0*1e6:.2f} um")
    except Exception as e:
        raise ValueError(f"Error loading radius file: {e}")

    return t_p, p_raw_acoustic, t_r, r_r, R0

# ==============================================================================
# 3. PHYSICS MODEL (Rayleigh-Plesset + Marmottant)
# ==============================================================================
def get_marmottant_pressure(R, Rdot, chi, kappa_s, R0, sigma_0=0.0):
    """Calculates the surface pressure term including shell viscosity."""
    if chi < 1e-5: chi = 1e-5
    
    # Critical Radii
    R_buck = R0 / np.sqrt(sigma_0/chi + 1)
    R_rupt = R_buck * np.sqrt(1 + GAMMA/chi)

    # 1. Shell Viscosity
    term_viscosity = -4 * kappa_s * Rdot / (R**2)

    # 2. Dynamic Surface Tension
    sigma = 0
    if R <= R_buck:
        sigma = 0
    elif R < R_rupt:
        sigma = chi * ((R/R_buck)**2 - 1)
    else:
        sigma = GAMMA
    
    # Pressure term from tension
    term_tension = -2 * sigma / R
    
    return term_tension + term_viscosity

def rp_solver(t, y, chi, kappa_s, R0, p_interp_func, use_marmottant=True):
    """The ODE system for solve_ivp."""
    R, Rdot = y
    if R <= 1e-9: R = 1e-9 

    # 1. Inertial & Viscous Terms (Liquid)
    inertial = 1.5 * Rdot**2
    viscous = -4 * MU * Rdot / R
    
    # 2. Gas Pressure Physics (Fix for Clean Bubble)
    if use_marmottant:
        # Marmottant: Initial state is Buckled (Sigma=0)
        # So internal pressure balances ambient pressure exactly.
        p_gas_0 = P_INF_STATIC
    else:
        # Clean Bubble: Initial state has water surface tension (Sigma=0.072)
        # Internal pressure must be HIGHER to balance Laplace pressure.
        p_gas_0 = P_INF_STATIC + (2 * GAMMA / R0)

    # 3. Driving Pressure
    p_driving = p_interp_func(t)
    pressure = p_gas_0 * (R0/R)**(3*K_POLY) - p_driving

    # 4. Surface Term
    if use_marmottant:
        surface = get_marmottant_pressure(R, Rdot, chi, kappa_s, R0)
    else:
        # Clean bubble (Water only)
        surface = -2 * GAMMA / R

    # RP Equation: rho * (R*Rddot + 3/2*Rdot^2) = Terms
    numerator = pressure + viscous + surface
    Rddot = (numerator / RHO - inertial) / R
    
    return [Rdot, Rddot]

# ==============================================================================
# 4. OPTIMIZATION CORE
# ==============================================================================
def run_optimization(t_p, p_raw, t_r, r_r_um, R0):
    """Optimizes Chi, Kappa_s, and Pressure Scale."""
    
    # Prepare Fitting Data
    mask = (t_r >= FIT_WINDOW[0]) & (t_r <= FIT_WINDOW[1])
    t_fit = t_r[mask]
    r_fit_m = r_r_um[mask] * 1e-6 
    
    print("="*60)
    print(f"🚀 Starting Optimization (Window: {FIT_WINDOW[0]*1e6:.1f}-{FIT_WINDOW[1]*1e6:.1f} us)")
    print(f"   Algorithm: Nelder-Mead (Robust for physics fitting)")

    def objective(params):
        """Cost function: RMSE between sim and exp."""
        # Unpack and unscale
        chi_val = abs(params[0])       # Ensure positive
        ks_val  = abs(params[1]) / SCALE_KS
        p_scale = abs(params[2])

        # Create dynamic pressure function
        p_total = P_INF_STATIC + (p_raw * p_scale)
        p_interp = interp1d(t_p, p_total, kind='linear', bounds_error=False, fill_value=P_INF_STATIC)

        # Solve ODE
        try:
            sol = solve_ivp(
                lambda t, y: rp_solver(t, y, chi_val, ks_val, R0, p_interp, True),
                (t_fit[0], t_fit[-1]), [R0, 0], 
                method='RK45', rtol=1e-3
            )
        except:
            return 1e9

        if not sol.success or len(sol.t) < 5: return 1e9

        # Calculate Error
        f_eval = interp1d(sol.t, sol.y[0], kind='linear', fill_value="extrapolate")
        r_sim = f_eval(t_fit)
        rmse = np.sqrt(np.mean((r_sim - r_fit_m)**2))
        
        return rmse * 1e9 # Huge scaling to ensure optimizer sees the gradient

    # Initial Guess: [Chi=0.5, Ks_scaled=4.0, P_scale=1.0]
    x0 = [0.5, 4.0, 1.0]
    
    # Run Minimize (Nelder-Mead is better when gradients are flat/noisy)
    # Note: Nelder-Mead in older scipy doesn't support bounds, so we use abs() inside objective
    # to enforce positivity, but generally it stays within reasonable range.
    res = minimize(objective, x0, method='Nelder-Mead', tol=1e-4)

    best_chi = abs(res.x[0])
    best_ks  = abs(res.x[1]) / SCALE_KS
    best_ps  = abs(res.x[2])

    print(f"✅ Optimization Finished")
    print(f"   Best Chi      : {best_chi:.4f} N/m")
    print(f"   Best Ks       : {best_ks:.4e} kg/s")
    print(f"   Best P_Scale  : {best_ps:.4f}")
    print("="*60)

    return best_chi, best_ks, best_ps

# ==============================================================================
# 5. MAIN EXECUTION & PLOTTING
# ==============================================================================
if __name__ == "__main__":
    # 1. Load Data
    t_p, p_raw, t_r, r_r, R0 = load_data()

    # 2. Optimize
    chi_opt, ks_opt, p_scale_opt = run_optimization(t_p, p_raw, t_r, r_r, R0)

    # 3. Final Simulation (Full Duration)
    # Create optimized pressure function
    p_final = P_INF_STATIC + (p_raw * p_scale_opt)
    p_func_final = interp1d(t_p, p_final, kind='linear', bounds_error=False, fill_value=P_INF_STATIC)

    t_span = (t_p[0], t_p[-1])
    t_eval = np.linspace(t_p[0], t_p[-1], 5000)

    # Sim 1: With Marmottant (Optimized)
    print("Running Final Simulation (With Shell)...")
    sol_m = solve_ivp(
        lambda t, y: rp_solver(t, y, chi_opt, ks_opt, R0, p_func_final, use_marmottant=True),
        t_span, [R0, 0], t_eval=t_eval, method='RK45', max_step=1e-8
    )

    # Sim 2: Without Marmottant (Clean Bubble)
    print("Running Final Simulation (Without Shell)...")
    sol_clean = solve_ivp(
        lambda t, y: rp_solver(t, y, chi_opt, ks_opt, R0, p_func_final, use_marmottant=False),
        t_span, [R0, 0], t_eval=t_eval, method='RK45', max_step=1e-8
    )

    # 4. Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Pressure
    ax1.plot(t_p * 1e6, p_final / 1000, 'k-', alpha=0.8, label=f'Optimized P (Scale={p_scale_opt:.2f})')
    ax1.set_ylabel('Pressure [kPa]')
    ax1.set_title('Driving Pressure Signal')
    ax1.grid(True, alpha=0.5)
    ax1.legend()

    # Plot Radius
    # Experiment
    ax2.plot(t_r * 1e6, r_r, 'g.', markersize=4, alpha=0.6, label='Experiment Data')
    
    # Simulation: With Shell
    ax2.plot(sol_m.t * 1e6, sol_m.y[0] * 1e6, 'b-', linewidth=2, label='Marmottant Model')
    
    # Simulation: Clean Bubble (Dashed Red)
    ax2.plot(sol_clean.t * 1e6, sol_clean.y[0] * 1e6, 'r--', linewidth=1.5, alpha=0.8, label='Rayleigh-Plesset (No Shell)')
    
    # Highlight fitting region
    ax2.axvspan(FIT_WINDOW[0]*1e6, FIT_WINDOW[1]*1e6, color='yellow', alpha=0.15, label='Fitting Window')

    ax2.set_xlabel('Time [us]')
    ax2.set_ylabel('Radius [um]')
    ax2.set_title(f'Result: R0={R0*1e6:.2f}um | Chi={chi_opt:.3f} | Ks={ks_opt:.2e}')
    ax2.grid(True, alpha=0.5)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()