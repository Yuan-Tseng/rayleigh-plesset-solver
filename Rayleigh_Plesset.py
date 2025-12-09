import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d 
from scipy.optimize import minimize

#============================== Water Properties ==============================#
rho = 1000              # density [kg/m^3]
mu = 1e-3               # viscosity [Pa.s]
gamma = 0.072           # surface tension of water [N/m]

#============================= Initial Properties =============================#
sigma_0 = 0.0               # initial surface tension [N/m] (Assumption: Buckled)
p_inf_static = 100000       # ambient pressure [Pa]
k = 1.0                     # polytropic index (1.07 for isothermal-ish, 1.0 for isothermal, 1.4 for adiabatic)
hydrophone_sensitivity = 39 # [mV/MPa]
pressure_scaling = 1        # Pressure Scaling

#================================= CSV File ===================================#
HYD_FILE = 'Data/cleaned_F1--1.5mhz-100mV-4.csv'                    # Input pressure detected by hydrophone, signal averaged
EXP_CSV = 'Data/AVI24/Camera_15_02_48/Camera_15_02_48_radius.csv'   # temporal radius data computed
# 'Data/AVI24/Camera_15_02_48/Camera_15_02_48_radius.csv'
# 'Data/AVI24/Camera_15_08_33/Camera_15_08_33_radius.csv'
# 'Data/AVI24/Camera_15_16_28/Camera_15_16_28_radius.csv'

try:
    df = pd.read_csv(HYD_FILE)
    time = df['Time_s'].values # [s]
    voltage = df['Voltage_V'].values * 1e3 # [mV]
    pressure = voltage / hydrophone_sensitivity * 1e6 * pressure_scaling # [Pa]
    print(f"Processed Signal: {len(time)} Data Points, Time Interval: {time[0]*1e6:.1f} - {time[-1]*1e6:.1f} us")
except Exception as e:
    print(f"Error: Cannot read CSV, check column or path\n{e}")
    exit()

try:
    df_r = pd.read_csv(EXP_CSV)
    t_r_exp = df_r['Time_s'].values
    r_r_exp = df_r['Radius_um'].values
    
    # --- Time Alignment ---
    camera_start_offset = 55.75e-6 
    t_r_exp_aligned = t_r_exp + camera_start_offset
    
    # Take the first value to be R0 (Initial Radius)
    R0_exp = r_r_exp[0] * 1e-6 # [m]
    R0 = R0_exp
    p_gas_0 = p_inf_static + (2 * sigma_0 / R0) # initial gas pressure [Pa]
    print(f" Successful Read Radius: {len(t_r_exp)} Data Points")
    print(f" Initial Radius: {R0_exp*1e6:.2f} um")
    
except Exception as e:
    print(f"Failed Reading Experiment CSV: {e}")
    exit()

#============================ Acoustic Properties =============================#
p_total = p_inf_static + pressure
P_interp_func = interp1d(time, p_total, kind='linear', bounds_error=False, fill_value=p_inf_static) # Make it continuous
def P_driving(t): 
    """Return absolute pressure at that timestep"""
    return P_interp_func(t)

# 1. Image Processing
def R_effective(Area):
    """This is the R resolved by cv2 from image processing"""
    return np.sqrt(Area/np.pi)

# 2. Marmottant Model
def get_marmottant_term(R, Rdot, chi, kappa_s):
    """
    chi: Elastic Modulus [N/m]
    kappa_s: Shell Viscosity [kg/s]
    Returns the combined surface tension and shell viscosity term
    """
    
    # Calculate critical radii based on current Elastic Modulus (chi)
    R_buck = R0 / np.sqrt(sigma_0/chi + 1)
    R_rupt = R_buck * np.sqrt(1 + gamma/chi)

    # 1. Shell Viscosity Term (always present)
    shell_visc_term = -4 * kappa_s * Rdot / (R**2)

    # 2. Dynamic Surface Tension Term
    sigma = 0
    if R <= R_buck:     # Buckled regime
        sigma = 0
    elif R > R_buck and R < R_rupt:   # Elastic regime
        sigma = -2 * chi * ((R/R_buck)**2 - 1)/R
    elif R >= R_rupt:   # Ruptured regime
        sigma = -2 * gamma/R
    
    return sigma + shell_visc_term

def no_marmottant_term(R):
    sigma = -2 * gamma/R

    return sigma

# 3. RP Linearization
"""
Rayleigh-Plesset equation First Order ODE system, including Marmottant model for shell properties
y[0]: bubble radius R
y[1]: radial velocity Rdot

Modified RP with Marmottant model:
LHS(Inertial terms) = (P_bubble - P_infinity) - P_viscous - P_surface_tension(Marmottant term)
--------------------------------------------
dR/dt = Rdot
dRdot/dt = (1/R) * ( (1/rho)[(pB(R) - p_inf) - (4*mu*Rdot)/(R) + (Marmottant term)] - (3/2)*Rdot^2)
--------------------------------------------
Returns: [dR/dt, dRdot/dt]
"""
def rp_equation(t, y, chi, kappa_s, marmottant):
    R = y[0]
    Rdot = y[1]
    
    # Safety check to prevent negative radius calculation errors
    if R <= 0: R = 1e-9

    # 1. Inertial Term (LHS related)
    inertial_term = 1.5 * Rdot**2
    
    # 2. Viscous Term (Liquid)
    viscous_term = -4 * mu * Rdot / R
    
    # 3. Gas Pressure Term
    # Using p_gas_0 to be physically consistent
    pressure_term = p_gas_0 * (R0/R)**(3*k) - P_driving(t)
    
    # 4. Marmottant Term (Shell Viscosity + Surface Tension)
    marmottant_term = get_marmottant_term(R, Rdot, chi, kappa_s)

    # 5. No Marmottant Term (Surface Tension)
    without_marmottant = no_marmottant_term(R)

    # Combine RHS terms, Note: viscous_term and marmottant_term already calculate negative forces
    if marmottant:
        RHS = pressure_term + viscous_term + marmottant_term
    else:
        RHS = pressure_term + viscous_term + without_marmottant
    
    # Solve for Rddot
    Rddot = (RHS/rho - inertial_term) / R
    
    return [Rdot, Rddot]

#============================== Optimization Module ===========================#

# 1. Define Fitting Window (55us - 71us)
mask = (t_r_exp_aligned >= 55e-6) & (t_r_exp_aligned <= 71e-6)
t_fit = t_r_exp_aligned[mask]
r_fit_m = r_r_exp[mask] * 1e-6 

print("==================================================")
print(f"Optimization Window: {len(t_fit)} data points between 55us and 71us")
print("Running Automatic Optimization... Please wait.")

# --- SCALING FACTORS (CRITICAL!) ---
# Easier to optimize if parameters scaled
SCALE_CHI = 1.0       
SCALE_KS  = 1e9       # Ks is in the order of magnitude of 1e-9, so * 1e9  

def objective_function(params_scaled):
    """
    params_scaled: [chi, kappa_s_scaled]
    """
    # return real values
    chi_val = params_scaled[0]
    ks_val = params_scaled[1] / SCALE_KS 
    
    # Solve ODE
    t_span = (t_fit[0], t_fit[-1])
    try:
        sol = solve_ivp(lambda t, y: rp_equation(t, y, chi_val, ks_val, True), 
                        t_span, [R0, 0], method='RK45', rtol=1e-3)
    except:
        return 1e6 

    if not sol.success or len(sol.t) < 5:
        return 1e6

    f_interp = interp1d(sol.t, sol.y[0], kind='linear', fill_value="extrapolate")
    r_sim_at_fit_times = f_interp(t_fit)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((r_sim_at_fit_times - r_fit_m)**2))
    
    # Multiply error by a huge number to make gradients visible
    # RMSE is usually ~1e-7, we scale it to ~0.1
    return rmse * 1e6 

# 3. Run Optimization
# Initial Guesses (Scaled)
# Chi = 0.5, Ks = 4e-9 -> Scaled Ks = 4.0
initial_guess_scaled = [0.5, 4.0] 

# Bounds (Scaled)
# Chi: 0.3~2.0
# Ks: 1.5e-9 ~ 12e-9 -> Scaled: 1.5 ~ 12.0
bounds_scaled = ((0.0, 2.0), (0.0, 20.0))

# Try 'Nelder-Mead' method first (It is derivative-free, more robust for "bumpy" landscapes)
result = minimize(objective_function, initial_guess_scaled, bounds=bounds_scaled, method='Nelder-Mead', tol=1e-4)

# Get best parameters (Remember to un-scale Ks!)
best_chi = result.x[0]
best_ks = result.x[1] / SCALE_KS

print(f"Optimization Success: {result.success}")
print(f"Best Chi (Elasticity): {best_chi:.4f} N/m")
print(f"Best Ks  (Viscosity) : {best_ks:.4e} kg/s")
print(f"Final RMSE error     : {result.fun / 1e6:.2e}") # Un-scale error for display
print("==================================================")

#============================== Final Plotting ================================#

# Run final simulation with BEST parameters over the full time range
t_start = time[0]
t_end = time[-1]
t_eval = np.linspace(t_start, t_end, 5000)
t_span_full = (t_start, t_end)

sol_m = solve_ivp(lambda t, y: rp_equation(t, y, best_chi, best_ks, True), 
                t_span_full, [R0, 0], t_eval=t_eval, method='RK45', max_step=1e-8)

sol_no_m = solve_ivp(lambda t, y: rp_equation(t, y, best_chi, best_ks, False), 
                t_span_full, [R0, 0], t_eval=t_eval, method='RK45', max_step=1e-8)

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Upper Plot Driving Pressure
ax1.plot(time * 1e6, p_total / 1000, 'k-', alpha=0.6, label='Driving Pressure')
ax1.set_ylabel('Pressure [kPa]')
ax1.set_title(f'Driving Signal (Sensitivity: {hydrophone_sensitivity} mV/MPa)')
ax1.grid(True)
ax1.legend()

# Lower Plot Bubble Dynamics
# 1. Draw simulation with marmottant
ax2.plot(sol_m.t * 1e6, sol_m.y[0] * 1e6, 'b-', linewidth=1.5, label='With Marmottant')

# 2. Draw simulation without marmottant
ax2.plot(sol_no_m.t * 1e6, sol_no_m.y[0] * 1e6, '--', linewidth=1.5, label='Without Marmottant')

# 3. Add green experiment dots
ax2.plot(t_r_exp_aligned * 1e6, r_r_exp, 'r.', markersize=4, label='Experiment Data')

# 4. Highlight Fitting Region
ax2.axvspan(55, 71, color='yellow', alpha=0.1, label='Fitting Window (55-71us)')

ax2.set_xlabel('Time [us]')
ax2.set_ylabel('Radius [um]')
ax2.set_title(f'Optimized Fitting: R0={R0*1e6:.2f}um\nChi={best_chi:.3f} N/m, Ks={best_ks:.2e} kg/s')
ax2.grid(True)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()