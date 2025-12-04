import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d 

#============================== Water Properties ==============================#
rho = 1000              # density [kg/m^3]
mu = 1e-3               # viscosity [Pa.s]
gamma = 0.072           # surface tension of water [N/m]

#============================= Initial Properties =============================#
sigma_0 = 0.0               # initial surface tension [N/m] (Assumption: Buckled)
p_inf_static = 98325        # ambient pressure [Pa]
k = 1.0                     # polytropic index (1.07 for isothermal-ish, 1.0 for isothermal, 1.4 for adiabatic)
hydrophone_sensitivity = 39 # [mV/MPa]
current_kappa_s = 12e-9     # Example value: [1.5 12]e-9 kg/s 
current_chi = 2             # Example value: [0.3 2.0] N/m
pressure_scaling = 0.5

#================================= CSV File ===================================#
HYD_FILE = 'Data/cleaned_F1--1.5mhz-100mV-4.csv'
EXP_CSV = 'Data/AVI24/Camera_15_02_48/Camera_15_02_48_radius.csv'

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
    elif R >= R_rupt:   # Ruptured regime
        sigma = -2 * gamma/R
    else:               # Elastic regime
        sigma = -2 * chi * ((R/R_buck)**2 - 1)/R

    return sigma + shell_visc_term

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
def rp_equation(t, y, chi, kappa_s):
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

    # Combine RHS terms, Note: viscous_term and marmottant_term already calculate negative forces
    RHS = pressure_term + viscous_term + marmottant_term
    
    # Solve for Rddot
    Rddot = (RHS/rho - inertial_term) / R
    
    return [Rdot, Rddot]

#============================== Main Solve Loop ===============================#

# Follow the CSV time: start with 55us
t_start = time[0]
t_end = time[-1]
t_span = (t_start, t_end)

t_eval = np.linspace(t_start, t_end, 5000)
y0 = [R0, 0] # Initial conditions

print(f"Start Simulation... (chi={current_chi}, kappa_s={current_kappa_s})")
sol = solve_ivp(lambda t, y: rp_equation(t, y, current_chi, current_kappa_s), 
                t_span, y0, t_eval=t_eval, method='RK45', max_step=1e-8)

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Upper Plot Driving Pressure
ax1.plot(time * 1e6, p_total / 1000, 'k-', alpha=0.6, label='Driving Pressure')
ax1.set_ylabel('Pressure [kPa]')
ax1.set_title(f'Driving Signal (Sensitivity: {hydrophone_sensitivity} mV/MPa)')
ax1.grid(True)
ax1.legend()

# Lower Plot Bubble Dynamics
# 1. Draw simulation (RP + M)
ax2.plot(sol.t * 1e6, sol.y[0] * 1e6, 'b-', linewidth=1.5, label='Simulation (Marmottant)')

# 2. Add green experiment dots
ax2.plot(t_r_exp_aligned * 1e6, r_r_exp, 'g.', markersize=4, label='Experiment Data')

ax2.set_xlabel('Time [us]')
ax2.set_ylabel('Radius [um]')
ax2.set_title(f'Fitting: R0={R0*1e6:.2f}um, chi={current_chi}, Ks={current_kappa_s:.1e}\nOffset={camera_start_offset*1e6}us')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()