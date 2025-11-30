import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#============================== Water Properties ==============================#
rho = 1000              # density [kg/m^3]
mu = 1e-3               # viscosity [Pa.s]
gamma = 0.072           # surface tension of water [N/m]

#============================= Initial Properties =============================#
sigma_0 = 0.0           # initial surface tension [N/m] (Assumption: Buckled)
R0 = 10e-6              # initial radius [m]
p_inf_static = 101325   # ambient pressure [Pa]
k = 1.07                # polytropic index (1.07 for isothermal-ish, 1.4 for adiabatic)
p_gas_0 = p_inf_static + (2 * sigma_0 / R0) # initial gas pressure [Pa]

#============================ Acoustic Properties =============================#
f = 2e6                 # frequency [Hz]
omega = 2 * np.pi * f   # angular frequency [rad/s]
P_A = 200e3             # pressure amplitude [Pa]
def P_driving(t):
    """Time-varying driving pressure P_inf(t)"""
    return p_inf_static + P_A * np.sin(omega * t)

#============================= Image Processing  ==============================#
def R_effective(Area):
    """This is the R resolved by cv2 from image processing"""
    return np.sqrt(Area/np.pi)

#============================= Marmottant Model ===============================#
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

#====================== Raleigh Plesset Liniearization =======================#
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
# Example: Select ONE set of parameters to solve
current_kappa_s = 2.4e-9  # Example value: 2.4e-9 kg/s
current_chi = 1.0         # Example value: 1.0 N/m

t_span = (0, 10/f) # Simulate 10 cycles
t_eval = np.linspace(t_span[0], t_span[1], 1000)
y0 = [R0, 0] # Initial conditions: R=R0, Rdot=0

# Use lambda to pass parameters into solve_ivp
sol = solve_ivp(lambda t, y: rp_equation(t, y, current_chi, current_kappa_s), 
                t_span, y0, t_eval=t_eval, method='RK45')

# Plotting
plt.figure()
plt.plot(sol.t * 1e6, sol.y[0] * 1e6)
plt.xlabel('Time [us]')
plt.ylabel('Radius [um]')
plt.title(f'Marmottant Model (chi={current_chi}, Ks={current_kappa_s})')
plt.grid(True)
plt.show()