import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

rho = 1000      # density [kg/m^3]
mu = 1e-3       # viscosity [Pa.s]
sigma = 0.072   # surface tension [N/m]
p_inf = 1e5     # ambient pressure [Pa]
R0 = 10e-6      # initial radius [m]

def pB(R):
    # gas pressure inside bubble (polytropic)
    p0 = 3e5    # initial gas pressure [Pa]
    kappa = 1.4 # polytropic index
    return (p0 + 2*sigma/R0) * (R0 / R)**(3*kappa)

def rp_equation(t, y):
    """
    Rayleigh-Plesset equation First Order ODE system.
    y[0]: bubble radius R
    y[1]: radial velocity Rdot
    --------------------------------------------
    dR/dt = Rdot
    dRdot/dt = (1/R) * ((1/rho)[ (pB(R) - p_inf) - (2*sigma)/(R) - (4*mu*Rdot)/(R)] - (3/2)*Rdot^2)
    --------------------------------------------
    Returns: [dR/dt, dRdot/dt]
    """

    R, Rdot = y                                             # y[0]: radius, y[1]: radial velocity
    term = (3/2)*Rdot**2                                    # inertial term
    RHS = (pB(R) - p_inf) - 2*sigma/R - 4*mu*Rdot/R         # right-hand side
    Rddot = (RHS/rho - term) / R
    return [Rdot, Rddot]

def main():
    # initial radius 10 microns, initial velocity 0 m/s
    # solve the ODE from t=0 to t=50 microseconds by integrating rp_equation (Rdot and Rddot)
    # with a maximum step size of 10 nanoseconds
    sol = solve_ivp(rp_equation, 
                    [0, 50e-6], 
                    [10e-6, 0], 
                    max_step=1e-8, 
                    method='RK45', 
                    dense_output=True)
    return sol     

if __name__ == "__main__":
    sol = main()
    plt.plot(sol.t, sol.y[0])
    plt.xlabel("time (s)")
    plt.ylabel("R(t)")
    plt.show()