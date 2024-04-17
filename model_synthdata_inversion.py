# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:56:16 2023

@author: klein
"""

import numpy as np


def derivs(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
           Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    #Ca 1001 J/kg/K, Cp water vapour at 95 C 1880 J/kg/K:
    #https://www.engineeringtoolbox.com/water-vapor-d_979.html
    """
    axial distance s, state vector V, 
    entrainment coefficient without wind ks, entrainment coefficient with wind 
    kw (from Woodhouse et al. 2013)
    specific heat capacity of atmosphere Ca, Specific heat capacity of plume Cp
    Air temperature (K) Ta0: average air T Sanner last 2 years
    specific gas constant of dry air Ra = 287 J/kg/K
    specific gas constant of water vapour Rp = 461.5 J/kg/K

    order of state vector V:  Q, M, th, E, Pa, n (mass flux, momentum flux, 
    plume angle, enthalpy flux, atmospheric pressure, mass fraction air)
    """
    Ta   = Ta0
    W    = wind_profile(s, V)
    rhoa = density_atm(s, V, Ra=Ra)
    rho  = density_fume(s, V)
    Ue   = entrainment_vel(s, V)   # entrainment velocity normal to plume axis
    Q, M, E, th, Pa, n = V

    dQ  = 2 * rhoa * Ue * Q / (np.sqrt(rho * M))

    dM  = g * (rhoa - rho) * Q**2 / (rho * M) * np.sin(th) \
        + 2 * rhoa * Q / (np.sqrt(rho * M)) * Ue * W * np.cos(th)
    
    dE  = (Ca * Ta + (Ue**2) / 2)* dQ + M**2 / (2 * Q**2) * dQ \
        - rhoa/rho * Q * g * np.sin(th) \
        - 2 * rhoa * np.sqrt(M / rho) * Ue * W * np.cos(th)

    dth = g * (rhoa-rho) * Q**2 / (rho * M**2) * np.cos(th) \
        - 2 * rhoa * Q / (M * np.sqrt(rho * M)) * Ue * W * np.sin(Q)

    dPa = -g * Pa/ (Ra * Ta) * np.sin(th)

    dn  = (1 - n) / Q * dQ  # mass fraction of not steam

    return np.array([dQ, dM, dE, dth, dPa, dn])


def entrainment_vel(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
                    Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    W  = wind_profile(s, V)
    Q, M, _, th, _, _ = V  # Don't need E, Pa, n
    return ks * np.abs(M / Q - W * np.cos(th)) + kw * np.abs(W * np.sin(th))

    
def wind_profile(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
                 Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05, W1=5, H1=5):
    """
    Define atmospheric properties:
    wind speed: wind_profile(s,V): constant wind shear between ground and 
    height H1, then constant speed W1
    air density: density_atm depending on spec. gas constant of air Ra and air 
    temperature Ta (assumed constant at about 18 deg C)
    """
    # average wind speed at Sanner ~38 km/h -> 10.556 m/s
    theta = V[3]
    z = s * np.sin(theta)
    # constant wind shear, 0 m/s at ground up to W1    
    return 0.01 * np.ones_like(s)  # np.where(z < H1, W1 * z / H1, W1) 


def density_atm(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
                Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    Ta = Ta0
    Pa = V[4]
    return Pa / (Ra * Ta)
    

def density_fume(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
                 Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    '''Returns the density of the fumarole, based on ideal gas
    '''
    T  = temperature_fume(s, V)
    Pa = V[4]  # atmospheric pressure
    n  = V[5]
    Rp = bulk_gas_constant(s, V)
    return Pa / (Rp * T) 


def heat_capacity(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
                  Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    """
    Define specific heat capacity, Cp, of plume, as a function of the dry air 
      mass fraction, n, for initial 95% of vapour 
    Ca:  spec. heat capacity of air
    Cp0: spec. heat capacity of vapour
    """  
    return Ca + (Cp0 - Ca) * (1 - V[5]) / (1 - n0)


def bulk_gas_constant(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
                      Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    """
    Define bulk gas constant, Rp, of the plume, as a function of the dry air 
    mass fraction in the plume, n.
    Ra:  gas constant of dry air
    Rp0: gas constant of vapour 
    """
    n = V[5]  # dry air mass fraction
    return Ra + (Rp0 - Ra) * (n0 * (1 - n)) / (n * (1 - n0))  


def temperature_fume(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
                     Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    #specific gas constant water vapour: 461.5 J/(kgK)
    """
    Define fumarole plume properties
    plume temperature: temperature_fume, as a function of the mass flux, 
    enthalpy flux and specific heat capacity of vapour.
    plume density: density_fume as a function of spec. gas constant of 
    vapour, Rp, atmospheric pressure and plume temperature T
    entrainment velocity as a function of plume velocity U = V[1] / V[0], 
    plume angle theta: V[2], 
    windspeed W
    entrainment coefficients: ks and kw
    """
    Q, _, E, *_ = V
    Cp = heat_capacity(s, V)
    return E / (Q * Cp)  


if __name__ == '__main__':
    from itertools import product
    from myiapws import iapws1992, iapws1995
    from matplotlib import pyplot as plt
    #from mpl_toolkits import mplot3d
    from scipy.integrate import solve_ivp
    from scipy.optimize import fmin

    import pandas as pd
    """
    Solve differential equations for a set of initial values Q0, M0, theta0, 
    E0, Pa0, n0
    """
    plt.close('all')

    # fixed parameters
    v0   = 10    # definitely don't know what this is
    rho0 = 0.5   # careful, density will vary with temperature!
    Cp0  = 1885  # careful, Cp0 will vary with temperatur!e
    Rp0  = 462   # careful, Rp0 will vary with temperatur!e
    Pa0  = 86000 # Atmospheric pressure at altitude of la Soufrière
    theta0 = np.pi/2
    n0   = 0.05  # Fumarole plumes are always 95% vapour?
    Ta0  = 291   # Tair 2-year average at Sanner
    s    = np.linspace(0, 250, 501)
    
    # variables
    Tt = iapws1995.Tt  # Triple point of water
    R0 = np.linspace(0.1, 1, 11)  # It goes to 11!
    T0 = np.linspace(80 + Tt, 160 + Tt, 11)

    R0true = .5
    T0true = Tt + 96
    Q0true = rho0 * v0 * R0true**2 * np.pi
    M0true = Q0true * v0
    E0true = Q0true * Cp0 * T0true
    V0true = [Q0true, M0true, E0true, theta0, Pa0, n0]
 
    sol_true = solve_ivp(derivs, [s[0], s[-1]], V0true, t_eval=s)

    cm = 1/2.54
    fig0, ax0 = plt.subplots(figsize=(8*cm, 8*cm))
    ax0.plot(sol_true.y.T, sol_true.t, '-')
    ax0.set_xlabel('Plume parameters')
    ax0.set_ylabel(r'Distance along plume axis, $s$')
    ax0.legend((r'$Q$', r'$M$', r'$E$', r'$\theta$', r'$P_a$', r'$n$'))

    sol  = sol_true
    rho  = density_fume(sol.t, sol.y)
    T    = temperature_fume(sol.t, sol.y) - Tt
    Cp   = heat_capacity(sol.t, sol.y)
    b    = sol.y[0] / np.sqrt(rho * sol.y[1])
    u    = sol.y[1] / sol.y[0] 
    solp = np.array([b, u, T])
    fig1, ax1 = plt.subplots(figsize=(8*cm, 8*cm))
    ax1.plot(solp.T, sol.t, '-')
    ax1.set_xlabel('Plume parameters')
    ax1.set_ylabel(r'Distance along plume axis, $s$')
    ax1.legend((r'$b$', r'$u$', r'$T$'))

    # #empty dictionaries and lists
    # varlist = [(t0,r0) for (t0,r0) in product(T0, R0)]
    # m = {} 
    # d = {}
    # E = []

    # #get all possible combinations of initial conditions
    # for var in varlist:
    #     t0 = var[0]
    #     r0 = var[1]
    #     Q0 = rho0 * v0 * r0**2 * np.pi
    #     M0 = Q0 * v0
    #     E0 = Q0 * Cp0 * t0
        
    #     V0 = [Q0, M0, E0, theta0, Pa0, n0]
        
    #     s = np.linspace(0, 50, 101)
    #     sol = solve_ivp(derivs, [s[0], s[-1]], V0, t_eval=s)
        
    #     # out = [sol.t, sol.y[0], sol.y[1], sol.y[2],
    #     #        sol.y[3], sol.y[4], sol.y[5]]
    #     # m_out[var] = pd.DataFrame(out, index =['s', 'Q', 'M', 'theta',
    #     #                                        'E', 'Pa', 'n'])
        
    #     """
    #     Transform model output into "useful" variables

    #     plume temperature T=E/(Q cp)
    #     plume density: density_fume(T)
    #     plume velocity U=M/Q
    #     plume radius R=Q/(sqrt(rho M))
    #     plume angle theta
    #     plume height z
    #     """

    #     T     = temperature_fume(sol.t, sol.y)
    #     rho   = density_fume(sol.t, sol.y)
    #     U     = sol.y[1] / sol.y[0]
    #     R     = sol.y[0] / np.sqrt(rho * sol.y[1]) 
    #     theta = sol.y[3]
    #     z     = sol.t*np.sin(theta)
        
    #     # m[var] = pd.DataFrame([s, T, R, z, U, rho, theta, n],
    #     #                       index =['s', 'T', 'R', 'z', 'U',
    #     #                               'rho', 'theta', 'n'])
    #     m[var] = pd.DataFrame([T, R, theta], index =['T', 'R', 'theta'])
        
    #     #add noise
    #     noise_perc = 0.1
    #     T_noised = T + (np.random.normal(0, T.std(), len(T)) * noise_perc)
    #     R_noised = R + (np.random.normal(0, R.std(), len(R)) * noise_perc)
    #     theta_noised = theta + (
    #         np.random.normal(0, theta.std(), len(theta)) * noise_perc)
        
    #     d[var] = pd.DataFrame([T_noised, R_noised, theta_noised],
    #                           index =['T', 'R', 'theta'])
        
    #     # Calculate error between model and synthetc data:
    #     E_T = (abs(T-T_noised))/T.std()
    #     E_R = (abs(R-R_noised))/R.std()
    #     E_theta = (abs(theta-theta_noised))/theta.std()
        
    #     E_sum = E_T.sum() + E_R.sum() + E_theta.sum()
    #     E.append(E_sum)  # list with absolute error for each of the starting
    #                      # conditions
        

    # t0, r0 = zip(*varlist)
    # size = [1.2**n for n in E]

    # fig1, ax = plt.subplots()    
    # cs = ax.scatter(t0, r0, s=size, c=E, cmap='viridis')  #c=colors, alpha=0.5
    # ax.set_xlabel('T0', fontsize=12)
    # ax.set_ylabel('R0', fontsize=12)
    # cbar = plt.colorbar(cs)
    # plt.show() 
