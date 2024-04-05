# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:56:16 2023

@author: klein
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
from scipy.optimize import fmin


def derivs(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
           #Ca 1001 J/kg/K, Cp water vapour at 95 C 1880 J/kg/K:
           #https://www.engineeringtoolbox.com/water-vapor-d_979.html
           Ta=291, Ra=287, Pa0=86000):
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
        
    W  = wind(s, V)
    rhoa = density_atm(s, V, Ra=Ra)
    rho  = density_fume(s, V)
    Ue   = fct_ue(s, V)   # entrainment velocity normal to plume axis

    dQ  = 2 * rhoa * Ue * V[0] / (np.sqrt(rho * V[1]))

    dM  = g * (rhoa - rho) * V[0]**2 / (rho * V[1]) * np.sin(V[2]) \
        + 2 * rhoa * V[0] / \
        (np.sqrt(rho * V[1])) * Ue * W * np.cos(V[2])

    dth = g * (rhoa-rho) * V[0]**2 / (rho * V[1]**2) * np.cos(V[2]) \
        - 2 * rhoa * V[0] / (V[1] * np.sqrt(rho * V[1])) \
        * Ue * W * np.sin(V[2])

    dE  = (Ca * Ta + (Ue**2) / 2)* dQ + V[1]**2 / (2 * V[0]**2) * dQ \
        - rhoa/rho * V[0] * g * np.sin(V[2]) \
        - 2 * rhoa * np.sqrt(V[1]/rho) * Ue * W * np.cos(V[2])

    dPa = -g * V[4]/ (Ra * Ta) * np.sin(V[2])

    dn  = (1 - V[5]) / V[0] * dQ  # mass fraction of not steam

    return np.array([dQ, dM, dth, dE, dPa, dn])


def wind(s, V):
    """
    Define atmospheric properties:
    wind speed: wind(s,V): constant wind shear between ground and height H1 
    (here 2m as meteorological station), then constant speed W1
    air density: density_atm depending on spec. gas constant of air Ra and air 
    temperature Ta (assumed constant at about 18 deg C)
    """
    theta = V[2]
    z = s * np.sin(theta)
    H1 = 10 # 2m height of meteorological station at Sanner 
    W1 = 10 # average wind speed at Sanner ~38 km/h -> 

    # constant wind shear, 0 m/s at ground up to W1    
    return np.where(z < H1, W1 * z / H1, W1) 



def density_atm(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885, Ta0=291, Ra=287, Pa0=86000):
    Ta = Ta0
    return V[4] / (Ra * Ta)
    
    

   
def Cp(s, V, n0=0.05, Ca=1006, Cp0=1885):
    """
    Define specific heat capacity, Cp, of plume, as a function of the dry air 
      mass fraction, n, for initial 95% of vapour 
    Ca:  spec. heat capacity of air
    Cp0: spec. heat capacity of vapour
    """  
    return Ca + (Cp0 - Ca) * (1 - V[5]) / (1 - n0)


def Rp(s, V, n0=0.05, Ra=287, Rp0=462):
    """
    Define bulk gas constant, Rp, of the plume, as a function of the dry air 
    mass fraction in the plume, n.
    Ra:  gas constant of dry air
    Rp0: gas constant of vapour 
    """

    return Ra + (Rp0 - Ra) * (n0 * (1 - V[5])) / (V[5] * (1 - n0))  


def temperature_fume(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
                     Ta0=291, Ra=287, Pa0=86000):  #specific gas constant water vapour: 461.5 J/(kgK)
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
    return V[3] / (V[0] * Cp(s, V))  


def density_fume(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
                     Ta0=291, Rp0=462, Pa0=86000):
    T  = temperature_fume(s, V)
    return   V[4]/ (Rp(s, V) * T) 

    
def fct_ue(s, V, ks=0.09, kw=0.9):
    W  = wind(s, V)
    return ks * np.abs(V[1] / V[0] - W * np.cos(V[2])) + kw * np.abs(W * np.sin(V[2]))

    
if __name__ == '__main__':
    from Modelling.fumarolePlumeModel.Model_and_synthdata_inversion_updated import *
    from itertools import product
    """
    Solve differential equations for a set of initial values Q0, M0, theta0, 
    E0, Pa0, n0
    """
    # fixed parameters
    v0   = 10   # definitely don't know what this is
    rho0 = 0.5  # careful, density will vary with temperature!
    Cp0  = 1885 # careful, Cp0 will vary with temperatur!e
    Rp0  = 462  # careful, Rp0 will vary with temperatur!e
    Pa0  = 86000 # Atmospheric pressure at altitude of la Soufrière
    theta0 = np.pi/2
    n0   = 0.05
    
    # variables
    R0 = np.linspace(0.1, 1, 11)  # It goes to 11!
    T0 = np.linspace(100+273.15, 200+273.15, 11)
    
    #empty dictionaries and lists
    varlist = [(t0,r0) for (t0,r0) in product(T0, R0)]
    m = {} 
    d = {}
    E = []

    #get all possible combinations of initial conditions
    for var in varlist:
        t0 = var[0]
        r0 = var[1]
        Q0 = rho0 * v0 * r0**2 * np.pi
        M0 = Q0 * v0
        E0 = Q0 * Cp0 * t0
        
        V0 = [Q0,M0,theta0,E0,Pa0,n0]
        
        s = np.linspace(0, 50, 100)
        sol = solve_ivp(derivs, [s[0], s[-1]], V0, t_eval=s)
        
        # out = [sol.t, sol.y[0], sol.y[1], sol.y[2],
        #        sol.y[3], sol.y[4], sol.y[5]]
        # m_out[var] = pd.DataFrame(out, index =['s', 'Q', 'M', 'theta',
        #                                        'E', 'Pa', 'n'])
        
        """
        Transform model output into "useful" variables

        plume temperature T=E/(Q cp)
        plume density: density_fume(T)
        plume velocity U=M/Q
        plume radius R=Q/(sqrt(rho M))
        plume angle theta
        plume height z
        """

        T     = temperature_fume(sol.t, sol.y)
        rho   = density_fume(sol.t, sol.y)
        U     = sol.y[1] / sol.y[0]
        R     = sol.y[0] / np.sqrt(rho * sol.y[1]) 
        theta = sol.y[2]
        z     = sol.t*np.sin(theta)
        
        # m[var] = pd.DataFrame([s, T, R, z, U, rho, theta, n],
        #                       index =['s', 'T', 'R', 'z', 'U',
        #                               'rho', 'theta', 'n'])
        m[var] = pd.DataFrame([T, R, theta], index =['T', 'R', 'theta'])
        
        #add noise
        noise_perc = 0.1
        T_noised = T + (np.random.normal(0, T.std(), len(T)) * noise_perc)
        R_noised = R + (np.random.normal(0, R.std(), len(R)) * noise_perc)
        theta_noised = theta + (
            np.random.normal(0, theta.std(), len(theta)) * noise_perc)
        
        d[var] = pd.DataFrame([T_noised, R_noised, theta_noised],
                              index =['T', 'R', 'theta'])
        
        # Calculate error between model and synthetc data:
        E_T = (abs(T-T_noised))/T.std()
        E_R = (abs(R-R_noised))/R.std()
        E_theta = (abs(theta-theta_noised))/theta.std()
        
        E_sum = E_T.sum() + E_R.sum() + E_theta.sum()
        E.append(E_sum)  # list with absolute error for each of the starting
                         # conditions
        

    t0, r0 = zip(*varlist)
    size = [1.2**n for n in E]

    fig1, ax = plt.subplots()    
    cs = ax.scatter(t0, r0, s=size, c=E, cmap='viridis')  #c=colors, alpha=0.5
    ax.set_xlabel('T0', fontsize=12)
    ax.set_ylabel('R0', fontsize=12)
    cbar = plt.colorbar(cs)
    plt.show() 
