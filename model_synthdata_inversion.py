# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:56:16 2023

Modified by D. E. Jessop, 2024-04-16

@author: klein
"""

import numpy as np

from myiapws import iapws1995

Tt = iapws1995.Tt  # Triple point of water


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


def produce_Gm(sol, s, cutoff=None):
    """Returns the model predictions in a format suitable for comparison with 
    data"""
    rho  = density_fume(sol.t, sol.y)
    T    = temperature_fume(sol.t, sol.y) - Tt
    Cp   = heat_capacity(sol.t, sol.y)
    b    = sol.y[0] / np.sqrt(rho * sol.y[1])
    u    = sol.y[1] / sol.y[0]
    theta = sol.y[3]
    # insert solution into array of large values where soln is valid
    L = len(sol.t)
    if L > cutoff:
        L = cutoff
    solp  = -9999 * np.ones((3, cutoff))
    solp[:, :L] = np.array([theta[:L],
                            b[:L],
                            T[:L]])

    if cutoff is not None:
        solp = solp[:,:cutoff]
    return solp.flatten()


def objective_fn(model, data, errors, mode='lstsq'):
    """Calculate and return the objective (model misfit) function"""

    Gm_d   = model - data
    if mode != 'lstsq':
        # errors should be in 1D array form in this case,
        # apply the absolute value
        return (-np.abs(Gm_d) * np.sqrt(errors)).sum()
    Cd_inv = errors.copy()
    if len(errors.shape) != 2:
        Cd_inv = np.diag(1/errors**2)
    return -.5 * Gm_d @ Cd_inv @ Gm_d


def parallel_job(u0, R0, T0):
    warnings.filterwarnings('ignore')
    V0   = [1, 1, Cp0 * T0, 1, 1, .05]  # required for routines
    rho0 = density_fume(0, V0)
    Q0   = rho0 * np.pi * R0**2 * u0
    M0   = Q0 * u0
    E0   = Q0 * Cp0 * T0
    V0   = [Q0, M0, E0, np.pi/2, Pa0, .05]

    sol  = solve_ivp(derivs, [s[0], s[-1]], V0, t_eval=s)
    Gm   = produce_Gm(sol, s)

    return objective_fn(Gm, d, Cd_inv, mode='lstsq'), V0


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
    rho0 = 0.5   # careful, density will vary with temperature!
    Cp0  = 1885  # careful, Cp0 will vary with temperatur!e
    Rp0  = 462   # careful, Rp0 will vary with temperatur!e
    Pa0  = 86000 # Atmospheric pressure at altitude of la Soufrière
    theta0 = np.pi/2
    n0   = 0.05  # Fumarole plumes are always 95% vapour?
    Ta0  = 291   # Tair 2-year average at Sanner
    s    = np.linspace(0, 250, 501)
    # variables
    rho0true = .5
    R0true = .5
    T0true = Tt + 96
    u0true = 10
    Q0true = rho0true * u0true * R0true**2 * np.pi
    M0true = Q0true * u0true
    E0true = Q0true * Cp0 * T0true
    V0true = [Q0true, M0true, E0true, theta0, Pa0, n0]
 
    sol_true = solve_ivp(derivs, [s[0], s[-1]], V0true, t_eval=s)

    cm = 1/2.54

    # Produce "true" data values
    sol  = sol_true
    rho  = density_fume(sol.t, sol.y)
    T    = temperature_fume(sol.t, sol.y) - Tt
    Cp   = heat_capacity(sol.t, sol.y)
    b    = sol.y[0] / np.sqrt(rho * sol.y[1])
    u    = sol.y[1] / sol.y[0]
    theta = sol.y[3]
    
    sigtheta, sigb, sigT = np.pi / 20, .5, .5  # rad, m and K
    Cd_inv = np.diag(np.array([1 / sigtheta**2 * np.ones_like(theta),
                               1 / sigb**2 * np.ones_like(b),
                               1 / sigT**2 * np.ones_like(T)]).ravel())

    solp = np.array([theta, b, T])
    noise = np.random.randn(*solp.shape)  # Gaussian noise, _N_(0,1)
    sol_noise = (solp.T + noise.T * (sigtheta, sigb, sigT)).T
    d    = sol_noise.flatten()  # array of data
    Gm   = produce_Gm(sol)
    Gm_d = Gm - d
    
    fig0, ax0 = plt.subplots(figsize=(10*cm, 10*cm))
    ax0.plot(solp.T, s, '-')
    ax0.set_xlabel('Plume parameters')
    ax0.set_ylabel(r'Distance along plume axis, $s$')
    ax0.legend((r'$\theta$', r'$b$', r'$T$'))

    plt.gca().set_prop_cycle(None)  # reset colour cycle
    ax0.plot(sol_noise.T, s, '.')

    fig0.tight_layout()
    
    E = np.exp(objective_fn(Gm, d, Cd_inv))
    print(f'Value of objective function: {E}')
    
    ## Run jobs in parallel in order to calculate objective function at
    ## each combination of possible source values
    from joblib import Parallel, delayed

    import time
    import warnings

    njobs = 16
    ngrid = 51  # Number of grid points

    R0 = np.linspace(0.1, 1, ngrid)  
    T0 = np.linspace(80 + Tt, 160 + Tt, ngrid)
    u0 = np.linspace(0.1, 100, ngrid)

    t = time.time()
    sequence = [u0, R0, T0]
    results = Parallel(n_jobs=njobs)(delayed(parallel_job)(u0, R0, T0) 
                                     for (u0,R0,T0) in list(product(*sequence)))
    print("Job ran in %.3f s" % (time.time() - t))

    ## Deal out the results so as to 
    objFn, initialConds = [], []

    for result in results:
        objFn.append(result[0])
        initialConds.append(result[1])

    initialConds = np.array(initialConds)
    objFn = np.array(objFn).reshape((-1, nGrid))
               
    # #empty list for storing objective fn
    # objFn = []
    # for (T0_,R0_) in product(T0, R0):
    #     V0   = [1, 1, Cp0 * T0_, np.pi, Pa0, .05]  # required for routines
    #     rho0 = density_fume(0, V0)
    #     Q0   = rho0 * np.pi * R0_**2 * v0
    #     M0   = Q0 * v0
    #     E0   = Q0 * Cp0 * T0_
    #     V0   = [Q0, M0, E0, np.pi, Pa0, .05]

    #     sol  = solve_ivp(derivs, [s[0], s[-1]], V0, t_eval=s)
        
    #     Gm   = produce_Gm(sol)
        
    #     Gm_d = Gm - d
    #     objFn.append(objective_fn(Gm, d, sigma, mode='lstsq'))

    # objFn = np.array(objFn).reshape((-1, N))

    fig, ax = plt.subplots(figsize=(10*cm, 10*cm))
    ax.pcolor(R0, T0, objFn, label='objective function')
    ax.plot(R0true, T0true, 'wo', label='true conditions')
    # argmax gives element number.  To convert to row number, do Euclidian
    # division wrt N and take modulo N for columns.
    R0opt, T0opt = R0[objFn.argmax() % N], T0[objFn.argmax() // N]
    ax.plot(R0opt, T0opt, 'ko', label='opt. conditions')
    
    ax.set_xlabel(r'Vent radius, $R_0/$[m]')
    ax.set_ylabel(r'Vent temperature, $T_0$/[K]')

    ax.legend()

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
