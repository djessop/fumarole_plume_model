# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:56:16 2023

Modified by D. E. Jessop, 2024-04-16

@author: klein
"""

from itertools import product
from myiapws import iapws1992, iapws1995
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
#from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from joblib import Parallel, delayed
#from IPython.display import IFrame
from functools import partial

import numpy as np
import time
#import pandas as pd
#import plotly.graph_objects as go


Tt = iapws1995.Tt  # Triple point of water
cm = 1/2.54


def derivs(s, V, ks=0.09, kw=0.5, g=9.81, Ca=1006, Cp0=1885,
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


def entrainment_vel(s, V, ks=0.09, kw=0.5, g=9.81, Ca=1006, Cp0=1885,
                    Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    W  = wind_profile(s, V)
    Q, M, _, th, _, _ = V  # Don't need E, Pa, n
    return ks * np.abs(M / Q - W * np.cos(th)) + kw * np.abs(W * np.sin(th))

    
def wind_profile(s, V, ks=0.09, kw=0.5, g=9.81, Ca=1006, Cp0=1885,
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
    return 1 * np.ones_like(s)  # np.where(z < H1, W1 * z / H1, W1) 


def density_atm(s, V, ks=0.09, kw=0.5, g=9.81, Ca=1006, Cp0=1885,
                Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    Ta = Ta0
    Pa = V[4]
    return Pa / (Ra * Ta)
    

def density_fume(s, V, ks=0.09, kw=0.5, g=9.81, Ca=1006, Cp0=1885,
                 Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    '''Returns the density of the fumarole, based on ideal gas
    '''
    T  = temperature_fume(s, V)
    Pa = V[4]  # atmospheric pressure
    n  = V[5]
    Rp = bulk_gas_constant(s, V)
    return Pa / (Rp * T) 


def heat_capacity(s, V, ks=0.09, kw=0.5, g=9.81, Ca=1006, Cp0=1885,
                  Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    """
    Define specific heat capacity, Cp, of plume, as a function of the dry air 
      mass fraction, n, for initial 95% of vapour 
    Ca:  spec. heat capacity of air
    Cp0: spec. heat capacity of vapour
    """  
    return Ca + (Cp0 - Ca) * (1 - V[5]) / (1 - n0)


def bulk_gas_constant(s, V, ks=0.09, kw=0.5, g=9.81, Ca=1006, Cp0=1885,
                      Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    """
    Define bulk gas constant, Rp, of the plume, as a function of the dry air 
    mass fraction in the plume, n.
    Ra:  gas constant of dry air
    Rp0: gas constant of vapour 
    """
    n = V[5]  # dry air mass fraction
    return Ra + (Rp0 - Ra) * (n0 * (1 - n)) / (n * (1 - n0))  


def temperature_fume(s, V, ks=0.09, kw=0.5, g=9.81, Ca=1006, Cp0=1885,
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


def produce_Gm(s, sol):
    """Returns the model predictions in a format suitable for comparison with 
    data"""
    rho  = density_fume(sol.t, sol.y)
    T    = temperature_fume(sol.t, sol.y) - Tt
    Cp   = heat_capacity(sol.t, sol.y)
    b    = sol.y[0] / np.sqrt(rho * sol.y[1])
    u    = sol.y[1] / sol.y[0]
    theta = sol.y[3]
    # insert solution into array of large values where soln is valid

    L = len(s)
    Lsol  = len(sol.t)
    solp  = -9999 * np.ones((3, L))
    solp[:, :Lsol] = np.array([theta[:Lsol],
                               b[:Lsol],
                               T[:Lsol]])

    return solp.flatten()


def objective_fn(model, data, errors, mode='leastsq', exponentiate=False):
    """Calculate and return the objective (model misfit) function"""

    Gm_d   = model - data
    errors = np.array(errors)
    if mode != 'abs' and mode != 'leastsq' and mode != 'lstsq':
        raise TypeError('Unknown mode: should be either "abs" or "leastsq"')
    
    # Laplacian distributed erros - absolute value of model - data
    if mode == 'abs':
        # errors should be in 1D array form in this case,
        if len(errors.shape) == 2:
            errors = np.diag(errors)
        # apply the absolute value
        S = (np.abs(Gm_d) * np.sqrt(errors)).sum()

    # Gaussian distributed errors
    if mode == 'leastsq' or mode == 'lstsq':
        if len(errors.shape) != 2:
            errors = np.diag(1/errors**2)
        S = .5 * Gm_d @ errors @ Gm_d

    if not exponentiate:
        return S
    return 1 - np.exp(-S)


def parallel_job(U0, R0, T0, s, d, Cd_inv): 
    import warnings
    from scipy.integrate import solve_ivp

    # errors/Cd_inv are not passed explicitly as arguements
    warnings.filterwarnings('ignore')
    n0   = .05
    Cp0  = heat_capacity(0, [0, 0, 0, 0, 0, n0])
    Pa0  = 86000
    V0   = [1, 1, Cp0 * T0, 1, Pa0, n0]  # required for routines
    rho0 = density_fume(0, V0)
    Q0   = rho0 * np.pi * R0**2 * U0
    M0   = Q0 * U0
    E0   = Q0 * Cp0 * T0
    V0   = [Q0, M0, E0, np.pi/2, Pa0, .05]

    sol  = solve_ivp(derivs, [s[0], s[-1]], V0, t_eval=s)
    Gm   = produce_Gm(s, sol)

    return objective_fn(Gm, d, Cd_inv, mode='lstsq'), V0


def solve_system(x0, s, data, errors):
    """Helper function for the minimisation problem using 
    scipy.optimize.minimize
    """
    sol = solve_ivp(derivs, [s[0], s[-1]], x0, t_eval=s)
    Gm  = produce_Gm(s, sol)
    return objective_fn(Gm, d, errors, mode='lstsq')
    

def do_plots(dimensionality, T0, R0, objFn, exponentiate=False):
    fig0, ax0 = plt.subplots(figsize=(10*cm, 10*cm))
    ax0.plot(solp.T, s, '-')
    ax0.set_xlabel('Plume parameters')
    ax0.set_ylabel(r'Distance along plume axis, $s$')
    ax0.legend((r'$\theta$', r'$b$', r'$T$'))

    plt.gca().set_prop_cycle(None)  # reset colour cycle
    ax0.plot(sol_noise.T, s, '.')

    fig0.tight_layout()
    
    S = objective_fn(Gm, d, Cd_inv, 'leastsq', False)
    print(f'Value of misfit function: {S}')
    
    fig, ax = plt.subplots(figsize=(10*cm, 10*cm))
    if dimensionality == 2:
        Ropt_ind, Topt_ind = np.where(objFn == objFn.min())
        im = ax.pcolor(T0, R0, np.exp(-objFn), norm=LogNorm(),
                       label='misfit function')
    if dimensionality == 3:
        Uopt_ind, Ropt_ind, Topt_ind = np.where(objFn == objFn.min())
        im = ax.pcolor(T0, R0, np.exp(-objFn[Uopt_ind]), norm=LogNorm(),
                       label='misfit function')
    p = ax.plot(T0true, R0true, 'wo', label='true conditions')
    # q = ax.plot(T0[Topt_ind], R0[Ropt_ind], 'y*', mec='k', mew=.25,
    #             label='opt. conditions')
    
    ax.set_xlabel(r'Vent radius, $R_0/$[m]')
    ax.set_ylabel(r'Vent temperature, $T_0$/[K]')

    ax.legend()

    return ax0, ax
    
if __name__ == '__main__':
    import sys

    ## Helper functions for plotting
    def get_the_slice(x,y,z, surfacecolor):
        return go.Surface(x=x,
                          y=y,
                          z=z,
                          surfacecolor=surfacecolor,
                          coloraxis='coloraxis')

    def get_lims_colors(surfacecolor):# color limits for a slice
        return np.min(surfacecolor), np.max(surfacecolor)


    def colorax(vmin, vmax):
        return dict(cmin=vmin,
                    cmax=vmax)

    """
    Solve differential equations for a set of initial values Q0, M0, theta0, 
    E0, Pa0, n0
    """
    plt.close('all')

    ##  Job options
    ncore =  12  # Number of processors/cores
    ngrid = 101  # Number of grid points
    if len(sys.argv) == 2:
        ncore   = int(sys.argv[1])
    if len(sys.argv) >= 3:
        ncore   = int(sys.argv[1])
        ngrid   = int(sys.argv[2])
    inversion   = False
    nelder_mead = False
    plots       = False
    if len(sys.argv) == 4:
        if sys.argv[3] == 'True':
            inversion = True

    R0 = np.linspace(0.1, 1, ngrid)  
    T0 = np.linspace(80 + Tt, 160 + Tt, ngrid)
    U0 = np.linspace(0.1, 100, ngrid)

    ##  fixed parameters
    nsol   =   101   # number of points at which "observations" will be made
    Cp0    =  1885   # careful, Cp0 will vary with temperatur!e
    Pa0    = 86000   # Atmospheric pressure at altitude of la Soufriere
    theta0 = np.pi/2
    n0     = 0.05  # Fumarole plumes are always 95% vapour?
    Ta0    = 291   # Tair 2-year average at Sanner
    s      = np.linspace(0, 100, nsol)

    ##  variables
    rho0true = .5
    R0true = .5
    T0true = Tt + 96
    U0true = 10
    Q0true = rho0true * U0true * R0true**2 * np.pi
    M0true = Q0true * U0true
    E0true = Q0true * Cp0 * T0true
    V0true = [Q0true, M0true, E0true, theta0, Pa0, n0]
 
    sol_true = solve_ivp(derivs, [s[0], s[-1]], V0true, t_eval=s)

    # Produce "true" data values
    sol   = sol_true
    rho   = density_fume(sol.t, sol.y)
    T     = temperature_fume(sol.t, sol.y) - Tt
    Cp    = heat_capacity(sol.t, sol.y)
    b     = sol.y[0] / np.sqrt(rho * sol.y[1])
    u     = sol.y[1] / sol.y[0]
    theta = sol.y[3]

    noise_level = 1
    sigtheta, sigb, sigT = (np.pi / 10 * noise_level,
                            .5 * noise_level, 1 * noise_level)  # rad, m and K
    Cd_inv = np.diag(np.array([1 / sigtheta**2 * np.ones_like(theta),
                               1 / sigb**2 * np.ones_like(b),
                               1 / sigT**2 * np.ones_like(T)]).ravel())

    solp  = np.array([theta, b, T])
    noise = np.random.randn(*solp.shape)  # Gaussian noise, _N_(0,1)
    sol_noise = (solp.T + noise.T * (sigtheta, sigb, sigT)).T
    d    = sol_noise.flatten()  # array of data
    Gm   = produce_Gm(s, sol)
    Gm_d = Gm - d
    
    if inversion:
        """
        Run jobs in parallel in order to calculate objective function at
        each combination of possible source values.
        """
        # "wrapper" partial function to simplify notation below
        pj = partial(parallel_job, s=s, d=d, Cd_inv=Cd_inv)
        t  = time.perf_counter()

        dimensionality = 2
        if dimensionality == 2:
            u0 = U0true
            sequence = [R0, T0]
            results = Parallel(n_jobs=ncore)(delayed(pj)(u0, r0, t0) for (
                r0, t0) in list(product(*sequence)))

            print("Job ran in %.3f s using %2d processors" % (
                time.perf_counter() - t, ncore))

            ## Deal out the results
            objFn, initialConds = [], []

            for result in results:
                objFn.append(result[0])
                initialConds.append(result[1])

            initialConds = np.array(initialConds)
            objFn = np.array(objFn).reshape((-1, ngrid))

        if dimensionality == 3:
            sequence = [U0, R0, T0]
            results = Parallel(n_jobs=ncore)(delayed(pj)(u0, r0, t0) for (
                u0, r0, t0) in list(product(*sequence)))

            print("Job ran in %.3f s using %2d processors" % (
                time.perf_counter() - t, ncore))

            ## Deal out the results
            objFn, initialConds = [], []

            for result in results:
                objFn.append(result[0])
                initialConds.append(result[1])

            initialConds = np.array(initialConds)
            objFn = np.array(objFn).reshape((-1, ngrid, ngrid))

    if nelder_mead:
        x0  = V0true  # [1, 1, 1, np.pi/2 , 86000, 0.05]
        t   = time.perf_counter()
        res = minimize(solve_system, x0=V0true, args=(s, d, Cd_inv),
                       method='Nelder-Mead',
                       bounds=((0, np.inf), (0, np.inf), (0, np.inf),
                               (np.pi/2, np.pi/2), (Pa0, Pa0), (n0, n0)))
        print("Solution found using Nelder-Mead in %.3f s" % (
            time.perf_counter() - t))

    if plots:
        dp = partial(do_plots, T0=T0, R0=R0, objFn=objFn)
        dp(dimensionality)
