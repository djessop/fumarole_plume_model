# -*- coding: utf-8 -*-
"""
@filename: model_synthdata_inversion.py
@authors: avklein, dejessop
@created: Fri Nov 24 09:56:16 2023
"""

from itertools import product
from myiapws import iapws1992, iapws1995
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
#from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from joblib import Parallel, delayed
from functools import partial

import numpy as np
import time
import warnings
#import pandas as pd
#import plotly.graph_objects as go


Tt = iapws1995.Tt  # Triple point of water
cm = 1/2.54


def derivs(s, V, *args):
    #Ca 1001 J/kg/K, Cp water vapour at 95°C, 1880 J/kg/K:
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
    ks, kw, g, Ca, Cp0, Ta0, Ra, Rp0, Pa0, n0, wind, W1, H1, mode = args 
    Ta   = Ta0
    W    = wind_profile(s, V, *args)
    rhoa = density_atm(s, V, *args)
    rho  = density_fume(s, V, *args)
    Ue   = entrainment_vel(s, V, *args)   # entrainment velocity
    Q, M, E, th, Pa, n, x, z = V
    
    ## Definition of derivatives, as per Woodhouse et al., 2013 (JGR)
    dQ  = 2 * rhoa * Ue * Q / (np.sqrt(rho * M))

    dM  = g * (rhoa - rho) * Q**2 / (rho * M) * np.sin(th) \
        + 2 * rhoa * Q / (np.sqrt(rho * M)) * Ue * W * np.cos(th)
    
    dth = g * (rhoa - rho) * Q**2 / (rho * M**2) * np.cos(th) \
        - 2 * rhoa * Q / (M * np.sqrt(rho * M)) * Ue * W * np.sin(th)

    dE  = (Ca * Ta + (Ue**2) / 2)* dQ + M**2 / (2 * Q**2) * dQ \
        - rhoa/rho * Q * g * np.sin(th) \
        - 2 * rhoa * np.sqrt(M / rho) * Ue * W * np.cos(th)

    dPa = -g * Pa/ (Ra * Ta) * np.sin(th)

    dn  = (1 - n) / Q * dQ  # mass fraction of not steam

    dx  = np.cos(th)
    dz  = np.sin(th)

    return np.array([dQ, dM, dE, dth, dPa, dn, dx, dz])


def entrainment_vel(s, V, *args): # ks=0.09, kw=0.5, g=9.81, Ca=1006, Cp0=1885,
                    # Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05,
                    # wind=True, W1=5, H1=5, mode='leastsq'):
    ks, kw, g, Ca, Cp0, Ta0, Ra, Rp0, Pa0, n0, wind, W1, H1, mode = args
    # print(ks, kw)
    W  = wind_profile(s, V, *args)
    Q, M, _, th, *_ = V  # Don't need E, Pa, n
    return ks * np.abs(M / Q - W * np.cos(th)) + kw * np.abs(W * np.sin(th))

    
def wind_profile(s, V, *args): 
    """
    Define atmospheric properties:
    wind speed: wind_profile(s,V): constant wind shear between ground and 
    height H1, then constant speed W1
    air density: density_atm depending on spec. gas constant of air Ra and air 
    temperature Ta (assumed constant at about 18 deg C)
    """
    ks, kw, g, Ca, Cp0, Ta0, Ra, Rp0, Pa0, n0, wind, W1, H1, mode = args
    if not wind:
        return np.zeros_like(s)
    # average wind speed at Sanner ~38 km/h -> 10.556 m/s
    # constant wind shear, 0 m/s at ground up to W1    
    z = V[-1]
    # z = s * np.sin(theta)  # dz/dx = sin(theta) -/-> z = s * sin(theta) !

    return np.where(z < H1, W1 * z / H1, W1) 


def density_atm(s, V, *args):
    ks, kw, g, Ca, Cp0, Ta0, Ra, Rp0, Pa0, n0, wind, W1, H1, mode = args
    Ta = Ta0
    Pa = V[4]
    return Pa / (Ra * Ta)
    

def density_fume(s, V, *args):
    '''Returns the density of the fumarole, based on ideal gas
    '''
    ks, kw, g, Ca, Cp0, Ta0, Ra, Rp0, Pa0, n0, wind, W1, H1, mode = args
    T  = temperature_fume(s, V, *args)
    Pa = V[4]  # atmospheric pressure
    n  = V[5]
    Rp = bulk_gas_constant(s, V, *args)
    return Pa / (Rp * T) 


def heat_capacity(s, V, *args):
    """
    Define specific heat capacity, Cp, of plume, as a function of the dry air 
      mass fraction, n, for initial 95% of vapour 
    Ca:  spec. heat capacity of air
    Cp0: spec. heat capacity of vapour
    """  
    ks, kw, g, Ca, Cp0, Ta0, Ra, Rp0, Pa0, n0, wind, W1, H1, mode = args
    return Ca + (Cp0 - Ca) * (1 - V[5]) / (1 - n0)


def bulk_gas_constant(s, V, *args):
    """
    Define bulk gas constant, Rp, of the plume, as a function of the dry air 
    mass fraction in the plume, n.
    Ra:  gas constant of dry air
    Rp0: gas constant of vapour 
    """
    ks, kw, g, Ca, Cp0, Ta0, Ra, Rp0, Pa0, n0, wind, W1, H1, mode = args
    n = V[5]  # dry air mass fraction
    return Ra + (Rp0 - Ra) * (n0 * (1 - n)) / (n * (1 - n0))  


def temperature_fume(s, V, *args):
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
    ks, kw, g, Ca, Cp0, Ta0, Ra, Rp0, Pa0, n0, wind, W1, H1, mode = args
    Q, _, E, *_ = V
    Cp = heat_capacity(s, V, *args)
    return E / (Q * Cp)  


def produce_Gm(s, sol, *args):
    """Returns the model predictions in a format suitable for comparison with 
    data"""
    rho  = density_fume(sol.t, sol.y, *args)
    T    = temperature_fume(sol.t, sol.y, *args) - Tt
    Cp   = heat_capacity(sol.t, sol.y, *args)
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
        mode = 'lstsq'  
        warnings.warn('Unknown mode: should be either "abs" or "leastsq.  ' +
                      'Defaulting to leastsq"')
    
    
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


def parallel_job(U0, R0, T0, W1, s, d, Cd_inv, *args): 
    from scipy.integrate import solve_ivp

    # errors/Cd_inv are not passed explicitly as arguements
    warnings.filterwarnings('ignore')
    args = list(args)
    args[-2] = W1
    _, _, _, _, Cp0, _, _, _, Pa0, n0, wind, _, H1, mode = args
    V0   = [1, 1, Cp0 * T0, 1, Pa0, n0, 0, 0]  # required for routines
    rho0 = density_fume(0, V0, *args)
    Q0   = rho0 * np.pi * R0**2 * U0
    M0   = Q0 * U0
    E0   = Q0 * Cp0 * T0
    V0   = [Q0, M0, E0, np.pi/2, Pa0, n0, 0, 0]
    
    ## Extract mode from args if it has been included
    sol  = solve_ivp(derivs, [s[0], s[-1]], V0, t_eval=s, args=args)
    Gm   = produce_Gm(s, sol, *args)

    return objective_fn(Gm, d, Cd_inv, mode=mode), V0, W1


def solve_system(x0, s, data, errors, *args):
    """Helper function for the minimisation problem using 
    scipy.optimize.minimize
    """
    _, _, _, _, Cp0, _, _, _, Pa0, n0, wind, _, H1, mode = args
    args = list(args)
    x0p  = x0[:-1]
    wind = x0[-1]
    args[-2] = wind

    R0, U0, T0, _ = x0
    V0   = [1, 1, Cp0 * T0, 1, Pa0, n0, 0, 0]  # required for routines
    rho0 = density_fume(0, V0, *args)
    Q0   = rho0 * np.pi * R0**2 * U0
    M0   = Q0 * U0
    E0   = Q0 * Cp0 * T0
    V0   = [Q0, M0, E0, np.pi/2, Pa0, n0, 0, 0]

    sol  = solve_ivp(derivs, [s[0], s[-1]], V0, t_eval=s, args=args)
    Gm   = produce_Gm(s, sol, *args)

    ## Extract mode from args if it has been included
    mode = 'lstsq'
    if type(args[-1]) is str:
        mode = args[-1]

    return objective_fn(Gm, data, errors, mode=mode)
    

def do_plots(ndims, T0, R0, objFn, exponentiate=False):
    ## Helper functions for plotting in plotly
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


    fig0, ax0 = plt.subplots(figsize=(10*cm, 10*cm))
    ax0.plot(solp.T, s, '-')
    ax0.set_xlabel('Plume parameters')
    ax0.set_ylabel(r'Distance along plume axis, $s$')
    ax0.legend((r'$\theta$', r'$b$', r'$T$'))

    plt.gca().set_prop_cycle(None)  # reset colour cycle
    ax0.plot(sol_noise.T, s, '.')

    fig0.tight_layout()
    
    S = objective_fn(Gm, d, Cd_inv, 'leastsq', False)
        
    fig, ax = plt.subplots(figsize=(10*cm, 10*cm))
    if ndims == 2:
        Ropt_ind, Topt_ind = np.where(objFn == objFn.min())
        im = ax.pcolor(T0, R0, np.exp(-objFn), norm=LogNorm(),
                       label='misfit function')
    if ndims == 3:
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
    """
    To run a calculation, do
    python model_synthdata_inversion.py ncore npts inversion print wind

    parameters
    ----------
    ncore : int
       Number of cores/processors to run on
    ngrid : int
       Number of grid points per parameter (all parameters have equal number 
       of grid points)
    ndims : int
       Number of parameter dimensions for inversion
    inversion : bool
       Do invsion
    plots : bool
       Show plots after calculations have run
    wind  : bool
       Include wind in simulations
    """
    from pathlib import Path
    
    import sys

    home_str = str(Path.home())

    """
    Solve differential equations for a set of initial values Q0, M0, theta0, 
    E0, Pa0, n0
    """
    
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
    wind        = False
    # Parameter dimensions in which to find solution
    # 2=(R0, T0), 3=(U0, R0, T0), 4=(W1, U0, R0, T0)
    if len(sys.argv) >= 4:
        ndims   = int(sys.argv[3])
    if len(sys.argv) >= 5:
        if sys.argv[4] == 'True':
            inversion = True
    if len(sys.argv) >= 6:
        if sys.argv[5] == 'True':
            plots = True
    if len(sys.argv) == 7:
        if sys.argv[6] == 'True':
            wind = True

    ##  fixed parameters
    nsol   =    51    # number of points at which "observations" will be made
    Cp0    =  1885    # Heat capacity of plume contents
    Pa0    = 86000    # Atmospheric pressure at vent altitude
    theta0 = np.pi/2  # Initial plume angle
    n0     =     0.05 # (Dry) gas content of plume
    Ta0    = 18 + Tt  # Air temperature at vent altitude
    s      = np.linspace(0, 15, nsol)  # Follow plume up to 15 m

    ##  variables
    rho0true = .5
    R0true = .5
    T0true = Tt + 96
    U0true = 10
    Q0true = rho0true * U0true * R0true**2 * np.pi
    M0true = Q0true * U0true
    E0true = Q0true * Cp0 * T0true
    V0true = [Q0true, M0true, E0true, theta0, Pa0, n0, 0, 0]
    # Wind profile - simple shear to z=H1, constant wind beyond
    H1     = 5  # Height for end of shear profile
    W1 = np.linspace(0, 10, ngrid)             # Wind speed at H1 (= 5 )m
    args = (.09, .5, 9.81, 1006, 1885, 291, 287, 462, 86000, 0.05,
                wind, W1[0], H1, 'lstsq')  # 'abs'
    if ndims < 4:
        W1   = 5
        args = (.09, .5, 9.81, 1006, 1885, 291, 287, 462, 86000, 0.05,
                wind, W1, H1, 'lstsq')  # 'abs'
    

    # ks, kw, g, Ca, Cp0, Ta0, Ra, Rp0, Pa0, n0, wind, W1, H1, mode

    R0 = np.linspace(0.1, 1, ngrid)             # Vent radius
    T0 = np.linspace(50 + Tt, 150 + Tt, ngrid)  # Vent temperature
    U0 = np.linspace(5, 15, ngrid)              # Vent speed

    sol_true = solve_ivp(derivs, [s[0], s[-1]], V0true, t_eval=s, args=args)

    # Produce "true" data values
    sol   = sol_true
    rho   = density_fume(sol.t, sol.y, *args)
    T     = temperature_fume(sol.t, sol.y, *args) #- Tt
    Cp    = heat_capacity(sol.t, sol.y, *args)
    b     = sol.y[0] / np.sqrt(rho * np.pi * sol.y[1])
    u     = sol.y[1] / sol.y[0]
    theta = sol.y[3]

    noise_level = 1
    sigtheta, sigb, sigT = (np.pi / 10 * noise_level,   # rad
                            .05 * noise_level,          # m
                            1 * noise_level)            # K
    Cd_inv = np.diag(np.array([1 / sigtheta**2 * np.ones_like(theta),
                               1 / sigb**2 * np.ones_like(b),
                               1 / sigT**2 * np.ones_like(T)]).ravel())

    solp  = np.array([theta, b, T])
    noise = np.random.randn(*solp.shape)  # Gaussian noise, _N_(0,1)
    sol_noise = (solp.T + noise.T * (sigtheta, sigb, sigT)).T
    d    = sol_noise.flatten()  # array of data
    Gm   = produce_Gm(s, sol, *args)
    Gm_d = Gm - d
    
    if inversion:
        """
        Run jobs in parallel in order to calculate objective function at
        each combination of possible source values.
        """
        # "wrapper" partial function to simplify notation below
        pj = partial(parallel_job, s=s, d=d, Cd_inv=Cd_inv, args=args)
        t  = time.perf_counter()

        if ndims == 2:
            u0 = U0true
            w1 = W1
            sequence = [R0, T0]
            results  = Parallel(n_jobs=ncore)(delayed(parallel_job)(
                 u0, r0, t0, w1, s, d, Cd_inv, *args) for (
                     r0, t0) in list(product(*sequence)))

            tstop = time.perf_counter()

            ## Deal out the results
            objFn, initialConds, winds = [], [], []

            for result in results:
                objFn.append(result[0])
                initialConds.append(result[1])
                winds.append(result[2])

            initialConds = np.array(initialConds)
            objFn = np.array(objFn).reshape((-1, ngrid))

        if ndims == 3:
            sequence = [U0, R0, T0]
            w1       = W1
            results  = Parallel(n_jobs=ncore)(delayed(parallel_job)(
                 u0, r0, t0, w1, s, d, Cd_inv, *args) for (
                     u0, r0, t0) in list(product(*sequence)))

            tstop    = time.perf_counter()

            ## Deal out the results
            objFn, initialConds, winds = [], [], []

            for result in results:
                objFn.append(result[0])
                initialConds.append(result[1])
                winds.append(result[2])

            initialConds = np.array(initialConds)
            objFn = np.array(objFn).reshape((-1, ngrid, ngrid))

        if ndims == 4:
            u0 = U0true

            sequence = [W1, U0, R0, T0]
            results  = Parallel(n_jobs=ncore)(delayed(parallel_job)(
                 u0, r0, t0, w1, s, d, Cd_inv, *args) for (
                     w1, u0, r0, t0) in list(product(*sequence)))

            tstop = time.perf_counter()

            ## Deal out the results
            objFn, initialConds, winds = [], [], []

            for result in results:
                objFn.append(result[0])
                initialConds.append(result[1])
                winds.append(result[2])

            initialConds = np.array(initialConds)
            objFn = np.array(objFn).reshape((ngrid, ngrid, ngrid, ngrid))


        print("%dD job ran in %.3f s using %2d processors" % (
            ndims, tstop - t, ncore))


        ## Save variables for later
        save_str = home_str + f'/Modelling/fumarolePlumeModel/' + \
            f'objFn_soln_{ndims}D_{ngrid:04d}pts_{ncore:03d}cores'
        if wind:
            save_str += '_wind'
        np.savez(save_str, W1=W1, R0=R0, T0=T0, U0=U0, objFn=objFn, solp=solp)
        ## To retrieve variables, use np.load which will parse them as a
        ## dict-like object, i.e.
        ## container = np.load(filename)
        ## R0, T0, U0, objFn = container.values()

    if nelder_mead:
        x0  = V0true  # [1, 1, 1, np.pi/2 , 86000, 0.05]
        t   = time.perf_counter()
        res = minimize(solve_system, x0=V0true, args=(s, d, Cd_inv),
                       method='Nelder-Mead',
                       bounds=((0, None), (0, None), (0, None),
                               (0, None)))
        print("Solution found using Nelder-Mead in %.3f s" % (
            time.perf_counter() - t))

    if plots:
        dp = partial(do_plots, T0=T0, R0=R0, objFn=objFn)
        dp(ndims)
