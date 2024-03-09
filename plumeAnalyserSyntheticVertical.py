#!/usr/bin/env python
# coding: utf-8

from bentPlumeAnalyser import *
from fumarolePlumeModel import *
from scipy.io.matlab import loadmat
from scipy.integrate import ode, solve_ivp
from itertools import product

import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, json


# Set numpy options, notably for the printing of floating point numbers
np.set_printoptions(precision=6)

# Set matplotlib options
mpl.rcParams['figure.dpi'] = 300

#get_ipython().run_line_magic('reload_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


def integrator(V0, p, x=None):
    if x is None:
        x   = np.linspace(0, 25, 21)
    sol = solve_ivp(derivs, [x[0], x[-1]], V0, args=(p,), t_eval=x)
    return sol.t, sol.y.T


if __name__ == "__main__":
    run = 3

    # Import table of experimental conditions
    GCTA = pandas.read_excel('./data/ExpPlumes_for_Dai/TableA1.xlsx',
                             sheet_name='CGSdata', skiprows=2, 
                             names=('exptNo', 'rhoa0', 'sig_rhoa0',
                                    'N', 'sig_N', 'rho0', 'sig_rho0',
                                    'U0', 'sig_U0', 'W', 'sig_W', 
                                    'gp', 'sig_gp',
                                    'Q0','sig_Q0', 'M0', 'sig_M0',
                                    'F0', 'sig_F0', 'Ri_0', 'sig_Ri_o',
                                    'W*','sig_W*'))

    # Extract densities of ambient and plume, and calculate g' at the source
    expt  = GCTA[GCTA['exptNo'] == run]
    rhoa0 = expt['rhoa0']
    rho0  = expt['rho0']
    g = 981 #cm/s²
    gp0   = (rhoa0 - rho0) / rhoa0 * g

    parameters = pandas.read_excel('./data/ExpPlumes_for_Dai/TableA1.xlsx',
                                   sheet_name='CGSparameters')
    b0theoretical = parameters[parameters['property'] == 'nozzleSize']\
        ['value'].values[0]
    u0theoretical = expt['U0'].values[0]


    # ### Create a synthetic dataset for a vertical plume

    exptNo      = 3
    plotResults = True

    V0, p = loadICsParameters(pathname, exptNo, alpha=0.09, beta=0, m=1)

    p = list(p)
    p[1], p[4] = 0., 0.
    p = tuple(p)

    t1 = 30.
    dt = .1
    sexp = np.arange(0., t1 + dt, dt)
    dsexp = np.diff(sexp)


    ## Set initial conditions
    Q0, M0, F0, theta0 = V0
    s, V = integrator(V0, p)

    sexp = np.copy(sexp)

    Q, M, F, theta = [V[:,i] for i in range(4)]
    b, u, gp = Q / np.sqrt(M), M / Q, F / Q

    V2    = np.array([b, u, gp]).T
    sexp  = np.copy(s)
    dsexp = np.diff(sexp)

    # ### Add noise to signal
    V3 = V2.copy()
    for i in range(3):
        noise = np.random.normal(0, .3, len(sexp))
        V3[:,i] = V2[:,i] + noise

    bexp, uexp, gpexp = [V3[:,i] for i in range(3)]
    gp0 = V0[2] / V0[0]


    #p = (0.05, .5, .012, 2., 4.)
    nGrid = 201   # Number of grid points
    b0Vec = np.linspace(.02, 2, nGrid) #cm
    u0Vec = np.linspace(2, 20, nGrid) #cm/s
    Q0Vec = u0Vec * b0Vec**2 #cm3/s
    M0Vec = Q0Vec * u0Vec #cm4/s2

    theta0 = np.pi / 2

    objFn, initialConds = [], []

    sequence = [Q0Vec, M0Vec]

    for (Q0, M0) in list(product(*sequence)):
        F0 = Q0 * gp0
        V0 = [Q0, M0, F0, theta0]
    
        # Call the 'integrator' function (defined above) to solve
        # the model
        s, V = integrator(V0, p, x=sexp)
        Q, M, F, theta = [V[:,i] for i in range(4)]
    
        #######################################
    
        b  = Q / np.sqrt(M) # Factor of sqrt{2} to correspond with top-hat model
        u  = M / Q
        gp = F / Q
    
        Vexp = np.array([bexp, gpexp]).T[:len(b)].T.ravel()
        Vsyn = np.array([b, gp]).ravel()
        
        objFn.append(objectiveFn(Vexp[:-1], Vsyn[:-1], p=p))
        initialConds.append(V0)

    # Transform initialConds and objFn from lists to arrays, 
    # reshaping the latter 
    initialConds = np.array(initialConds)
    objFn = np.array(objFn).reshape((nGrid, nGrid))
    

    np.save("objFn.npy", objFn)
    np.save("initialConds.npy", initialConds)
