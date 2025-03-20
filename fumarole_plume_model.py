#!/usr/bin/env python3
# coding=utf-8

""" 
fumarole_plume_model.py

Provides solutions to the model of Aubry et al (2017a)* which models the 
variation of volume flux, q, momentum flux, m, buoyancy flux, f, and plume 
angle, :math: `\\theta' which describe the rise of a buoyant plume subject to a side 
wind. 

Provided functions:
-------------------
- derivs
    Description of the forward model.
- wind
    Wind at altitude, z.
- objectiveFn
    Misfit function between the synthetic and "experimental" data.

Model description:
------------------
dx/ds = \cos(\\theta), dz/ds = \sin(\\theta),                    (4)
d(\\rho u r**2)/ds = 2 \\rho_a r u_e,                            (5)
d(\\rho u**2 r**2)/ds = (\\rho_a - \\rho) g r**2 \\sin(\\theta) 
    + w\\cos(\\theta) d(\\rho u r**2)/ds                          (6)
(\\rho u**2 r**2) d\\theta/ds = (\\rho_a - \\rho) g r**2 \\cos(\\theta) 
    - w\\sin(\\theta) d(\\rho u r**2)/ds                          (7)
d(g'ur**2)/ds = -N**2 u r**2 \\sin(\\theta),                     (8)

with
    g'  = g (\\rho_a - \\rho) / \\rho_a,
    u_e = \\alpha_e |u - w\\sin\\theta| + \\beta_e |w\\cos\\theta|
    N   = \\sqrt{-g/\\rho_0 d\\rho/dz}

Note that \\theta is measured relative to the horizontal plane.  

Applying the Boussinesq approximation (density variations are small enough to 
be negligible, except when a density term is multiplied by gravity) and upon 
making the following substitutions:
Q = u r**2, M = u**2 r**2, F = g' u r**2,
we obtain
dQ/ds = 2 Q/\\sqrt{M} (\\alpha_e\\abs{M/Q - w\\cos\\theta} 
	+ \beta_e\\abs{w\\sin\\theta})
dM/ds = FQ/M\\sin\\theta + w\\cos\\theta dQ/ds
dF/ds = -N**2 \\sqrt{M} \\sin\\theta
d\theta/ds = FQ/M**2\\cos\\theta - w/M\\sin\\theta dQ/ds

References:
-----------
* Aubry, T. J., Carazzo, G., & Jellinek, A. M. (2017a). 
Turbulent entrainment into volcanic plumes: new constraints from laboratory
experiments on buoyant jets rising in a stratified crossflow. 
Geophys. Res. Lett., 44, 10,198--10,207.
https://dx.doi.org/10.1002/2017GL075069

see also:
Aubry, T. J., Jellinek, A. M., Carazzo, G., Gallo, R., Hatcher, K., 
Dunning, J. (2017b)
A new analytical scaling for turbulent wind-bent plumes: comparison of scaling 
laws with analog experiments and a new database of eruptive conditions for 
predicting the height of volcanic plumes
J. Volcanol. Geotherm. Res., 343, 233--251
http://dx.doi.org/10.1016/j.jvolgeores.2017.07.006

Woodhouse, M. J., A. J. Hogg, J. C. Phillips, and R. S. J. Sparks (2013)
Interaction between volcanic plumes and wind during the 2010 Eyjafjallaj\"okull
eruption, Iceland
J. Geophys. Res. Solid Earth, 118, 92--109
https://dx.doi.org/10.1029/2012JB009592

changes log:
------------
2018-06-25      Clipping model range to be equal to the experimental data
2018-06-26      Introduced command line option to plot data.  
                Introduced "main()" function to contain the bod
2019-05-09      Rewriting objective function to allow scipy.optimize.fmin to 
                call it
                
to do:
------
- Calculate experimental plume widths etc. in local reference (i.e. rotated)
  reference frame.
- Interpolate model data to be evaluated at same points as the experimental 
  data.
- Compare experimental and model data.
- Something odd happening in models of most expts: predicted plume trajectory
  changes direction quite abruptly.  Derivitives (theta) change sign here.
  Stop the calculation at this point?xs
"""

from scipy.integrate import ode, solve_ivp
from scipy.interpolate import interp1d
from scipy.io.matlab import loadmat
from bentPlumeAnalyser import (plume_trajectory,
                               dist_along_path,
                               plume_angle,
                               true_location_width)

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pandas

# Set matplotlib font to be "computer modern" and use TeX rendering
font = {'family' : 'serif',
        'serif': ['computer modern roman']}
#        'fontsize': 16}
# plt.rc('font', **font)
# plt.rc('text', usetex=True)

eps         = 1e-3
scaleFactor = 38
pathname    = '/home/david/Modelling/fumarolePlumeModel/data/'

# PHYSICAL CONSTANTS
g = 981                     # CGS

    
def derivs(s, V, p=(.09, .6, .1, 1., None)):
    """ 
    Description of the forward model.  Returns the derivatives of the state 
    vector, V, for the Aubry et al., 2017 (GRL) wind-bent plume model.  
    V contains the following state variables:
    Q           volume flux (specific mass flux) = b**2 u
    M           specific momentum flux = b**2 u**2
    F           specific buoyancy flux = b**2 u g' 
    :math: `\\theta'      local deflection of the plume axis with respect to 
        vertical
    
    Parameters
    ----------
    s	distance along the plume axis
    V   state variable consisting of q, m, f and :math: `\\theta'
    p   tuple containing model parameters:
        alpha   perpendicular entrainment coefficient (default = 0.09)
        beta    wind entrianment coefficient (0.6)
        N       Brunt-Väisälä frequency for the stratified environment (0.1)
        m       Devenish's coefficient, used in calculating u_e (1.0)
        w       "wind" option.  If None, the function wind will be called.
        
        Default parameters are as defined in GCs email to me

    Returns
    -------
    dVds	an array containing the derivatives

    Changes log
    -----------
    2018-06-25  swapped np.sqrt(m) term for q in df/ds (see Aubry et al.)
	
    """
    Q, M, F, theta = V
    b, u, gp = Q / np.sqrt(M), M / Q, F / Q
    alpha, beta, N, m, w = p

    # Get the wind speed at the current altitude
    if w is None:
        w = wind(s, V)

    # calculate the entrainment velocity
    u_e = ((alpha * abs(u - w * np.cos(theta))**m)
		   + (beta * abs(w * np.sin(theta)))**m) ** (1./m)

    # Define the derivatives
    dQds     =  2. * b * u_e
    dMds     =  F * Q / M * np.sin(theta) + w * np.cos(theta) * dQds
    dFds     = -N**2 * Q * np.sin(theta) 
    dthetads =  F * Q / M**2 * np.cos(theta) - w / M * np.sin(theta) * dQds

    dVds    = np.zeros(4)
    dVds[0] = dQds
    dVds[1] = dMds
    dVds[2] = dFds
    dVds[3] = dthetads

    return dVds


def integrator(p, s0, V0, derivs=derivs):
    """ 
    Return the solution (s, V) of derivs.  Wrapper for the numerical 
    integration of the model defined by derivs, using the 
    scipy.integrate.ode module.
    """
    # Initialise an integrator object
    r = ode(derivs).set_integrator('lsoda', nsteps=1e6)
    r.set_initial_value(V0, 0.)
    r.set_f_params(p)
    
    # Define state vector and axial distance
    V = []    # State vector
    s = []    # Axial distance
    V.append(V0)
    s.append(s0)
    
    # Define the individual variables - these will be calculated at run time
    Q, M, F, theta = V0  # 0., 0., 0., 0.
    Q = np.float64(Q)
    M = np.float64(M)
    F = np.float64(F)
    theta = np.float64(theta)
    
    ####################################

    # Integrate, whilst successful, until the domain size is reached
    ind = 0
    while r.successful() and r.t < t1 and M >= 0.:
        dt = dsexp[ind]
        r.integrate(r.t + dt)
        V.append(r.y)
        s.append(r.t)
        Q, M, F, theta = r.y
        ind += 1
    s = np.array(s)
    V = np.float64(np.array(V))
    return s, V


def integrator2(V0, p, x=None):
    if x is None:
        x   = np.linspace(0, 25, 21)
    sol = solve_ivp(derivs, [x[0], x[-1]], V0, args=(p,), t_eval=x)
    return sol.t, sol.y.T


def wind(s, V):
    """ 
    Functions that define the wind at altitude, z.
    
    Currently a constant value is returned
    """

    theta = V[3]

    return .01


def objectiveFn(Vexp, Vsyn, cov=None, p=(.09, .6, .1, 1., None),
                mode='lsq'):
    """
    Returns the objective (misfit/cost) function as the either (weighted) 
    sum of square differences between the synthetic and "experimental" data, 
    or the (weighted) absolute difference.
    
    Parameters
    ----------
    Vsyn : list or array_like
        state vector of sythetic (model) data
    Vexp : list or array_like	
        state vector of experimental (or natural) data
    cov : array_like (optional)
        covariance matrix of the experimental data for weighting.  
    p : tuple or array_like
        vector of state variables
    mode : str (optional)
        switch for choosing the form of the objective function.  Choices 
        are 'least-squares' (lsq) or 'absolute differences' (abs).  Default 
        is 'lsq'.

    TO DO
    -----
    Check for dimensional coherence - "enforce" statement at beginning 
    of code?
    
    """
    # If no stdr /  dev is provided for the experimental data,
    # set unit weights.
    if cov is None:
        sigma = np.ones_like(Vexp)
        cov   = np.eye(len(Vexp))
    else:
        sigma = np.sqrt(np.diag(cov))

    Vsyn = Vsyn.ravel()
    Vexp = Vexp.ravel()
    
    # Do some checking on the size of inputs
    if any(len(row) != len(cov) for row in cov):
        raise TypeError('Covariance matrix must be square. ')
    if (len(Vexp) != len(Vsyn)):
        raise TypeError('Model and data vectors must be of equal length')


    # Check that the synthetic and experimental data are of the same dimension
    # then define a residual (difference).
    if len(Vsyn) == len(Vexp):
        r = Vsyn - Vexp
    else:
        raise Warning('input vectors must be of the same dimension')
	
    if mode == 'lsq':
        # Note that the "@" syntax is not recognised for python < 3.5
        # In this case use r.T.dot(invSig).dot(r)
        objFn = .5 * r.T @ np.linalg.inv(cov) @ r  # Quadratic form
    elif mode == 'abs':
        objFn = (np.abs(r) / sigma).sum()
    else:
        raise Warning('Covariance matrix of unknown dimensions')
	#print(objFn)

    return np.exp(-objFn)


def objectiveFn2(V0, derivs, p, sexp, dexp, sig_dexp=None, mode='lsq'):
    """
    Return the objective (misfit/cost) function as the either sum of 
    square differences between the synthetic and "experimental" data, 
    or the absolute difference.  If the vector of standard errors on 
    the data, 'sig_dexp', is supplied, the objective function will be
    built from weights derived from these.
    
    Parameters
    ----------
    V0 : list or array_like
        Current "guess" of the source conditions.
    derivs : callable
        function that describes the model.
    p : tuple or array_like
        model parameters.
    sexp : list or array_like
        independent variable (sexp) of experimental (or natural) data.
    dexp : list or array_like
        state vector of experimental (or natural) data consisting of width 
        (bexp) and angle (thetaexp).
    sig_dexp : list or array_like (optional)
        covariance matrix of the experimental data for weighting.  
    mode : str (optional)
        switch for choosing the form of the objective function.  Choices 
        are 'least-squares' (lsq) or 'absolute differences' (abs).  Default 
        is 'lsq'.

    Notes
    -----
    1)  The covariance matrix of data is assumed to be diagonal though there 
    will be a strong correlation between several variables, for example width 
    and plume angle.  Non (main) diagonal covariances can be dealt with by 
    rewritting the code to allow for the covariance matrix to be explicitly 
    passed to the function.
    2)  The synthetic data is flattened ('ravelled') according to 'C' 
    (row-major) format.  Thus care should be taken to flatten the natural data 
    using the same format.

    TO DO
    -----
    - Check for dimensional coherence
    - Explicitly pass the covariance matrix

    """
    # Initialise integrator object, set intial conditions and model params.
    r = ode(derivs).set_integrator('lsoda', nsteps=1e6)
    r.set_initial_value(V0, 0).set_f_params(p)

    # Domain of integration and integration step. 
    # No need to go further than extent of the experimental plume
    t1 = sexp.max()             
    dt = .1
    dsexp = np.diff(sexp)
    s, Q, M, F, theta = [], [], [], [], []
    s.append(0.)
    Q.append(V0[0])
    M.append(V0[1])
    F.append(V0[2])
    theta.append(V0[3])

    # Solve the model for the current initial conditions
    ind = 0
    while r.successful() and r.t < t1:
        dt = dsexp[ind]         # to get model output at expt. points
        r.integrate(r.t + dt)
        s.append(r.t)
        Q_, M_, F_, theta_ = r.y
        Q.append(Q_)
        M.append(M_)
        F.append(F_)
        theta.append(theta_)
        ind += 1
    Q, M, F, theta = np.array(Q), np.array(M), np.array(F), np.array(theta)

    # Convert plume flux parameters into basic params
    b, u, gp = Q / np.sqrt(M), M / Q, F / Q

    d   = np.array([theta]).ravel(order='C')
    res = dexp - d

    # Kernel definition depends on mode
    if mode == 'lsq':
        if sig_dexp is None:
            kernel = .5 * res.dot(res)
        else:
            # Build and invert covariance matrix
            invCd = np.linalg.inv(np.diag(sig_dexp**2))  # Cov. is std**2
            kernel = .5 * res @ invCd @ res  # Requires python >= 3.5
    else:
        if sig_dexp is None:
            kernel = np.abs(res).sum()
        else:
            kernel = np.abs(res / sig_dexp).sum()

    return kernel


def objectiveFn3(V0, derivs, p, sexp, dexp, sig_dexp=None, mode='lsq'):
    """
    Return the objective (misfit/cost) function as the either sum of 
    square differences between the synthetic and "experimental" data, 
    or the absolute difference.  If the vector of standard errors on 
    the data, 'sig_dexp', is supplied, the objective function will be
    built from weights derived from these.
    
    Parameters
    ----------
    V0 : list or array_like
        Current "guess" of the source conditions.
    derivs : callable
        function that describes the model.
    p : tuple or array_like
        model parameters.
    sexp : list or array_like
        independent variable (sexp) of experimental (or natural) data.
    dexp : array_like
        state vector of experimental (or natural) data consisting of width 
        (bexp) and reduced gravity (gpexp).  Should be an array of size 
        nobs-by-2 where nobs is the number of observations.
    sig_dexp : list or array_like (optional)
        covariance matrix of the experimental data for weighting.  
    mode : str (optional)
        switch for choosing the form of the objective function.  Choices 
        are 'least-squares' (lsq) or 'absolute differences' (abs).  Default 
        is 'lsq'.

    Notes
    -----
    1)  The covariance matrix of data is assumed to be diagonal though there 
    will be a strong correlation between several variables, for example width 
    and plume angle.  Non (main) diagonal covariances can be dealt with by 
    rewritting the code to allow for the covariance matrix to be explicitly 
    passed to the function.
    2)  The synthetic data is flattened ('ravelled') according to 'C' 
    (row-major) format.  Thus care should be taken to flatten the natural data 
    using the same format.

    TO DO
    -----
    - Check for dimensional coherence
    - Explicitly pass the covariance matrix

    """
    from scipy.integrate import solve_ivp
    
    sol = solve_ivp(derivs, [sexp[0], sexp[-1]], V0, args=(p,), t_eval=sexp)
    s, (Q, M, F, theta) = sol.t, sol.y

    # Convert plume flux parameters into basic params
    b, u, gp = Q / np.sqrt(M), M / Q, F / Q

    d   = np.array([b, gp]).T
    if len(dexp) >= len(d):
        res = (dexp[:len(d)] - d).ravel()
    else:
        res = (dexp - d[:len(dexp)]).ravel()

    # Kernel definition depends on mode
    if mode == 'lsq':
        if sig_dexp is None:
            kernel = .5 * res.dot(res)
        else:
            # Build and invert covariance matrix
            invCd = np.linalg.inv(np.diag(sig_dexp**2))  # Cov. is std**2
            kernel = .5 * res @ invCd @ res  # Requires python >= 3.5
    else:
        if sig_dexp is None:
            kernel = np.abs(res).sum()
        else:
            kernel = np.abs(res / sig_dexp).sum()

    return -np.exp(-kernel)


def loadICsParameters(pathname, run, alpha=.09, beta=.6, m=1.5):
    """ 
    Returns initial conditions and model parameters determined from CGTA's 
    experimental data
    """
    filename = pathname + 'ExpPlumes_for_Dai/TableA1.xlsx'
    # Load all the data (in CGS units)
    df = pandas.read_excel(filename, sheet_name='CGSdata', skiprows=1, 
                           names=('run','rhoa','rhoa_2sig','N','N_2sig', 
                                  'rho0', 'rho0_2sig', 'U0', 'U0_2sig',
                                  'W', 'W_2sig', 'gp', 'gp_2sig', 
                                  'Q0', 'Q0_2sig', 'M0', 'M0_2sig',
                                  'F0', 'F0_2sig', 'Ri0', 'Ri0_2sig', 
                                  'Wstar', 'Wstar_2sig'))
    params = pandas.read_excel(filename, sheet_name='CGSparameters')

    # Define source conditions
    expt  = df.loc[df['run'] == run]
    rhoa0 = expt['rhoa'].values[0]
    rho0  = expt['rho0'].values[0]
    N     = expt['N'].values[0]
    u0    = expt['U0'].values[0]
    W     = expt['W'].values[0]
    r0    = params[params['property'] == 'nozzleSize'].value.values[0]
    
    # Define function parameters to pass to derivs.  
    # Tuple p contains alpha, beta, N, m, w
    p = (alpha, beta, N, m, W)

    gp0 = -(rhoa0 - rho0) / rhoa0 * g
        
    Q0 = r0**2 * u0 
    M0 = r0**2 * u0**2 
    F0 = gp0 * Q0
    theta0 = np.pi / 2.

    V0 = np.array([Q0, M0, F0, theta0])
    return V0, p


def loadExptData(run):
    # Load the experimental data and image
    dataDirName = pathname + 'ExpPlumes_for_Dai/exp%02d/' % run
    # load the experimental image
    data = np.flipud(loadmat(dataDirName + 'gsplume.mat')['gsplume'])
    # Load the locations of the centre of the plume (values in pixels)
    xexp = loadmat(dataDirName + 'xcenter.mat')['xcenter'][0]
    zexp = loadmat(dataDirName + 'zcenter.mat')['zcenter'][0]
    
    # Define the pixel coordinates of the origin as the first points in x & z
    origin = (xexp[0], zexp[0])
    # Now calculate the world extent of the data, using a conversion factor 
    # There's probably some neater and more pythonic way of calculating the
    # world extent list but at least it works as is.
    extentInPix = [0, data.shape[1], 0, data.shape[0]]
    extent = np.array([(extentInPix[:2] - origin[0])/scaleFactor,
                       (origin[1] - extentInPix[2:])/scaleFactor])
    extent = extent.flatten().tolist()

    # Convert experimental trajectories to physical units
    xexp = (xexp - xexp[0]) / scaleFactor
    zexp = (zexp[0] - zexp) / scaleFactor
    return data, xexp, zexp, extent


if __name__ == '__main__':
    # Look for command line arguments that tell us which experimental run
    # to analyse.  If the command line argument is 'all', set allFlag to True
    # and run the whole set.  If 
    allFlag = False
    run     = None
    if len(sys.argv) >= 2:
        run = sys.argv[1]
        if run == 'all':
            allFlag = True
        else:
            run = int(run)
        if len(sys.argv) == 3:
            plotResults = sys.argv[2]
            if plotResults == 'True':
                plotResults = True
            if plotResults == 'False':
                plotResults = False
        else:
            plotResults = True  # Default behaviour is to print the solution
    else:
        run = 3                 # Default expt
        plotResults = True      # Default behaviour is to print the solution
    # Test if plotResults is a boolean
    if not isinstance(plotResults, bool):
        raise TypeError('plotReults must be a boolean')

    # Load experimental data and image from file
    data, xexp, zexp, extent = loadExptData(run)
    xexp = xexp[::10]
    zexp = zexp[::10]
    n, m = data.shape
    
    # Distance along plume axis and plume angle
    sexp     = distAlongPath(xexp, zexp)
    thetaexp, sig_thetaexp = plumeAngle(xexp, zexp, errors=[1/scaleFactor]*2)
    
    xexpPix = (xexp - extent[0]) * scaleFactor
    zexpPix = n + (extent[-1] - zexp) * scaleFactor
    trueLocn, bexp, sig_trueLocn, sig_bexp = trueLocationWidth(
        np.array([xexpPix, zexpPix]).T,
        data,
        errors=[1/scaleFactor])

    bexp /= scaleFactor
    sig_bexp /= scaleFactor

    # Form initial conditions and model parameters from CGTA expt data
    V0, p = loadICsParameters(pathname, run, alpha=.075, beta=.5, m=2.)

    
    # Define state vector and axial distance
    V  = []
    s  = []
    V.append(V0)
    s.append(0.)

    # Define the individual variables - these will be calculated at run time
    Q, M, F, theta = [], [], [], []
    Q0, M0, F0, theta0 = V0
    Q.append(Q0)
    M.append(M0)
    F.append(F0)
    theta.append(theta0)

    r = ode(derivs).set_integrator('lsoda', nsteps=1e6)
    r.set_initial_value(V0, 0)
    r.set_f_params(p)           # Set non default values for p

    # Domain of integration and integration step. 
    # No need to go further than extent of the experimental plume
    t1 = sexp.max()             
    dsexp = np.diff(sexp)
	
    ind = 0
    while r.successful() and r.t < t1 and M[-1] >= 0.:
        dt = dsexp[ind]         # to get model output at expt. points
        r.integrate(r.t + dt)
        V.append(r.y)
        s.append(r.t)
        Q_, M_, F_, theta_ = r.y
        Q.append(Q_)
        M.append(M_)
        F.append(F_)
        theta.append(theta_)
        ind += 1

    # Convert variables to numpy arrays for convenience
    s, V, Q, M, F, theta = (np.array(s), np.array(V), np.array(Q), 
                            np.array(M), np.array(F), np.array(theta))

    filename = pathname + 'fumarolePlumeModel_expt%02d.csv' % run
    try:
        if not os.path.isfile(filename):
            with open(filename, 'w') as fido:
                fido.write('# Solution to run % 2d\n' % run)
                for _s, line in zip(s, V):
                    fido.write('%6.2f,%8.4f,%8.4f,%8.4f,%8.4f\n' % (_s,
							            line[0],
                                                                    line[1],
							            line[2],
                                                                    line[3]))
    except FileNotFoundError:
        print('Directory does not exist')

    # Convert plume flux parameters into basic params
    b, u, gp = Q / np.sqrt(M), M / Q, F / Q

    # Calculate the model predictions
    xmod, zmod = [0.], [0.]
    xup, zup   = [0.], [0.]
    xlo, zlo   = [0.], [0.]
    ds_ = np.diff(s)
    for (ds, b_, th) in zip(ds_, b, theta):
        xmod.append(xmod[-1] + ds * np.cos(th))
        zmod.append(zmod[-1] + ds * np.sin(th))
        # Add plume width to obtain outer envelope
        xup.append(xup[-1] + ds * np.cos(th) + b_ * np.sin(th))
        zup.append(zup[-1] + ds * np.sin(th) + b_ * np.cos(th))
        xlo.append(xlo[-1] + ds * np.cos(th) - b_ * np.sin(th))
        zlo.append(zlo[-1] + ds * np.sin(th) - b_ * np.cos(th))

    if plotResults:
        # Plot the image, experimental and model data
        plt.close('all')
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # Plume greyscale intensity image 
        ax[0].imshow(data, extent=extent, cmap=plt.cm.gray)
        ax[0].set_xlabel(r'$x$/[cm]')
        ax[0].set_ylabel(r'$z$/[cm]')
        XLim = ax[0].get_xlim()
        YLim = ax[0].get_ylim()
        # Experimentally-determined plume axis trajectory
        ax[0].plot(xexp, zexp, 'r--', label='expt', lw=2)

        # Model plume axis trajectory
        ax[0].plot(xmod, zmod, 'g-', label='model', lw=2)
        ax[0].axes.invert_yaxis() # set_ylim(YLim[::-1])
        ax[0].legend(loc=4, framealpha=.6)
        ax[0].grid(True)

        # Create interpolation structures based on the model solution which
        # will be used to recast the model at the experimental data points.
        # PLUME WIDTH
        ax[1].plot(b - bexp, s,
                   label=r'$b_\mathrm{model} - b_\mathrm{exp}$')
        # PLUME ANGLE
        ax[1].plot(theta - thetaexp, # / sig_thetaexp,
                   s, '-', #c='C3',
                   label=r'$\theta_\mathrm{model} - \theta_\mathrm{exp}$')
        ax[1].set_title('Differences between model and expt data')
        ax[1].legend(loc='best',
                     framealpha=.6)
        ax[1].grid(True)
        ax[1].set_ylabel(r'$s$/[cm]')

        dVds = []
        for s_, V_ in zip(s, V):                                               
            dVds.append(derivs(s_, V_)) 
        dVds = np.array(dVds)

        plt.show()
