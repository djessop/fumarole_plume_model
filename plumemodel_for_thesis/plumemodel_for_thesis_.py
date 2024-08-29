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


def produce_Gm(sol, cutoff=None):
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
    # errors/Cd_inv are not passed explicitly as arguements
    warnings.filterwarnings('ignore')
    V0   = [1, 1, Cp0 * T0, 1, 1, .05]  # required for routines
    rho0 = density_fume(0, V0)
    Q0   = rho0 * np.pi * R0**2 * u0
    M0   = Q0 * u0
    E0   = Q0 * Cp0 * T0
    V0   = [Q0, M0, E0, np.pi/2, Pa0, .05]

    sol  = solve_ivp(derivs, [s[0], s[-1]], V0, t_eval=s)
    Gm   = produce_Gm(sol, s)

    return objective_fn(Gm, d, Cd_inv, mode='lstsq'), V0import numpy as np
