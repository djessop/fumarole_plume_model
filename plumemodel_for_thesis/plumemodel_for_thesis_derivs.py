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
