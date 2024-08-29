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
