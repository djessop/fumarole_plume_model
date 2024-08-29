def density_fume(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
                 Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    '''Returns the density of the fumarole, based on ideal gas
    '''
    T  = temperature_fume(s, V)
    Pa = V[4]  # atmospheric pressure
    n  = V[5]
    Rp = bulk_gas_constant(s, V)
    return Pa / (Rp * T)
