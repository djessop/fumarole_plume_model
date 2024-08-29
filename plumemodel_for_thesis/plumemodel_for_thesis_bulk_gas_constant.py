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
