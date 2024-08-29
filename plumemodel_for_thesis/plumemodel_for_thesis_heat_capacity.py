def heat_capacity(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
                  Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    """
    Define specific heat capacity, Cp, of plume, as a function of the dry air 
      mass fraction, n, for initial 95% of vapour 
    Ca:  spec. heat capacity of air
    Cp0: spec. heat capacity of vapour
    """  
    return Ca + (Cp0 - Ca) * (1 - V[5]) / (1 - n0)
