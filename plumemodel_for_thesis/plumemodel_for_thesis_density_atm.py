def density_atm(s, V, ks=0.09, kw=0.9, g=9.81, Ca=1006, Cp0=1885,
                Ta0=291, Ra=287, Rp0=462, Pa0=86000, n0=0.05):
    Ta = Ta0
    Pa = V[4]
    return Pa / (Ra * Ta)
