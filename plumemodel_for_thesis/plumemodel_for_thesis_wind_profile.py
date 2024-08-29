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
    return 0.01 * np.ones_like(s)
