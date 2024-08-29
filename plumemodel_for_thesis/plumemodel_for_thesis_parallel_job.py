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
