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
