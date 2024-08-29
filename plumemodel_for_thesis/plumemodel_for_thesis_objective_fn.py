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
