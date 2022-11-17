"""
Some utils functions of weakident to predict partial/ordinary different equations.
"""

def set_hist_bins(is_1d_ode: bool) -> int:
    """This function set up number of bins used in histogram when finding highly dynamic region.

    Args:
        is_1d_ode (bool): whether or not given data is 1d ode data.

    Returns:
        int: number of bins.
    """
    if is_1d_ode:
        return 100
    else:
        return 200

def set_sparsity_level(is_1d_ode: bool, num_of_u: int, dim_x: int) -> int:
    """This function set maximum sparsity level for support recovery.

    Args:
        is_1d_ode (bool): whether or not given data is 1d ode data.
        num_of_u (int): number of variables.
        dim_x (int): spatial dimension.

    Returns:
        int: sparsity level
    """
    if is_1d_ode:
        if num_of_u <= 2:
            sparsity = 10
        else:
            sparsity = 15
    else:
        if num_of_u == 2 and dim_x == 2:
            sparsity = 25
        else:
            sparsity = 10
    return sparsity

