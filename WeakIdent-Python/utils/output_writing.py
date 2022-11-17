import numpy as np
import pandas as pd
from utils.feature_library_building import build_eqn_str, build_feature_latex_tags
from utils.calculations import compute_err_inf, compute_err_l2, compute_err_res, compute_tpr_ppv

"""
This file stores all the functions used to write output table for WeakIdent.

Code author: Mengyi Tang Rajchel.

Copyright 2022, All Rights Reserved
"""

def write_true_and_identified_equations(c_true: np.array, c_pred: np.array, tags_lhs: list, tags_rhs: list) -> pd.core.frame.DataFrame:
    """
    This function write identified and true equations into a pandas table.
    Args:
        c_true (np.array): true coefficient vector.
        c_pred (np.array): predicted(identified) coefficient vector.
        tags_lhs(list): a list of string tag for lhs feature(s).
        tags_rhs(list): a list of string tag for rhs feature(s).

    Returns:
        pd.core.frame.DataFrame: a table of identified and true equation. 
        Here is an example:
        +----+-----------------------------------+----------------------------------------+
        |    | True equation                     | Predicted equation                     |
        +====+===================================+========================================+
        |  0 | u_t = + 2.0 u - 2.05 uv - 2.0 u^2 | u_t = + 1.991 u - 2.036 uv - 1.988 u^2 |
        +----+-----------------------------------+----------------------------------------+
        |  1 | v_t = + 2.0 v - 2.0 v^2 - 2.05 uv | v_t = + 1.99 v - 1.989 v^2 - 2.031 uv  |
        +----+-----------------------------------+----------------------------------------+
    """
    true_eqn_str = build_eqn_str(tags_lhs, tags_rhs, c_true)
    predicted_eqn_str = build_eqn_str(tags_lhs, tags_rhs, c_pred)
    df_eqns = pd.DataFrame(data={
        'True equation': true_eqn_str,
        'Predicted equation': predicted_eqn_str
    })

    return df_eqns


def write_coefficient_table(num_of_u: int, c_true: np.array, c_pred: np.array, tags_lhs: list, tags_rhs: list) -> pd.core.frame.DataFrame:
    """
    This function writes the feature tags and values for true and identifed equation into a table.

    Args:
        num_of_u (int): number of variables.
        c_true (np.array): true coefficient vector.
        c_pred (np.array): predicted(identified) coefficient vector.
        tags_lhs(list): a list of string tag for lhs feature(s).
        tags_rhs(list): a list of string tag for rhs feature(s).

    Returns:
        pd.core.frame.DataFrame: a table of feature tags and values for true and identifed equation.
        Here is an example:
        +----+-----------+------------+------------+------------+------------+
        |    | feature   |   true u_t |   pred u_t |   true v_t |   pred v_t |
        +====+===========+============+============+============+============+
        |  0 | 1         |       0    |    0       |       0    |    0       |
        +----+-----------+------------+------------+------------+------------+
        |  1 | v         |       0    |    0       |       2    |    1.9856  |
        +----+-----------+------------+------------+------------+------------+
        |  2 | v^2       |       0    |    0       |      -2    |   -1.98639 |
        +----+-----------+------------+------------+------------+------------+
        |  3 | v^3       |       0    |    0       |       0    |    0       |
        +----+-----------+------------+------------+------------+------------+
        |  4 | v^4       |       0    |    0       |       0    |    0       |
        +----+-----------+------------+------------+------------+------------+
        |  5 | u         |       2    |    2.00513 |       0    |    0       |
        +----+-----------+------------+------------+------------+------------+
        |  6 | uv        |      -2.05 |   -2.04292 |      -2.05 |   -2.02336 |
        +----+-----------+------------+------------+------------+------------+
        |  7 | uv^2      |       0    |    0       |       0    |    0       |
        +----+-----------+------------+------------+------------+------------+
        |  8 | uv^3      |       0    |    0       |       0    |    0       |
        +----+-----------+------------+------------+------------+------------+
        |  9 | u^2       |      -2    |   -2.00625 |       0    |    0       |
        +----+-----------+------------+------------+------------+------------+
        | 10 | u^2v      |       0    |    0       |       0    |    0       |
        +----+-----------+------------+------------+------------+------------+
        | 11 | u^2v^2    |       0    |    0       |       0    |    0       |
        +----+-----------+------------+------------+------------+------------+
        | 12 | u^3       |       0    |    0       |       0    |    0       |
        +----+-----------+------------+------------+------------+------------+
        | 13 | u^3v      |       0    |    0       |       0    |    0       |
        +----+-----------+------------+------------+------------+------------+
        | 14 | u^4       |       0    |    0       |       0    |    0       |
        +----+-----------+------------+------------+------------+------------+
    """
    d_coe = {}
    d_coe['feature'] = list(tags_rhs)
    for i in range(num_of_u):
        d_coe['true ' + tags_lhs[i]] = list(c_true[:, i])
        d_coe['pred ' + tags_lhs[i]] = list(c_pred[:, i])
    df_coe = pd.DataFrame(d_coe)
    return df_coe


def write_identification_err_table(c_true: np.array, c_pred: np.array, w: np.array, b: np.array, idx_highly_dynamic: np.array) -> pd.core.frame.DataFrame:
    """This function writes multiple type of identification error into a table.

    Args:
        c_true (np.array): true coefficient vector.
        c_pred (np.array): predicted(identified) coefficient vector.
        w (np.array): feature matrix (rhs).
        b (np.array): dynamic variable (lhs).
        idx_highly_dynamic (np.array): row index of features located in highly dynamic region.

    Returns:
        pd.core.frame.DataFrame: a table of identification errors. Here is an example:
        +----+------------+----------------+-------------+---------+---------+
        |    |      $e_2$ |   $e_{\infty}$ |   $e_{res}$ |   $tpr$ |   $ppv$ |
        +====+============+================+=============+=========+=========+
        |  0 | 0.00706426 |      0.0129936 |    0.162743 |       1 |       1 |
        +----+------------+----------------+-------------+---------+---------+
    """
    e2 = compute_err_l2(c_true, c_pred)
    e_inf = compute_err_inf(c_true, c_pred)
    e_tpr, e_ppv = compute_tpr_ppv(c_true, c_pred)
    e_res = compute_err_res(w[idx_highly_dynamic, :], c_pred,
                            b[idx_highly_dynamic])
    d_errs = {
        '$e_2$': [e2],
        '$e_{\infty}$': [e_inf],
        '$e_{res}$': [e_res],
        '$tpr$': [e_tpr],
        '$ppv$': [e_ppv]
    }
    df_errs = pd.DataFrame(data=d_errs)
    return df_errs


def write_output_tables(num_of_variables: int, c_true: np.array, dim_x: int,
                        is_1d_ode: bool, dict_list: np.array,
                        lhs_ind: np.array, idx_highly_dynamic: np.array,
                        w: np.array, b: np.array, c_pred: np.array):
    """This function write output tables as identification results.

    Args:
        num_of_variables (int): total number of variables.
        c_true (np.array): true coefficient vector.
        dim_x (int): spatial dimension of given data.
        is_1d_ode (bool): whether or not given data is 1d ode data.
        dict_list (np.array): a list of feature in the dictioinary.
        lhs_ind (np.array): index of lhs features.
        idx_highly_dynamic (np.array): row index of features located in highly dynamic region.
        w (np.array): feature matrix (rhs).
        b (np.array): dynamic variable (lhs).
        c_pred (np.array): predicted coefficient vector.

    Returns:
    Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame, pd.core.frame.DataFrame, float]: 
        equation table, identification error table,  coefficient vector table.
    """
    latex_tags_lhs, latex_tags_rhs = build_feature_latex_tags(
        num_of_variables, dim_x, is_1d_ode, dict_list, lhs_ind)
    df_eqns = write_true_and_identified_equations(c_true, c_pred,
                                                  latex_tags_lhs,
                                                  latex_tags_rhs)
    df_coe = write_coefficient_table(num_of_variables, c_true, c_pred, latex_tags_lhs,
                                     latex_tags_rhs)
    df_errs = write_identification_err_table(
        c_true, c_pred, w, b, idx_highly_dynamic)
    return df_eqns, df_coe, df_errs
