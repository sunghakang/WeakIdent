import numpy as np
import pandas as pd
from utils.feature_library_building import build_eqn_str, build_feature_latex_tags
from utils.calculations import compute_err_inf, compute_err_l2, compute_err_res, compute_tpr_ppv



def write_true_and_identified_equations(c_true, c_pred, tags_lhs, tags_rhs):
    true_eqn_str = build_eqn_str(tags_lhs, tags_rhs, c_true)
    predicted_eqn_str = build_eqn_str(tags_lhs, tags_rhs, c_pred)
    df_eqns = pd.DataFrame(data={
        'True equation': true_eqn_str,
        'Predicted equation': predicted_eqn_str
    })

    return df_eqns


def write_coefficient_table(num_of_u, c_true, tags_lhs, tags_rhs, c_pred):
    d_coe = {}
    d_coe['feature'] = list(tags_rhs)
    for i in range(num_of_u):
        d_coe['true ' + tags_lhs[i]] = list(c_true[:, i])
        d_coe['pred ' + tags_lhs[i]] = list(c_pred[:, i])
    df_coe = pd.DataFrame(d_coe)
    return df_coe


def write_identification_err_table(c_true, idx_highly_dynamic, w, b, c_pred):
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
        idx_highly_dynamic (np.array): row index of features located in highly dynamic region
        w (np.array): feature matrix (rhs)
        b (np.array): dynamic variable (lhs)
        c_pred (np.array): predicted coefficient vector

    Returns:
    Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame, pd.core.frame.DataFrame, float]: 
        equation table, identification error table,  coefficient vector table,
    """
    latex_tags_lhs, latex_tags_rhs = build_feature_latex_tags(
        num_of_variables, dim_x, is_1d_ode, dict_list, lhs_ind)
    df_eqns = write_true_and_identified_equations(c_true, c_pred,
                                                  latex_tags_lhs,
                                                  latex_tags_rhs)
    df_coe = write_coefficient_table(num_of_variables, c_true, latex_tags_lhs,
                                     latex_tags_rhs, c_pred)
    df_errs = write_identification_err_table(c_true, idx_highly_dynamic, w, b,
                                             c_pred)
    return df_eqns, df_coe, df_errs