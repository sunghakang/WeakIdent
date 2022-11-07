from utils import get_tags_v2, compute_test_function_para, circshift, drange, compute_err_inf, compute_err_l2, compute_err_res, compute_tpr_ppv, two_piece_fit_v2, build_tag_ode, build_tag_pde, build_eqn_str
import numpy as np
import scipy.linalg
# import matplotlib.pyplot as plt
import numpy_indexed as npi
import sys
import pandas as pd
import time
from typing import Tuple

"""

modeling functions of weakident to predict partial/ordinary different equations.

Copyright 2022, All Rights Reserved
Code by Mengyi Tang Rajchel
For Paper, "WeakIdent: Weak formulation for Identifying
Differential Equation using Narrow-fit and Trimming"
by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

"""

def weak_ident_pred(
        u_hat: np.array, xs: np.array, true_coefficients: np.array,
        max_dx: int, max_poly: int, skip_x: int, skip_t: int,
        use_cross_der: bool,
        tau) -> Tuple[np.array, np.array, pd.core.frame.DataFrame, float]:
    """
    This is the main function to perform prediction for partial differential equation or 
    ordinary differential equations using weak_ident.
    
    Args:
        u_hat (np.array): array of shape (n,) (given data with n variable).
        phi_xs (np.array): array of shape (max_dx +1, 2m_x + 1), fourier transform phi and phi^(i)(x) for i = 0,1,...,max_dx.
        true_coefficients: array of shape (n,).
        max_dx (int): maximum total order of partial derivatives.
        max_poly (int): maximum total order of monomials.
        skip_x (int): skipping steps in spatial domain when downsampling feature matrix.
        skip_t (int): skipping steps in temporal domain when downsampling feature matrix.
        use_cross_der (bool): whether allow partial derivatives.
        tau (float): trimming threshold.

    Returns:
        Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame, pd.core.frame.DataFrame, float]: 
        identification error table,  equation table, coefficient vector table,
        running time        
    """
    start_time = time.time()
    num_of_u = u_hat.shape[0]
    dims = u_hat[0].shape  # (Nx, Nt)
    dim_x_and_t = len(dims)
    dim_x = dim_x_and_t - 1
    is_1d_ode = False
    t_mesh = xs[-1].reshape(-1, 1)
    dt = t_mesh[1] - t_mesh[0]

    if dims[0] == 1:
        # check this part later
        is_1d_ode = True

    if is_1d_ode:
        num_of_bins = 100
        lhsIdx = np.block([
            np.diag(np.ones(num_of_u)),
            np.zeros((num_of_u, 1)),
            np.ones((num_of_u, 1))
        ])
        if num_of_u <= 2:
            sparsity = 10
        else:
            sparsity = 15
    else:
        num_of_bins = 200
        lhsIdx = np.block([
            np.diag(np.ones(num_of_u)),
            np.zeros((num_of_u, dim_x)),
            np.ones((num_of_u, 1))
        ])
        if num_of_u == 2 and dim_x == 2:
            sparsity = 25
        else:
            sparsity = 10
        xmesh = xs[0].reshape(-1, 1)
        dx = xmesh[1] - xmesh[0]

    dict_list, lhs_ind, c_true = get_tags_v2(num_of_u, dim_x, lhsIdx, max_dx,
                                             max_poly, use_cross_der,
                                             true_coefficients)
    if is_1d_ode:
        m_x, _, p_x, _ = compute_test_function_para(u_hat, xs, max_dx)
    else:
        m_x, m_t, p_x, p_t = compute_test_function_para(u_hat, xs, max_dx)
    if is_1d_ode:
        tags_lhs, tags_rhs = build_tag_ode(num_of_u,  dict_list, lhs_ind)
        w_b_large, s_b_large = compute_feature_and_scale_matrix_ode(
            m_x, p_x, skip_x, dict_list, u_hat, dt)
    else:
        tags_lhs, tags_rhs = build_tag_pde(num_of_u, dim_x, dict_list, lhs_ind)
        w_b_large, s_b_large = compute_feature_and_scale_matrix_pde(
            dims, m_x, m_t, p_x, p_t, skip_x, skip_t, dict_list,
            u_hat, dx, dt, max_dx)
    idx_interesting_features = find_idx_of_interesting_feature(
        num_of_u, dim_x, is_1d_ode)
    idx_highly_dynamic = find_highly_dynamic_region(dict_list,
                                                    idx_interesting_features,
                                                    s_b_large, num_of_bins)
    print("The number of rows in the highly dynamic region is ",
          idx_highly_dynamic.shape[0])
    scales_w_b = np.mean(np.abs(s_b_large[idx_highly_dynamic, :]), axis=0)
    l = w_b_large.shape[1]
    rhs_ind = [i for i in range(l) if i not in lhs_ind]
    rhs_ind = np.array(rhs_ind)
    scales_feature = scales_w_b[rhs_ind]
    scales_feature = 1 / scales_feature.flatten()
    scales_b = scales_w_b[lhs_ind]
    w = w_b_large[:, rhs_ind]
    b = w_b_large[:, lhs_ind]
    w_tilda = w * scales_feature
    b_tilda = b * scales_b
    b_narrow_tilda = b_tilda[idx_highly_dynamic, :]
    w_narrow_tilda = w_tilda[idx_highly_dynamic, :]
    c_pred = np.zeros((l - num_of_u, num_of_u))
    for num_var in range(num_of_u):
        b_large_ = b[:, num_var].reshape(-1, 1)
        b_large_tilda_ = b_tilda[:, num_var].reshape(-1, 1)
        b_narrow_tilda_ = b_narrow_tilda[:, num_var].reshape(-1, 1)
        scale_b_ = scales_b[num_var]
        support_pred = weak_ident_feature_selection(w, b_large_,
                                                    b_large_tilda_, w_tilda,
                                                    sparsity, scales_feature,
                                                    scale_b_, w_narrow_tilda,
                                                    b_narrow_tilda_, tau)
        print(" ")
        print(
            'Finished support trimming and narrow fit for variable no.%d . A support is found.'
            % (num_var + 1))
        relative_scale = scales_feature / scale_b_
        coeff_sp = least_square_adp(
            w_narrow_tilda[:, support_pred],
            b_narrow_tilda_) * (relative_scale[support_pred].reshape(-1, 1))
        c_pred[support_pred, num_var] = coeff_sp.flatten()
    run_time = time.time() - start_time
    true_eqn_str = build_eqn_str(tags_lhs, tags_rhs, c_true)
    predicted_eqn_str = build_eqn_str(tags_lhs, tags_rhs, c_pred)
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
    df_eqns = pd.DataFrame(data= {'True equation': true_eqn_str, 'Predicted equation': predicted_eqn_str})
    d_coe = {}
    d_coe['feature'] = list(tags_rhs)
    for i in range(num_of_u):
        d_coe['true ' + tags_lhs[i]] = list(c_true[:,i])
        d_coe['pred ' + tags_lhs[i]] = list(c_true[:,i])
    df_coe = pd.DataFrame(d_coe)
    return df_errs,  df_eqns, df_coe, run_time


def compute_discrete_phi(m: int, d: int, p: int) -> np.array:
    """
    This function will compute the discretized 0th - dth order partial derivative of 
    1d test function f(x) = (1-x^2)^p on a localized domain centered around 0 with grid
    size 2m+1.

    Args:
        m (int): 2*m+1 is the local intergration grid size for weak features in one dimension.
        d (int): the highest order of partial derivatives allowed in the dictionary.
        p (int): a smoothness parameter in the test function f(x) = (1-x^2)^p.

    Remark: The script is modified from Matlab code for Paper, "Weak SINDy for Partial Differential
            Equations" by D. A. Messenger and D. M. Bortz.

    Returns:
        np.array: array of shape (d+1, 2*m+1) where i-th (i = 1,...,d, d+1) row represents
                  d^(i-1)(f)/dx^(i-1).
    """
    t = np.arange(m + 1) / m
    t_l = np.zeros((d + 1, m + 1))
    t_r = np.zeros((d + 1, m + 1))
    for j in range(m):
        t_l[:, j] = np.power(
            1 + t[j],
            np.fliplr(np.arange(p - d, p + 1).reshape(1, -1)).flatten())
        t_r[:, j] = np.power(
            1 - t[j],
            np.fliplr(np.arange(p - d, p + 1).reshape(1, -1)).flatten())
    ps = np.ones((d + 1, 1))
    for q in range(d):
        ps[q + 1] = (p - (q + 1) + 1) * ps[q]
    t_l = ps * t_l
    t_r = (np.power(-1, np.arange(d + 1)).reshape(-1, 1) * ps) * t_r
    cfs = np.zeros((d + 1, 2 * m + 1))
    cfs[0, :] = np.block([
        np.fliplr((t_l[0, :] * t_r[0, :]).reshape(1, -1)),
        t_l[0, 1:] * t_r[0, 1:]
    ])
    p = np.fliplr(scipy.linalg.pascal(d + 1))
    for k in range(d):
        binoms = np.diag(p, d - k - 1)
        cfs_temp = np.zeros((1, m + 1))
        for j in range(k + 2):
            cfs_temp = cfs_temp + binoms[j] * t_l[k + 1 - j, :] * t_r[j, :]
        cfs[k + 1, :] = np.block([
            (-1)**(k + 1) * np.fliplr(cfs_temp.reshape(1, -1)),
            cfs_temp[0, 1:].reshape(1, -1)
        ])
    return cfs


def compute_feature_and_scale_matrix_ode(
        m: int, p: int, skip: int, dict_list: np.array, u_hat: np.array,
        dt: np.array) -> Tuple[np.array, np.array]:
    """
    This function will compute the feature matrix W and scale matrix S for both lhs and rhs features for ODEs.

    Note: (1) the current version of ode equations refer to 1d ode problems where spatial derivatives do not exits.
          (2) for odes, we consider the scale matrix S to be identical to feature matrix W.
    Args:
        m (int): one-side length of local integration area
        p (int): parameter in the test function
        skip (int): skipping steps when downsampling a feature matrix
        dict_list(np.ndarray): array of shape (L, n + 2). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial in each feature
                               column n + 1 has 0s.
                               column n + 2 represents the order of partial derivatives along temporal
                               domain. We take 0 or 1 for this value in WeakIdent.
        u_hat (np.array): array of shape (n,) (given noisy data).
        dt (np.array): delta t (temporal increase) of given data .

    Returns:
        np.array, np.array: feature matrix W and sclae matrix S (including lhs)
    """
    n_t = u_hat[0].shape[1]
    subsampling_idx = np.arange(0, n_t - 2 * m, skip)
    phi_xs = compute_discrete_phi(m, 1, p)
    mm, nn = phi_xs.shape[0], phi_xs.shape[1]
    temp = np.block([
        np.zeros((mm, n_t - nn)),
        np.power(m * dt, -np.arange(mm)).reshape(-1, 1) * phi_xs / nn
    ])
    fft_phis = np.fft.fft(temp)
    w_b_large = compute_w_ode(dict_list, u_hat, fft_phis, subsampling_idx)
    s_b_large = w_b_large.copy()
    s_b_large[s_b_large == 0] = 1
    return w_b_large, s_b_large


def compute_w_ode(dict_list: np.array, u_hat: np.array, fft_phis: np.array,
                  subsampling_idx: np.array) -> np.array:
    """
    Args:
        dict_list(np.ndarray): array of shape (L, n + 2). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial in each feature
                               column n + 1 has 0s.
                               column n + 2 represents the order of partial derivatives along temporal
                               domain. We take 0 or 1 for this value in WeakIdent.
        u_hat (np.array): given noisy data with shape (n,).
        fft_phis (np.array): array of shape (2, mathbbNx). This is the fast fourier transform of the test function and
                             it's first order derivative. 
        subsampling_idx (np.array): array of shape (Nx , ) which stores the index of subsampled feature matrix.

    Returns:
        np.array: array of shape (N, L + n)
    """
    print('The number Nx  in the subsampled feature matrix is',
          subsampling_idx.shape[0], '.')
    print('Start building feature matrix W:')
    L = dict_list.shape[0]
    w = np.zeros((subsampling_idx.shape[0], L))
    n = u_hat.shape[0]
    ind = 0
    while ind < L:
        tags = dict_list[ind, :n]
        fcn = np.power(u_hat[0], tags[0])
        for k in range(1, n):
            fcn = fcn * np.power(u_hat[k], tags[k])
        while np.all(dict_list[ind, :n] == tags):
            temp = fft_phis
            test_conv_cell = temp[int(dict_list[ind, n + 1]), :].reshape(-1, 1)
            fcn_conv = conv_fft_v2(fcn, test_conv_cell, subsampling_idx)
            w[:, ind] = fcn_conv.flatten()
            sys.stdout.write('\r')
            sys.stdout.write("[{:{}}] {:.1f}%".format("=" * ind, L - 1,
                                                      (100 / (L - 1) * ind)))
            sys.stdout.flush()
            ind += 1
            if ind >= L:
                break
    return w


def compute_feature_and_scale_matrix_pde(
        dims: tuple, m_x: int, m_t: int, p_x: int, p_t: int, skip_x: int,
        skip_t: int, dict_list: np.array, u_hat: np.array, dx: np.array,
        dt: np.array, max_dx: int) -> Tuple[np.array, np.array]:
    """
    This function will compute the feature matrix W and scale matrix S (including lhs and rhs) for given data.

    Args:
        dims (tuple): (N_x, N_t) or (N_x, N_y, N_t).
        m_x (int): 1-side size of integrating region in spatial domain.
        m_t (int): 1-side size of integrating region in temporal domain.
        p_x (int): parameter in test function in terms of x.
        p_t (int): parameter in test function in terms of t.
        skip_x (int): # skipping steps in spatial domain when downsampling feature matrix
        skip_t (int): # skipping steps in temporal domain when downsampling feature matrix
        dict_list(np.ndarray): array of shape (L, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain,
                               and column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. We take 0 or 1 in WeakIdent.
        u_hat (np.array): given data of shape (n,).
        dx (np.array): delta t (spatial increase) of given data.
        dt (np.array): delta t (temporal increase) of given data.
        max_dx (int): maximum total order of partial derivatives.

    Returns:
        Tuple[np.array, np.array]: feature matrix W and scale matrix S (including lhs).
    """
    dim_x = len(dims) - 1
    subsampling_idx = []
    shrink_size = np.ones((1, dim_x + 1))
    shrink_size = np.block(
        [np.ones((1, dim_x)) * m_x * 2,
         np.array([[m_t * 2]])]).flatten()
    ss = np.block([np.ones((1, dim_x)) * skip_x,
                   np.array([[skip_t]])]).flatten()
    for j in range(len(dims)):
        subsampling_idx.append(np.arange(0, dims[j] - shrink_size[j], ss[j]))
    phi_xs, phi_ts = compute_test_funs(m_x, m_t, p_x, p_t, max_dx)
    fft_phis = compute_fft_test_fun(dims, phi_xs, phi_ts, m_x, m_t, dx, dt)
    w_b_large = compute_w(dict_list, u_hat, fft_phis, subsampling_idx)
    s_b_large = compute_s_v2(w_b_large, dict_list, u_hat, fft_phis,
                             subsampling_idx)
    return w_b_large, s_b_large


def compute_s_v2(w: np.array, dict_list: np.array, u_hat: np.array,
                 fft_phis: np.array, subsampling_idx: np.array) -> np.array:
    """
    This function will compute the scale matrix S (including lhs) for given data.

    Args:
        w (np.array): (feature matrix array) of shape (N, L + n) where L is the number of rhs features and n 
                      is the number of variables
        dict_list(np.ndarray): array of shape (L, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable, 
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain, 
                               and column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. We take 0 or 1 in WeakIdent.
        u_hat (np.array): array of shape (n,) (given noisy data).
        fft_phis (np.array): array of shape (max_dx+1, mathbbNx). This is the fast fourier transform of a test function
                             and it's derivatives. 
        subsampling_idx (np.array): array of shape (Nx , ) which stores the index of subsampled feature matrix.

    Returns:
        np.array: (scale matrix) array of shape (N, L + n). 
    """
    print(" ")
    print('Start building scale matrix S:')
    n = u_hat.shape[0]
    dim = len(u_hat[0].shape)
    l = dict_list.shape[0]
    if dim == 2:
        s = np.ones((len(subsampling_idx[0]) * len(subsampling_idx[1]), l))
    elif dim == 3:
        s = np.ones((len(subsampling_idx[0])**2 * len(subsampling_idx[2]), l))
    ind = 1
    while ind < l:
        tags = dict_list[ind, :n]
        beta_u_plus_beta_v = np.sum(tags)
        if beta_u_plus_beta_v > 1:
            fcn = compute_base_of_s(u_hat, tags)
        while np.all(dict_list[ind, :n] == tags):
            if beta_u_plus_beta_v == 1:
                s[:, ind] = w[:, ind]
            else:
                test_conv_cell = []
                if np.sum(dict_list[ind, n:n + dim]) == 0:
                    for k in range(dim):
                        temp = fft_phis[k]
                        test_conv_cell.append(temp[0, :].reshape(-1, 1))
                else:
                    for k in range(dim):
                        temp = fft_phis[k]
                        test_conv_cell.append(
                            temp[int(dict_list[ind, n + k]), :].reshape(-1, 1))
                fcn_conv = conv_fft(fcn, test_conv_cell, subsampling_idx)
                if dim == 2:
                    s[:, ind] = np.transpose(fcn_conv, (1, 0)).flatten()
                elif dim == 3:
                    s[:, ind] = np.transpose(fcn_conv, (2, 1, 0)).flatten()
            if dim == 3 or l > 100:
                print(" Progress {:2.1%}".format((ind + 1) / l), end="\r")
            else:
                sys.stdout.write('\r')
                sys.stdout.write("[{:{}}] {:.1f}%".format(
                    "=" * ind, l - 1, (100 / (l - 1) * ind)))
                sys.stdout.flush()
            ind += 1
            if ind >= l:
                break
    print(" ")
    return s


def compute_base_of_s(u_hat: np.array, tags: np.array) -> np.array:
    """
    This function will compute the base of the scale (for error normalization) for a given features. 
    The definition of base follows the idea of leading coefficient of epsilon in WeakIdent paper.

    Args:
        u_hat (np.array): array of shape (n,) (given data).
        tags (np.array): the tags of betas (monomial order) of a given feature.

    Returns:
        np.array: base of scale for a given feature.
    """
    n = u_hat.shape[0]
    if n == 1:
        fcn = tags[0] * np.power(u_hat[0], tags[0] - 1)
    elif n >= 2:
        fcn = np.zeros_like(u_hat[0])
        for ii in range(n):
            if tags[ii] == 0:
                temp = 0
            else:
                if tags[ii] == 1:
                    temp = 1
                else:
                    temp = tags[ii] * np.power(u_hat[ii], tags[ii] - 1)
                for k in range(n):
                    if k != ii and tags[k] > 0:
                        temp = temp * np.power(u_hat[k], tags[k])
            fcn = fcn + temp
    return fcn


def compute_s(dict_list: np.array, u_hat: np.array, fft_phis: np.array,
              subsampling_idx: np.array, m_x: int, m_t: int, dx: np.array,
              dt: np.array):
    """
    This function compute the scale matrix S (including lhs) for given data.
    [IMPORTANT]: This function is not used anymore since it only applies to the case of 1 variable.
                 See compute_s_v2 for the most updated version. 
    Args:
        w (np.array): (feature matrix array) of shape (N, L + n) where L is the number of rhs features and n 
                      is the number of variables
        dict_list(np.ndarray): array of shape (L, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each feature,
                               column n+1 - column n+dim_x represents the order of partial derivatives,
                               along each spatial domain,
                               and column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. We take 0 or 1 in WeakIdent.
        u_hat (np.array): array of shape (n,) (given data).
        fft_phis (np.array): array of shape (max_dx+1, mathbbNx). This is the fft of test functions and it's derivatives. 
        subsampling_idx (np.array): array of shape (Nx , ) which stores the index of subsampled feature matrix.
        m_x (int): 1-side size of integrating region in spatial domain.
        m_t (int): 1-side size of integrating region in temporal domain.
        dx (np.array): delta t (spatial increase) of given data .
        dt (np.array): delta t (temporal increase) of given data .
    Returns:
        np.array: (scale matrix including lhs) array of shape (N, L + n). 
    """
    print(" ")
    print('Start building scale matrix S:')
    n = u_hat.shape[0]
    dim = len(u_hat[0].shape)
    L = dict_list.shape[0]
    if dim == 2:
        S = np.ones((len(subsampling_idx[0]) * len(subsampling_idx[1]), L))
    elif dim == 3:
        S = np.ones((len(subsampling_idx[0])**2 * len(subsampling_idx[2]), L))
    ind = 1
    while ind < L:
        tags = dict_list[ind, :n]
        beta_u_plus_beta_v = np.sum(tags)
        if beta_u_plus_beta_v <= 1:
            fcn = 1
            for k in range(n):
                fcn = fcn * np.power(u_hat[k], tags[k])
        else:
            fcn = 0
            for k in range(n):
                if tags[k] > 0:
                    fcn = fcn + (tags[k] - 1) * np.power(u_hat[k], tags[k] - 1)
        while np.all(dict_list[ind, :n] == tags):
            test_conv_cell = []
            scale_factor = 1
            if np.sum(dict_list[ind, n:n + dim]) == 0:
                for k in range(dim):
                    temp = fft_phis[k]
                    test_conv_cell.append(temp[0, :].reshape(-1, 1))
                    if k + 1 < dim:
                        scale_factor = scale_factor * \
                            np.power(m_x * dx, dict_list[ind, n+k])
                    else:
                        scale_factor = scale_factor * \
                            np.power(m_t * dt, dict_list[ind, n+k])
            else:
                scale_factor = 1
                for k in range(dim):
                    temp = fft_phis[k]
                    test_conv_cell.append(
                        temp[int(dict_list[ind, n + k]), :].reshape(-1, 1))
                    if k + 1 < dim:
                        scale_factor = scale_factor * \
                            np.power(m_x * dx, dict_list[ind, n+k])
                    else:
                        scale_factor = scale_factor * \
                            np.power(m_t * dt, dict_list[ind, n+k])
            fcn_conv = conv_fft(fcn, test_conv_cell, subsampling_idx)
            if dim == 2:
                S[:, ind] = np.transpose(fcn_conv, (1, 0)).flatten()
            elif dim == 3:
                S[:, ind] = np.transpose(fcn_conv, (2, 1, 0)).flatten()
            if dim == 3 or L > 100:
                print(" Progress {:2.1%}".format((ind + 1) / L), end="\r")
            else:
                sys.stdout.write('\r')
                sys.stdout.write("[{:{}}] {:.1f}%".format(
                    "=" * ind, L - 1, (100 / (L - 1) * ind)))
                sys.stdout.flush()
            ind += 1
            if ind >= L:
                break
    print(" ")
    return S


def compute_w(dict_list: np.array, u_hat: np.array, fft_phis: np.array,
              subsampling_idx: np.array):
    """
    This function will computes the feature matrix W (including lhs) for given data for pde equations.

    Args:
        dict_list(np.ndarray): array of shape (L, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable,
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain,
                               and column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. We take 0 or 1 in WeakIdent.
        u_hat (np.array): array of shape (n,) (given data).
        fft_phis (np.array): array of shape (max_dx+1, mathbbNx). This is the fft of a test functions and it's derivatives. 
        subsampling_idx (np.array): array of shape (Nx , ) which stores the index of subsampled feature matrix.

    Returns:
        np.array: (feature matrix including lhs) array of shape (N, L + n). 
    """
    print('Start building feature matrix W:')
    n = u_hat.shape[0]
    dim = len(u_hat[0].shape)
    L = dict_list.shape[0]
    if dim == 2:
        W = np.zeros((len(subsampling_idx[0]) * len(subsampling_idx[1]), L))
    elif dim == 3:
        W = np.zeros((len(subsampling_idx[0])**2 * len(subsampling_idx[2]), L))
    ind = 0
    while ind < L:
        tags = dict_list[ind, :n]
        fcn = np.power(u_hat[0], tags[0])
        for k in range(1, n):
            fcn = fcn * np.power(u_hat[k], tags[k])
        while np.all(dict_list[ind, :n] == tags):
            test_conv_cell = []
            # test_conf_cell store the fft of a test function(or it's derivatives) in all spatial
            # and temporal dimension
            for k in range(dim):
                temp = fft_phis[k]
                test_conv_cell.append(temp[int(dict_list[ind,
                                                         n + k]), :].reshape(
                                                             -1, 1))
            fcn_conv = conv_fft(fcn, test_conv_cell, subsampling_idx)
            if dim == 2:
                W[:, ind] = np.transpose(fcn_conv, (1, 0)).flatten()
            elif dim == 3:
                W[:, ind] = np.transpose(fcn_conv, (2, 1, 0)).flatten()
            if dim == 3 or L > 100:
                print(" Progress {:2.1%}".format((ind + 1) / L), end="\r")
            else:
                sys.stdout.write('\r')
                sys.stdout.write("[{:{}}] {:.1f}%".format(
                    "=" * ind, L - 1, (100 / (L - 1) * ind)))
                sys.stdout.flush()
            ind += 1
            if ind >= L:
                break
    return W


def conv_fft_v2(fu: np.array, fft_phi: np.array,
                subsampling_idx: np.array) -> np.array:
    """
    This function takes f(u) and fft(d^i(phi)) as input and returns the convolution u * d^i(phi) as
    as a result of weak feature.
    Note: this function is called only when we have 1d ode systems. See conv_fft for multi-dimensional
          convlution for the case of PDEs or multi-dimensional ODEs. 

    Args:
        fu (np.array): the base of a feature (the product of monomials).
        fft_phi (np.array): This is the fft of test functions (derivative) fft(d^i(phi)).
        subsampling_idx (np.array): array of shape (Nx, ) which stores the index of subsampled feature matrix.

    Returns:
        np.array: convolution u * d^i(phi)
    """
    ifft_phi = fft_phi.flatten()
    temp = np.fft.fft(fu).flatten()
    fu = np.fft.ifft(ifft_phi * temp, axis=0)
    inds = subsampling_idx.astype(int)
    fu = fu[inds]
    return np.real(fu)


def conv_fft(fu: np.array, fft_phis: np.array, subsampling_idx: np.array):
    """
    This function takes f(u) and fft(d^i(phi)) as input and returns the 
    multi-dimennional convolution u * d^i(phi) as
    as a weak feature.

    Args:
        fu (np.array): the base of feature (the product of monomials)
        fft_phi (np.array): This is the fft of test functions (derivative) fft(d^i(phi)) in each spatial
                            and temporal domain.
        subsampling_idx (np.array): stores the index of subsampled feature matrix in each spatial and 
                                    and temporal domain.

    Returns:
        np.array: convolution u * d^i(phi)
    """
    Ns = fu.shape
    dim = len(Ns)
    X = np.copy(fu)
    for k in range(dim):
        col_ifft = fft_phis[k]
        shift = circshift([i for i in range(dim)], 1 - (k + 1))
        shift_back = circshift([i for i in range(dim)], -1 + (k + 1))
        Y = np.fft.fft(X.transpose(tuple(shift)), axis=0)
        if dim == 3:
            col_ifft = col_ifft.reshape(-1, 1, 1)
        X = np.fft.ifft(col_ifft * Y, axis=0)
        inds = subsampling_idx[k].astype(int)
        if dim == 2:
            X = X[inds, :]
        elif dim == 3:
            X = X[inds, :, :]
        X = X.transpose(tuple(shift_back))
    return np.real(X)


def compute_fft_test_fun(dims: tuple, phi_xs: np.array, phi_ts: np.array,
                         m_x: int, m_t: int, dx: np.array,
                         dt: np.array) -> list:
    """
    This function computes fast fourier transform of test function phi and it's partial derivatives w.r.t x and t.

    Args:
        dims (tuple): (N_x, N_t) or (N_x, N_y, N_t).
        phi_xs (np.array): array of shape (max_dx +1, 2m_x + 1), fourier transform phi and phi^(i)(x) for i = 0,1,...,max_dx.
        phi_ts (np.array): array of shape (2, 2m_x + 1), fourier transform phi and phi^(i)(t) for i = 0,1,...,max_dt.
        m_x (int): 1-side size of integrating region in spatial domain.
        m_t (int): 1-side size of integrating region in temporal domain.
        dx (np.array): delta t (spatial increase) of given data.
        dt (np.array): delta t (temporal increase) of given data.

    Returns:
        list: [np.array, np.array] where each element is fft(phi_xs) and fft(phi_ts).
    """
    dimxandt = len(dims)
    fft_phis = []
    mm, nn = phi_xs.shape[0], phi_xs.shape[1]
    for k in range(dimxandt - 1):
        temp = np.block([
            np.zeros((mm, dims[k] - nn)),
            np.power(m_x * dx, -np.arange(mm)).reshape(-1, 1) * phi_xs / nn
        ])
        fft_phis.append(np.fft.fft(temp))
    mm, nn = phi_ts.shape[0], phi_ts.shape[1]
    temp = np.block([
        np.zeros((mm, dims[dimxandt - 1] - nn)),
        np.power(m_t * dt, -np.arange(mm)).reshape(-1, 1) * phi_ts / nn
    ])
    fft_phis.append(np.fft.fft(temp))
    return fft_phis


def compute_test_funs(m_x: int, m_t: int, p_x: int, p_t: int,
                      max_dx: np.array) -> Tuple[np.array, np.array]:
    """
    This function will compute the discretized test function for both spatial dimension and temporal 
    dimension.

    Args:
        m_x (int): 1-side size of integrating region in spatial domain.
        m_t (int): 1-side size of integrating region in temporal domain.
        p_x (int): parameter in test function in terms of x.
        p_t (int): parameter in test function in terms of t.
        max_dx (int): maximum total order of partial derivatives.

    Returns:
        [np.array, np.array] : phi(x) = (1-x^2)^p_x and phi(t) (1-t^2)^p_t.
    """
    phi_xs = compute_discrete_phi(m_x, max_dx, p_x)
    phi_ts = compute_discrete_phi(m_t, 1, p_t)
    return phi_xs, phi_ts


def weak_ident_feature_selection(w: np.array,
                                 b: np.array,
                                 b_tilda: np.array,
                                 w_tilda: np.array,
                                 k: int,
                                 s_rhs: np.array,
                                 s_lhs: np.float64,
                                 w_nar_tilda: np.array,
                                 b_nar_tilda: np.array,
                                 tau=0.05) -> np.ndarray:
    """
    This function returns the support predicted by WeakIdent. See details in paper "WeakIdent: Weak formulation for Identifying
    Differential Equation using Narrow-fit and Trimming" by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang
    
    Note: There is another version (2.0) of this function coming very soon which reduced cpu by caching the considered supports.
    
    Args:
        w (np.array): (feature matrix array) of shape (N, L) where L is the number of rhs features and n 
                      is the number of variables.
        b (np.array): (dynamic variable u_t) of shape (N, 1).
        b_tilda (np.array): error normalized vector b of shape (N, 1).
        w_tilda (np.array): error normalized matrix w of shape (N, L).
        k (np.array): maximum allowed sparsity level.
        s_rhs (np.array): scale vector s for rhs features, array of shape (L, ).
        s_lhs (np.float64): scale number for lhs feature (u_t).
        w_nar_tilda (np.array): \Tilda{W}_narrow, arrow normalized feature matrix, array of shape (NxNt, L).
        b_nar_tilda (np.array): \Tilda{b}_narrow, arrow normalized dynamic variable, array of shape (NxNt, 1).
        tau (float, optional): trimming score. Defaults to 0.05.

    Returns:
        np.ndarray: a support vector that stores the index of finalized candiate features for each variable.
    """
    display_tex = False
    support_list = {}
    cv_err_list = []
    w_column_norm = np.linalg.norm(w, axis=0).reshape(1, -1)
    column_normalized_w = w / w_column_norm
    print("\n Start finding support: ")
    for i in range(k):
        sys.stdout.write('\r')
        sys.stdout.write("[{:{}}] {:.1f}%".format("=" * i, k - 1,
                                                  (100 / (k - 1) * i)))
        sys.stdout.flush()
        support = subspace_persuit(column_normalized_w,
                                   b / np.linalg.norm(b, 2), i)
        if display_tex:
            print('SP -- Sparsity level is    :', i + 1)
            print('SP -- Support (Coef index) :', support)
        c_pred = narrow_fit(w_nar_tilda, b_nar_tilda, support, s_rhs, s_lhs)
        trim_score = compute_trim_score(w_column_norm, support, c_pred)
        if display_tex:
            print('The trimming scores are: ', trim_score)
        support_list[i] = support
        cv_err_list.append(
            compute_cross_validation_err_v2(support, w_tilda, b_tilda))
        if display_tex:
            print('The CV residual error is: ', cv_err_list)
        while len(support) > 1:
            idx_least_imp_feature = np.where(trim_score == trim_score.min())[0]
            if trim_score[idx_least_imp_feature[0]] > tau:
                break
            idx_least_imp_feature = idx_least_imp_feature[0]
            support = np.delete(support, idx_least_imp_feature)
            c_pred = narrow_fit(w_nar_tilda, b_nar_tilda, support, s_rhs,
                                s_lhs)
            trim_score = compute_trim_score(w_column_norm, support, c_pred)
            if display_tex:
                print('For a smaller support  ', support, ',')
                print('we have trimming score to be ', trim_score)
            cv_err_list[i] = compute_cross_validation_err_v2(
                support, w_tilda, b_tilda)
            support_list[i] = support
            if display_tex:
                print(
                    'We update the current smallest residual error for sparsity level',
                    i + 1, ' to be ', cv_err_list[i])
    cv_err_list = np.array(cv_err_list)
    cross_idx = np.argmin(cv_err_list)
    support_pred = support_list[int(cross_idx)]
    return support_pred


def compute_cross_validation_err_v2(support: np.array,
                                    w: np.array,
                                    b: np.array,
                                    iter_max=30) -> np.float64:
    """
    This function returns the cross validation error for the support of vector c in the least square problem Wc = b.
    Note: We compute cross-validation error 30 times and take mean + std as our final result for a stablized error.
    Args:
        support (np.array): a support vector that stores the index of candiate features for each variable.
        w (np.array): error normalized matrix w of shape (N, L).
        b (np.array): error normalized vector b of shape (N, 1).
        iter_max (int, optional): maximum number of iterations. Defaults to 30.

    Returns:
        np.float64: cross-validation error (modified version)
    """
    err_cross_accu = []
    for _ in range(iter_max):
        err_cross_accu.append(compute_cross_validation_error(support, w, b))
    err_cross_accu = np.array(err_cross_accu)
    err = np.mean(err_cross_accu) + np.std(err_cross_accu)
    return err


def compute_cross_validation_error(support: np.array,
                                   w: np.array,
                                   b: np.array,
                                   ratio=1 / 100) -> np.float64:
    """
    This function computes the cross validation error (from a random split w.r.t. ratio 1/100).

    Args:
        support (np.array): a support vector that stores the index of candiate features for each variable.
        w (np.array): error normalized matrix w of shape (N, L).
        b (np.array): error normalized vector b of shape (N, 1).
        ratio (_type_, optional): ratio between two partitions of w. Defaults to 1/100.

    Returns:
        np.float64: cross-validation error
    """
    n = len(b)
    inds = np.random.permutation(n)
    k = int(np.floor(n * ratio - 1))  # end of part 1
    w = w[inds, :]
    b = b[inds, :]
    coeff = np.zeros(w.shape[1])
    coeff[support] = least_square_adp(w[:k, support], b[:k]).flatten()
    e1 = np.linalg.norm(w[k:, :] @ coeff.reshape(-1, 1) - b[k:])
    coeff = np.zeros(w.shape[1])
    coeff[support] = least_square_adp(w[k:, support], b[k:]).flatten()
    e2 = np.linalg.norm(w[:k, :] @ coeff.reshape(-1, 1) - b[:k])
    return e1 * (1 - ratio) + e2 * ratio


def narrow_fit(w: np.array, b: np.array, support: np.array, s_rhs: np.array,
               s_lhs: np.float64) -> np.array:
    """
    This function perform narrow least square fit for the problem wb = c.
    Args:

        w (np.array): \Tilda{W}_narrow, arrow normalized feature matrix, array of shape (NxNt, L).
        b (np.array): \Tilda{b}_narrow, arrow normalized dynamic variable, array of shape (NxNt, 1).
        support (np.array): a support vector that stores the index of candiate features for each variable.
        s_rhs (np.array): scale vector s for rhs features, array of shape (L, ).
        s_lhs (np.float64): scale number for lhs feature (u_t).

    Returns:
        np.array: the coefficient values for a given support
    """
    c_pred = least_square_adp(w[:, support], b)
    c_pred = np.absolute(c_pred * s_rhs[support].reshape(-1, 1) / s_lhs)
    return c_pred


def compute_trim_score(w_column_norm: np.array, support: np.array,
                       c_pred: np.array) -> np.array:
    """
    This function computes the trimming score vector for given support and coefficient values
    Args:
        w_column_norm (np.array): column norm vector of feature matrix w
        support (np.array): a support vector that stores the index of candiate features for each variable.
        c_pred (np.array): a coefficient vector that stores the values of candiate features for each variable.

    Returns:
        np.array: a trimming score vector for each candidate feature
    """
    trim_score = w_column_norm.flatten()[support] * c_pred.flatten()
    trim_score = trim_score / np.max(trim_score)
    return trim_score


def subspace_persuit(phi: np.array, b: np.array, k: int) -> np.array:
    """
    This function finds the support for a sparse vector c such that phi * x = b where k numbers in c 
    are nonzero. 

    Args:
        phi (np.array): array of shape (N, L)
        b (np.array): array of shape (N, 1)
        k (int): sparse level

    Returns:
        np.array: array of shape (k,). a support vector that stores the index of candiate features for each variable.
    """
    itermax = 15
    n = len(phi[0])
    is_disp = 0
    b_t = np.transpose(b)
    cv = np.absolute(np.matmul(b_t, phi))
    cv_index = np.argsort(-cv).flatten()
    lda = cv_index[:(k + 1)]
    phi_lda = phi[:, lda]
    if is_disp:
        print(np.sort(lda))
    x = least_square_adp(np.matmul(np.transpose(phi_lda), phi_lda),
                         np.matmul(np.transpose(phi_lda), b).reshape(-1, 1))
    r = b - np.matmul(phi_lda, x)
    res = np.linalg.norm(r, 2)
    if res < 10**(-12):
        X = np.zeros(n)  # it should be a column
        X[lda] = np.transpose(x)
        supp = [i for i in range(phi.shape[1]) if X[i] != 0]
        return np.array(supp)
    usedlda = np.zeros(n)
    usedlda[lda] = 1
    for _ in range(itermax):
        res_old = res
        cv = np.absolute(np.matmul(np.transpose(r), phi))
        cv_index = np.argsort(-cv).flatten()
        Sga = np.union1d(lda, cv_index[:k + 1])
        Phi_Sga = phi[:, Sga]
        x_temp = least_square_adp(np.matmul(np.transpose(Phi_Sga), Phi_Sga),
                                  np.matmul(np.transpose(Phi_Sga), b))
        x_temp_index = np.argsort(-np.absolute(x_temp.flatten())).flatten()
        lda = Sga[x_temp_index[:k + 1]]
        phi_lda = phi[:, lda]
        usedlda[lda] = 1
        x = least_square_adp(np.matmul(np.transpose(phi_lda), phi_lda),
                             np.matmul(np.transpose(phi_lda), b))
        r = b - np.matmul(phi_lda, x)
        res = np.linalg.norm(r, 2)
        X = np.zeros(n)
        X[lda] = np.transpose(x)
        if res / res_old >= 1 or res < 10**(-12):
            supp = [i for i in range(phi.shape[1]) if X[i] != 0]
            return np.array(supp)


def find_highly_dynamic_region(dict_list: np.array,
                               idx_interesting_features: np.array,
                               s_b: np.array, num_of_bins: int) -> np.array:
    """
    This function returns the index of rows in the feature matrix where (x,t) is located in the 
    highly dynamic region.

    Args:
        dict_list(np.ndarray): array of shape (L, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each feature,
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain,
                               and column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. We take 0 or 1 for this value in WeakIdent.
        idx_interesting_features (np.array): array of shape (m, n + 2) where each row provides the a tag 
                                             for an interesting feature.
        s_b (np.array): scale matrix S (inlcuding lhs).
        num_of_bins (int): number of bins used in histogram.

    Returns:
        np.array: np.array of shape (m*,)
    """
    idxs = np.zeros(idx_interesting_features.shape[0])
    dict_list = dict_list.astype(int)
    for i in range(idx_interesting_features.shape[0]):
        idxs[i] = np.flatnonzero(
            npi.contains([idx_interesting_features[i, :]], dict_list))
    idxs = idxs.astype(int)
    scales_sum = np.sum(np.abs(s_b[:, idxs]), axis=1)
    ind = scales_sum.argsort()
    hist, _ = np.histogram(scales_sum, bins=num_of_bins)
    hist = np.cumsum(hist)
    y = np.block([1, hist])
    transition_group_idx = two_piece_fit_v2(hist)
    idx_highly_dynamic = ind[y[transition_group_idx] - 1:y[-1]]
    return idx_highly_dynamic


def find_idx_of_interesting_feature(n: int, dim: int,
                                    is_1d_ode: bool) -> np.array:
    """
    This function returns the tags for interesting features which are used in error-normalization
    for Narrow-fit.

    Args:
        n (int): number of variable
        dim (int): spatial dimension (1 or 2)
        is_1d_ode (bool): whether or not given data is 1d ode data.

    Returns:
        np.array: array which each row represents the tag of one interesting feature.
    """
    if is_1d_ode:
        idx = np.block([np.diag(np.ones(n)), np.zeros((n, 2))])
        idx = idx.astype(int)
    else:
        if n == 1 and dim == 1:
            idx = np.array([[2, 1, 0]])
        elif n == 1 and dim == 2:
            idx = np.array([[2, 1, 0, 0], [2, 0, 1, 0], [3, 1, 1, 0]])
        elif n == 2 and dim == 1:
            idx = np.array([[0, 2, 1, 0], [2, 0, 1, 0]])
        elif n == 2 and dim == 2:
            idx = np.array([[0, 2, 0, 1, 0], [2, 0, 0, 1, 0], [0, 2, 1, 0, 0],
                            [2, 0, 1, 0, 0], [2, 1, 1, 0, 0], [1, 2, 0, 1, 0]])
            # [3,0,1,1,0], \ # if not using cross derivative, then exclude this feature
            # [0,3,1,1,0]] \ # if not using cross derivative, then exclude this feature
    return idx


def least_square_adp(w: np.array, b: np.array) -> np.array:
    """
    This function returns a vector c where w * x = b by performing least square. In the case
    of w and b are constants, we simply perform divison.

    Args:
        w (np.array): array of shape (N, L)
        b (np.array): array of shape (N, 1)

    Returns:
        np.array: array of shape (L, 1)
    """
    if w.shape[0] == 1:
        x = b / w
    else:
        x, _, _, _ = np.linalg.lstsq(w, b, rcond=None)
    return x
