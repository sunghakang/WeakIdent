import numpy as np
import numpy_indexed as npi
from scipy.optimize import brentq as root
from typing import Tuple
from utils.calculations import find_corner_pts

"""
This file stores some util function used in feature building (tags and values).

Code author: Mengyi Tang Rajchel.
    
Copyright 2022, All Rights Reserved
"""

def build_library_tags(
    n: int,
    dim_x: int,
    max_dx: int,
    max_poly: int,
    use_cross_der: bool,
    lhs_idx: np.ndarray,
) -> np.ndarray:
    """
    This function builds the library tags given data in dim_xD spatial domain with maximum monomial order max_py,
    maximum derivative order max_poly for n variables

    Args:
        n (int): total number of variables.
        dim_x (int): spatial dimensions. This number is considered to be 1 or 2.
        max_dx (int): maximum total order of partial derivatives.
        max_poly (int): maximum total order of monomials.
        use_cross_der (bool): whether allow partial derivatives.
        lhs_idx (np.ndarray): the tag of left hand side features e.g. u_t, v_t.

    Returns:
        dict_list(np.ndarray): array of shape (L + n, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable,
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain, and
                               column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. 
    """
    polys = [i for i in range(max_poly + 1)]
    betas = build_monomials_list(n, polys)
    alphas = build_partial_der_list(dim_x, max_dx, max_poly, use_cross_der,
                                    polys)
    dict_list = build_dictionary_list(lhs_idx, betas, alphas)
    return dict_list


def build_partial_der_list(dim_x: int, max_dx: int, max_poly: int,
                           use_cross_der: bool, polys: list) -> list:
    """
    This function builds different combination of partial derivatives of features.
    Args:
        dim_x (int): spatial dimension. This number is considered to be 1 or 2 in WeakIdent.
        max_dx (int): maximum order of derivative allowed in the dictionary.
        max_poly (int): maximum total order of monomials.
        use_cross_der (bool): whether allow partial derivatives.
        polys (list): [0,1,2,...,max_poly].

    Returns:
        alphas(list): a list with shape (_, m) consiting different combination of partial derivatives of a feature
                      m = dim_x + 1. For example, if the given data is on 2D spatial domain, then m = 3.
    """
    alphas = []
    if use_cross_der:
        for p in range(1, max_poly + 2):
            if dim_x == 3:
                for i in range(polys[p - 1] + 1):
                    for j in range(0, polys[p - 1] + 1 - i):
                        k = polys[p - 1] - i - j
                        alphas.append([i, j, k])
            elif dim_x == 2:
                for i in range(polys[p - 1] + 1):
                    j = polys[p - 1] - i
                    alphas.append([i, j])
            elif dim_x == 1:
                alphas.append[[polys[p - 1]]]
    else:
        alphas = [[0 for i in range(dim_x)]]
        for i in range(dim_x):
            for j in range(max_dx):
                temp = [0 for _ in range(dim_x)]
                temp[i] = j + 1
                alphas.append(temp)
    return alphas


def build_monomials_list(n: int, polys: list) -> list:
    """
    This function builds the tag of a list of polynomials. For example, the base of a feature is u^2v, this corresponds
    to a row (2,1). Note that the sum of degree of polynomials (sum of each row) can not be larger than
    max_poly.

    Args:
        n (int): number of variables
        polys (list): [0,1,2,...,max_poly]

    Returns:
        list: (_, n), a list of bases of each feature, each column specifies the degree of polynomial
        for each variable. The sum of each row <= max_poly
    """
    betas = []
    for p in range(1, polys[-1] + 1):
        if n == 1:
            betas.append([polys[p]])
        elif n == 2:
            for i in range(0, polys[p] + 1):
                j = polys[p] - i
                betas.append([i, j])
        elif n == 3:
            for i in range(0, polys[p] + 1):
                for j in range(0, polys[p] - i + 1):
                    k = polys[p] - i - j
                    betas.append([i, j, k])

    return betas


def build_dictionary_list(lhs_tag: np.ndarray, betas: list,
                          alphas: list) -> np.array:
    """
    build a list of dictionary including the left hand side feature such as u_t and v_t.

    Args:
        lhs_tag (np.ndarray): the tag of left hand side feature (dynamic variable).
        betas (list): a list with shape (_, n), a list of bases of each feature, each column specifies the
                      degree of polynomial for each variable. The sum of each row <= max_poly. Here n is the
                      number of variables.
        alphas (list): a list with shape (_, m) consiting different combination of partial derivatives of a feature
                      m = dim_x + 1. For example, if the given data is on 2D spatial domain, then m = 3.

    Returns:
        np.array: array of shape (L + n, n + m). A list of tags of each feature in the library.
    """
    n1 = len(betas[0])
    n2 = len(alphas[0])
    dict_list = [[0 for i in range(n1 + n2 + 1)]]
    for i in range(len(betas)):
        for j in range(len(alphas)):
            dict_list.append(betas[i] + alphas[j] + [0])
    dict_list = np.array(dict_list)
    dict_list = np.block([[dict_list], [lhs_tag]])
    dict_list = dict_list[dict_list[:, dict_list.shape[1] - 1].argsort()]
    for i in range(dict_list.shape[1] - 2, -1, -1):
        dict_list = dict_list[dict_list[:, i].argsort(kind="mergesort")]
    return dict_list


def build_feature_vector_tags(
        n: int, dim: int, max_dx: int, max_poly: int, use_cross_der: bool,
        true_coefficients: np.ndarray, is_1d_ode: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    this function build the dictionary of features, find the row index of lhs feature in
    this diction, and rearrange the true coefficient in terms of a sparse coefficient
    vector.
    Args:
        n (int): number of variables.
        dim (int): spatial dimensions.  (suggested value: dim = 1 or 2 ).
        max_dx (int): maximum total order of partial derivatives.
        max_poly (int): maximum total order of monomials.
        use_cross_der (bool): whether allow partial derivatives.
        true_coefficients: array of shape (n_var,).

    Returns:
        dict_list(np.ndarray): array of shape (L + n, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable,
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain, and
                               column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. 
        lhs_ind(np.ndarray): shape of (n,) , row index of left hand side feature. 
        rhs_ind(np.ndarray): shape of (L,) , row index of right hand side features.
        c_true(np.ndarray):  shape of (L , 2) where L is the total number of features on the rhs of the equation.
    """
    lhs_tags = set_lhs_idx(is_1d_ode, n, dim)
    dict_list = build_library_tags(n, dim, max_dx, max_poly, use_cross_der,
                                   lhs_tags)
    lhs_ind = np.zeros(lhs_tags.shape[0])
    for i in range(lhs_tags.shape[0]):
        lhs_ind[i] = np.flatnonzero(npi.contains([lhs_tags[i, :]], dict_list))
    lhs_ind = lhs_ind.astype(int)
    c_true = compute_true_coefficient_vector(n, true_coefficients, lhs_tags,
                                             dict_list, lhs_ind)
    rhs_ind = np.array(
        [i for i in range(dict_list.shape[0]) if i not in lhs_ind])
    return dict_list, lhs_ind, rhs_ind, c_true

def set_lhs_idx(is_1d_ode: bool, num_of_u: int, dim_x: int) -> np.array:
    """This function find row index left-hand-side variable tag in the dictionary.

    Args:
        is_1d_ode (bool): whether or not given data is 1d ode data.
        num_of_u (int): number of variables
        dim_x (int): spatial dimension of given data.

    Returns:
        np.array: tag(s) for left-hand-side variable(s).
    """
    if is_1d_ode:
        lhsIdx = np.block([
            np.diag(np.ones(num_of_u)),
            np.zeros((num_of_u, 1)),
            np.ones((num_of_u, 1))
        ])
    else:
        lhsIdx = np.block([
            np.diag(np.ones(num_of_u)),
            np.zeros((num_of_u, dim_x)),
            np.ones((num_of_u, 1))
        ])
    return lhsIdx


def compute_true_coefficient_vector(n: int, true_coefficients: np.array,
                                    lhs_tags: np.array, dict_list: np.array,
                                    lhs_ind: np.array) -> np.array:
    """This function compute the true sparse coefficient vector for given data. (tags and values are provided)

    Args:
        n (int): number of variables
        true_coefficients: array of shape (n_var,)
        lhs_tags (np.ndarray): the tag of left hand side feature (dynamic variable)
        dict_list(np.ndarray): array of shape (L + n, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable.
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain, and 
                               column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain.  
        lhs_ind(np.ndarray): shape of (n,) , row index of left hand side feature. 

    Returns:
        c_true(np.ndarray):  shape of (L , 2) where L is the total number of features on the rhs of the equation.
    """
    dict_list_without_lhs = np.delete(dict_list, lhs_ind, 0)
    c_true = np.zeros((dict_list.shape[0] - n, lhs_tags.shape[0]))
    for i in range(n):
        dict_of_true_supp = true_coefficients[i][:, :-1].astype("float")
        vals_of_true_coef = true_coefficients[i][:, -1]
        for j in range(true_coefficients[i].shape[0]):
            temp_ind = np.flatnonzero(
                npi.contains([dict_of_true_supp[j, :]], dict_list_without_lhs))
            c_true[temp_ind, i] = vals_of_true_coef[j]
    return c_true



def build_eqn_str(tags_lhs: list, tags_rhs: list, c: np.ndarray) -> list:
    """
    This function builds string for an equation using tags of lhs and rhs features.

    Args:
        tags_lhs (list): a list of lhs features.
        tags_rhs (list): a list of rhs feature.
        c (np.ndarray): a list of coefficient values.

    Returns:
        list: a list of equations.
    """
    n = c.shape[1]
    eqn_strs = []
    for i in range(n):
        eqn_str = tags_lhs[i] + ' ='
        for l in range(c.shape[0]):
            coe = c[l, i]
            if coe == 0:
                continue
            if tags_rhs[l] == '1':
                eqn_str += ' ' + str(coe)
            else:
                eqn_str += ' ' + ["-", "+"][coe > 0] + ' ' + \
                    str(round(abs(coe), 3)) + ' ' + tags_rhs[l]
        eqn_strs.append(eqn_str)
    return eqn_strs


def build_feature_latex_tags(num_of_variables: int, dim_x: int,
                             is_1d_ode: bool, dict_list: np.array,
                             lhs_ind: np.array):
    """
    This function will build str tag for each lhs and rhs feature.

    Args:
        num_of_variables (int): number of variable
        dim_x (int): number of spatial dimension
        is_1d_ode (bool): whether or not given data is 1d ode data.
        dict_list(np.ndarray): array of shape (L + n, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable,
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain, and
                               column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. 
        lhs_ind(np.ndarray): shape of (n,)

    Returns:
        tags_lhs(list): a list of string tag for lhs feature(s)
        tags_rhs(list): a list of string tag for rhs feature(s)
    """
    if is_1d_ode:
        tags_lhs, tags_rhs = build_tag_1d_ode(num_of_variables, dict_list,
                                              lhs_ind)
    else:
        tags_lhs, tags_rhs = build_tag_de(num_of_variables, dim_x, dict_list,
                                          lhs_ind)

    return tags_lhs, tags_rhs


def build_tag_1d_ode(n: int, dict_list: np.ndarray,
                     lhs_ind: np.ndarray) -> Tuple[list, list]:
    """
    This function will build str tag for each lhs and rhs feature for 1d odes

    Args:
        n (int): number of variable
        dict_list(np.ndarray): array of shape (L + n, n + 2). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain,
                               column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. 
        lhs_ind(np.ndarray): shape of (n,)

    Returns:
        tags_lhs(list): a list of string tag for lhs feature(s)
        tags_rhs(list): a list of string tag for rhs feature(s)
    """
    base_names = [['', 'x'] + ['x^' + str(i) for i in range(2, 10)],
                  ['', 'y'] + ['y^' + str(i) for i in range(2, 10)],
                  ['', 'z'] + ['z^' + str(i) for i in range(2, 10)]]
    lhs_names = ['dx/dt', 'dy/dt', 'dz/dt']
    tags_lhs = lhs_names[:n]
    tags_rhs = []
    dict_rhs = np.ndarray.tolist(lhs_ind)
    for i in range(0, dict_list.shape[0]):
        if i not in dict_rhs:
            if np.sum(dict_list[i, :]) == 0:
                tags_rhs.append('1')
            else:
                base = ''
                for j in range(n):
                    base += base_names[j][int(dict_list[i, j])]
                tags_rhs.append(base)
    return tags_lhs, tags_rhs


def build_tag_de(n: int, dim_x: int, dict_list: np.ndarray,
                 lhs_ind: np.ndarray) -> Tuple[list, list]:
    """
    This function will build str tag for each lhs and rhs feature for pdes or multi-dimensioinal odes

    Args:
        n (int): number of variable
        dim_x (int): number of spatial dimension
        dict_list(np.ndarray): array of shape (L + n, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable,
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain, and
                               column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain.  
        lhs_ind(np.ndarray): shape of (n,)

    Returns:
        tags_lhs(list): a list of string tag for lhs feature(s)
        tags_rhs(list): a list of string tag for rhs feature(s)
    """
    base_names = [['', 'u'] + ['u^' + str(i) for i in range(2, 10)],
                  ['', 'v'] + ['v^' + str(i) for i in range(2, 10)]]
    der_names = [['x' * i for i in range(7)], ['x' * i for i in range(7)]]
    lhs_names = ['u_t', 'v_t']
    tags_lhs = lhs_names[:n]
    tags_rhs = []
    dict_rhs = np.ndarray.tolist(lhs_ind)
    for i in range(0, dict_list.shape[0]):
        if i not in dict_rhs:
            if np.sum(dict_list[i, :]) == 0:
                tags_rhs.append('1')
            else:
                base = ''
                for j in range(n):
                    base += base_names[j][int(dict_list[i, j])]
                if np.sum(dict_list[i, n:n + dim_x]) > 0:
                    if len(base) > 1:
                        base = '(' + base + ')'
                    base += "_{"
                    for k in range(n, n + dim_x):
                        der = int(dict_list[i, k])
                        if der > 0:
                            base += der_names[k - n][der]
                    base += '}'
                tags_rhs.append(base)
    return tags_lhs, tags_rhs

def compute_test_function_para(u: np.array, xs: np.array,
                               max_dx: int) -> Tuple[int, int, int, int]:
    """
    This function computes the size of test function and the order p in the test function. 
    Note: In each spatial domain, the size of integration region of a test function is 2m_x + 1.
          In the temporal domain, the size of integration region of a test fucntino is 2m_t + 1.
          The localized test function is phi(x,t) = (1-x^2)^p_x* (1-t^2)^p_t.

    Remark: The method follows "Weak SINDy" to find proper parameters for test functions. The script is 
            modified from Matlab code for Paper, "Weak SINDy for Partial 
            Differential Equations" by D. A. Messenger and D. M. Bortz

    Args:
        u (np.array): array of shape (n_var,)
        xs (np.array): array of shape (dim_x + 1,)
        max_dx (int): maximum total order of partial derivatives

    Returns:
        Tuple[int, int, int, int]: m_x, m_t, p_x, p_t
    """
    m_x = []
    m_t = 0
    p_x, p_t, maxd = 0, 0, 1
    if u[0].shape[0] > 1:
        T = max(xs[-1].shape)
        maxd = xs.shape[0] - 1
    tau, tau_hat = 1e-10, 2
    L = max(xs[0].shape)
    for n in range(u.shape[0]):
        corners = find_corner_pts(u[n], xs)
        for d in range(maxd):
            N = max(xs[d].shape)
            k = corners[d]
            mstar1 = np.sqrt(3) * N / np.pi / 2 / k * tau_hat
            mstar2 = (
                1 / np.pi * tau_hat * N / 2 / k *
                np.sqrt(np.log(np.power(np.exp(1), 3) / np.power(tau, 8))))
            mnew = root(
                lambda m: np.log((2 * m - 1) / m**2) *
                (4 * np.pi**2 * k**2 * m**2 - 3 * N**2 * tau_hat**2
                 ) - 2 * N**2 * tau_hat**2 * np.log(tau),
                mstar1,
                mstar2,
            )
            if mnew > N / 2 - 1:
                mnew = N / 2 / k
            m_x.append(mnew)
            L = min(L, N)
        k = corners[-1]
        if u[0].shape[0] > 1:
            mnew = root(
                lambda m: np.log((2 * m - 1) / m**2) *
                (4 * np.pi**2 * k**2 * m**2 - 3 * T**2 * tau_hat**2
                 ) - 2 * T**2 * tau_hat**2 * np.log(tau),
                1,
                2 / np.sqrt(tau),
            )
            if mnew > T / 2 - 1:
                mnew = T / 2 / k
            m_t = mnew
    if len(m_x) == 1:
        m_x = min((L - 1) // 2, np.ceil(m_x[0]))
    else:
        m_x = min((L - 1) // 2, np.ceil(np.mean(np.asarray(m_x))))
    p_x = max(max_dx + 2,
              np.floor(np.log(tau) // np.log(1 - np.power(1 - 1 / m_x, 2))))
    if u[0].shape[0] > 1:
        m_t = min((T - 1) // 2, np.ceil(m_t))
        p_t = max(1 + 2,
                  np.floor(np.log(tau) / np.log(1 - np.power(1 - 1 / m_t, 2))))
    return int(m_x), int(m_t), int(p_x), int(p_t)