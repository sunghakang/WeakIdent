import numpy as np
import numpy_indexed as npi
from scipy.optimize import brentq as root
from typing import Tuple
from scipy.integrate import solve_ivp
import pandas as pd
"""

Utils functions of weakident to predict partial/ordinary different equations.

Copyright 2022, All Rights Reserved
Code by Mengyi Tang Rajchel
For Paper, "WeakIdent: Weak formulation for Identifying
Differential Equation using Narrow-fit and Trimming"
by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

"""


def reaction_diffusion_rhs(t, uvt, K22, d1, d2, beta, n, N):
    ut = uvt[:N].reshape(n, n, order='F')
    vt = uvt[N:].reshape(n, n, order='F')
    u = np.real(np.fft.ifft2(ut))
    v = np.real(np.fft.ifft2(vt))
    u3 = np.power(u, 3)
    v3 = np.power(v, 3)
    u2v = np.power(u, 2) * v
    uv2 = np.power(v, 2) * u
    utrhs = np.fft.fft2(u - u3 - uv2 + beta * u2v + beta * v3).reshape(
        -1, 1, order='F').flatten()
    vtrhs = np.fft.fft2(v - u2v - v3 - beta * u3 - beta * uv2).reshape(
        -1, 1, order='F').flatten()
    rhs = np.block([[-d1 * K22 * uvt[:N] + utrhs],
                    [-d2 * K22 * uvt[N:] + vtrhs]])
    return rhs.flatten()


def simulate_reaction_diffusion_equation():
    t = np.arange(0, 10, 0.0498)
    d1 = 0.1
    d2 = 0.1
    beta = 1.0
    L = 20
    n = 256
    N = n**2
    x2 = np.linspace(-L / 2, L / 2, n + 1)
    x = x2[:n]
    y = x.copy()
    kx = 2 * np.pi / L * np.block(
        [np.arange(n / 2).reshape(1, -1),
         np.arange(-n / 2, 0)])
    ky = kx.copy()
    X, Y = np.meshgrid(x, y)
    KX, KY = np.meshgrid(kx, ky)
    K2 = np.power(KX, 2) + np.power(KY, 2)
    K22 = K2.reshape(-1, 1, order='F').flatten()
    m = 2
    u_t = np.zeros((len(x), len(y), len(t)))
    v_t = np.zeros((len(x), len(y), len(t)))
    temp = X + 1j * Y
    temp2 = np.sqrt(np.power(X, 2) + np.power(Y, 2))
    u0 = np.tanh(temp2) * np.cos(m * np.angle(temp) - temp2)
    v0 = np.tanh(temp2) * np.sin(m * np.angle(temp) - temp2)
    u_t[:, :, 0] = u0
    v_t[:, :, 0] = v0
    uvt = np.block([[np.fft.fft2(u0).reshape(-1, 1, order='F')],
                    [np.fft.fft2(v0).reshape(-1, 1, order='F')]])

    yinit = uvt.flatten()
    sol = solve_ivp(reaction_diffusion_rhs, [t[0], t[-1]],
                    yinit,
                    args=(K22, d1, d2, beta, n, N),
                    method='RK45',
                    t_eval=t)

    for i in range(len(t) - 1):
        ut = sol.y[:N, i + 1].reshape(n, n, order='F')
        vt = sol.y[N:, i + 1].reshape(n, n, order='F')
        u_t[:, :, i + 1] = np.real(np.fft.ifft2(ut))
        v_t[:, :, i + 1] = np.real(np.fft.ifft2(vt))

    U = np.concatenate((np.array([u_t]), np.array([v_t])), axis=0)
    xs = np.array([x.reshape(-1, 1),
                   x.reshape(-1, 1),
                   t.reshape(1, -1)],
                  dtype=object)
    true_coefficients = np.array([
        np.array([[1., 0., 2., 0., 0., d1], [1., 0., 0., 2., 0., d1],
                  [1., 2., 0., 0., 0., -1.], [3., 0., 0., 0., 0., -1.],
                  [0., 3., 0., 0., 0., 1.], [2., 1., 0., 0., 0., 1.],
                  [1., 0., 0., 0., 0., 1.]]),
        np.array([[0., 1., 2., 0., 0., d2], [0., 1., 0., 2., 0., d2],
                  [0., 1., 0., 0., 0., 1.], [1., 2., 0., 0., 0., -1.],
                  [3., 0., 0., 0., 0., -1.], [0., 3., 0., 0., 0., -1.],
                  [2., 1., 0., 0., 0., -1.]])
    ],
                                 dtype=object)

    with open('dataset-Python/' + 'rxnDiff' + '.npy', 'wb') as f:
        np.save(f, U)
        np.save(f, xs)
        np.save(f, true_coefficients)


def load_data(filepath: str,
              filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function loads given data u, time sampling points
    and spatial sampling points xs, and true coefficients
    Assume given data has n_var varialbes on dim_x-D spatial domain.
    Args:
        filepath (str): path of dataset
        filename (str): the filename of given dataset

    Returns:
        u (np.array): array of shape (n_var,)
        xs (np.array): array of shape (dim_x + 1,)
        true_coefficients (np.array): array of shape (n_var,)
    """
    if filename == 'rxnDiff.npy':
        print('Simulating equation...')
        simulate_reaction_diffusion_equation()
        print('Simulation finished.')

    filename = filepath + "/" + filename
    with open(filename, "rb") as f:
        u = np.load(f, allow_pickle=True)
        xs = np.load(f, allow_pickle=True)
        true_coefficients = np.load(f, allow_pickle=True)
    return u, xs, true_coefficients


def rms(x: np.ndarray) -> np.float64:
    """
    Calculate the relative root mean square of given data.
    rms(x) = sqrt(sum((x_i - (max(x_i) + min(x_i))/2)^2))

    Note: we are using relative rms instead of rms to enforce noise since the given data may
          be centered around a large scale
    Args:
        x (np.ndarray): given clean data

    Returns:
        np.float64: relative root mean square
    """
    x_max = np.max(x)
    x_min = np.min(x)
    x_bar = x - (x_max + x_min) / 2
    return np.sqrt(np.mean(np.power(x_bar, 2)))


def add_noise(u: np.ndarray, sigma_nsr: float) -> np.ndarray:
    """
    This function add Gaussian noise with 0 mean and std = noise-sigal-ratio * relative-rms of given data
    on given clean data with n variables based on the relative root mean square of given data.

    Args:
        u (np.ndarray): given data, array of shape (n,)
        sigma_nsr (float): noise-signal-ratio

    Returns:
        u_hat (np.ndarray): noisy data, array of shape (n,)
    """
    n = u.shape[0]
    # random.seed(seed_number)
    if sigma_nsr > 0:
        u_hat = np.zeros_like(u)
        for k in range(n):
            std = rms(u[k])
            u_hat[k] = u[k] + \
                np.random.normal(0, sigma_nsr * std, size=u[0].shape)
    else:
        u_hat = u
    return u_hat


def build_library_tags(
    n: int,
    dim_x: int,
    max_dx: int,
    max_poly: int,
    use_cross_der: bool,
    lhs_idx: np.ndarray,
) -> np.ndarray:
    """
    build the library tags given data in dim_xD spatial domain with maximum monomial order max_py,
    maximum derivative order max_poly for n variables

    Args:
        n (int): total number of variables
        dim_x (int): spatial dimensions. In weakIdent, we consider dim_x = 1 or 2
        max_dx (int): maximum total order of partial derivatives
        max_poly (int): maximum total order of monomials
        use_cross_der (bool): whether allow partial derivatives
        lhs_idx (np.ndarray): the tag of left hand side features e.g. u_t, v_t

    Returns:
        dict_list(np.ndarray): array of shape (L + n, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain
                               column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. We take 0 or 1 in WeakIdent.
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
    build different combination of partial derivatives of features.
    Args:
        dim_x (int): spatial dimension. We consider 1 or 2 in WeakIdent.
        max_dx (int): maximum order of derivative allowed in the dictionary
        max_poly (int): maximum total order of monomials
        use_cross_der (bool): whether allow partial derivatives
        polys (list): [0,1,2,...,max_poly]

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
    build the tag of a list of polynomials. For example, the base of a feature is u^2v, this corresponds
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
    build a list of dictionary including the left hand side feature such as u_t and v_t

    Args:
        lhs_tag (np.ndarray): the tag of left hand side feature (dynamic variable)
        betas (list): a list with shape (_, n), a list of bases of each feature, each column specifies the
                      degree of polynomial for each variable. The sum of each row <= max_poly. Here n is the
                      number of variables
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
    vector
    Args:
        n (int): number of variables
        dim (int): spatial dimensions. In weakIdent, we consider dim = 1 or 2
        max_dx (int): maximum total order of partial derivatives
        max_poly (int): maximum total order of monomials
        use_cross_der (bool): whether allow partial derivatives
        true_coefficients: array of shape (n_var,)

    Returns:
        dict_list(np.ndarray): array of shape (L + n, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain
                               column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. We take 0 or 1 in WeakIdent.
        lhs_ind(np.ndarray): shape of (n,) , row index of left hand side feature 
        rhs_ind(np.ndarray): shape of (L,) , row index of right hand side features
        c_true(np.ndarray):  shape of (L , 2) where L is the total number of features on the rhs of the equation
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


def compute_true_coefficient_vector(n: int, true_coefficients: np.array,
                                    lhs_tags: np.array, dict_list: np.array,
                                    lhs_ind: np.array) -> np.array:
    """This function compute the true sparse coefficient vector for given data. (tags and values are provided)

    Args:
        n (int): number of variables
        true_coefficients: array of shape (n_var,)
        lhs_tags (np.ndarray): the tag of left hand side feature (dynamic variable)
        dict_list(np.ndarray): array of shape (L + n, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain
                               column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. We take 0 or 1 in WeakIdent.
        lhs_ind(np.ndarray): shape of (n,) , row index of left hand side feature 

    Returns:
        c_true(np.ndarray):  shape of (L , 2) where L is the total number of features on the rhs of the equation
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


def compute_test_function_para(u: np.array, xs: np.array,
                               max_dx: int) -> Tuple[int, int, int, int]:
    """
    compute the size of test function and the order p in the test function 
    Note: In each spatial domain, the size of integration region of a test function is 2m_x + 1.
          In the temporal domain, the size of integration region of a test fucntino is 2m_t + 1.
          The localized test function is phi(x,t) = (1-x^2)^p_x* (1-t^2)^p_t.

    Remark: We follow the method in "Weak SINDy" to find proper parameters for test functions. The script is 
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


def circshift(l: list, d: int) -> list:
    """
    compute the permutation of a list l by shifting all the elements to the left d units
    Args:
        l (list):  given list
        d (int):   number of shifting

    Returns:
        list: shifted list
    """
    if d == 0:
        return l
    else:
        return l[-d:] + l[:-d]


def drange(x: np.array) -> np.float64:
    """
    compute the scale of range of given data x

    Args:
        x (np.array): given data

    Returns:
        np.float64: max(x) - min(x)
    """
    return np.max(x) - np.min(x)


def find_corner_pts(u: np.array, xs: np.array) -> list:
    """
    This function will find a transition frequency mode of given data in each dimension (spatial / temporal)
    Args:
        u (np.array): array of shape (n_x, n_y, n_t) or (n_x, n_t) for one variable in a given system
        xs (np.array): array of shape (dim_x + 1,)

    Remark: We follow the method in "Weak SINDy" to find transition frequency mode. The script is modified from 
            Matlab code for Paper, "Weak SINDy for Partial Differential Equations" by D. A. Messenger and D. M. Bortz

    Returns:
        list: [k_1^*, k_2^*, k_3^*, ... ] where each k^* represent a corner frequency mode, the order of this 
              list should be for x, y(if applicable), t
    """
    if u.shape[0] == 1:
        dim = 1
    else:
        dims = u.shape
        dim = len(dims)
    corners = []
    for d in range(dim):
        if dim == 1:
            shift = [0, 1]
        else:
            shift = circshift([i for i in range(dim)], 1 - (d + 1))
        x = xs[d].reshape(-1, 1)
        L = len(x)
        range_of_x = np.absolute(x[-1] - x[0])
        wn = ((np.arange(L) - L // 2) * 2 * np.pi / range_of_x).reshape(-1, 1)
        NN = (len(wn) + 1) // 2
        xx = wn[:NN]
        if dim >= 2:
            ufft = np.absolute(
                np.fft.fftshift(np.fft.fft(u.transpose(tuple(shift)), axis=0)))
        else:
            ufft = np.absolute(np.fft.fftshift(np.fft.fft(u)))
        if dim == 3:
            ufft = ufft.reshape((L, -1))
        if dim >= 2:
            ufft = np.mean(ufft, axis=1).reshape((-1, 1))
        else:
            ufft = ufft.reshape((-1, 1))
        ufft = np.cumsum(ufft).reshape((-1, 1))
        ufft = ufft[:(L + 1) // 2]
        errs = np.zeros(NN - 6)
        for k in range(4, NN - 2):
            subinds1 = np.arange(k)
            subinds2 = np.arange(k - 1, NN)
            ufft_av1 = ufft[subinds1].reshape(-1, 1)
            ufft_av2 = ufft[subinds2].reshape(-1, 1)
            m1 = drange(ufft_av1) / drange(xx[subinds1])
            m2 = drange(ufft_av2) / drange(xx[subinds2])
            L1 = np.min(ufft_av1) + m1 * (xx[subinds1] - xx[0])
            L2 = np.max(ufft_av2) + m2 * (xx[subinds2] - xx[-1])
            errs[k - 4] = np.sqrt(
                np.sum(np.power((L1 - ufft_av1) / ufft_av1, 2)) +
                np.sum(np.power((L2 - ufft_av2) / ufft_av2, 2)))
        idx = np.argmin(errs)
        corners.append(NN - idx - 4)
    return corners


def compute_err_res(W: np.array, c: np.array, b: np.array) -> np.float64:
    """
    compute the residual error of a system

    Args:
        W (np.array): feature matrix, array of shape (N, L)
        c (np.array): predicted sparse coefficient vector, array of shape (L, n)
        b (np.array): dynamic variables, array of shape (N, n)

    Returns:
        np.float64: residual error
    """
    err = np.linalg.norm(W @ c - b) / np.linalg.norm(b)
    return err


def compute_tpr_ppv(c_true: np.array,
                    c: np.array) -> Tuple[np.float64, np.float64]:
    """
    compute the True Positive Rate (TPR) and Positive Predictive Value (PPV) of 
    predicted sparse coefficient vector c
    Note: n is the number of variables in a given system.

    Args:
        c_true (np.array): array of shape (L,n)
        c (np.array): array of shape (L,n)

    Returns:
        np.float64: tpr
        np.float64: ppv
    """
    p = np.sum(c != 0)
    tp = np.sum((c_true * c) != 0)
    fp = p - tp
    t = np.sum(c_true != 0)
    fn = t - tp
    tpr = tp / (tp + fn)
    ppv = tp / (tp + fp)
    return tpr, ppv


def compute_err_inf(c_true: np.array, c: np.array) -> np.float64:
    """
    compute the l_infty norm of error of predicted sparse coefficient vector c
    Note: n is the number of variables in a given system.

    Args:
        c_true (np.array): array of shape (L,n)
        c (np.array): array of shape (L,n)

    Returns:
        np.float64: relative l_infty norm error
    """
    if np.sum((c * c_true) != 0) == 0:
        err = 0
    else:
        err = np.max(
            np.abs(c[(c * c_true) != 0] / c_true[(c * c_true) != 0] - 1))
    return err


def compute_err_l2(c_true: np.array, c: np.array) -> np.float64:
    """
    compute the relative l2 norm error of predicted coefficient vector c
    Note: n is the number of variables in a given system.

    Args:
        c_true (np.array): array of shape (L,n)
        c (np.array): array of shape (L,n)

    Returns:
        np.float64: relative l_2 norm error
    """
    err = np.linalg.norm(c - c_true) / np.linalg.norm(c_true)
    return err


def least_square_adp(A: np.array, b: np.array) -> np.array:
    """
    This script solve least square problem Ax = b. In the case of A and b both being a constant number,
    we simply perform division x = b / A

    Args:
        A (np.array): array of shape (N, L)
        b (np.array): array of shape (N, n)

    Returns:
        np.array: array of shape (L, n)
    """

    if A.shape[0] == 1:
        x = b / A
    else:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x


def two_piece_fit_v2(y: np.array) -> np.int64:
    """
    This function returns the index i as a breaking point for a 2 piece linear fit for y vector.
    This function is used to find a transition point that partition given data into highly dynamic
    region and other region.

    Args:
        y (np.array): array of shape (num, )

    Returns:
        np.int64: index of breaking point
    """
    NN = len(y)
    y = y.reshape(-1, 1)
    xx = np.arange(NN)
    errs = np.zeros(NN - 2)
    for k in range(2, NN):
        subinds1 = np.arange(k)
        subinds2 = np.arange(k - 1, NN)
        y_av1 = y[subinds1].reshape(-1, 1)
        y_av2 = y[subinds2].reshape(-1, 1)
        m1 = drange(y_av1) / drange(xx[subinds1])
        m2 = drange(y_av2) / drange(xx[subinds2])
        L1 = np.min(y_av1) + m1 * (xx[subinds1] - xx[0])
        L2 = np.max(y_av2) + m2 * (xx[subinds2] - xx[-1])
        errs[k - 2] = np.sqrt(
            np.sum(np.power((L1 - y_av1) / y_av1, 2)) +
            np.sum(np.power((L2 - y_av2) / y_av2, 2)))
    idx = np.argmin(errs)
    idx_highly_dynamic = idx
    return idx_highly_dynamic


def build_tag_de(n: int, dim_x: int, dict_list: np.ndarray,
                 lhs_ind: np.ndarray) -> Tuple[list, list]:
    """
    This function will build str tag for each lhs and rhs feature for pdes or multi-dimensioinal odes

    Args:
        n (int): number of variable
        dim_x (int): number of spatial dimension
        dict_list(np.ndarray): array of shape (L + n, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain
                               column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. We take 0 or 1 in WeakIdent.
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


def build_tag_1d_ode(n: int, dict_list: np.ndarray,
                     lhs_ind: np.ndarray) -> Tuple[list, list]:
    """
    This function will build str tag for each lhs and rhs feature for 1d odes

    Args:
        n (int): number of variable
        dict_list(np.ndarray): array of shape (L + n, n + 2). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain
                               column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. We take 0 or 1 in WeakIdent.
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
                               column 1 - column n represents the degree of monomial for each variable
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain
                               column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. We take 0 or 1 in WeakIdent.
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