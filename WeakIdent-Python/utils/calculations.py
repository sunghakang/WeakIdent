import numpy as np 
from typing import Tuple


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
