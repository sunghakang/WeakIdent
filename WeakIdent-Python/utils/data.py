import numpy as np 
from scipy.integrate import solve_ivp
from typing import Tuple
from utils.calculations import rms
'''
This file stores all the functions needed to simulate or load data. 

Code author: Mengyi Tang Rajchel.

I provided two function to simulate reaction-diffusion type of equation and Lottera-Volterra Equation.
For other equations, datasets are already provided in dataset-Python.

'''
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
        print('Simulating equation... \n this is going to take a 1-3 minutes')
        simulate_reaction_diffusion_equation()
        print('Simulation finished.')
    elif filename == 'LotkaVolterra2D.npy':
        print('Simulating equation... \n this is going to take a few seconds.')
        simulate_lotka_volterra_2d_equation()
        print('Simulation finished.')

    filename = filepath + "/" + filename
    with open(filename, "rb") as f:
        u = np.load(f, allow_pickle=True)
        xs = np.load(f, allow_pickle=True)
        true_coefficients = np.load(f, allow_pickle=True)
    return u, xs, true_coefficients

def simulate_reaction_diffusion_equation():
    """
    This function simulates reaction diffusion type of equation and save the dataset into dataset-Python.
    """
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


def reaction_diffusion_rhs(t: np.array, uvt: np.array, K22: np.array, d1: float, d2: float, beta: float, n: int, N: int) -> np.array:
    """
    This function calculates the right-hand-side of reaction diffusion type of equation.
    Args:
        t (np.array): spacing in time
        uvt (np.array): values of u and v at previous timestamp
        K22 (np.array): an operator used to compute laplace of u and laplace of v in frequency domain
        d1 (float): diffusion parameter for u
        d2 (float): diffusion parameter for v
        beta (flaot): paramter in the reaction diffusion type of equation
        n (int): number of points in spatial direction x or y
        N (int): number of points in spatial domain (x and y). N = n * n

    Returns:
        rhs (np.array): values of u and v at current timestamp
    """
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


def simulate_lotka_volterra_2d_equation():
    """
    This function simulate a dataset for lotka_volterra_2d equation.
    """
    max_t = 1
    dt = 0.01
    file_name = 'LotkaVolterra2D'
    t = np.linspace(0, max_t, int(max_t / dt + 1))
    r, s = 2, 2
    alpha_ab, alpha_ba = 0.05, 0.05
    L = 20
    n = 256
    N = n**2
    x2 = np.linspace(-L / 2, L / 2, n + 1)
    x = x2[:n]
    y = x.copy()
    u_t = np.zeros((len(x), len(y), len(t)))
    v_t = np.zeros((len(x), len(y), len(t)))
    u0 = 1 + np.random.rand(n,n) - 0.5
    v0 = 1 + np.random.rand(n,n) - 0.5
    u0 = u0 - np.min(u0)
    v0 = v0 - np.min(v0)
    u0 = u0 / np.max(u0)
    v0 = v0 / np.max(v0)
    u_t[:, :, 0] = u0
    v_t[:, :, 0] = v0
    uvt = np.block([[u0.reshape(-1, 1, order='F')],
                    [v0.reshape(-1, 1, order='F')]])
    yinit = uvt.flatten()
    sol = solve_ivp(lotka_volterra_rhs, [t[0], t[-1]],
                    yinit,
                    args=( r, s, alpha_ab, alpha_ba,  N),
                    method='RK45',
                    t_eval=t)
    for i in range(len(t) - 1):
        ut = sol.y[:N, i + 1].reshape(n, n, order='F')
        vt = sol.y[N:, i + 1].reshape(n, n, order='F')
        u_t[:, :, i + 1] = ut
        v_t[:, :, i + 1] = vt
    U = np.concatenate((np.array([u_t]), np.array([v_t])), axis=0)
    xs = np.array(
        [x.reshape(-1, 1), x.reshape(-1, 1),
        t.reshape(1, -1)], dtype=object)
    true_coefficients = np.array([
        np.array([[1., 0., 0., 0., 0., r],  [2., 0., 0., 0., 0., -s],
                [1., 1., 0., 0., 0., -(s+alpha_ab)]]),
        np.array([[0., 1., 0., 0., 0., r],  [0., 2., 0., 0., 0., -s],
                [1., 1., 0., 0., 0., -(s+alpha_ba)]])
    ],
        dtype=object)
    with open('dataset-Python/' + file_name + '.npy', 'wb') as f:
        np.save(f, U)
        np.save(f, xs)
        np.save(f, true_coefficients)
    return

def lotka_volterra_rhs(t, uvt,  r:float, s:float, alpha_ab: float, alpha_ba: float,  N: int) -> np.array:
    """
    This function calculates the right-hand-side of lotka_volterra type of equation.
    Args:
        t (np.array): spacing in time.
        uvt (np.array): values of u and v at previous timestamp.
        r, s, alpha_ab, alpha_ba (flaot): paramter in the reaction diffusion type of equation.
        N (int): number of points in spatial domain (x and y). 

    Returns:
        rhs (np.array): values of u and v at current timestamp.
    """
    u = uvt[:N]
    v = uvt[N:]
    u2 = np.power(u, 2)
    v2 = np.power(v, 2)
    uv = u * v
    utrhs = r * u - s * u2 - (s + alpha_ab) * uv
    vtrhs = r * v - s * v2 - (s + alpha_ba) * uv
    rhs = np.block([[ utrhs],
                    [ vtrhs]])
    return rhs.flatten()


def add_noise(u: np.ndarray, sigma_nsr: float) -> np.ndarray:
    """
    This function add Gaussian noise with 0 mean and std = noise-sigal-ratio * relative-rms of given data
    on given clean data with n variables based on the relative root mean square of given data.

    Args:
        u (np.ndarray): given data, array of shape (n,).
        sigma_nsr (float): noise-signal-ratio.

    Returns:
        u_hat (np.ndarray): noisy data, array of shape (n,).
    """
    n = u.shape[0]
    if sigma_nsr > 0:
        u_hat = np.zeros_like(u)
        for k in range(n):
            std = rms(u[k])
            u_hat[k] = u[k] + \
                np.random.normal(0, sigma_nsr * std, size=u[0].shape)
    else:
        u_hat = u
    return u_hat