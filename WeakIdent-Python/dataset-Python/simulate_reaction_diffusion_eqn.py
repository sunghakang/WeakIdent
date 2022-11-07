import numpy as np
# import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
uvt = np.block([[np.fft.fft2(u0).reshape(-1,1, order = 'F')], \
    [np.fft.fft2(v0).reshape(-1,1, order = 'F')]])


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
    rhs = np.block([[-d1 * K22 * uvt[:N]+ utrhs ]  , \
        [-d2 * K22 * uvt[N:] + vtrhs]])
    return rhs.flatten()


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

# U = np.array([np.array(u_t), np.array(v_t)], dtype = object)
U = np.concatenate((np.array([u_t]), np.array([v_t])), axis=0)
xs = np.array(
    [x.reshape(-1, 1), x.reshape(-1, 1),
     t.reshape(1, -1)], dtype=object)
true_coefficients = np.array([
    np.array([[1., 0., 2., 0., 0., 0.1], [1., 0., 0., 2., 0., 0.1],
              [1., 2., 0., 0., 0., -1.], [3., 0., 0., 0., 0., -1.],
              [0., 3., 0., 0., 0., 1.], [2., 1., 0., 0., 0., 1.],
              [1., 0., 0., 0., 0., 1.]]),
    np.array([[0., 1., 2., 0., 0., 0.1], [0., 1., 0., 2., 0., 0.1],
              [0., 1., 0., 0., 0., 1.], [1., 2., 0., 0., 0., -1.],
              [3., 0., 0., 0., 0., -1.], [0., 3., 0., 0., 0., -1.],
              [2., 1., 0., 0., 0., -1.]])
],
                             dtype=object)

with open('dataset-Python/' + 'rxnDiff' + '.npy', 'wb') as f:
    np.save(f, U)
    np.save(f, xs)
    np.save(f, true_coefficients)