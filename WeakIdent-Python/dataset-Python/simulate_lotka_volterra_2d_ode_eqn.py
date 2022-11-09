import numpy as np
from scipy.integrate import solve_ivp

max_t = 1
dt = 0.01
file_name = 'LotkaVolterra2D2'

t = np.linspace(0, max_t, int(max_t / dt + 1))

r = 2
s = 2
alpha_ab = 0.05
alpha_ba = 0.05

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

m = 2
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


def reaction_diffusion_rhs(t, uvt,  r, s, alpha_ab, alpha_ba,  N):
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


yinit = uvt.flatten()
sol = solve_ivp(reaction_diffusion_rhs, [t[0], t[-1]],
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
