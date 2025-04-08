
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

# ========== Convolution & Image Utilities ==========

def gaussian_kernel(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def fft_conv2d(eigvals, x):
    return np.fft.ifft2(eigvals * np.fft.fft2(x)).real

def periodic_conv_eigvals(kernel, shape):
    a = np.zeros(shape)
    a[0, 0] = 1
    Ra = gaussian_filter(a, sigma=2, mode='wrap')
    return np.fft.fft2(Ra)

def normalize_image(im):
    return (im - im.min()) / (im.max() - im.min())

# ========== Proximal Operators ==========

def box_prox(t, y):
    return np.minimum(np.maximum(y, 0), 1)

def l1prox(t, y, b):
    x = y - np.sign(y - b) * t
    x[np.abs(y - b) <= t] = b[np.abs(y - b) <= t]
    return x

def isoprox(t, g, w1, w2):
    lambda_ = t * g
    norm = np.sqrt(w1**2 + w2**2)
    alpha = np.maximum(1 - lambda_ / (norm + 1e-12), 0)
    return w1 * alpha, w2 * alpha

def conjugate_one(prox, y, t):
    return y - t * prox(y / t)

# ========== Base Solver Class ==========

class OptSolver:
    def __init__(self, k, b, **kwargs):
        self.b = b
        self.k = k
        self.shape = b.shape
        self.maxiter = kwargs.get('maxiter', 100)
        self.step_size = kwargs.get('step_size', 1.0)
        self.relax = kwargs.get('relax', 1.0)
        self.gamma = kwargs.get('gamma', 0.1)
        self.eigval = {
            'K': periodic_conv_eigvals(k, self.shape),
            'D1': periodic_conv_eigvals(np.array([[-1], [1]]), self.shape),
            'D2': periodic_conv_eigvals(np.array([[-1, 1]]), self.shape),
        }
        self.eigval.update({
            'K_T': np.conj(self.eigval['K']),
            'D1_T': np.conj(self.eigval['D1']),
            'D2_T': np.conj(self.eigval['D2'])
        })

    def applyA(self, x):
        return np.dstack([
            fft_conv2d(self.eigval['K'], x),
            fft_conv2d(self.eigval['D1'], x),
            fft_conv2d(self.eigval['D2'], x)
        ])

    def applyAT(self, y):
        return fft_conv2d(self.eigval['K_T'], y[:, :, 0]) + \
               fft_conv2d(self.eigval['D1_T'], y[:, :, 1]) + \
               fft_conv2d(self.eigval['D2_T'], y[:, :, 2])

    def apply_eigvals_mat(self, t):
        return 1 + t**2 * (np.abs(self.eigval['K'])**2 +
                           np.abs(self.eigval['D1'])**2 +
                           np.abs(self.eigval['D2'])**2)

    def fft_invert(self, eigvals_mat, x):
        return np.fft.ifft2(np.fft.fft2(x) / eigvals_mat).real

# ========== Algorithm 1: Douglas-Rachford Primal ==========

class DouglasRachfordPrimal(OptSolver):
    def solve(self):
        z1 = fft_conv2d(self.eigval['K'], self.b)
        z2 = self.applyA(self.b)
        t = self.step_size
        eigvals_mat = self.apply_eigvals_mat(t)
        eps = np.zeros(self.maxiter)

        for i in range(self.maxiter):
            x = box_prox(t, z1)
            y1 = l1prox(t, z2[:, :, 0], self.b)
            y2, y3 = isoprox(t, self.gamma, z2[:, :, 1], z2[:, :, 2])
            y = np.dstack([y1, y2, y3])

            u = self.fft_invert(eigvals_mat, 2*x - z1 + self.applyAT(2*y - z2))
            v = self.applyA(u)

            z1 += self.relax * (u - x)
            z2 += self.relax * (v - y)
            eps[i] = np.linalg.norm(fft_conv2d(self.eigval['K'], x) - self.b)

        return box_prox(t, z1), eps

# ========== Algorithm 2: Douglas-Rachford Dual (corrected) ==========

class DouglasRachfordDual(OptSolver):
    def solve(self):
        t = self.step_size
        rho = self.relax
        pk = np.zeros_like(self.b)
        qk = self.applyA(np.zeros_like(self.b))
        eps = np.zeros(self.maxiter)

        for i in range(self.maxiter):
            xk = box_prox(t, pk)
            y1 = l1prox(t, qk[:, :, 0], self.b)
            y2, y3 = isoprox(t, self.gamma, qk[:, :, 1], qk[:, :, 2])
            zk = np.dstack([y1, y2, y3])

            temp_zq = 2 * zk - qk
            temp_xp = 2 * xk - pk

            wk = self.fft_invert(self.apply_eigvals_mat(t), temp_xp - t * self.applyAT(temp_zq))
            vk = temp_zq + t * self.applyA(self.fft_invert(self.apply_eigvals_mat(t), temp_xp)) \
                 - t**2 * self.applyA(self.fft_invert(self.apply_eigvals_mat(t), self.applyAT(temp_zq)))

            pk += rho * (wk - xk)
            qk += rho * (vk - zk)

            eps[i] = np.linalg.norm(fft_conv2d(self.eigval['K'], xk) - self.b)

        return xk, eps

# ========== Algorithm 3: Chambolle-Pock (corrected) ==========

class ChambollePock(OptSolver):
    def solve(self):
        tau = sigma = self.step_size
        x = np.zeros_like(self.b)
        y = self.applyA(x)
        z = x.copy()
        eps = np.zeros(self.maxiter)

        def prox_g(y):
            y1 = l1prox(sigma, y[:, :, 0], self.b)
            y2, y3 = isoprox(sigma, self.gamma, y[:, :, 1], y[:, :, 2])
            return np.dstack([y1, y2, y3])

        def prox_g_conj(y):
            return conjugate_one(prox_g, y, sigma)

        for i in range(self.maxiter):
            y = prox_g_conj(y + sigma * self.applyA(z))
            x_old = x.copy()
            x = box_prox(tau, x - tau * self.applyAT(y))
            z = 2 * x - x_old

            eps[i] = np.linalg.norm(fft_conv2d(self.eigval['K'], x) - self.b)

        return x, eps

# ========== Algorithm 4: ADMM ==========

class ADMM(OptSolver):
    def solve(self):
        t = self.step_size
        z = np.zeros_like(self.b)
        u = np.zeros_like(self.b)
        eps = np.zeros(self.maxiter)

        for i in range(self.maxiter):
            x = box_prox(1, z - u)
            v = x + u
            rhs = self.applyAT(np.dstack([
                l1prox(t, fft_conv2d(self.eigval['K'], v), self.b),
                *isoprox(t, self.gamma, v, v)
            ]))
            z = self.fft_invert(self.apply_eigvals_mat(t), rhs)
            u += x - z
            eps[i] = np.linalg.norm(fft_conv2d(self.eigval['K'], x) - self.b)

        return x, eps

# === Test: run all algorithms ===

image = color.rgb2gray(data.astronaut())
image = resize(image, (256, 256), anti_aliasing=True)
image = normalize_image(image)

kernel = gaussian_kernel(size=15, sigma=2.0)
k_shape = image.shape
eig_K = periodic_conv_eigvals(kernel, k_shape)
blurred = fft_conv2d(eig_K, image)

params = {
    'k': kernel,
    'b': blurred,
    'maxiter': 50,
    'step_size': 1.0,
    'relax': 1.0,
    'gamma': 0.1
}

solvers = {
    'Douglas-Rachford Primal': DouglasRachfordPrimal(**params),
    'Douglas-Rachford Dual': DouglasRachfordDual(**params),
    'Chambolle-Pock': ChambollePock(**params),
    'ADMM': ADMM(**params)
}

results = {}
for name, solver in solvers.items():
    print(f"Running {name}...")
    sol, err = solver.solve()
    results[name] = (sol, err)

fig, axes = plt.subplots(1, len(solvers) + 1, figsize=(20, 5))
axes[0].imshow(blurred, cmap='gray')
axes[0].set_title("Blurred")
axes[0].axis('off')

for i, (name, (rec, _)) in enumerate(results.items(), 1):
    axes[i].imshow(np.clip(rec, 0, 1), cmap='gray')
    axes[i].set_title(name)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

plt.figure()
for name, (_, err) in results.items():
    plt.plot(err, label=name)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Convergence')
plt.legend()
plt.grid(True)
plt.show()
