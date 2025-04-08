import numpy as np
import prox_util as proxops
import periodic_mat_util as mat
from optutil import OptUtil, build_eigval_store

prox_store = {
    'l1': proxops.l1prox,
    'l2': proxops.l2prox,
    'iso': proxops.iso_prox,
    'box': proxops.box_prox,
    }


kernel_D1 = np.array([-1, 1])[:, None]
kernel_D2 = np.array([-1, 1])[None, :]

class OptSolver:
    def __init__(self, k, b, 
                 deblurring_objective='l1',
                 maxiter=500,
                 step_size=0.1,
                 relax=0.5,
                 gamma=0.1
                 ):
        """
        Parameters:
            k: description of k
            b: description of b
            deblurring_objective: description...
            maxiter: description...
            relax: description...
            step_size: description...
            gamma: description...
        """
        self.b = b
        self.proxdbl = prox_store.get(deblurring_objective, 'l1')
        self.maxiter = maxiter
        self.relax = relax
        self.step_size = step_size
        self.gamma = gamma
        self.prox_box = prox_store['box']
        self.prox_iso = prox_store['iso']
        self.eps = np.zeros(self.maxiter)

        self.eigval = build_eigval_store(
            key_kernel_dict={'K': k, 
                             'D1': kernel_D1, 
                             'D2': kernel_D2}, 
            shape=b.shape
            )
        self.conj_eigval = {key: np.conjugate(val) for key, val in self.eigval.items()}

        self.err_ord = {
            'l1': 1,
            'l2': 2
            }.get(deblurring_objective, 'l1')
        
        self.util = OptUtil(eigval=self.eigval, conj_eigval=self.conj_eigval)


class DouglasRachfordPrimal(OptSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # cache expensive (static) operations
        self.z1 = mat.fft_conv2d(eigvals=self.eigval['K'], x=self.b)
        self.z2 = self.util.applyA(self.b)
        self.eigvals_mat = self.util.apply_eigvals_mat(t=1.0)
        self.proxf = prox_store.get("box")
        self.proxg2 = prox_store.get("iso") # g2(y2, y3) = gamma||(y2, y3)||_iso

    def solve(self, track_objective=False):
        """
        f(x) = indicator of {x, a finite dimenional vector: 0 <= x_i <= 1 for all dimenions i}.
        g(y) = ||y1 − b||_l1 + gamma||(y2, y3)||_iso, (y1, (y2, y3)) = [K, D]z
        """
        b, z1, z2, t = self.b, self.z1, self.z2, self.step_size
        eps = []
        
        for _ in range(self.maxiter):
            x, y = self.resolvent_A(z1, z2, b, t)
            u, v = self.resolvent_B(x, y, z1, z2)
            z1 = z1 + self.relax * (u - x)
            z2 = z2 + self.relax * (v - y)

            if track_objective:
                # For now this is just ||Kx-b||
                # @TODO implement iso and add it here
                eps.append(self.util.get_objective(x=x, b=b, ord=self.err_ord))

        return self.proxf(t=t, x=z1), eps or [self.util.get_objective(x=x, b=self.b, ord=self.err_ord)]

    
    def resolvent_A(self, z1, z2, b, t):
        return (self.proxf(t=t, x=z1), 
                mat.cat_mats([
                    self.proxdbl(t=t, 
                                 y=z2[:,:,0], 
                                 b=b),  # shape: (b.shape[0], b.shape[1])
                    *self.proxg2(t=t, 
                                 g=self.gamma, 
                                 w1=z2[:,:,1], 
                                 w2=z2[:,:,2]) # (2, b.shape[0], b.shape[1])
                ]))

    def resolvent_B(self, x, y, z1, z2):
        # (I + A^TA)(-1)(2x_k − z1_(k-1)+ A^T(2y_k − z2_(k-1)))
        u = mat.fft_invert(eigvals_mat=self.eigvals_mat, 
                              x=2 * x - z1 + self.util.applyAT(2 * y - z2))
        return (u, self.util.applyA(u))


class DouglasRachfordDual(OptSolver):
    def __init__(self, k, b, **kwargs):
        super().__init__(k, b, **kwargs)
        # cache expensive (static) operations
        self.eigvalmat_t = self.util.apply_eigvals_mat(self.step_size)
        self.p = np.zeros_like(self.b)
        self.q = self.util.applyA(np.zeros_like(self.b))
        self.t, self.tsq, self.tre = self.step_size, self.step_size**2, 1/self.step_size
        
    def solve(self, track_objective=False):
        # initialization
        t, tsq, tre = self.step_size, self.step_size**2, 1/self.step_size
        rho = self.relax
        pk, qk = self.p, self.q
        eps = []

        for _ in range(self.maxiter):
            """
            prox[t*g_conj](q) = q - [prox((1/t)*g)](q/t),
            as described in 'Primal-Dual Decomposition by Operator 
            Splitting and Applications to Image Deblurring' (O'Connor et al., 2014).
            """
            x, z = self.resolvantA(pk, qk, t)
            w, v = self.resolvantB(x,z,pk,qk)
            pk += rho * (w - x).real
            qk += rho * (v - z).real
            if track_objective:
                eps.append(self.util.get_objective(x=x, b=self.b, ord=self.err_ord))
                
        return self.prox_box(t, pk), eps or [self.util.get_objective(x=x, b=self.b, ord=self.err_ord)]
    
    def resolvantA(self, pk, qk, t):
        z = np.dstack([self.proxdbl(self.tre, qk[:, :, 0]/t, self.b),
                       *self.prox_iso(self.tre, self.gamma, qk[:, :, 1]/t, qk[:, :, 2]/t)])
                       
        return (self.prox_box(t, pk), (qk - t * z))
    
    def resolvantB(self, x, z, pk, qk):
        temp_zq = 2 * z - qk
        temp_xp = 2 * x - pk
        return (
            mat.fft_invert(self.eigvalmat_t, temp_xp - self.t * self.util.applyAT(temp_zq)),
            (temp_zq 
                  + self.t * self.util.applyA(mat.fft_invert(self.eigvalmat_t, 
                                                        temp_xp)) 
                  - self.tsq * self.util.applyA(mat.fft_invert(self.eigvalmat_t, 
                                                          self.util.applyAT(temp_zq))))
        )
    


class ADMM(OptSolver):
    def __init__(self, k, b, **kwargs):
        super().__init__(k, b, **kwargs)
        
    def solve(self):
        t = self.step_size
        z = np.zeros_like(self.b)
        u = np.zeros_like(self.b)

        for i in range(self.maxiter):
            x = self.prox_box(1, z - u)
            v = x + u
            rhs = self.util.applyAT(np.dstack([
                self.proxdbl(t, mat.fft_conv2d(self.eigval['K'], v), self.b),
                *self.prox_iso(t, self.gamma, v, v)
            ]))
            z = mat.fft_invert(self.util.apply_eigvals_mat(t), rhs).real
            u += x - z
            self.eps[i] = np.linalg.norm(mat.fft_conv2d(self.eigval['K'], x) - self.b)

        return x, self.eps
 

class ChambollePock(OptSolver):
    def __init__(self, k, b, step_size2, **kwargs):
        super().__init__(k, b, **kwargs)
        self.step_size2 = step_size2
    def solve(self):
        tau = self.step_size
        sigma = self.step_size2
        x = np.zeros_like(self.b)
        y = self.util.applyA(x)
        z = x.copy()
        eps = np.zeros(self.maxiter)

        for i in range(self.maxiter):
            y = self.prox_g_conj(y + sigma * self.util.applyA(z))
            x_old = x.copy()
            x = self.prox_box(tau, x - tau * self.util.applyAT(y))
            z = 2 * x - x_old

            eps[i] = np.linalg.norm(mat.fft_conv2d(self.eigval['K'], x) - self.b)

        return x, eps

    def prox_g(self, y, sigma):
        y1 = self.proxdbl(sigma, y[:, :, 0], self.b)
        y2, y3 = self.prox_iso(sigma, self.gamma, y[:, :, 1], y[:, :, 2])
        return np.dstack([y1, y2, y3])

    def prox_g_conj(self, y, sigma, prox_g):
        return self.conjugate_prox(prox_g, y, sigma)
    

if __name__ == "__main__":
    # usage
    params = {'deblurring_objective': 'l2', 'maxiter': 2}
    # solver = DouglasRachfordPrimal(k=np.array([1, 2, 3])[:, None], b=np.random.random((128, 128)), **params)
    # solver = DouglasRachfordDual(k=np.array([1, 2, 3])[:, None], b=np.random.random((128, 128)), **params)
    solver = ADMM(k=np.array([1, 2, 3])[:, None], b=np.random.random((128, 128)), **params)
    solver.solve()