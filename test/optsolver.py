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
    def __init__(self, 
                 k, 
                 b,
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

        self.err_ord = {
            'l1': 1,
            'l2': 2
            }.get(deblurring_objective, 'l1')
        
        self.util = OptUtil(eigval=self.eigval,t=step_size)


class DouglasRachfordPrimal(OptSolver):
    def __init__(self, k, b, **kwargs):
        super().__init__(k, b, **kwargs)
        # cache some operations
        self.z1 = mat.fft_conv2d(eigvals=self.eigval['K'], x=self.b)
        self.z2 = self.util.applyA(self.b)

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
                eps.append(self.util.get_objective(x=x, b=b, ord=self.err_ord))

        return self.prox_box(t=t, x=z1), eps or [self.util.get_objective(x=x, b=self.b, ord=self.err_ord)]

    def resolvent_A(self, z1, z2, b, t):
        return (self.prox_box(t=t, x=z1), # box constraint f(x)
                mat.cat_mats([ # objective g(y)
                    self.proxdbl(t=t, y=z2[:,:,0], b=b),  # shape: (b.shape[0], b.shape[1])
                    *self.prox_iso(t=t, g=self.gamma, w1=z2[:,:,1], w2=z2[:,:,2]) # (2, b.shape[0], b.shape[1])
                ]))

    def resolvent_B(self, x, y, z1, z2):
        # (I + A^TA)(-1)(2x_k − z1_(k-1)+ A^T(2y_k − z2_(k-1)))
        u = self.util.applyBig1(x=2 * x - z1 + self.util.applyAT(2 * y - z2))
        return (u, self.util.applyA(u))


class DouglasRachfordPrimalDual(OptSolver):
    def __init__(self, k, b, **kwargs):
        super().__init__(k, b, **kwargs)
        # cache some operations
        self.p = np.zeros_like(self.b)
        self.q = self.util.applyA(np.zeros_like(self.b))
        self.t, self.tsq, self.trec = self.step_size, self.step_size**2, 1/self.step_size
        
    def solve(self, track_objective=False):
        # initialization
        t = self.step_size
        rho = self.relax
        p, q = self.p, self.q
        eps = []

        for _ in range(self.maxiter):
            """
            prox[t*g_conj](q) = q - [prox((1/t)*g)](q/t),
            as described in 'Primal-Dual Decomposition by Operator 
            Splitting and Applications to Image Deblurring' (O'Connor et al., 2014).
            """
            x, z = self.resolvantA(p, q)
            w, v = self.resolvantB(x, z, p, q)
            p += rho * (w - x).real
            q += rho * (v - z).real

            if track_objective:
                eps.append(self.util.get_objective(x=x, b=self.b, ord=self.err_ord))
                
        return self.prox_box(t, p), eps or [self.util.get_objective(x=x, b=self.b, ord=self.err_ord)]
    
    def resolvantA(self, p, q):
        q1t = self.trec * q
        z = np.dstack([self.proxdbl(self.trec, q1t[:,:,0], self.b),
                       *self.prox_iso(self.trec, self.gamma, q1t[:,:,1], q1t[:,:,2])])
                       
        return (self.prox_box(self.t, p), (q - self.t * z))
    
    def resolvantB(self, x, z, p, q):
        zq = 2 * z - q
        # (I+tA^TA)^(-1)(xq-tA^T(zq))
        combined = self.util.applyBig(2 * x - p - self.t * self.util.applyAT(zq))
        # [0,zq] + [I,tA] * tempc
        return (combined, (zq + self.t * self.util.applyA(combined)))
 

class ADMM(OptSolver):
    def __init__(self, k, b, **kwargs):
        super().__init__(k, b, **kwargs)
        # use for intialization only
        like_b = self.b.copy()
        like_b_triple_cat = mat.cat_mats([self.b.copy() for _ in range(3)])
        self.all_init = (
            like_b,             # x
            like_b,             # u
            like_b_triple_cat,  # y
            like_b,             # w
            like_b_triple_cat,  # z
            self.step_size      # t
            )
        self.trec = 1/self.step_size 
        self.compress = 1 - self.relax

    def solve(self, track_objective=False): # save w/t
        x, u, y, w, z, t = self.all_init
        eps = []

        for _ in range(self.maxiter):
            w1t, z1t = self.trec*w, self.trec*z
            x, u = self.resolvent_A(x, y, u, w1t, z1t)
            Ax, y = self.resolvent_B(x, y, z1t)
            w = w + t * (x - u)
            z = z + t * (Ax - y)

            if track_objective:
                eps.append(self.util.get_objective(x, self.b, ord=self.err_ord))

        return self.composite_op(u, y, self.trec*w, self.trec*z), eps
    
    def composite_op(self, u, y, w1t, z1t):
        return self.util.applyBig1(u + self.util.applyAT(y) - (w1t + self.util.applyAT(z1t)))
    
    def resolvent_A(self, x, y, u, w1t, z1t):
        x = self.composite_op(u, y, w1t, z1t)
        return x, self.prox_box(t=self.trec, x=self.relax*x + self.compress*u + w1t)
    
    def resolvent_B(self, x, y, z1t):
        Ax = self.util.applyA(x)
        y_arg = self.relax*Ax + self.compress*y + z1t
        return Ax, mat.cat_mats([
            self.proxdbl(t=self.trec, y=y_arg[:,:,0], b=self.b),
            *self.prox_iso(t=self.trec, g=self.gamma, w1=y_arg[:,:,1], w2=y_arg[:,:,2])
            ])
    

class ChambollePock(OptSolver):
    def __init__(self, k, b, step_size2, **kwargs):
        super().__init__(k, b, **kwargs)
        self.step_size2 = step_size2
    def solve(self, track_objective=False):
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