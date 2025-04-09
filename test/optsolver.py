import numpy as np
import prox_util as prox
import periodic_mat_util as mat
from optutil import OptUtil
from solvertemplate import SolverTemplate


def construct_solver(optsolver):
    return SolverTemplate(
        initial_iterates=getattr(optsolver, 'initial_iterates', None),
        scaling=getattr(optsolver, 'scaling', None),
        util=getattr(optsolver, 'util', None),
        maxiter=getattr(optsolver, 'maxiter', None),
        b=getattr(optsolver, 'b', None),
        err_ord=getattr(optsolver, 'err_ord', None),
        step_size=getattr(optsolver, 'step_size', None),
    )

kernel_D1 = np.array([-1, 1])[:, None]
kernel_D2 = np.array([-1, 1])[None, :]

# algorithm state and resolvent methods
class OptSolver:
    """
    main wrapper class for the algorithms,
    meant to be the interface for the user.
    """
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
        # user's hyperparameters
        self.b = b
        self.proxdbl = {'l1': prox.l1prox,
                        'l2': prox.l2prox
                        }.get(deblurring_objective, 'l1')
        # order of deblurring objective
        self.err_ord = {'l1': 1,
                        'l2': 2
                        }.get(deblurring_objective, 'l1')
        self.maxiter = maxiter
        self.relax = relax
        self.step_size = step_size
        self.gamma = gamma
        # prox operators used in all methods
        self.prox_box = prox.box_prox
        self.prox_iso = prox.iso_prox
        # helper class for (stateful) matrix and proximal operations
        self.util = OptUtil(
            key_kernel_dict={'K': k, 'D1': kernel_D1, 'D2': kernel_D2},
            shape=b.shape,
            t=step_size,
            deblurring_prox=self.proxdbl)
    

class DouglasRachfordPrimal(OptSolver):
    def __init__(self, k, b, **kwargs):
        super().__init__(k, b, **kwargs)
        # cache initialization
        self.scaling = self.relax
        self.initial_iterates = [b, self.util.applyA(self.b)]
        self.solver = construct_solver(self)
        
    def solve(self, track_objective=False):
        return self.solver.douglasrachford_main(resolvent_A=self.resolvent_A,
                                                resolvent_B=self.resolvent_B,
                                                track_objective=track_objective)

    def resolvent_A(self, z1, z2):
        return (self.prox_box(t=self.step_size, x=z1), 
                self.util.objective_prox(y=z2, b=self.b, t=self.step_size, g=self.gamma))

    def resolvent_B(self, x, y, z1, z2):
        # (I + A^TA)(-1)(2x_k − z1_(k-1)+ A^T(2y_k − z2_(k-1)))
        u = self.util.applyBig1(x=2 * x - z1 + self.util.applyAT(2 * y - z2))
        return u, self.util.applyA(u)
    

class DouglasRachfordPrimalDual(OptSolver):
    def __init__(self, k, b, **kwargs):
        super().__init__(k, b, **kwargs)
        # 1/t is used multiple times, just compute once
        self.t, self.trec = self.step_size, 1/self.step_size
        self.scaling = self.relax
        self.initial_iterates = [self.b, self.util.applyA(self.b)]
        self.solver = construct_solver(self)

    def solve(self, track_objective=False):
        return self.solver.douglasrachford_main(resolvent_A=self.resolvent_A,
                                                resolvent_B=self.resolvent_B,
                                                track_objective=track_objective)
    
    def resolvent_A(self, p, q):
        z = self.util.objective_prox(y=self.trec * q, b=self.b, t=self.trec, g=self.gamma)             
        return (self.prox_box(self.t, p), (q - self.t * z))
    
    def resolvent_B(self, x, z, p, q):
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
        # 1/t, 1/rho used multiple times, just compute once
        self.trec = 1/self.step_size 
        self.compress = 1 - self.relax
        self.scaling = self.step_size
        # x, u, y, w, z
        self.initial_iterates = [like_b, like_b, like_b_triple_cat, like_b, like_b_triple_cat]
        self.solver = construct_solver(self)

    def solve(self, track_objective=False):
        return self.solver.admm_main(resolvent_A=self.resolvent_A,
                                     resolvent_B=self.resolvent_B,
                                     final_out=self.composite_op,
                                     track_objective=track_objective)
    
    def composite_op(self, u, y, w1t, z1t):
        return self.util.applyBig1(u + self.util.applyAT(y) - (w1t + self.util.applyAT(z1t)))
    
    def resolvent_A(self, x, y, u, w1t, z1t):
        x = self.composite_op(u, y, w1t, z1t)
        return x, self.prox_box(t=self.trec, x=self.relax*x + self.compress*u + w1t)
    
    def resolvent_B(self, x, y, z1t):
        Ax = self.util.applyA(x)
        return Ax, self.util.objective_prox(y=self.relax*Ax + self.compress*y + z1t, b=self.b, t=self.trec, g=self.gamma)
    

class ChambollePock(OptSolver):
    def __init__(self, k, b, step_size2=0.1, **kwargs):
        super().__init__(k, b, **kwargs)
        self.t, self.s = self.step_size, step_size2 # step sizes
        self.srec = 1/self.s # 1/s used multiple times, just compute once
        self.initial_iterates = [self.b.copy(), self.util.applyA(b), self.b.copy()] # x, y, z
        self.solver = construct_solver(self)

    def solve(self, track_objective=False):
        return self.solver.chambollepock_main(prox_g_conj=self.prox_g_conj,
                                              track_objective=track_objective)
    
    def prox_g_conj(self, y, z):
        q = y + self.s * self.util.applyA(z)
        return q - self.s * self.util.objective_prox(y=self.srec*q, 
                                                     b=self.b, 
                                                     t=self.s, 
                                                     g=self.gamma)
    


if __name__ == "__main__":
    # usage
    params = {'deblurring_objective': 'l2', 'maxiter': 2}
    k = np.array([1, 2, 3])[:, None]
    b = np.random.random((128, 128))
    # solver = DouglasRachfordPrimal(k=np.array([1, 2, 3])[:, None], b=np.random.random((128, 128)), **params)
    # solver = DouglasRachfordPrimalDual(k=np.array([1, 2, 3])[:, None], b=np.random.random((128, 128)), **params)
    # solver = ADMM(k=np.array([1, 2, 3])[:, None], b=np.random.random((128, 128)), **params)
    solver = ChambollePock(k=k, b=b, **params)
    solver.solve()