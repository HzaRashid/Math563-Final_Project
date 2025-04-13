import numpy as np
import prox_util as prox
import periodic_mat_util as mat
from optutil import OptUtil
from solvertemplate import SolverTemplate


def construct_solver(optsolver):
    return SolverTemplate(
        scaling=getattr(optsolver, 'scaling', None),
        maxiter=getattr(optsolver, 'maxiter', None),
        err_ord=getattr(optsolver, 'err_ord', None),
        step_size=getattr(optsolver, 'step_size', None),
        util=getattr(optsolver, 'util', None),
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
                 k, # kernel of convolution
                 shape, # shape of image to be deblurred
                 deblurring_objective='l1',
                 maxiter=100,
                 step_size=0.1,
                 relax=0.5,
                 gamma=0.1
                 ):
        """
        Parameters:
            k: kernel of convolution 
            shape: shape of image
            deblurring_objective: 'l1' or 'l2'
            maxiter: number of iterations to run
            relax: hyperparameter rho
            step_size: hyperparameter t
            step_size2: hyperparameter s (only for ChambollePock)
            gamma: hyperparameter gamma
        """
        # user's hyperparameters
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
            shape=shape,
            t=step_size,
            deblurring_prox=self.proxdbl)
        
    def get_objective(self, x, b, ord):
       """
       Returns the objective value
       Args:
            x: current image
            b: blurred image
            ord: indicates 1 or 2-norm
       """
       y = self.util.applyA(x)
       # |y[0]-b| using 1 or 2 norm
       eps1 = np.linalg.norm(
             x = y[:,:,0] - b, 
             ord=ord
             )
       # iso(y[1],y[2])
       eps2 = np.sum(np.sqrt(np.abs(y[:,:,1])**2 + np.abs(y[:,:,2])**2))
       return (eps1 + self.gamma*eps2)/np.size(x)
    

class DouglasRachfordPrimal(OptSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # cache initialization
        self.scaling = self.relax
        self.solver = construct_solver(self)
        
    def solve(self, b, if_track=False):
        return self.solver.douglasrachford_main(b, 
                                                resolvent_A=self.resolvent_A,
                                                resolvent_B=self.resolvent_B,
                                                if_track=if_track,
                                                get_obj = self.get_objective)

    def resolvent_A(self, z1, z2, b):
        # Computes prox tf(z1), prox tg(z2)
        return (self.prox_box(t=self.step_size, x=z1), 
                self.util.objective_prox(y=z2, b=b, t=self.step_size, g=self.gamma))

    def resolvent_B(self, x, y, z1, z2):
        # (I + A^TA)(-1)(2x_k − z1_(k-1)+ A^T(2y_k − z2_(k-1)))
        u = self.util.applyBig(x=2 * x - z1 + self.util.applyAT(2 * y - z2))
        return u, self.util.applyA(u)
    

class DouglasRachfordPrimalDual(OptSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 1/t is used multiple times, just compute once
        self.t, self.trec = self.step_size, 1/self.step_size
        self.scaling = self.relax
        self.solver = construct_solver(self)

    def solve(self, b, if_track=False): # same as DRP in structure, different resolvents
        return self.solver.douglasrachford_main(b,
                                                resolvent_A=self.resolvent_A,
                                                resolvent_B=self.resolvent_B,
                                                if_track=if_track,
                                                get_obj=self.get_objective)
    
    def resolvent_A(self, p, q, b):
        # z = prox_g/t(q/t)
        z = self.util.objective_prox(y=self.trec * q, b=b, t=self.trec, g=self.gamma)
        # Note prox_tg*(q) = q - t*prox_g/t(q/t)        
        return (self.prox_box(self.t, p), (q - self.t * z))
    
    def resolvent_B(self, x, z, p, q):
        zq = 2 * z - q
        # (I+tA^TA)^(-1)(xq-tA^T(zq))
        combined = self.util.applyBig(2 * x - p - self.t * self.util.applyAT(zq), t=self.t)
        # [0,zq] + [I,tA] * combined
        return (combined, (zq + self.t * self.util.applyA(combined)))
 

class ADMM(OptSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 1/t, 1-rho used multiple times, just compute once
        self.trec = 1/self.step_size 
        self.compress = 1 - self.relax
        self.scaling = self.step_size
        self.solver = construct_solver(self)

    def solve(self, b, if_track=False):
        return self.solver.admm_main(b,
                                     resolvent_A=self.resolvent_A,
                                     resolvent_B=self.resolvent_B,
                                     final_out=self.composite_op,
                                     if_track=if_track,
                                     get_obj=self.get_objective)
    
    def composite_op(self, u, y, w1t, z1t):
        # (I+ATA)^(-1)(u+A^Ty-(w/t+A^Tz/t))
        return self.util.applyBig(u + self.util.applyAT(y) - (w1t + self.util.applyAT(z1t)))
    
    def resolvent_A(self, x, y, u, w1t, z1t):
        # Computes prox_f/t(rho*x + (1-rho)*u + w/t)
        x = self.composite_op(u, y, w1t, z1t)
        return x, self.prox_box(t=self.trec, x=self.relax*x + self.compress*u + w1t)
    
    def resolvent_B(self, x, y, z1t, b):
        # Computes prox_g/t(rho*Ax + (1-rho)*y + z/t)
        Ax = self.util.applyA(x)
        return Ax, self.util.objective_prox(y=self.relax*Ax + self.compress*y + z1t, b=b, t=self.trec, g=self.gamma)
    

class ChambollePock(OptSolver):
    def __init__(self, step_size2=0.1, **kwargs):
        super().__init__(**kwargs)
        self.t, self.s = self.step_size, step_size2 # step sizes
        self.srec = 1/self.s # 1/s used multiple times, just compute once
        self.solver = construct_solver(self)

    def solve(self, b, if_track=False):
        return self.solver.chambollepock_main(b,
                                              prox_g_conj=self.prox_g_conj,
                                              prox_f=self.prox_f,
                                              if_track=if_track,
                                              get_obj=self.get_objective)
    
    def prox_g_conj(self, y, z, b):
        q = y + self.s * self.util.applyA(z)
        # Uses prox_sg*(q) = q - s*prox_g/s(q/s) to compute prox_sg*(q)
        return q - self.s * self.util.objective_prox(y=self.srec*q, 
                                                     b=b, 
                                                     t=self.s, 
                                                     g=self.gamma)
    def prox_f(self, x, y):
        return self.prox_box(t=self.t, x = x - self.t * self.util.applyAT(y))
    


if __name__ == "__main__":
    # usage
    params = {'deblurring_objective': 'l2', 'maxiter': 2}
    k = np.array([1, 2, 3])[:, None]
    b = np.random.random((128, 128))
    # solver = DouglasRachfordPrimal(k=np.array([1, 2, 3])[:, None], b=np.random.random((128, 128)), **params)
    # solver = DouglasRachfordPrimalDual(k=np.array([1, 2, 3])[:, None], b=np.random.random((128, 128)), **params)
    # solver = ADMM(k=np.array([1, 2, 3])[:, None], b=np.random.random((128, 128)), **params)
    solver = ChambollePock(k=k, shape=(128,128), **params)
    solver.solve(b)