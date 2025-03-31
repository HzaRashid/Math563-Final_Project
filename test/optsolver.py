import numpy as np
import periodic_mat_util as matops
import prox_util as proxops

kernel_D1 = np.array([-1, 1])[:, None]
kernel_D2 = np.array([-1, 1])[None, :]

class solve():
    def __init__(self,
                 k,
                 x,
                 objective='l1',
                 maxiter=500,
                 relax=1.0,
                 step_size=0.1,
                 gamma=0.5
                 ):
        self.x = x
        size = np.shape(x)
        self.eigvalK = matops.periodic_conv_eigvals(k, size)
        self.eigvalD1 = matops.periodic_conv_eigvals(kernel_D1, size)
        self.eigvalD2 = matops.periodic_conv_eigvals(kernel_D2, size)
        self.conjK = np.conjugate(self.eigvalK)
        self.conjD1 = np.conjugate(self.eigvalD1)
        self.conjD2 = np.conjugate(self.eigvalD2)
        self.objective = objective
        self.maxiter = maxiter
        self.relax = relax
        self.step_size = step_size
        self.gamma = gamma

    def applyA(self,x):
        '''
        Applies Ax
        '''
        return matops.apply_A(self.eigvalK,self.eigvalD1,self.eigvalD2,x)
    
    def applyAT(self,y):
        '''
        Applies A^Ty
        '''
        return matops.fft_conv2d(self.conjK,y) \
            + matops.fft_conv2d(self.conjD1,y) \
                + matops.fft_conv2d(self.conjD2,y)
    
    def applyBig(self,t,x):
        '''
        Applies (I+A^TA)^(-1)(x)
        '''
        eigvals = matops.eigvals_mat(self.conjK,self.eigvalK,self.conjD1,self.conjD1,
                                     self.conjD2,self.eigvalD2,t)
        return matops.fft_invert(eigvals,x)
    
    def applyBigT(self,t,y):
        '''
        Applies (I+AA^T)^(-1)(y)
        '''
        pass

    def DouglasRachfordPrimal(self):
        """
        f(x) = indicator of {x, a finite dimenional vector: 0 <= x_i <= 1 for all dimenions i}.
        g(y) = ||y1 − b||_l1 + gamma||(y2, y3)||_iso, (y1, (y2, y3)) = [K, D]z
        """
        pass