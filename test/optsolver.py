import numpy as np
import periodic_mat_util as matops
import prox_util as proxops


kernel_D1 = np.array([-1, 1])[:, None]
kernel_D2 = np.array([-1, 1])[None, :]

class OptSolver():
    def __init__(self, k, x, **kwargs):
        self.x = x
        self.objective = kwargs.get('objective', 'l1')
        self.maxiter = kwargs.get('maxiter', 500)
        self.relax = kwargs.get('relax', 1.0)
        self.step_size = kwargs.get('step_size', 0.1)
        self.gamma = kwargs.get('gamma', 0.5)
        self.eigvalK = matops.periodic_conv_eigvals(k, x.shape)
        self.eigvalD1 = matops.periodic_conv_eigvals(kernel_D1, x.shape)
        self.eigvalD2 = matops.periodic_conv_eigvals(kernel_D2, x.shape)
        self.conjK = np.conjugate(self.eigvalK)
        self.conjD1 = np.conjugate(self.eigvalD1)
        self.conjD2 = np.conjugate(self.eigvalD2)


    def applyA(self,x):
        '''
        Applies Ax

        Args:
            x (np.ndarray): x represented as an NxN matrix
        '''
        return matops.apply_A(self.eigvalK,self.eigvalD1,self.eigvalD2,x)
    
    def applyAT(self,y):
        """
        Applies A^Ty

        Args:
            y (np.ndarray): y represented as [y1,y2,y3] concatenated in a
            third dimension, each yi represented as an NxN matrix
        """
        return matops.fft_conv2d(self.conjK,y[0]) \
            + matops.fft_conv2d(self.conjD1,y[1]) \
                + matops.fft_conv2d(self.conjD2,y[2])
    
    def applyBig(self,t,x):
        '''
        Applies (I+tA^TA)^(-1)(x)

        Args:
            t (int): step size
            x (np.ndarray): x represented as an NxN matrix
        '''
        eigvals = matops.eigvals_mat(self.conjK,self.eigvalK,self.conjD1,self.conjD1,
                                     self.conjD2,self.eigvalD2,t)
        return matops.fft_invert(eigvals,x)
    
    def applyBigT(self,t,y):
        '''
        Applies (I+tAA^T)^(-1)(y)
        Args:
            t (int): step size
            y (np.ndarray): y represented as [y1,y2,y3] concatenated in a
            third dimension, each yi represented as an NxN matrix       
        '''
        # Compute (I+tAA^T)y
        newy = y + t * self.applyA(self.applyAT(y))
        # Compute (I+tAA^T)^(-1)y
        return np.fft.ifft2(np.square(np.fft.fft2(y)) / newy)


class DouglasRachfordPrimal(OptSolver):
    def __init__(self, k, x, **kwargs):
        super().__init__(k, x, **kwargs)
        self.foo = kwargs.get('foo', 'bar')

    def solve(self):
        """
        f(x) = indicator of {x, a finite dimenional vector: 0 <= x_i <= 1 for all dimenions i}.
        g(y) = ||y1 − b||_l1 + gamma||(y2, y3)||_iso, (y1, (y2, y3)) = [K, D]z
        """
        pass


if __name__ == "__main__":
    # usage
    solver = DouglasRachfordPrimal(k=np.array([1, 2, 3])[:, None], x=np.random.random((128, 128)), objective='l2', maxiter=1000)