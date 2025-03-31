import numpy as np
import periodic_mat_util as matops
import prox_util as proxops


kernel_D1 = np.array([-1, 1])[:, None]
kernel_D2 = np.array([-1, 1])[None, :]
prox_store = {
    'l1': proxops.l1prox,
    'l2': proxops.l2prox,
    'iso': proxops.iso_prox,
    'box': proxops.box_prox,
    }

class OptSolver():
    def __init__(self, k, b, **kwargs):
        self.b = b
        self.proxdbl = prox_store.get(kwargs.get('deblurring_objective', 'l1'))
        self.maxiter = kwargs.get('maxiter', 500)
        self.relax = kwargs.get('relax', 1.0)
        self.step_size = kwargs.get('step_size', 0.1)
        self.gamma = kwargs.get('gamma', 0.5)
        self.eigvalK = matops.periodic_conv_eigvals(k, b.shape)
        self.eigvalD1 = matops.periodic_conv_eigvals(kernel_D1, b.shape)
        self.eigvalD2 = matops.periodic_conv_eigvals(kernel_D2, b.shape)
        self.eigvalK_cj = np.conjugate(self.eigvalK)
        self.eigvalD1_cj = np.conjugate(self.eigvalD1)
        self.eigvalD2_cj = np.conjugate(self.eigvalD2)


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
        return matops.fft_conv2d(self.eigvalK_cj,y[0]) \
            + matops.fft_conv2d(self.eigvalD1_cj,y[1]) \
                + matops.fft_conv2d(self.eigvalD2_cj,y[2])
    
    def applyBig(self,t,x):
        '''
        Applies (I+tA^TA)^(-1)(x)

        Args:
            t (int): step size
            x (np.ndarray): x represented as an NxN matrix
        '''
        eigvals = matops.eigvals_mat(self.eigvalK_cj,
                                     self.eigvalK,
                                     self.eigvalD1_cj,
                                     self.eigvalD1_cj,
                                     self.eigvalD2_cj,
                                     self.eigvalD2,
                                     t)
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
    def __init__(self, k, b, **kwargs):
        super().__init__(k, b, **kwargs)
        self.foo = kwargs.get('foo', 'bar')
        self.z1 = matops.fft_conv2d(eigvals=self.eigvalK, x=b)
        self.z2 = matops.apply_A(kernel_eigvals=self.eigvalK,
                                  D1_eigvals=self.eigvalD1, 
                                  D2_eigvals=self.eigvalD2, 
                                  x=b)
        self.proxf = prox_store.get("box")
        self.proxg2 = prox_store.get("iso")
        self.eigvals_mat = matops.eigvals_mat(kernel_eigvals_conj=self.eigvalK_cj, 
                                              kernel_eigvals=self.eigvalK, 
                                              grad1_eigvals_conj=self.eigvalD1_cj, 
                                              grad1_eigvals=self.eigvalD1, 
                                              grad2_eigvals_conj=self.eigvalD2_cj, 
                                              grad2_eigvals=self.eigvalD2, 
                                              t=1.0)

    def solve(self):
        """
        f(x) = indicator of {x, a finite dimenional vector: 0 <= x_i <= 1 for all dimenions i}.
        g(y) = ||y1 − b||_l1 + gamma||(y2, y3)||_iso, (y1, (y2, y3)) = [K, D]z
        """
        b, z1, z2 = self.b, self.z1, self.z2
        print(z1.shape)
        # t = self.step_size
        # for k in range(self.maxiter):
        #     x = self.proxf(t=t, x=z1)
        #     y = self.proxdbl(t=t, y=z2[:,:,0], b=b) + self.proxg2(t=t, g=self.gamma, w1=z2[:,:,1], w2=z2[:,:,2])

        #     in_vector = 2 * x - z1 + matops.apply_A(kernel_eigvals=self.eigvalK_cj, D1_eigvals=self.eigvalD1_cj, D2_eigvals=self.eigvalD2_cj, x=(2 * y - z2))
        #     u = matops.fft_invert(eigvals_mat=self.eigvals_mat, x=in_vector)
        #     v = matops.apply_A(kernel_eigvals=self.eigvalK, D1_eigvals=self.eigvalD1, D2_eigvals=self.eigvalD2, x=u)

        #     z1 = z1 + self.relax * (u - x)

        #     z2 = z2 + self.relax * (v - y)


        # return self.proxf(t=t, x=z1)


if __name__ == "__main__":
    # usage
    params = {'deblurring_objective': 'l2', 'maxiter': 500}
    solver = DouglasRachfordPrimal(k=np.array([1, 2, 3])[:, None], x=np.random.random((128, 128)), **params)