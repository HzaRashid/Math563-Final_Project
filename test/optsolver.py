import numpy as np
import periodic_mat_util as matops
import prox_util as proxops

prox_store = {
    'l1': proxops.l1prox,
    'l2': proxops.l2prox,
    'iso': proxops.iso_prox,
    'box': proxops.box_prox,
    }


kernel_D1 = np.array([-1, 1])[:, None]
kernel_D2 = np.array([-1, 1])[None, :]

def build_eigval_store(key_kernel_dict, shape):
    return {key: matops.periodic_conv_eigvals(val, shape) 
            for key, val in key_kernel_dict.items()}


class OptSolver():
    def __init__(self, k, b, **kwargs):
        self.b = b
        self.dbl = kwargs.get('deblurring_objective', 'l1')
        self.proxdbl = prox_store.get(self.dbl)
        self.maxiter = kwargs.get('maxiter', 500)
        self.relax = kwargs.get('relax', 0.5)
        self.step_size = kwargs.get('step_size', 0.1)
        self.gamma = kwargs.get('gamma', 0.1)

        self.eigval = build_eigval_store({'K': k, 'D1': kernel_D1, 'D2': kernel_D2}, b.shape)

        self.conj_eigval = {key: np.conjugate(val) for key, val in self.eigval.items()}


    def applyA(self,x):
        '''
        Applies Ax

        Args:
            x (np.ndarray): x represented as an NxN matrix
        '''
        return matops.apply_A(kernel_eigvals=self.eigval['K'], 
                              eigvals_D1=self.eigval['D1'], 
                              eigvals_D2=self.eigval['D2'],
                              x=x)
    
    def applyAT(self,y):
        """
        Applies A^Ty

        Args:
            y (np.ndarray): y represented as [y1,y2,y3] concatenated in a
            third dimension, each yi represented as an NxN matrix
        """
        return matops.apply_A_conj(eigval_conj_arr=[self.conj_eigval['K'], 
                                                    self.conj_eigval['D1'], 
                                                    self.conj_eigval['D2']
                                                    ], y=y)
    
    def applyBig(self,t,x):
        '''
        Applies (I+tA^TA)^(-1)(x)

        Args:
            t (int): step size
            x (np.ndarray): x represented as an NxN matrix
        '''
        eigvals = self.apply_eigvals_mat(t)
        return matops.fft_invert(eigvals,x)
    
    def applyBigT(self,t,y):
        '''
        Applies (I+tAA^T)^(-1)(y)
        Args:
            t (int): step size
            y (np.ndarray): y represented as [y1,y2,y3] concatenated in a
            third dimension, each yi represented as an NxN matrix       
        '''
        
        newy = y + t * self.applyA(self.applyAT(y)) # (I+tAA^T)y
        return np.fft.ifft2(np.square(np.fft.fft2(y)) / newy) # (I+tAA^T)^(-1)y

    def apply_eigvals_mat(self, t):
        return matops.eigvals_mat(conj_eigvals_K=self.conj_eigval['K'],
                                  eigvals_K=self.eigval['K'], 
                                  conj_eigvals_D1=self.conj_eigval['D1'], 
                                  eigvals_D1=self.eigval['D1'], 
                                  conj_eigvals_D2=self.conj_eigval['D2'], 
                                  eigvals_D2=self.eigval['D2'], 
                                  t=t)


class DouglasRachfordPrimal(OptSolver):
    def __init__(self, k, b, **kwargs):
        super().__init__(k, b, **kwargs)
        self.z1 = matops.fft_conv2d(eigvals=self.eigval['K'], x=b)
        self.z2 = self.applyA(b)
        self.proxf = prox_store.get("box")
        self.proxg2 = prox_store.get("iso")
        self.eigvals_mat = self.apply_eigvals_mat(t=1.0)

    def solve(self):
        """
        f(x) = indicator of {x, a finite dimenional vector: 0 <= x_i <= 1 for all dimenions i}.
        g(y) = ||y1 − b||_l1 + gamma||(y2, y3)||_iso, (y1, (y2, y3)) = [K, D]z
        """
        b, z1, z2, t = self.b, self.z1, self.z2, self.step_size

        eps = np.zeros(self.maxiter)
        # Whether eps is computed using l1 or l2
        err_ord = int(self.dbl[1])
        
        for i in range(self.maxiter):
            x, y = self.resolvent_A(z1, z2, b, t)
            u, v = self.resolvent_B(x, y, z1, z2)
            z1 = z1 + self.relax * (u - x)
            z2 = z2 + self.relax * (v - y)

            # For now this is just ||Kx-b||
            # @TODO implement iso and add it here
            Kx = matops.fft_conv2d(self.eigval['K'],x)
            eps[i] = np.linalg.norm(Kx-b, ord=err_ord)
        return self.proxf(t=t, x=z1), eps

    
    def resolvent_A(self, z1, z2, b, t):
        return (self.proxf(t=t, x=z1), 
                matops.cat_mats([
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
        u = matops.fft_invert(eigvals_mat=self.eigvals_mat, 
                              x=2 * x - z1 + self.applyAT(2 * y - z2))
        return (u, self.applyA(u))


if __name__ == "__main__":
    # usage
    params = {'deblurring_objective': 'l2', 'maxiter': 500}
    solver = DouglasRachfordPrimal(k=np.array([1, 2, 3])[:, None], b=np.random.random((128, 128)), **params)