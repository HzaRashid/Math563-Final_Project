import periodic_mat_util as mat
import prox_util
import numpy as np


class OptUtil:
    """
    algorithm utility functions using saved state (leading to faster algorithms), 
    such as image shape, eigenvalues of 2D DFT
    of convolution kernel and discrete gradient operator,
    and the proximal operator of the user-selected deblurring objective.
    """
    def __init__(self, key_kernel_dict, shape, t, deblurring_prox):
        # Eigenvalue for K,D1,D2
        self.eigval = build_eigval_store(key_kernel_dict=key_kernel_dict, shape=shape)
        # Eigenvalue for K^T,D1^T,D2^T
        self.conj_eigval = {key: np.conjugate(val) for key, val in self.eigval.items()}
        # Eigenvalue for A^TA
        self.big_eigval = mat.eigvals_mat(conj_eigvals_K=self.conj_eigval['K'],
                                           eigvals_K=self.eigval['K'], 
                                           conj_eigvals_D1=self.conj_eigval['D1'], 
                                           eigvals_D1=self.eigval['D1'], 
                                           conj_eigvals_D2=self.conj_eigval['D2'], 
                                           eigvals_D2=self.eigval['D2'])
        self.proxdbl = deblurring_prox
        self.proxiso = prox_util.iso_prox

    def applyA(self,x):
        '''
        Applies Ax

        Args:
            x (np.ndarray): x represented as an NxN matrix
        '''
        return mat.apply_A(kernel_eigvals=self.eigval['K'], 
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
        return mat.apply_A_conj(eigval_conj_arr=[self.conj_eigval['K'], 
                                                 self.conj_eigval['D1'], 
                                                 self.conj_eigval['D2']], 
                                y=y)
    
    def applyBig(self,x,t=1):
        '''
        Applies (I+tA^TA)^(-1)(x)

        Args:
            x (np.ndarray): x represented as an NxN matrix
            t (int): step size
        '''
        big = self.big_eigval
        # Eigenvalues of I+tA^TA
        eigval = np.ones_like(big) + t*big
        return mat.fft_invert(eigval, x)

    def conjugate_prox(self, prox_op, y, t):
        """
        Returns [prox(t*f_conj)](x) for 
        some convex, proper, and lower semi-continuous function (t*f)(x)
        a using moreau decomposition variant (t > 0 a scalar, * denotes multiplication).

        Args:
            prox_op (function: np.ndarray -> np.ndarray): 
                proximal operator of a function f.
            y (ndarray): input to the conjugate proximal operator.
            t (float): positive scalar.
        """
        return y - t * prox_op(y / t)
    
    
    def objective_prox(self, y, b, t, g):
        '''
        Computes prox_tg(y), where g = |y[0]-b| + gamma*iso(y[1],y[2])
        Args:
            y: matrix y concatenated in the 3rd dimension
            b: matrix of blurred image
            t: step size for tg
            g: gamma
        Returns:
            prox_tg(y) concatenated in the 3rd dimension
        '''
        return mat.cat_mats([
            self.proxdbl(t=t, y=y[:,:,0], b=b),
            *self.proxiso(t=t, g=g, w1=y[:,:,1], w2=y[:,:,2])
            ])


def build_eigval_store(key_kernel_dict, shape):
    """
    Mutates the given dictionary 
    such that the corresponding values are the eigenvalues
    of the kernel operators (assuming periodic boundary conditions).

    Args:
        key_kernel_dict (dict): keys=variable name (str), 
                                values=kernel of convolution (ndarray).
        shape (tuple): shape of image to be convolved with

    Returns:
        dictionary whose keys are the same as the input, 
        but values are the the eigenvalues of the kernel operator (dict).
    """
    key_kernel_dict = {key: mat.periodic_conv_eigvals(val, shape) 
                       for key, val in key_kernel_dict.items()} 
    return key_kernel_dict


