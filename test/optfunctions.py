import numpy as np
import periodic_mat_util as matops
from prox_util import iso_prox, box_prox

def solve(self, b, z1, z2, t, track_objective=False, eps=None):
    """
    f(x) = indicator of {x, a finite dimenional vector: 0 <= x_i <= 1 for all dimenions i}.
    g(y) = ||y1 − b||_l1 + gamma||(y2, y3)||_iso, (y1, (y2, y3)) = [K, D]z
    """
    
    for i in range(self.maxiter):
        x, y = self.resolvent_A(z1, z2, b, t)
        u, v = self.resolvent_B(x, y, z1, z2)
        z1 = z1 + self.relax * (u - x)
        z2 = z2 + self.relax * (v - y)

        if eps is not None:
            # For now this is just ||Kx-b||
            # @TODO implement iso and add it here
            Kx = matops.fft_conv2d(self.eigval['K'],x)
            eps[i] = np.linalg.norm(Kx-b, ord=self.err_ord)

    return self.proxf(t=t, x=z1), eps or np.linalg.norm(matops.fft_conv2d(self.eigval['K'],x) - b, ord=self.err_ord)

def resolvent_A(self, z1, z2, b, t):
    return (self.proxf(t=t, x=z1), 
            matops.cat_mats([
                self.proxdbl(t=t, 
                                y=z2[:,:,0], 
                                b=b),  # shape: (width, height) (of b)
                *self.proxg2(t=t, 
                                g=self.gamma, 
                                w1=z2[:,:,1], 
                                w2=z2[:,:,2]) # (2, width, height)
            ]))

def resolvent_B(self, x, y, z1, z2):
    # (I + A^TA)(-1)(2x_k − z1_(k-1)+ A^T(2y_k − z2_(k-1)))
    u = matops.fft_invert(eigvals_mat=self.eigvals_mat, 
                            x=2 * x - z1 + self.applyAT(2 * y - z2))
    return (u, self.applyA(u))