import numpy as np
import periodic_mat_util as matops
import prox_util as proxops

kernel_D1 = np.array([-1, 1])[:, None]
kernel_D2 = np.array([-1, 1])[None, :]

class DouglasRachfordPrimal():
    def __init__(self,
                 objective='l1',
                 maxiter=500,
                 relax=1.0,
                 step_size=0.1,
                 gamma=0.5
                 ):
        self.objective = objective
        self.maxiter = maxiter
        self.relax = relax
        self.step_size = step_size
        self.gamma = gamma

    def solve(self, kernel, blurred_image):
        """
        f(x) = indicator of {x, a finite dimenional vector: 0 <= x_i <= 1 for all dimenions i}.
        g(y) = ||y1 − b||_l1 + gamma||(y2, y3)||_iso, (y1, (y2, y3)) = [K, D]z
        """
        pass