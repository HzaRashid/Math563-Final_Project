# usage
from blurkit.optsolver import ADMM
import numpy as np

k = np.array([1, 2, 3])[:, None] # dummy kernel
b = np.random.random((128, 128)) # dummy image

params = {'deblurring_objective': 'l1', 'maxiter': 100}
solver = ADMM(k=np.array([1, 2, 3])[:, None], shape=(128,128), **params)
solver.solve(b=b, # blurred image
             if_track=True, # whether or not to track objective
             stop_criterion=1e-1 # halt and return when objective reaches this value
            )