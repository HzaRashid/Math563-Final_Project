import os
from blurkit import kernel
from blurkit.test_util import blur_image, test_solver
from blurkit.optsolver import DouglasRachfordPrimal, DouglasRachfordPrimalDual, ADMM, ChambollePock

cur_dir = os.path.dirname(__file__)
my_path = './testimages/cameraman.jpg'
path_to_image = os.path.join(cur_dir, my_path)

shape = (256, 256)
k = kernel.gaussian_kernel()
noise_args = {
    's&p_noise': {
        'mode': 's&p',
        'amount': 0.1
    },
    'gaussian_noise':{
        'mode':'gaussian',
        'mean':0.0,
        'var':0.001
        }
    }

b, x = blur_image(path_to_image=path_to_image,
                  shape=shape,
                  show_before=False,
                  show_after=False,
                  kernel=k,
                  noise_args=noise_args['s&p_noise']
                  )
   
# You can change hyperparameters here
params = {"relax": 1.5, "step_size": 0.8, "gamma": 0.03}
params_champock = {"relax": 1.8, "step_size": 0.4, "step_size2": 0.4, "gamma": 0.03}

dr_primal = DouglasRachfordPrimal(k=k, shape=shape, maxiter=100, deblurring_objective='l1',
                                **params)

dr_dual = DouglasRachfordPrimalDual(k=k, shape=shape, maxiter=100, deblurring_objective='l1',**params)

admm = ADMM(k=k, shape=shape, maxiter=100, deblurring_objective='l1', **params)

cham_pock = ChambollePock(k=k, shape=shape, maxiter=100, deblurring_objective='l1', **params_champock)

test_solver(dr_primal, "Primal", b, x)
test_solver(dr_dual, "Dual", b, x)
test_solver(admm, "ADMM", b, x)
test_solver(cham_pock, "ChambollePock", b, x)