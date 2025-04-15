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
   
# Our default hyperparameters are optimized hyperparameter for the default kernel and noise using the optuna package
params_drp = {
            "relax": 1.971031969842028,
            "step_size": 0.3338244230304595,
            "gamma": 0.03880489242481133
        }
params_drpd = {
            "relax": 1.1780472793954466,
            "step_size": 0.9879224190808178,
            "gamma": 0.04411059218704158
        }
params_admm = {
            "relax": 1.1390324895476753,
            "step_size": 1.9544374002339948,
            "gamma": 0.03960543951522183
        }

params_champock = {
            "step_size": 0.8332046663820682,
            "gamma": 0.03481857181257442,
            "step_size2": 0.3858625337245462
        }

dr_primal = DouglasRachfordPrimal(k=k, shape=shape, maxiter=100, deblurring_objective='l1',
                                **params_drp)

dr_dual = DouglasRachfordPrimalDual(k=k, shape=shape, maxiter=100, deblurring_objective='l1',**params_drpd)

admm = ADMM(k=k, shape=shape, maxiter=100, deblurring_objective='l1', **params_admm)

cham_pock = ChambollePock(k=k, shape=shape, maxiter=100, deblurring_objective='l1', **params_champock)

test_solver(dr_primal, "Primal", b, x)
test_solver(dr_dual, "Dual", b, x)
test_solver(admm, "ADMM", b, x)
test_solver(cham_pock, "ChambollePock", b, x)