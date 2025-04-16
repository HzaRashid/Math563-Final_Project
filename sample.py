import os
import numpy as np
from blurkit import kernel
import matplotlib.pyplot as plt
from blurkit.test_util import blur_image # blurs image
from blurkit.optsolver import DouglasRachfordPrimal, DouglasRachfordPrimalDual, ADMM, ChambollePock

# --------* wrapper for algorithm evaluation *--------
def test_solver(solver, name, b, x):
    """
    Runs the chosen algorithm and shows
    the deblurred image and a graph for the objective value.

    direct copy of 'test_solver' from blurkit.test_util
    """
    # --------* Sample usage (each algorithms's solve method has the same signature) *--------
    # res=recovered image, eps=list of objective values across iterations (if if_track==True)
    res, eps = solver.solve(b, # blurred image
                            if_track=True, # whether or not to track objective
                            stop_criterion=3.5e-3 # halt early and return when objective reaches this value
                            ) 
    
    # displays original image
    plt.subplot(2,3,1)
    plt.imshow(x, cmap='gray')
    plt.title("Orignal")
    plt.axis('off') 
    # displays blurred image
    plt.subplot(2,3,2)
    plt.imshow(b, cmap='gray')
    plt.title("Blurred")
    plt.axis('off')
    # displays recovered image
    plt.subplot(2,3,3)
    plt.imshow(np.real(res), cmap='gray')
    plt.title(name)
    plt.axis('off')

    # plot objective value across iterations
    plt.subplot(2,1,2)
    plt.plot(eps)
    plt.yscale("log")
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Convergence')
    plt.grid(True)
    plt.show()

    # x0=recovered image, eps=list[final objective value] when if_track==False (default)
    x0, eps = solver.solve(b) 
    print(name, 
          'average 1-norm difference from unblurred image:', 
          np.linalg.norm(x-x0, ord=1)/np.size(x))
    print(name, 
          'average 2-norm difference from unblurred image:', 
          np.linalg.norm(x-x0, ord=2)/np.size(x)) 
    # we normalize by image size to get per-pixel error


# --------* load image, change my_path accordingly *--------
cur_dir = os.path.dirname(__file__)
my_path = './testimages/cameraman.jpg'
path_to_image = os.path.join(cur_dir, my_path)

# --------* image reshaping and blur settings *--------
shape = (256, 256) # image shape
k = kernel.gaussian_kernel() # kernel of convolution
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
    } # noise types and settings

# --------* apply image blurring + noising *--------
# b=blurred image, x=true (grayscaled, normalized, and reshaped) image
b, x = blur_image(path_to_image=path_to_image,
                  shape=shape,
                  show_before=False,
                  show_after=False,
                  kernel=k,
                  noise_args=noise_args['s&p_noise']
                  ) # blurs image with specified kernel and noise settings
# we encourage the reader to look at our implementation of blur_image in blurkit/test_util.py

# --------* Solver hyperparameters *--------
# obtained with optuna for default Gaussian kernel and s&p noise
# not expected to work as well under different settings
params_drp = { # for Primal Douglas-Rachford
            "relax": 1.971031969842028,
            "step_size": 0.3338244230304595,
            "gamma": 0.03880489242481133
            }
params_drpd = { # for Primal Dual Douglas-Rachford
            "relax": 1.1780472793954466,
            "step_size": 0.9879224190808178,
            "gamma": 0.04411059218704158
            }
params_admm = { # for ADMM
            "relax": 1.1390324895476753,
            "step_size": 1.9544374002339948,
            "gamma": 0.03960543951522183
            }
params_champock = { # for Chambolle Pock
            "step_size": 0.8332046663820682,
            "gamma": 0.03481857181257442,
            "step_size2": 0.3858625337245462
            }

# --------* instantiate solvers *--------
# Primal Douglas-Rachford
dr_primal = DouglasRachfordPrimal(k=k, 
                                  shape=shape, 
                                  maxiter=100, 
                                  deblurring_objective='l1', 
                                  **params_drp)
# Primal Dual Douglas-Rachford
dr_dual = DouglasRachfordPrimalDual(k=k, 
                                    shape=shape, 
                                    maxiter=100, 
                                    deblurring_objective='l1',
                                    **params_drpd)
# ADMM
admm = ADMM(k=k, 
            shape=shape, 
            maxiter=100, 
            deblurring_objective='l1', 
            **params_admm)
# Chambolle Pock
cham_pock = ChambollePock(k=k, 
                          shape=shape,
                          maxiter=100, 
                          deblurring_objective='l1', 
                          **params_champock)

# --------* run solvers *--------
test_solver(dr_primal, "Primal", b, x)
test_solver(dr_dual, "Dual", b, x)
test_solver(admm, "ADMM", b, x)
test_solver(cham_pock, "ChambollePock", b, x)