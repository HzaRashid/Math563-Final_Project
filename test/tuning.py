import optuna
from preprocess_image import *
from test_util import blur_image
from optsolver import DouglasRachfordPrimal, DouglasRachfordPrimalDual, ADMM, ChambollePock
import os
import numpy as np
from kernel import motion_kernel, gaussian_kernel, disk_kernel

kernels = [motion_kernel, gaussian_kernel, disk_kernel]
cur_dir = os.path.dirname(__file__)
my_path = '../testimages'
path_to_images = os.path.join(cur_dir, my_path)
image_paths = [os.path.join(path_to_images, fname) for fname in os.listdir(path_to_images)]
images = [normalize_image(image_to_numpy(rgb2gray(path_to_image=img_path))) for img_path in image_paths]
one_norms = [np.linalg.norm(x, ord=1) for x in images]

def objective(trial):
    params = {
        # select one objective at a time, otherwise bias towards l2
        "deblurring_objective": trial.suggest_categorical('deblurring_objective', ['l1']),
        "maxiter": trial.suggest_int('maxiter', 100, 200),
        "relax": trial.suggest_float('relax', 1e-5, 2.0),
        "step_size": trial.suggest_float('step_size', 1e-5, 1e-1),
        "gamma": trial.suggest_float('gamma', 5e-2, 1.0)
    }
    # if algo_name == 'ChambollePock':
    #     params.update({"step_size2": trial.suggest_float('step_size2', 1e-2, 2.0)})
    #     params.pop("relax")

    total_loss = 0
    img = images[0]
    imgnorm = one_norms[0]
    # for img, imgnorm in zip(images, one_norms):
    """
    we can test the algorithm on many images, 
    kernels, noise types noise densities, etc, 
    to determine which setting generalize best 
    but end up sacrificing a bit for any 
    particular one of these settings and images
    """
    for kernel in kernels: 
        k = kernel() # initial kernel with default params
        solver = DouglasRachfordPrimal(k=k, shape=(256, 256), **params)

        b, x = blur_image(image=img,
                          kernel=k,
                          noise_args={'mode': 's&p', 'amount': 0.1}
                          )
        

        out, loss = solver.solve(b, if_track=True)

        # total_loss += loss[-1]
        total_loss += np.linalg.norm((out - x)/imgnorm, ord=1)

    return total_loss


    
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5) 
    print()
    print("best hyperparameters:", study.best_params)