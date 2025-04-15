import os
from skimage import util
# import preprocess_image as pci
from . import preprocess_image as pci
import matplotlib.pyplot as plt
from . import kernel
import time
from scipy.signal import convolve2d
import numpy as np

default_noise_args = {
    'mode':'s&p',
    'amount':0.1
}

def blur_image(image=None,
               path_to_image=None,
               shape = (256,256), 
               show_before=False, 
               show_after=False, 
               kernel=kernel.gaussian_kernel([15, 15], 5),
               noise_args=default_noise_args
               ):
    """
    Blurs and adds noise to an image
    """
    if path_to_image:
        # Convert to grayscale and normalize
        gray_img = pci.rgb2gray(path_to_image=path_to_image, shape=shape) # PIL Image
        gray_img_np = pci.image_to_numpy(gray_img)
        x = pci.normalize_image(gray_img_np)
    else: 
        x = image # already processed

    # blur step
    Kx = convolve2d(x, 
                    kernel,
                    mode='same',
                    boundary='wrap' # 'wrap' == periodic boundary conditions
                    )  

    # noise step
    Kx_plus_n = util.random_noise(Kx, **noise_args)

    if show_before:
        plt.figure("Image before blurring")
        plt.imshow(x, cmap='gray')
        plt.axis('off')
        plt.show()
    if show_after:
        plt.figure("Image after blurring and noise")
        plt.imshow(Kx_plus_n, cmap='gray')
        plt.axis('off')
        plt.show()

    return Kx_plus_n, x


def test_solver(solver, name, b, x):
# This wrapper runs the chosen algorithm and shows:
# the deblurred image and a graph for the objective value
    res, eps = solver.solve(b, if_track=True, 
                            stop_criterion=3.5e-3
                            )
    plt.subplot(2,3,1)
    plt.imshow(x, cmap='gray')
    plt.title("Orignal")
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.imshow(b, cmap='gray')
    plt.title("Blurred")
    plt.axis('off')
    
    plt.subplot(2,3,3)
    plt.imshow(np.real(res), cmap='gray')
    plt.title(name)
    plt.axis('off')

    plt.subplot(2,1,2)
    plt.plot(eps)
    plt.yscale("log")
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Convergence')
    plt.grid(True)
    plt.show()

    start = time.time()
    x0, eps = solver.solve(b)
    end = time.time()
    print(name, "time:", end-start)
    print(name, 'average 1-norm difference from unblurred image:', np.linalg.norm(x-x0, ord=1)/np.size(x))
    print(name, 'average 2-norm difference from unblurred image:', np.linalg.norm(x-x0, ord=2)/np.size(x)) 

    
if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    my_path = '../testimages/cameraman.jpg'
    # my_path = 'testimages\\mcgill.jpg' # <-- MODIFY THIS AS NEEDED
    path_to_image = os.path.join(cur_dir, my_path)
    gaussian_noise_args = {
    'mode':'gaussian',
    'mean':0.0,
    'var':0.001
    }

    sp_noise_args = {
        'mode': 's&p',
        'amount': 0.1
    }
    shape = (256, 256)
    k = kernel.gaussian_kernel()
    b, x = blur_image(path_to_image=path_to_image,
                      shape=shape,
                      show_before=False,
                      show_after=False,
                      kernel=k,
                      noise_args=sp_noise_args
                      )

    
    """
    Sample test
    """
    from optsolver import DouglasRachfordPrimal, DouglasRachfordPrimalDual, ADMM, ChambollePock
    import numpy as np

    
    # You can change hyperparameters here
    params = {"relax": 1.5, "step_size": 0.8, "gamma": 0.03}
    params_champock = {"step_size": 0.4, "step_size2": 0.4, "gamma": 0.03}

    dr_primal = DouglasRachfordPrimal(k=k, shape=shape, maxiter=100, deblurring_objective='l1',
                                   **params)

    dr_dual = DouglasRachfordPrimalDual(k=k, shape=shape, maxiter=100, deblurring_objective='l1',**params)

    admm = ADMM(k=k, shape=shape, maxiter=100, deblurring_objective='l1', **params)

    cham_pock = ChambollePock(k=k, shape=shape, maxiter=100, deblurring_objective='l1', **params_champock)
    
    test_solver(dr_primal, "Primal", b)
    test_solver(dr_dual, "Dual", b)
    test_solver(admm, "ADMM", b)
    test_solver(cham_pock, "ChambollePock", b)