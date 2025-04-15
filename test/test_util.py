import os
from skimage import util
import preprocess_image as pci
import matplotlib.pyplot as plt
import kernel
import time
from scipy.signal import convolve2d

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

    def test_solver(solver, name, b):
    # This wrapper runs the chosen algorithm and shows:
    # the deblurred image and a graph for the objective value
        res, eps = solver.solve(b, if_track=True)
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
        plt.loglog(eps)
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

    # You can change hyperparameters here
    best_hps = {
    'DouglasRachfordPrimal': {
            "relax": 1.8542801426300373,
            "step_size": 0.5303865420997071,
            "gamma": 0.037467313515114786
        },
    'DouglasRachfordPrimalDual':{
            "relax": 1.549599094560696,
            "step_size": 0.940056822261643,
            "gamma": 0.02280159605226321
        },
    'ADMM':{
            "relax": 1.160219522869758,
            "step_size": 1.6288522037705238,
            "gamma": 0.03851698444156989
        },
    'ChambollePock':{
            "relax": 1.888640583759617,
            "step_size": 0.36069547095563936,
            "step_size2": 0.4993573208985405,
            "gamma": 0.045716177124372946
        }
}

    dr_primal = DouglasRachfordPrimal(k=k, shape=shape, maxiter=100, deblurring_objective='l1',
                                   **best_hps['DouglasRachfordPrimal'])

    dr_dual = DouglasRachfordPrimalDual(k=k, shape=shape, maxiter=100, deblurring_objective='l1',**best_hps['DouglasRachfordPrimalDual'])

    admm = ADMM(k=k, shape=shape, maxiter=100, deblurring_objective='l1', **best_hps['ADMM'])

    cham_pock = ChambollePock(k=k, shape=shape, maxiter=100, deblurring_objective='l1', **best_hps['ChambollePock'])
    
    test_solver(dr_primal, "Primal", b)
    test_solver(dr_dual, "Dual", b)
    test_solver(admm, "ADMM", b)
    test_solver(cham_pock, "ChambollePock", b)