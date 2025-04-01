import os
from skimage import util
import preprocess_image as pci
import matplotlib.pyplot as plt
import kernel
import time
from scipy.signal import convolve2d


def blur_image(path_to_image, 
               show_before=False, 
               show_after=False, 
               kernel=kernel.gaussian_kernel([15, 15], 5),
               noise_mode="s&p",
               noise_density=0.1
               ):
    """
    Blurs and adds noise to an image
    """
    # Convert to grayscale and normalize
    gray_img = pci.rgb2gray(path_to_image=path_to_image) # PIL Image
    gray_img_np = pci.image_to_numpy(gray_img)
    x = pci.normalize_image(gray_img_np)

    # blur step
    Kx = convolve2d(x, 
                    kernel,
                    mode='same',
                    boundary='wrap' # 'wrap' == periodic boundary conditions
                    )  

    # noise step
    Kx_plus_n = util.random_noise(Kx, 
                                  mode=noise_mode, 
                                  amount=noise_density
                                  )

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
    # my_path = 'testimages\\cameraman.jpg' # <-- MODIFY THIS AS NEEDED
    path_to_image = os.path.join(cur_dir, my_path)

    k = kernel.motion_kernel(len=15)
    b, x = blur_image(path_to_image=path_to_image,
                      show_before=True,
                      show_after=True,
                      kernel=k)

    
    """
    Sample test
    """
    from optsolver import DouglasRachfordPrimal
    import numpy as np

    start = time.time()
    drp_solver = DouglasRachfordPrimal(k=k, b=b, maxiter=500, deblurring_objective='l1')

    result, eps = drp_solver.solve()
    end = time.time()
    print("time: ", end-start)    

    plt.figure("algo output")
    plt.imshow(np.real(result), cmap='gray')
    plt.axis('off')
    plt.show()
    plt.figure("Error")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.loglog(eps)
    plt.show()
