import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import preprocess_image as pci
from kernel import gaussian_kernel
from scipy.linalg import circulant
from skimage import util
from PIL import Image

def blur_image(path_to_image, 
               show_before=False, 
               show_after=False, 
               kernel=gaussian_kernel([15, 15], 5),
               noise_mode="s&p",
               noise_density=0.1
               ):
    """
    Blurs and adds noise to an image
    """
    # Convert to grayscale and normalize
    gray_img = pci.rgb2gray(path_to_image=path_to_image) # PIL Image
    gray_img_np = pci.image_to_numpy(gray_img)
    x = pci.box_prox(gray_img_np)

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
    path_to_image = os.path.join(cur_dir, '../testimages/cameraman.jpg')

    b, x = blur_image(path_to_image=path_to_image,
                      show_before=True,
                      show_after=True
                      )
    k = gaussian_kernel([15, 15], 5)

    
    """
    - algorithms are given: x=true_image, k=kernel, b=blurred_and_noised_image
    - algorithms start by initializing Kx and Dx
    - get Kx -> pass the kernel to scipy.signal.convolve2d
    - get Dx -> use a python equivalent of their matlab code
    """