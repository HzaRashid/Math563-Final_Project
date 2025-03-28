import cv2
import numpy as np

def gaussian_kernel(hsize=[15,15], sigma=1.0):
    """
    Creates a 2D Gaussian kernel similar to MATLAB's fspecial('gaussian', hsize, sigma).

    Args:
        hsize (list[int]): Kernel of size (hsize[0] x hsize[1])
                                     will be created.
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        np.ndarray: A 2D Gaussian kernel normalized so that its sum is 1.
    """
    # create matrix of zeros with middle pixel set to 1.0
    base = np.zeros(shape=(hsize[0], hsize[1])) 
    centerX = hsize[0] // 2
    centerY = hsize[1] // 2
    base[centerX, centerY] = 1.0
    # applying gaussian blur to such matrix is returned (equivalent to the kernel)
    return cv2.GaussianBlur(base, ksize=(hsize[0], hsize[1]), sigmaX=sigma, sigmaY=sigma)


if __name__ == "__main__":
    # Example usage:
    kernel = gaussian_kernel([15,15], 5)
    print(kernel)
