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


def disk_kernel(r=8):
    """
    Creates a 2D disk kernel similar to MATLAB's fspecial('disk', radius).

    Args:
        radius (int): Radius of a disk-shaped filter

    Returns:
        np.ndarray: A 2D disk kernel normalized so that its sum is 1.
    """

    base = np.zeros(shape=(2*r+1, 2*r+1))
    # matrix of zeros with a disk of ones in the middle
    h = cv2.circle(base, center=(r,r), radius=r, color=(1,), thickness=-1)
    # average to one and return
    return h/h.sum()


def motion_kernel(len=9, theta=0.0):
    """
    Creates a 2D motion kernel similar to MATLAB's fspecial('motion', len, theta), but quadruple the size.

    Args:
        len (int): Length of motion
        theta (float): Angle of motion in degrees counterclockwise from horizontal pointing right 

    Returns:
        np.ndarray: A 2D motion kernel normalized so that its sum is 1.
    """
    base = np.zeros(shape=(2*len,2*len))
    # radians
    theta = theta/180*np.pi
    # compute the location of the second point
    p = (int(len+np.cos(theta)*len),int(len+np.sin(theta)*len))
    # matrix of zeros with a line of ones at the given angle
    h = cv2.line(base, pt1=(len,len), pt2=p, color=(1,), thickness=1)
    # average to 1 and return
    return h/h.sum()

if __name__ == "__main__":
    # Example usage:
    kernel = motion_kernel()
    print(kernel)