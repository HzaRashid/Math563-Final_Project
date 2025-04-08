
import numpy as np
from PIL import Image


def image_to_numpy(image_mat):
    """
    Converts raw image to numpy array

    Args:
        image_mat (list[list[float]]): raw image as 2D matrix (typically PIL format)

    Returns:
        np array of image pixels
    """
    return np.array(image_mat)


def normalize_image(image_mat):
    """
    Scales image pixels between 0 and 1 with min-max scaling:

    Let I := input image (ndarray),
    m := min pixel value in I,
    M:= Max pixel value in I,

    Returns (I - m)/(M - m), where the operations are pointwise

    Args: 
        image (ndarray): 2D numpy array of image

    Returns: 
        Min-Max scaled image (pixels in range (0,1)) (ndarray).
    """
    min_ = image_mat.min()
    return (image_mat - min_)/(image_mat.max() - min_)


def rgb2gray(path_to_image):
    """
    Converts RGB image to grayscale and saves it

    Args:
        path_to_image (str): file path to original image
        destination_path (str): file path to save grayscaled image

    Returns:
        None
    """
    return Image.open(fp=path_to_image).convert('L')