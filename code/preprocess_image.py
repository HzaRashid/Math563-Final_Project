
import numpy as np
from PIL import Image


def image_to_numpy(image):
    """
    Converts raw image to numpy array

    Args:
        path_to_image (str): file path to original image

    Returns:
        np array of image pixels
    """
    return np.array(image)


def box_prox(image):
    """
    Scales image pixels between 0 and 1 with min-max scaling:

    Let I := input image (matrix),
    m := min pixel value in I,
    M:= Max pixel value in I,

    Returns (I - m)/(M - m), where the operations are pointwise

    Args: 
        image: PIL Image object

    Returns: 
        Min-Max scaled (from original-)image (pixels in range (0,1)) 
    """
    min_ = image.min()
    return (image - min_)/(image.max() - min_)


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