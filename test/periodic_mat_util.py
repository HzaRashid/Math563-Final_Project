import numpy as np
import scipy.ndimage as ndi
from typing import Sequence, List

def periodic_conv_eigvals(kernel: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    """
    Computes the eigenvalues of the 2D DFT of the convolution kernel.

    Args:
        kernel (np.ndarray): Correlation kernel (e.g. for Gaussian).
        image_shape (Sequence[int]): Typically 2D, image_shape[0] is the number of rows, image_shape[1] is the number of columns.

    Returns:
        np.ndarray: A complex ndarray containing the eigenvalues of the convolution kernel.
    """
    a = np.zeros(shape=image_shape)
    a[0, 0] = 1
    Ra = ndi.correlate(input=a, weights=kernel, mode='wrap')
    return np.fft.fft2(a=Ra)


def fft_conv2d(eigvals: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    For a given "unblurred" image x and the eigenvalue array for the blurring kernel, computes the blurred image.

    Args:
        eigvals (np.ndarray): m x n array of the eigenvalues of the 2D DFT of the convolution kernel (complex ndarray)
        x (np.ndarray): m x n image as an ndarray

    Returns:
        np.ndarray: (m x n) ndarray resulting from the convolution (via inverse FFT).
    """
    return np.fft.ifft2(eigvals * np.fft.fft2(x))  # (Hadamard product)


def cat_mats(mats: Sequence[np.ndarray]) -> np.ndarray:
    """
    Concatenates a list of identically shaped matrices along a third dimension.

    Args:
        mats (Sequence[np.ndarray]): Sequence of matrices (each an ndarray)

    Returns:
        np.ndarray: An ndarray with the matrices concatenated along the third axis.
    """
    return np.dstack(mats)


def apply_D(eigvals_D1: np.ndarray, eigvals_D2: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Computes Dx given the eigenvalues of D1 and D2.

    Args:
        eigvals_D1 (np.ndarray): Eigenvalues for the first derivative operator component.
        eigvals_D2 (np.ndarray): Eigenvalues for the second derivative operator component.
        x (np.ndarray): The image represented as an ndarray.

    Returns:
        np.ndarray: The computed Dx as an ndarray.
    """
    return cat_mats([
        fft_conv2d(eigvals=eigvals_D1, x=x),
        fft_conv2d(eigvals=eigvals_D2, x=x)
    ])


def apply_A(kernel_eigvals: np.ndarray, eigvals_D1: np.ndarray, eigvals_D2: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Computes Ax, where A is block matrix of three rows,
    using the eigenvalues of each block.

    Args:
        kernel_eigvals (np.ndarray): Eigenvalues of the kernel convolution operator.
        eigvals_D1 (np.ndarray): Eigenvalues of the first derivative operator.
        eigvals_D2 (np.ndarray): Eigenvalues of the second derivative operator.
        x (np.ndarray): The image as an ndarray.

    Returns:
        np.ndarray: The resulting Ax as an ndarray.
    """
    Dx = apply_D(eigvals_D1=eigvals_D1, eigvals_D2=eigvals_D2, x=x)
    D1x, D2x = Dx[:, :, 0], Dx[:, :, 1]
    Kx = fft_conv2d(eigvals=kernel_eigvals, x=x)
    return cat_mats([Kx, D1x, D2x])


def apply_D_conj(conj_eigvals_D1: np.ndarray, conj_eigvals_D2: np.ndarray, Dx: np.ndarray) -> np.ndarray:
    """
    Computes D^T D x, given that Dx has already been computed for some x.

    Args:
        conj_eigvals_D1 (np.ndarray): Conjugated eigenvalues of the first derivative operator.
        conj_eigvals_D2 (np.ndarray): Conjugated eigenvalues of the second derivative operator.
        Dx (np.ndarray): Precomputed Dx as an ndarray, with the first two channels corresponding to the derivative components.

    Returns:
        np.ndarray: The result of applying D^T D to x as an ndarray.
    """
    return fft_conv2d(eigvals=conj_eigvals_D1, x=Dx[:, :, 0]) + \
           fft_conv2d(eigvals=conj_eigvals_D2, x=Dx[:, :, 1])


def apply_A_conj(eigval_conj_arr: Sequence[np.ndarray], y: np.ndarray) -> np.ndarray:
    """
    Applies A^T to y, T denoting complex conjugation.

    Args:
        eigval_conj_arr (Sequence[np.ndarray]): A sequence (array) of complex conjugate eigenvalue arrays of some matrix A.
        y (np.ndarray): An ndarray where y is represented as [y1, y2, y3] concatenated along the third dimension, 
                        with each yi being an N x N matrix.

    Returns:
        np.ndarray: The result of applying A^T to y as an ndarray.
    """
    return sum(fft_conv2d(eigvals=eigval_conj_arr[i], x=y[:, :, i]) for i in range(len(eigval_conj_arr)))


def eigvals_mat(conj_eigvals_K: np.ndarray, 
                eigvals_K: np.ndarray, 
                conj_eigvals_D1: np.ndarray, 
                eigvals_D1: np.ndarray, 
                conj_eigvals_D2: np.ndarray, 
                eigvals_D2: np.ndarray) -> np.ndarray:
    """
    Computes the eigenvalues matrix 
    for the composite operator A^TA = K^TK + D1^TD1 + D2^TD2

    Args:
        conj_eigvals_K (np.ndarray): Conjugated eigenvalues of the kernel convolution operator.
        eigvals_K (np.ndarray): Eigenvalues of the kernel convolution operator.
        conj_eigvals_D1 (np.ndarray): Conjugated eigenvalues of the first derivative operator.
        eigvals_D1 (np.ndarray): Eigenvalues of the first derivative operator.
        conj_eigvals_D2 (np.ndarray): Conjugated eigenvalues of the second derivative operator.
        eigvals_D2 (np.ndarray): Eigenvalues of the second derivative operator.

    Returns:
        np.ndarray: The computed eigenvalues matrix.
    """
    return (conj_eigvals_K * eigvals_K) + (conj_eigvals_D1 * eigvals_D1) + (conj_eigvals_D2 * eigvals_D2)


def fft_invert(eigvals_mat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
     Computes A^(-1)x using fft and eigenvalues of some matrix A.

    Args:
        eigvals_mat (np.ndarray): The eigenvalues matrix of A (as computed from eigvals_mat or similar functions).
        x (np.ndarray): The image or vector to which the inverse operation is applied, represented as an ndarray.

    Returns:
        np.ndarray: The result of the inverse operation as an ndarray.
    """
    return np.fft.ifft2(np.fft.fft2(x) / eigvals_mat)


if __name__ == "__main__":
    from kernel import gaussian_kernel
    k = gaussian_kernel(hsize=[15, 15], sigma=1.0)
    out = periodic_conv_eigvals(kernel=k, image_shape=[128, 128])

    kernel_D1 = np.array([-1, 1])[:, None]
    kernel_D2 = np.array([-1, 1])[None, :]

    D1_eigvals = periodic_conv_eigvals(kernel=kernel_D1, image_shape=[128, 128])
    D2_eigvals = periodic_conv_eigvals(kernel=kernel_D2, image_shape=[128, 128])
    
    rand_img = np.random.random((128, 128))

    D = apply_D(eigvals_D1=D1_eigvals, eigvals_D2=D2_eigvals, x=rand_img)
    
    # Sample calls for apply_A and others. Make sure the appropriate arrays are used.
    A_out = apply_A(kernel_eigvals=out, eigvals_D1=D1_eigvals, eigvals_D2=D2_eigvals, x=rand_img)
    print(D)
    print(A_out)