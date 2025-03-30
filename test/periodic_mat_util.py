import numpy as np
import cv2


def periodic_conv_eigvals(kernel, image_shape):
    """
    Computes the eigenvalues of the 2D DFT of the convolution kernel.

    Args:
        kernel : np.ndarray 
        Correlation kernel (e.g. for Gaussian)
        image_shape: list[int]
        typically 2D, with image_shape[0]==num_rows, image_shape[1]=num_cols

    Returns:
        matrix (same shape as image) containing the
        eigenvalues of the convolution kernel (complex ndarray).
    """
    a = np.zeros(shape=image_shape)
    a[0, 0] = 1
    Ra = cv2.filter2D(src=a, kernel=kernel, ddepth=-1, borderType=cv2.BORDER_WRAP)
    return np.fft.fft2(a=Ra)


def fft_conv2d(eigvals, x):
    """
    For a given "unblurred" image x and the eigenvalue array for
    the blurring kernel, computes the "blurred image" (e.g. Kx and Dx)

    Args:
        eigvals : complex ndarray
            m x n representing the eigenvalues of the 2D DFT of the convolution kernel
        x : ndarray
            m x n matrix representing the image

    Returns:
        m x n matrix of the 2d convolution of the kernel with image.
    """
    return np.fft.ifft2(eigvals * np.fft.fft2(x)) # (Hadamard product)


def cat_mats(x, y):
    return np.dstack([x, y])


def apply_grad_conj(D, D1_evconj, D2_evconj):
    return fft_conv2d(eigvals=D1_evconj, image=D[:, :, 0]) \
        + fft_conv2d(eigvals=D2_evconj, image=D[:, :, 1])


def apply_composite_op(kernel_eigvals, 
                       grad_stack, 
                       grad1_evconj,
                       grad2_evconj,
                       kernel_conv_x,
                       x):
    return x \
        + fft_conv2d(eigvals=np.conjugate(kernel_eigvals), image=kernel_conv_x) \
        + apply_grad_conj(D=grad_stack,  D1_evconj=grad1_evconj, D2_evconj=grad2_evconj)
    

def eigvals_mat(kernel_eigvals_conj, 
                kernel_eigvals, 
                grad1_eigvals_conj, 
                grad1_eigvals, 
                grad2_eigvals_conj, 
                grad2_eigvals, 
                t):
    """
    Computes the eigenvalues matrix for the composite operator.
    
    Args:
        kernel_eigvals_conj (np.ndarray):
            Conjugated eigenvalues of the kernel convolution operator.
        kernel_eigvals (np.ndarray):
            Eigenvalues of the kernel convolution operator.
        grad1_eigvals_conj (np.ndarray):
            Conjugated eigenvalues of the first gradient (derivative) operator.
        grad1_eigvals (np.ndarray):
            Eigenvalues of the first gradient (derivative) operator.
        grad2_eigvals_conj (np.ndarray):
            Conjugated eigenvalues of the second gradient (derivative) operator.
        grad2_eigvals (np.ndarray):
            Eigenvalues of the second gradient (derivative) operator.
        t (float):
            The step-size or scaling parameter for the operators.
            
    Returns:
        np.ndarray:
            The computed eigenvalues matrix.
    """
    ones_mat = np.ones_like(kernel_eigvals)
    return ones_mat + (t**2) * (kernel_eigvals_conj * kernel_eigvals) \
           + (t**2) * (grad1_eigvals_conj * grad1_eigvals) \
           + (t**2) * (grad2_eigvals_conj * grad2_eigvals)


def fft_invert(x, eigvals_mat):
    return np.fft.ifft2(np.fft.fft2(x) / eigvals_mat)


if __name__ == "__main__":
    from kernel import gaussian_kernel
    k = gaussian_kernel(hsize=[15, 15], sigma=1.0)
    out = periodic_conv_eigvals(kernel=k, image_shape=[128, 128])

    print(out)