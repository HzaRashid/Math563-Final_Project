import numpy as np
import scipy.ndimage as ndi


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
    Ra = ndi.correlate(input=a, weights=kernel, mode='wrap')
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


def cat_mats(mats):
    """
    Concatenates list of identically shaped matrices along a third dimension.
    Throws error if the matrices do not have the same dimensions.
    """
    return np.dstack(mats)


def apply_D(D1_eigvals, D2_eigvals, x):
    """
    Computes Dx if provided the eigenvalues of D1 and D2.
    If you want to compute D^Tx, then pass the complex
    conjugates of the eigenvalue arrays.
    """
    return cat_mats([fft_conv2d(eigvals=D1_eigvals, x=x), 
                     fft_conv2d(eigvals=D2_eigvals, x=x)])


def apply_A(kernel_eigvals, 
            D1_eigvals, 
            D2_eigvals,
            x):
    """
    Computes Ax, where A is block matrix of three rows,
    using the eigenvalues of each block.
    If you want to compute A^Tx, then pass the complex
    conjugates of the eigenvalue arrays.
    """
    Dx = apply_D(x=x, D1_eigvals=D1_eigvals, D2_eigvals=D2_eigvals)
    D1x, D2x = Dx[:, :, 0], Dx[:, :, 1]
    Kx = fft_conv2d(eigvals=kernel_eigvals, x=x)
    return cat_mats([Kx, D1x, D2x])


def apply_grad_conj(D1_evconj, D2_evconj, Dx):
    """
    Computes D^TDx, where Dx has aleady been computed for some x.
    """
    return fft_conv2d(eigvals=D1_evconj, x=Dx[:, :, 0]) \
        + fft_conv2d(eigvals=D2_evconj, x=Dx[:, :, 1])


def apply_composite_op(kernel_eigvals, 
                       Dx, 
                       grad1_evconj,
                       grad2_evconj,
                       kernel_conv_x,
                       x):
    
    """
    Computes (I + K^TK + D^TD)x, where Dx is already known.
    """
    return x \
        + fft_conv2d(eigvals=np.conjugate(kernel_eigvals), x=kernel_conv_x) \
        + apply_grad_conj(D1_evconj=grad1_evconj, D2_evconj=grad2_evconj, Dx=Dx)


def eigvals_mat(kernel_eigvals_conj, 
                kernel_eigvals, 
                grad1_eigvals_conj, 
                grad1_eigvals, 
                grad2_eigvals_conj, 
                grad2_eigvals, 
                t):
    """
    Computes the eigenvalues matrix for the composite operator
    I + t^2K^TK + t^2D^TD
    
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


def fft_invert(eigvals_mat, x):
    """
    Computes A^(-1)x using fft and eigenvalues of some matrix A.
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

    D = apply_D(D1_eigvals=D1_eigvals, D2_eigvals=D2_eigvals, x=rand_img)
    
    print(D)