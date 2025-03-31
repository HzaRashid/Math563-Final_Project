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


def apply_D(eigvals_D1, eigvals_D2, x):
    """
    Computes Dx if provided the eigenvalues of D1 and D2.
    """
    return cat_mats([fft_conv2d(eigvals=eigvals_D1, x=x), 
                     fft_conv2d(eigvals=eigvals_D2, x=x)])


def apply_A(kernel_eigvals, 
            eigvals_D1, 
            eigvals_D2,
            x):
    """
    Computes Ax, where A is block matrix of three rows,
    using the eigenvalues of each block.
    """
    Dx = apply_D(x=x, eigvals_D1=eigvals_D1, eigvals_D2=eigvals_D2)
    D1x, D2x = Dx[:, :, 0], Dx[:, :, 1]
    Kx = fft_conv2d(eigvals=kernel_eigvals, x=x)
    return cat_mats([Kx, D1x, D2x])


# we should make a general function for these conjugate handlers
def apply_D_conj(conj_eigvals_D1, conj_eigvals_D2, Dx):
    """
    Computes D^TDx, where Dx has aleady been computed for some x.
    """
    return fft_conv2d(eigvals=conj_eigvals_D1, x=Dx[:, :, 0]) \
        + fft_conv2d(eigvals=conj_eigvals_D2, x=Dx[:, :, 1])


def apply_A_conj(eigval_conj_arr, y):
    """
    Applies A^Ty

    Args:
        eigval_conj_arr: array of the complex conjugate eigenvalue arrays of some matrix A.
        y (np.ndarray): y represented as [y1,y2,y3] concatenated in a
        third dimension, each yi represented as an NxN matrix
    """

    return sum(fft_conv2d(eigval_conj_arr[i], y[:,:,i]) for i in range(len(eigval_conj_arr)))


def apply_composite_op(eigvals_K, 
                       Dx, 
                       conj_eigvals_D1,
                       conj_eigvals_D2,
                       kernel_conv_x,
                       x):
    
    """
    Computes (I + K^TK + D^TD)x, where Dx is already known.
    """
    return x \
        + fft_conv2d(eigvals=np.conjugate(eigvals_K), x=kernel_conv_x) \
        + apply_D_conj(conj_eigvals_D1=conj_eigvals_D1, conj_eigvals_D2=conj_eigvals_D2, Dx=Dx)


def eigvals_mat(conj_eigvals_K, 
                eigvals_K, 
                conj_eigvals_D1, 
                eigvals_D1, 
                conj_eigvals_D2, 
                eigvals_D2, 
                t):
    """
    Computes the eigenvalues matrix for the composite operator
    I + t^2K^TK + t^2D^TD
    
    Args:
        conj_eigvals_K (np.ndarray):
            Conjugated eigenvalues of the kernel convolution operator.
        eigvals_K (np.ndarray):
            Eigenvalues of the kernel convolution operator.
        conj_eigvals_D1 (np.ndarray):
            Conjugated eigenvalues of the first gradient (derivative) operator.
        eigvals_D1 (np.ndarray):
            Eigenvalues of the first gradient (derivative) operator.
        conj_eigvals_D2 (np.ndarray):
            Conjugated eigenvalues of the second gradient (derivative) operator.
        eigvals_D2 (np.ndarray):
            Eigenvalues of the second gradient (derivative) operator.
        t (float):
            The step-size or scaling parameter for the operators.
            
    Returns:
        np.ndarray:
            The computed eigenvalues matrix.
    """
    ones_mat = np.ones_like(eigvals_K)
    return ones_mat + (t**2) * (conj_eigvals_K * eigvals_K) \
           + (t**2) * (conj_eigvals_D1 * eigvals_D1) \
           + (t**2) * (conj_eigvals_D2 * eigvals_D2)


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

    D = apply_D(eigvals_D1=D1_eigvals, eigvals_D2=D2_eigvals, x=rand_img)
    
    print(D)