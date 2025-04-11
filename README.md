# Math563 – Final Project
To get started:
In test_util.py, change my_path to the path of the image to be tested. Then run test_util.py.

test_util.py contains code for importing the image, applying the blurring method, and a wrapper function for running and testing the different algorithms for time and epsilon.

Overview of the other files:
kernel.py contains 3 kernels for blurring: Gaussian, motion, and disk.
optsolver.py contains the classes for running the algorithms. The resolvents are implemented there.
optutil.py contains methods for various matrix operations using Fourier transform, with the matrix's eigenvalues already cached.
periodic_mat_util.py contains various Fourier transform methods.
preprocess_image.py contains methods for converting and resizing the image into a matrix.
prox_util.py contains the proximal operators.
solvertemplate.py contains templates for the algorithms, which optsolver.py calls.