# Math563 – Final Project
To get started:
In sample.py, change my_path to the path of the image to be tested. Then run sample.py.

sample.py contains code for importing the image, applying the blurring method, and a wrapper function for running and testing the different algorithms for time and epsilon.

The blurkit directory contains the finalized code. In it:
optsolver.py contains the interface for the algorithms to be used by the user.
optutil.py defines functions using some saved states that are frequently called by optsolver.
solvertemplate.py implements the core logic of the algorithms, in a structure resembling pseudocode, and implements early stopping.
kernel.py contains 3 kernels for blurring: Gaussian, motion, and disk.
periodic_mat_util.py contains various Fourier transform methods, mainly based on the provided Matlab code.
preprocess_image.py contains methods for converting and resizing the image into a matrix.

The test directory contains an outdated version of our source code, along with some files for testing and hyperparameter tuning