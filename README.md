# Math563 – Final Project
### To get started:
A working example is found in ```sample.py```, containing code for importing the image, applying the blurring method, and a wrapper function for running and testing the different algorithms for time and epsilon. Change ```my_path``` to the path of the desired image.

### blurkit
The ```blurkit``` directory contains the finalized code.
- ```optsolver.py```: Implements main wrapper class ```OptSolver``` for the algorithms, meant to be the interface for the user.

- ```optutil.py```: Implements a helper class, ```OptUtil```, for ```OptSolver```, abstracting commonly used matrix operations using the eigenvalues of the 2D DFTs
    of the convolution kernel and the discrete gradient operator. 

- ```solvertemplate.py```: Contains core algorithm logic, in a structure resembling pseudocode, and implements early stopping.

- ```periodic_mat_util.py```: Various Fourier transform methods, mainly based on the provided Matlab code. Used by ```optutil.py```.

- ```kernel.py```: Contains 3 kernels for blurring: Gaussian, motion, and disk.

- ```preprocess_image.py``` contains methods for converting and resizing the image into a matrix.

### Other
- ```./test```: Contains an outdated version of our source code, along with some files for testing and hyperparameter tuning


### Sample Usage
```python
# usage
from blurkit.optsolver import ADMM
import numpy as np

k = np.array([1, 2, 3])[:, None] # dummy kernel
b = np.random.random((128, 128)) # dummy image

params = {'deblurring_objective': 'l1', 'maxiter': 100}
solver = ADMM(k=k, shape=(128,128), **params)

solver.solve(b=b, # blurred image
             if_track=True, # whether or not to track objective
             stop_criterion=1e-1 # halt and return when objective reaches this value
            )
```
