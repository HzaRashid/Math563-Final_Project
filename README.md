# Math563 – Final Project
### To get started:
See ```sample.py``` for a full working example; loads the image, applies blurring, and a implements a function for testing the algorithms. Change ```my_path``` to the path of the desired image.

### blurkit
The ```blurkit``` directory contains the finalized code.

- ```optsolver.py```: Implements main wrapper class ```OptSolver``` for the algorithms, meant to be the interface for the user.

- ```optutil.py```: Implements a helper class, ```OptUtil```, for ```OptSolver```, abstracting commonly used matrix operations using the eigenvalues of the 2D DFTs
    of the convolution kernel and the discrete gradient operator. 

- ```solvertemplate.py```: Contains core algorithm logic, in a structure resembling pseudocode, and implements early stopping.

- ```periodic_mat_util.py```: Various Fourier transform methods, mainly based on the provided Matlab code. Used by ```optutil.py```.

- ```kernel.py```: Implements 3 kernels for blurring: Gaussian, motion, and disk.

- ```preprocess_image.py```: Methods for converting and resizing the image into a matrix.

- ```prox_util.py```: Implements proximal operators.

- ```test_util.py```: Helper functions for testing.


### Other
- ```./test```: Contains an outdated version of our source code, along with some files for testing and hyperparameter tuning

### Sample Usage
```python
# usage
from blurkit.optsolver import ADMM
import matplotlib.pyplot as plt
import numpy as np

k = np.array([1, 2, 3])[:, None] # dummy kernel
b = np.random.random((128, 128)) # dummy image

# instantiate solver
params = {'deblurring_objective': 'l1', 'maxiter': 100}
solver = ADMM(k=k, shape=(128,128), **params)

recovered_image, obj_vals = solver.solve(
    b=b, # blurred image
    if_track=True, # whether or not to track objective
    stop_criterion=1e-2 # halt and return when objective reaches this value
    )

recovered_image = np.real(recovered_image) # extract real part of image

# show recovered image (doesn't look like anything special)
plt.figure('Recovered Image')
plt.imshow(recovered_image, cmap='gray')
plt.axis('off')
plt.show()

# show objective across iterations
plt.figure('Objective Across Iterations')
plt.plot(obj_vals)
plt.yscale("log")
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.grid(True)
plt.show()
```

### Tracking Loss
If one wishes to track the objective, set the solver's ```solve``` method's key word argument ```if_track``` to ```True```, and use the list of objective values for downsteam analysis. See ```sample.py``` for details.
