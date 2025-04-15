import os
import time
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import your deblurring solvers, kernel generators, and image processing utilities.
from kernel import motion_kernel, gaussian_kernel, disk_kernel
from preprocess_image import image_to_numpy, rgb2gray, normalize_image
from test_util import blur_image
from optsolver import DouglasRachfordPrimal, DouglasRachfordPrimalDual, ADMM, ChambollePock

# -----------------------------
# Set up the fixed image input
# -----------------------------
cur_dir = os.path.dirname(__file__)
out_dir = os.path.join(cur_dir, 'KernelSizes_results')
os.makedirs(out_dir, exist_ok=True)

my_path = '../testimages'
img_path = os.path.join(cur_dir, my_path, 'cameraman.jpg')

# For this evaluation, pick the first image.
img = normalize_image(image_to_numpy(rgb2gray(path_to_image=img_path)))
imgnorm = np.linalg.norm(img, ord=1)

# -----------------------------
# Define best hyperparameters
# (Assume these were obtained from previous tuning.)
# -----------------------------
fixed_hps = {"deblurring_objective": 'l1',
             "maxiter": 100,
             }
best_hps = {
    'DouglasRachfordPrimal': {
            "relax": 1.8542801426300373,
            "step_size": 0.5303865420997071,
            "gamma": 0.037467313515114786
        },
    'DouglasRachfordPrimalDual':{
            "relax": 1.549599094560696,
            "step_size": 0.940056822261643,
            "gamma": 0.02280159605226321
        },
    'ADMM':{
            "relax": 1.160219522869758,
            "step_size": 1.6288522037705238,
            "gamma": 0.03851698444156989
        },
    'ChambollePock':{
            "step_size": 0.36069547095563936,
            "step_size2": 0.4993573208985405,
            "gamma": 0.045716177124372946
        }
}

# -----------------------------
# Map kernel names to their functions.
# -----------------------------
kernel_funcs = {
    'gaussian': gaussian_kernel,
    'disk': disk_kernel,
    'motion': motion_kernel
}

# -----------------------------
# Define kernel variants for the evaluation.
# Here we vary the "size" parameters for each kernel type.
# -----------------------------
kernel_variants = {
    'gaussian': [
         {'hsize': [15, 15], 'sigma': 1.0},  # default
         {'hsize': [21, 21], 'sigma': 1.0},
         {'hsize': [31, 31], 'sigma': 1.0},
    ],
    'disk': [
         {'r': 5},   # smaller disk
         {'r': 8},   # default
         {'r': 12},  # larger disk
    ],
    'motion': [
         {'len': 7, 'theta': 0.0},   # shorter motion blur
         {'len': 9, 'theta': 0.0},   # default
         {'len': 15, 'theta': 0.0},  # longer motion blur
    ]
}

# -----------------------------
# Define the algorithms to be evaluated.
# -----------------------------
algorithms = {
    'DouglasRachfordPrimal': DouglasRachfordPrimal,
    'DouglasRachfordPrimalDual': DouglasRachfordPrimalDual,
    'ADMM': ADMM,
    'ChambollePock': ChambollePock
}

# -----------------------------
# Define a function to evaluate one combination.
#
# This function will:
#  - Build the kernel with a given variant.
#  - Blur the image using that kernel, with a fixed noise setting.
#  - Create the solver using the best hyperparameters.
#  - Run the solver while recording the elapsed time.
#  - Compute the normalized error.
#
# It returns a dictionary of results.
# -----------------------------
def run_evaluation(algo_name, algo_class, kernel_name, kernel_params):
    # Generate the kernel using the corresponding kernel function and variant parameters.
    k_func = kernel_funcs[kernel_name]
    k = k_func(**kernel_params)
    
    # Create the blurred image (with salt & pepper noise at 10% density)
    # b is the blurred image and true_image is the ground truth.
    b, true_image = blur_image(image=img, kernel=k, noise_args={'mode': 's&p', 'amount': 0.1})
    
    # Instantiate the solver with the best hyperparameters.
    solver = algo_class(k=k, shape=(256, 256), 
                        deblurring_objective=fixed_hps['deblurring_objective'], 
                        maxiter=fixed_hps['maxiter'], 
                        **best_hps[algo_name])
    
    # Measure the solving time.
    start = time.perf_counter()
    out, loss_history = solver.solve(b, if_track=True)
    end = time.perf_counter()
    elapsed_time = end - start
    
    # Compute the error normalized by the true image's 1-norm.
    error = np.linalg.norm((out - true_image) / imgnorm, ord=1)
    
    # Return the collected metrics.
    return {
        'algorithm': algo_name,
        'kernel': kernel_name,
        'kernel_params': kernel_params,
        'time': elapsed_time,
        'error': error
    }

# -----------------------------
# Run the evaluations in parallel.
#
# For every algorithm and every kernel variant, schedule a job.
# -----------------------------
def main():
    results = []
    tasks = []
    # Using ProcessPoolExecutor to run experiments in parallel.
    with ProcessPoolExecutor() as executor:
        for algo_name, algo_class in algorithms.items():
            for kernel_name, variants in kernel_variants.items():
                for params in variants:
                    tasks.append(
                        executor.submit(run_evaluation, algo_name, algo_class, kernel_name, params)
                    )
        # Collect the results as they complete.
        for future in as_completed(tasks):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print("An error occurred during evaluation:", e)
    
    # Save results to a JSON file.
    json_filename = os.path.join(out_dir, "evaluation_results.json")
    with open(json_filename, "w") as f:
        json.dump(results, f, indent=4)
    print("Evaluation Results saved to:", json_filename)
    
    # Optionally, print the results.
    for r in results:
        print(r)

if __name__ == '__main__':
    main()
