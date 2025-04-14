import os
import time
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import your deblurring solvers, kernel generator, and image processing utilities.
from kernel import gaussian_kernel
from preprocess_image import image_to_numpy, rgb2gray, normalize_image
from test_util import blur_image
from optsolver import DouglasRachfordPrimal, DouglasRachfordPrimalDual, ADMM, ChambollePock

# -----------------------------
# Set up the fixed image input
# -----------------------------
cur_dir = os.path.dirname(__file__)
out_dir = os.path.join(cur_dir, 'NoiseTypes_results')
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
            "relax": 1.888640583759617,
            "step_size": 0.36069547095563936,
            "step_size2": 0.4993573208985405,
            "gamma": 0.045716177124372946
        }
}

# -----------------------------
# Define the fixed kernel: Gaussian with hsize=[15,15], sigma=1.0.
# -----------------------------
KERNEL=gaussian_kernel(hsize=[15, 15], sigma=1.0)

# -----------------------------
# Define noise variants for the evaluation.
# Here we vary the noise mode and noise density.
# -----------------------------
noise_variants = [
    {'mode': 's&p', 'amount': 0.1},
    {'mode': 's&p', 'amount': 0.2},
    {'mode': 's&p', 'amount': 0.5},
    {"mode": "gaussian", "mean": 0.0, "var": 0.005},
    {"mode": "gaussian", "mean": 0.1, "var": 0.005},
    {"mode": "gaussian", "mean": 0.2, "var": 0.005},
    {"mode": "gaussian", "mean": 0.0, "var": 0.005},
    {"mode": "gaussian", "mean": 0.0, "var": 0.01},
    {"mode": "gaussian", "mean": 0.0, "var": 0.02}
]

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
#  - Build the fixed Gaussian kernel.
#  - Blur the image using that kernel and the specified noise parameters.
#  - Create the solver using the best hyperparameters.
#  - Run the solver while recording the elapsed time.
#  - Compute the normalized error.
#
# It returns a dictionary of results.
# -----------------------------
def run_evaluation(algo_name, algo_class, noise_params):
    
    # Create the blurred image with the specified noise configuration.
    b, true_image = blur_image(
        image=img, 
        kernel=KERNEL, 
        noise_args=noise_params
    )
    
    # Instantiate the solver with the best hyperparameters.
    solver = algo_class(
        k=KERNEL, 
        shape=(256, 256), 
        deblurring_objective=fixed_hps['deblurring_objective'], 
        maxiter=fixed_hps['maxiter'], 
        **best_hps[algo_name]
    )
    
    # Measure the solving time.
    start = time.perf_counter()
    out, loss_history = solver.solve(b, if_track=False)
    end = time.perf_counter()
    elapsed_time = end - start
    
    # Compute the error normalized by the true image's 1-norm.
    error = np.linalg.norm((out - true_image) / imgnorm, ord=1)
    
    # Return the collected metrics.
    return {
        'algorithm': algo_name,
        'mode': noise_params['mode'],
        'amount': noise_params.get('amount', 'N/A'),
        'mean': noise_params.get('mean', 'N/A'),
        'var': noise_params.get('var', 'N/A'),
        'time': elapsed_time,
        'error': error
    }

# -----------------------------
# Run the evaluations in parallel.
#
# For every algorithm and every noise configuration, schedule a job.
# -----------------------------
def main():
    results = []
    tasks = []
    # Using ProcessPoolExecutor to run experiments in parallel.
    with ProcessPoolExecutor() as executor:
        for algo_name, algo_class in algorithms.items():
            for noise_params in noise_variants:
                tasks.append(
                    executor.submit(run_evaluation, algo_name, algo_class, noise_params)
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
