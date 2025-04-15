import os
import json
import numpy as np
import optuna
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import your deblurring solvers, kernel generator, and image processing utilities.
from kernel import gaussian_kernel
from preprocess_image import image_to_numpy, rgb2gray, normalize_image
from test_util import blur_image
from optsolver import DouglasRachfordPrimal, DouglasRachfordPrimalDual, ADMM, ChambollePock

# -----------------------------
# Set up output directory and load the image
# -----------------------------
cur_dir = os.path.dirname(__file__)
out_dir = os.path.join(cur_dir, 'best_noise_params')
os.makedirs(out_dir, exist_ok=True)

my_path = '../testimages/bright'
img_path = os.path.join(cur_dir, my_path, 'mcgill.jpg')

# Load and normalize the image.
img = normalize_image(image_to_numpy(rgb2gray(path_to_image=img_path)))
imgnorm = np.linalg.norm(img, ord=1)

# -----------------------------
# Fixed parameters and kernel definition.
# -----------------------------
fixed_hps = {
    "deblurring_objective": 'l1',
    "maxiter": 100,
}

# Build the fixed Gaussian kernel.
KERNEL = gaussian_kernel(hsize=[15, 15], sigma=1.0)

# -----------------------------
# Define noise variants for evaluation.
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
# Define the deblurring algorithms.
# -----------------------------
algorithms = {
    'DouglasRachfordPrimal': DouglasRachfordPrimal,
    'DouglasRachfordPrimalDual': DouglasRachfordPrimalDual,
    'ADMM': ADMM,
    'ChambollePock': ChambollePock
}

# -----------------------------
# Precompute blurred/noised images.
# For each noise configuration, compute the blurred image and the corresponding true image.
# The results are stored in the local_precomputed dictionary.
# -----------------------------
local_precomputed = {}
for noise_params in noise_variants:
    key = str(noise_params)
    b, true_image = blur_image(image=img, kernel=KERNEL, noise_args=noise_params)
    local_precomputed[key] = (b, true_image)
    print(f"Precomputed noise configuration: {key}")

# -----------------------------
# Define the Optuna objective function.
#
# This function now receives the precomputed dictionary as an argument.
# For each trial:
#   - Suggest hyperparameters.
#   - Retrieve the precomputed blurred image corresponding to the given noise configuration.
#   - Instantiate and run the solver.
#   - Return the normalized error.
# -----------------------------
def objective(trial, algo_name, algo_class, noise_params, precomputed):
    # Suggest hyperparameters.
    params = {
        "relax": trial.suggest_float('relax', 1e-2, 2.0),
        "step_size": trial.suggest_float('step_size', 1e-2, 2.0),
        "gamma": trial.suggest_float('gamma', 1e-2, 0.5)
    }
    if algo_name == 'ChambollePock':
        params.update({"step_size2": trial.suggest_float('step_size2', 1e-2, 2.0)})
        params.pop("relax")
    
    # Retrieve the precomputed blurred image and true image using the noise configuration key.
    noise_key = str(noise_params)
    b, true_image = precomputed[noise_key]
    
    # Instantiate the solver with fixed and trial-suggested hyperparameters.
    solver = algo_class(
        k=KERNEL,
        shape=(256, 256),
        deblurring_objective=fixed_hps['deblurring_objective'],
        maxiter=fixed_hps['maxiter'],
        **params
    )
    
    out, _ = solver.solve(b, if_track=False)
    # Compute and return the normalized error.
    error = np.linalg.norm((out.real - true_image) / imgnorm, ord=1)
    return error

# -----------------------------
# Function to run an Optuna study for one algorithm-noise setting.
# The precomputed dictionary is passed as an argument.
# -----------------------------
def run_study(algo_name, algo_class, noise_params, n_trials, precomputed):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, algo_name, algo_class, noise_params, precomputed),
        n_trials=n_trials
    )
    return (algo_name, str(noise_params), {"best_params": study.best_params, "best_error": study.best_value})

# -----------------------------
# Main optimization loop leveraging parallelism.
# Each independent study (one per algorithm-noise combination) is executed in parallel.
# -----------------------------
def main():
    n_trials = 50  # Number of trials per study; adjust as needed.
    best_hps_results = {}
    tasks = []
    with ProcessPoolExecutor() as executor:
        for algo_name, algo_class in algorithms.items():
            for noise_params in noise_variants:
                tasks.append(executor.submit(run_study, algo_name, algo_class, noise_params, n_trials, local_precomputed))
        
        for future in as_completed(tasks):
            try:
                algo_name, noise_key, best = future.result()
                if algo_name not in best_hps_results:
                    best_hps_results[algo_name] = {}
                best_hps_results[algo_name][noise_key] = best
                print(f"Completed optimization for {algo_name} with noise: {noise_key}")
                print("Best hyperparameters:", best['best_params'])
                print("Best error:", best['best_error'])
            except Exception as e:
                print("An error occurred during a study:", e)
    
    # Save all results to a JSON file.
    results_filename = os.path.join(out_dir, "noise_results.json")
    with open(results_filename, "w") as f:
        json.dump(best_hps_results, f, indent=4)
    
    print("\nOptuna hyperparameter tuning complete.")
    print("Results saved to:", results_filename)

if __name__ == '__main__':
    main()
