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

img_path = os.path.join(cur_dir, '../testimages', 'cameraman.jpg')
img = normalize_image(image_to_numpy(rgb2gray(path_to_image=img_path)))

# Precompute norms for relative‐error calculations
img_l1_norm = np.linalg.norm(img, ord=1)
img_l2_norm = np.linalg.norm(img, ord=2)

# -----------------------------
# Fixed solver settings
# -----------------------------
MAX_ITER = 100

# Best‐found hyperparameters (assumed same for both objectives)
best_hps = {
    'DouglasRachfordPrimal': {
        "relax": 1.8542801426300373,
        "step_size": 0.5303865420997071,
        "gamma": 0.037467313515114786
    },
    'DouglasRachfordPrimalDual': {
        "relax": 1.549599094560696,
        "step_size": 0.940056822261643,
        "gamma": 0.02280159605226321
    },
    'ADMM': {
        "relax": 1.160219522869758,
        "step_size": 1.6288522037705238,
        "gamma": 0.03851698444156989
    },
    'ChambollePock': {
        "step_size": 0.36069547095563936,
        "step_size2": 0.4993573208985405,
        "gamma": 0.045716177124372946
    }
}

# -----------------------------
# Kernel registry & variants
# -----------------------------
kernel_funcs = {
    'gaussian': gaussian_kernel,
    'disk': disk_kernel,
    'motion': motion_kernel
}

kernel_variants = {
    'gaussian': [
        {'hsize': [15, 15], 'sigma': 1.0},
        {'hsize': [21, 21], 'sigma': 1.0},
        {'hsize': [31, 31], 'sigma': 1.0},
    ],
    'disk': [
        {'r': 5},
        {'r': 8},
        {'r': 12},
    ],
    'motion': [
        {'len': 7, 'theta': 0.0},
        {'len': 9, 'theta': 0.0},
        {'len': 15, 'theta': 0.0},
    ]
}

# -----------------------------
# Algorithms to evaluate
# -----------------------------
algorithms = {
    'DouglasRachfordPrimal': DouglasRachfordPrimal,
    'DouglasRachfordPrimalDual': DouglasRachfordPrimalDual,
    'ADMM': ADMM,
    'ChambollePock': ChambollePock
}

# -----------------------------
# Which deblurring objectives?
# -----------------------------
objectives = ['l1', 'l2']

# -----------------------------
# Single‐run evaluation
# -----------------------------
def run_evaluation(algo_name, algo_class, kernel_name, kernel_params, deblur_obj):
    # 1) build kernel
    k = kernel_funcs[kernel_name](**kernel_params)

    # 2) blur + noise
    b, true_image = blur_image(
        image=img, 
        kernel=k,
        noise_args={'mode': 's&p', 'amount': 0.1}
    )

    # 3) instantiate solver
    solver = algo_class(
        k=k,
        shape=(256, 256),
        deblurring_objective=deblur_obj,
        maxiter=MAX_ITER,
        **best_hps[algo_name]
    )

    # 4) solve & time
    start = time.perf_counter()
    out, _ = solver.solve(b, if_track=True)
    elapsed = time.perf_counter() - start

    # 5) compute relative errors
    l1_rel = np.linalg.norm(out - true_image, ord=1) / img_l1_norm
    l2_rel = np.linalg.norm(out - true_image, ord=2) / img_l2_norm

    return {
        'objective': deblur_obj,
        'algorithm': algo_name,
        'kernel': kernel_name,
        'kernel_params': kernel_params,
        'time': elapsed,
        'l1_rel_error': l1_rel,
        'l2_rel_error': l2_rel
    }

# -----------------------------
# Dispatch all jobs in parallel
# -----------------------------
def main():
    results = []
    tasks = []

    with ProcessPoolExecutor() as executor:
        for obj in objectives:
            for algo_name, algo_class in algorithms.items():
                for kernel_name, variants in kernel_variants.items():
                    for params in variants:
                        tasks.append(
                            executor.submit(
                                run_evaluation,
                                algo_name, algo_class,
                                kernel_name, params,
                                obj
                            )
                        )

        for future in as_completed(tasks):
            try:
                results.append(future.result())
            except Exception as e:
                print("Error during evaluation:", e)

    # Save to JSON
    out_file = os.path.join(out_dir, "evaluation_results_all_objectives.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print("Saved results to", out_file)

    # (Optional) print a quick summary
    for r in results:
        print(r)

if __name__ == '__main__':
    main()
