import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import util
from test_util import blur_image
import kernel

# ---------------------------------------
# Deblurring Solvers
# ---------------------------------------
from optsolver import DouglasRachfordPrimal, DouglasRachfordPrimalDual, ADMM, ChambollePock

# ---------------------------------------
# Map algorithm names to solver classes
# ---------------------------------------
solver_classes = {
    "Primal": DouglasRachfordPrimal,
    "Dual": DouglasRachfordPrimalDual,
    "ADMM": ADMM,
    "ChambollePock": ChambollePock
}

# Parameters for the solvers (adjust as needed)
default_params = {"relax": 1.5, "step_size": 0.8, "gamma": 0.03}
params_champock = {"step_size": 0.4, "step_size2": 0.4, "gamma": 0.03}

# ---------------------------------------
# Deblurring Objectives
# ---------------------------------------
objectives = ['l1', 'l2']  # The solver’s internal objective.

# ---------------------------------------
# Kernels and Noise definitions
# ---------------------------------------
kernel_funcs = {
    'gaussian': kernel.gaussian_kernel,
    'disk': kernel.disk_kernel,
    'motion': kernel.motion_kernel
}

# Example kernel experiments:
#   Each entry has ONE kernel type and a list of variants to test.
kernel_experiments = {
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
    ],
}

# Example noise experiments:
#   We separate “s&p” vs. “gaussian” so as not to mix them in one heatmap.
noise_experiments = {
    's&p': [
        {'mode': 's&p', 'amount': 0.1},
        {'mode': 's&p', 'amount': 0.3},
        {'mode': 's&p', 'amount': 0.5},
    ],
    'gaussian': [
        {'mode': 'gaussian', 'mean': 0.0, 'var': 0.005},
        {'mode': 'gaussian', 'mean': 0.0, 'var': 0.01},
        {'mode': 'gaussian', 'mean': 0.0, 'var': 0.02},
    ]
}

# ---------------------------------------
# Setup & Paths
# ---------------------------------------
cur_dir = os.path.dirname(__file__)
results_dir = os.path.join(cur_dir, 'results_separate_experiments')
os.makedirs(results_dir, exist_ok=True)

# Image path and shape
img_path = os.path.join(cur_dir, '../testimages/cameraman.jpg')
shape = (256, 256)

def evaluate_solver(solver, blurred, original):
    """Runs the solver, returns recovered image, L1 error, L2 error."""
    start = time.time()
    recovered, _ = solver.solve(blurred)
    end = time.time()
    l1_err = np.linalg.norm(original - recovered, ord=1) / np.linalg.norm(original, ord=1)
    l2_err = np.linalg.norm(original - recovered, ord=2) / np.linalg.norm(original, ord=2)
    print(f"{solver.__class__.__name__} completed in {end - start:.3f}s")
    return recovered, l1_err, l2_err

# ---------------------------------------------------------
# 1) KERNEL EXPERIMENTS: For each kernel type, vary ONE parameter
# ---------------------------------------------------------
# We'll fix a single noise setting so we don't mix multiple variables in the same experiment.
fixed_noise = {'mode': 's&p', 'amount': 0.1}

algo_names = list(solver_classes.keys())

for kernel_type, kernel_variants_list in kernel_experiments.items():
    # For each kernel type, generate a separate experiment
    # so that the x-axis truly corresponds to that one family of parameters.
    for obj in objectives:
        # Prepare matrices for error metrics:
        num_algos = len(algo_names)
        num_variants = len(kernel_variants_list)
        l1_errors = np.zeros((num_algos, num_variants))
        l2_errors = np.zeros((num_algos, num_variants))

        # Build some labels for the x-axis
        # e.g. for a Gaussian kernel variant: 'hsize=15, sigma=1.0', etc.
        variant_labels = []

        for j, variant_dict in enumerate(kernel_variants_list):
            # Create a label summarizing the variant:
            # (You can adapt formatting based on disk vs. motion vs. gaussian.)
            if kernel_type == 'gaussian':
                variant_labels.append(f"hsize={variant_dict['hsize'][0]}, σ={variant_dict['sigma']}")
            elif kernel_type == 'disk':
                variant_labels.append(f"r={variant_dict['r']}")
            elif kernel_type == 'motion':
                variant_labels.append(f"len={variant_dict['len']}")

            # Construct the kernel from the dictionary:
            k = kernel_funcs[kernel_type](**variant_dict)

            # Blur the image once for this variant (same noise).
            b, x_true = blur_image(
                path_to_image=img_path,
                shape=shape,
                kernel=k,
                noise_args=fixed_noise,
                show_before=False,
                show_after=False
            )

            # For each algorithm, run the solver and measure both errors:
            for i, algo in enumerate(algo_names):
                if algo == "ChambollePock":
                    solver_instance = solver_classes[algo](k=k, shape=shape, maxiter=100,
                                                           deblurring_objective=obj,
                                                           **params_champock)
                else:
                    solver_instance = solver_classes[algo](k=k, shape=shape, maxiter=100,
                                                           deblurring_objective=obj,
                                                           **default_params)
                _, err_l1, err_l2 = evaluate_solver(solver_instance, b, x_true)
                l1_errors[i, j] = err_l1
                l2_errors[i, j] = err_l2
                print(f"Kernel={kernel_type}, Variant={variant_dict}, "
                      f"Algorithm={algo}, Objective={obj}, L1={err_l1:.3f}, L2={err_l2:.3f}")

        # Now that we have the data, let's produce the heatmaps.
        # 1) L1 error heatmap
        plt.figure(figsize=(7, 5))
        ax = sns.heatmap(l1_errors, annot=True, fmt=".3f", cmap='viridis',
                         xticklabels=variant_labels, yticklabels=algo_names)
        ax.set_xlabel(f"{kernel_type.capitalize()} Kernel Parameter(s)")
        ax.set_ylabel("Algorithm")
        ax.set_title(f"Deblurring Objective: {obj}\nRelative L1 Error for {kernel_type.capitalize()} Kernel Variants")
        plt.tight_layout()
        outname_l1 = os.path.join(
            results_dir,
            f"heatmap_{kernel_type}_{obj}_L1error.pdf"
        )
        plt.savefig(outname_l1)
        plt.show()

        # 2) L2 error heatmap
        plt.figure(figsize=(7, 5))
        ax = sns.heatmap(l2_errors, annot=True, fmt=".3f", cmap='viridis',
                         xticklabels=variant_labels, yticklabels=algo_names)
        ax.set_xlabel(f"{kernel_type.capitalize()} Kernel Parameter(s)")
        ax.set_ylabel("Algorithm")
        ax.set_title(f"Deblurring Objective: {obj}\nRelative L2 Error for {kernel_type.capitalize()} Kernel Variants")
        plt.tight_layout()
        outname_l2 = os.path.join(
            results_dir,
            f"heatmap_{kernel_type}_{obj}_L2error.pdf"
        )
        plt.savefig(outname_l2)
        plt.show()


# ---------------------------------------------------------
# 2) NOISE EXPERIMENTS: For each noise type, vary ONE parameter
# ---------------------------------------------------------
# We'll fix the kernel to default Gaussian so we don't mix multiple variables.
default_gaussian_kernel = kernel.gaussian_kernel()  # e.g. hsize=[15,15], sigma=1.0

for noise_type, noise_variants_list in noise_experiments.items():
    for obj in objectives:
        # Prepare error matrices
        num_variants = len(noise_variants_list)
        l1_errors = np.zeros((len(algo_names), num_variants))
        l2_errors = np.zeros((len(algo_names), num_variants))

        # Build labels for the noise axis
        variant_labels = []
        for j, noise_params in enumerate(noise_variants_list):
            if noise_type == 's&p':
                # label = "amount=0.1", etc.
                variant_labels.append(f"amount={noise_params['amount']}")
            else:
                # Gaussian noise: "var=0.005", etc.
                variant_labels.append(f"var={noise_params['var']}")

            # Blur once with this noise
            b_noise, x_true = blur_image(
                path_to_image=img_path,
                shape=shape,
                kernel=default_gaussian_kernel,
                noise_args=noise_params,
                show_before=False,
                show_after=False
            )

            # Evaluate solvers
            for i, algo in enumerate(algo_names):
                if algo == "ChambollePock":
                    solver_instance = solver_classes[algo](
                        k=default_gaussian_kernel,
                        shape=shape,
                        maxiter=100,
                        deblurring_objective=obj,
                        **params_champock
                    )
                else:
                    solver_instance = solver_classes[algo](
                        k=default_gaussian_kernel,
                        shape=shape,
                        maxiter=100,
                        deblurring_objective=obj,
                        **default_params
                    )

                _, err_l1, err_l2 = evaluate_solver(solver_instance, b_noise, x_true)
                l1_errors[i, j] = err_l1
                l2_errors[i, j] = err_l2
                print(f"Noise={noise_type}, Params={noise_params}, "
                      f"Algorithm={algo}, Objective={obj}, L1={err_l1:.3f}, L2={err_l2:.3f}")

        # Plot heatmaps for the chosen noise type
        # 1) L1 error
        plt.figure(figsize=(7, 5))
        ax = sns.heatmap(l1_errors, annot=True, fmt=".3f", cmap='viridis',
                         xticklabels=variant_labels, yticklabels=algo_names)
        ax.set_xlabel(f"{noise_type.upper()} Noise Parameter")
        ax.set_ylabel("Algorithm")
        ax.set_title(f"Deblurring Objective: {obj}\nRelative L1 Error for {noise_type.upper()} Noise Variants")
        plt.tight_layout()
        outname_l1 = os.path.join(
            results_dir,
            f"heatmap_{noise_type}_{obj}_L1error.pdf"
        )
        plt.savefig(outname_l1)
        plt.show()

        # 2) L2 error
        plt.figure(figsize=(7, 5))
        ax = sns.heatmap(l2_errors, annot=True, fmt=".3f", cmap='viridis',
                         xticklabels=variant_labels, yticklabels=algo_names)
        ax.set_xlabel(f"{noise_type.upper()} Noise Parameter")
        ax.set_ylabel("Algorithm")
        ax.set_title(f"Deblurring Objective: {obj}\nRelative L2 Error for {noise_type.upper()} Noise Variants")
        plt.tight_layout()
        outname_l2 = os.path.join(
            results_dir,
            f"heatmap_{noise_type}_{obj}_L2error.pdf"
        )
        plt.savefig(outname_l2)
        plt.show()
