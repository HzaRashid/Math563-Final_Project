import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import util
from test_util import blur_image
import kernel

if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    # Create results directory if it doesn't exist
    results_dir = os.path.join(cur_dir, 'l1vsl2_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Set path for the image file
    my_path = '../testimages/cameraman.jpg'
    path_to_image = os.path.join(cur_dir, my_path)

    # Choose noise parameters; here we use salt & pepper noise
    sp_noise_args = {'mode': 's&p', 'amount': 0.1}
    shape = (256, 256)

    # Create a blur kernel
    k = kernel.gaussian_kernel()
    
    # Create the blurred + noisy image (b) and obtain the original (x)
    b, x = blur_image(path_to_image=path_to_image,
                      shape=shape,
                      show_before=False,
                      show_after=False,
                      kernel=k,
                      noise_args=sp_noise_args)

    # Save the true and blurred images once
    true_filename = os.path.join(results_dir, "true.png")
    blurred_filename = os.path.join(results_dir, "blurred.png")
    plt.imsave(true_filename, x, cmap='gray')
    plt.imsave(blurred_filename, b, cmap='gray')

    # -------------------------
    # Import solvers
    # -------------------------
    from optsolver import DouglasRachfordPrimal, DouglasRachfordPrimalDual, ADMM, ChambollePock

    # Helper function to evaluate a given solver.
    def evaluate_solver(solver, blurred, original):
        """
        Runs the given solver on the blurred image 'blurred' and returns:
          - the recovered image,
          - the relative l1 error,
          - the relative l2 error.
        """
        start = time.time()
        recovered, _ = solver.solve(blurred)  # Run the solver; recovered is x0.
        end = time.time()
        l1_err = np.linalg.norm(original - recovered, ord=1) / np.linalg.norm(original, ord=1)
        l2_err = np.linalg.norm(original - recovered, ord=2) / np.linalg.norm(original, ord=2)
        print(f"{solver.__class__.__name__} completed in {end - start:.3f} seconds")
        return recovered, l1_err, l2_err

    # Map algorithm names to solver classes.
    solver_classes = {
        "Primal": DouglasRachfordPrimal,
        "Dual": DouglasRachfordPrimalDual,
        "ADMM": ADMM,
        "ChambollePock": ChambollePock
    }

    # Parameters for the solvers (adjust as needed)
    default_params = {"relax": 1.5, "step_size": 0.8, "gamma": 0.03}
    params_cp = {"relax": 1.8, "step_size": 0.4, "step_size2": 0.4, "gamma": 0.03}

    # Define the deblurring objectives to test.
    objectives = ['l1', 'l2']
    algo_names = list(solver_classes.keys())

    # Prepare matrices to store the error metrics
    l1_errors = np.zeros((len(algo_names), len(objectives)))
    l2_errors = np.zeros((len(algo_names), len(objectives)))

    # Loop over each algorithm and objective, evaluate the solver and save recovered images.
    for i, algo in enumerate(algo_names):
        for j, obj in enumerate(objectives):
            if algo == "ChambollePock":
                solver_instance = solver_classes[algo](k=k, shape=shape, maxiter=100,
                                                       deblurring_objective=obj,
                                                       **params_cp)
            else:
                solver_instance = solver_classes[algo](k=k, shape=shape, maxiter=100,
                                                       deblurring_objective=obj,
                                                       **default_params)
            recovered, err_l1, err_l2 = evaluate_solver(solver_instance, b, x)
            l1_errors[i, j] = err_l1
            l2_errors[i, j] = err_l2
            print(f"Algorithm: {algo}, Objective: {obj}, L1 Error: {err_l1:.4f}, L2 Error: {err_l2:.4f}")

            # Save the recovered image for this algorithm-objective combination.
            base_filename = os.path.join(results_dir, f"{algo}_{obj}")
            plt.imsave(f"{base_filename}_recovered.png", np.real(recovered), cmap='gray')

    # -------------------------
    # Create heat maps for the error metrics using Seaborn and save them.
    # -------------------------
    # Heatmap for Relative L1 Error
    plt.figure(figsize=(8, 6))
    ax1 = sns.heatmap(l1_errors, annot=True, fmt=".4f", cmap='viridis',
                      xticklabels=objectives, yticklabels=algo_names)
    ax1.set_xlabel("Deblurring Objective")
    ax1.set_ylabel("Algorithm")
    ax1.set_title("Relative L1 Error")
    heatmap_l1_filename = os.path.join(results_dir, "heatmap_l1_error.pdf")
    plt.tight_layout()
    plt.savefig(heatmap_l1_filename)
    plt.show()

    # Heatmap for Relative L2 Error
    plt.figure(figsize=(8, 6))
    ax2 = sns.heatmap(l2_errors, annot=True, fmt=".4f", cmap='viridis',
                      xticklabels=objectives, yticklabels=algo_names)
    ax2.set_xlabel("Deblurring Objective")
    ax2.set_ylabel("Algorithm")
    ax2.set_title("Relative L2 Error")
    heatmap_l2_filename = os.path.join(results_dir, "heatmap_l2_error.pdf")
    plt.tight_layout()
    plt.savefig(heatmap_l2_filename)
    plt.show()
