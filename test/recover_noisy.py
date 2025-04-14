import os
import json
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import your deblurring solvers, kernel generator, and image processing utilities.
from kernel import gaussian_kernel
from preprocess_image import image_to_numpy, rgb2gray, normalize_image
from test_util import blur_image
from optsolver import DouglasRachfordPrimal, DouglasRachfordPrimalDual, ADMM, ChambollePock

# -----------------------------
# Setup directories and load best hyperparameters.
# -----------------------------
cur_dir = os.path.dirname(__file__)
hp_results_file = os.path.join(cur_dir, 'best_noise_params', 'noise_results.json')
results_dir = os.path.join(cur_dir, 'apply_best_noise_params')
os.makedirs(results_dir, exist_ok=True)

with open(hp_results_file, 'r') as f:
    best_hp_data = json.load(f)

# -----------------------------
# Load and preprocess the image.
# -----------------------------
my_path = '../testimages/bright'
img_path = os.path.join(cur_dir, my_path, 'mcgill.jpg')
img = normalize_image(image_to_numpy(rgb2gray(path_to_image=img_path)))
imgnorm = np.linalg.norm(img, ord=1)

# -----------------------------
# Fixed parameters and kernel definition.
# -----------------------------
fixed_hps = {
    "deblurring_objective": 'l1',
    "maxiter": 100,
}
KERNEL = gaussian_kernel(hsize=[15, 15], sigma=1.0)

# -----------------------------
# Define deblurring algorithms mapping.
# -----------------------------
algorithms = {
    'DouglasRachfordPrimal': DouglasRachfordPrimal,
    'DouglasRachfordPrimalDual': DouglasRachfordPrimalDual,
    'ADMM': ADMM,
    'ChambollePock': ChambollePock
}

# -----------------------------
# Recovery experiments for each algorithm/noise configuration.
# -----------------------------
for algo_name, noise_dict in best_hp_data.items():
    for noise_key, hp_info in noise_dict.items():
        best_params = hp_info['best_params']
        # Convert the noise_key string back to dictionary.
        noise_params = ast.literal_eval(noise_key)
        
        # Precompute the blurred (noised) image and the corresponding true image.
        b, true_image = blur_image(image=img, kernel=KERNEL, noise_args=noise_params)
        
        # Instantiate the solver with fixed parameters and best hyperparameters.
        solver_params = best_params.copy()
        solver = algorithms[algo_name](
            k=KERNEL,
            shape=(256, 256),
            deblurring_objective=fixed_hps['deblurring_objective'],
            maxiter=fixed_hps['maxiter'],
            **solver_params
        )
        
        # Run the solver with convergence tracking.
        recovered_image, loss_list = solver.solve(b, if_track=True)
        recovered_image = recovered_image.real
        # -----------------------------
        # Save outputs.
        # -----------------------------
        # Create a sub-directory for this algorithm.
        algo_dir = os.path.join(results_dir, algo_name)
        os.makedirs(algo_dir, exist_ok=True)
        
        # Create a folder name for the noise configuration, cleaning up non-filesystem characters.
        noise_dir_name = noise_key.replace(" ", "").replace(":", "_")\
                                  .replace("{", "").replace("}", "")\
                                  .replace("'", "").replace(",", "_")
        noise_dir = os.path.join(algo_dir, noise_dir_name)
        os.makedirs(noise_dir, exist_ok=True)
        
        # Save the images.
        true_img_path = os.path.join(noise_dir, "true_image.png")
        blurred_img_path = os.path.join(noise_dir, "blurred_image.png")
        recovered_img_path = os.path.join(noise_dir, "recovered_image.png")
        
        plt.imsave(true_img_path, true_image, cmap='gray')
        plt.imsave(blurred_img_path, b, cmap='gray')
        plt.imsave(recovered_img_path, recovered_image, cmap='gray')
        
        # -----------------------------
        # Plot convergence using Seaborn.
        # -----------------------------
        sns.set_style("whitegrid")
        plt.figure()
        # Create an x-axis corresponding to iterations (starting at 1).
        iterations = list(range(1, len(loss_list) + 1))
        sns.lineplot(x=iterations, y=loss_list)
        # Set both axes to logarithmic scale.
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Iteration (log scale)")
        plt.ylabel("Objective Value (log scale)")
        plt.title(f"Convergence: {algo_name}\nNoise: {noise_key}")
        # Save as a PDF for high quality.
        convergence_plot_path = os.path.join(noise_dir, "convergence_plot.pdf")
        plt.savefig(convergence_plot_path, format='pdf')
        plt.close()
        
        print(f"Saved results for {algo_name} with noise settings: {noise_key}")

print("\nRecovery experiments complete. Check the 'Recovered_Results' directory for outputs.")
