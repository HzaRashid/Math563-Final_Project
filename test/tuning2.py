import os
import json
import numpy as np
import optuna
from preprocess_image import *         # custom preprocessing functions
from test_util import blur_image         # function that blurs images
from optsolver import DouglasRachfordPrimal
from kernel import motion_kernel, gaussian_kernel, disk_kernel
import matplotlib.pyplot as plt
import pandas as pd

# ------------------ Data Configuration & Loading ------------------

# Current directory and base path to test images (which are in subfolders: 'bright', 'dark', 'noisy')
cur_dir = os.path.dirname(__file__)
base_testimages_path = os.path.join(cur_dir, '../testimages')
image_categories = ['bright', 'dark', 'noisy']

# List of kernel functions used during tuning (for aggregated loss computation)
kernels = [motion_kernel, gaussian_kernel, disk_kernel]

def load_images(category_path):
    """
    Loads images from a given directory. Each image is converted to grayscale, normalized,
    and its one-norm is computed.
    """
    image_paths = [os.path.join(category_path, fname) for fname in os.listdir(category_path)]
    images = [normalize_image(image_to_numpy(rgb2gray(path_to_image=img_path))) for img_path in image_paths]
    norms = [np.linalg.norm(x, ord=1) for x in images]
    return images, norms

def get_objective(images, one_norms):
    """
    Returns an objective function that computes the total loss (summed over all images and kernels)
    for a given trial. In this example, three hyperparameters ('relax', 'step_size', 'gamma') are tuned.
    The solver parameters maxiter (150) and deblurring_objective ('l1') remain fixed.
    """
    def objective(trial):
        params = {
            "relax": trial.suggest_float('relax', 1e-5, 2.0),
            "step_size": trial.suggest_float('step_size', 1e-5, 1e-1),
            "gamma": trial.suggest_float('gamma', 5e-2, 1.0)
        }
        total_loss = 0
        for img, imgnorm in zip(images, one_norms):
            for kernel in kernels:
                k = kernel()  # instantiate each kernel with default parameters
                solver = DouglasRachfordPrimal(k=k, shape=(256, 256), 
                                               maxiter=150, deblurring_objective='l1',
                                               **params)
                # Create a blurred image with salt & pepper noise
                b, x = blur_image(image=img, kernel=k, noise_mode='s&p', noise_density=0.1)
                out, loss = solver.solve(b, if_track=True)
                # Accumulate the loss (normalized by one-norm)
                total_loss += np.linalg.norm((out - x) / imgnorm, ord=1)
        return total_loss
    return objective

# ------------------ Hyperparameter Tuning ------------------

results = {}  # To store best hyperparameters and corresponding loss for each study

# Run tuning for each image category separately
for category in image_categories:
    print(f"Processing category: {category}")
    category_path = os.path.join(base_testimages_path, category)
    images, one_norms = load_images(category_path)
    
    # Build the objective function for the current category
    objective = get_objective(images, one_norms)
    
    # Create and run the Optuna study (n_trials set low for demonstration; increase for real runs)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    
    results[category] = {
        "best_params": study.best_params,
        "best_value": study.best_value
    }
    print(f"Best params for {category}: {study.best_params}")
    print(f"Best loss for {category}: {study.best_value}\n")

# Now tune on all images across categories

all_images = []
all_norms = []
# Also record the category label per image for output naming later.
all_labels = []
for category in image_categories:
    category_path = os.path.join(base_testimages_path, category)
    imgs, norms = load_images(category_path)
    all_images.extend(imgs)
    all_norms.extend(norms)
    all_labels.extend([category] * len(imgs))

objective_all = get_objective(all_images, all_norms)
study_all = optuna.create_study(direction="minimize")
study_all.optimize(objective_all, n_trials=5)

results["all"] = {
    "best_params": study_all.best_params,
    "best_value": study_all.best_value
}
print("Best params for all images:", study_all.best_params)
print("Best loss for all images:", study_all.best_value)

# ------------------ Save Hyperparameter Tuning Results ------------------

# Save the hyperparameter tuning results in JSON
with open("best_hyperparams.json", "w") as f:
    json.dump(results, f, indent=4)
print("\nHyperparameter tuning results saved to best_hyperparams.json")

# ------------------ Visualizations ------------------

# 1. Bar chart of best loss values across each category and the combined study
categories_keys = list(results.keys())
loss_values = [results[cat]["best_value"] for cat in categories_keys]

plt.figure(figsize=(8, 6))
plt.bar(categories_keys, loss_values)
plt.ylabel("Best Loss")
plt.title("Comparison of Best Loss Across Image Categories")
plt.savefig("best_loss_comparison.png")
plt.show()

# 2. CSV file of the best hyperparameters for each study
df_params = pd.DataFrame(results).transpose()
print("\nBest Hyperparameters per Category:")
print(df_params)
df_params.to_csv("best_hyperparams.csv")
print("Best hyperparameters saved to best_hyperparams.csv")

# ------------------ Saving Deblurred Outputs ------------------

# For final output generation we choose a default kernel.
# (Note: the tuning aggregated results over multiple kernels.
# Here we pick the Gaussian kernel by default; change as desired.)
selected_kernel = gaussian_kernel

# Create the base output directory if it doesn't exist
output_base_dir = os.path.join(cur_dir, 'outputs')
os.makedirs(output_base_dir, exist_ok=True)

# Process each category individually with its best hyperparameters
for category in image_categories:
    category_path = os.path.join(base_testimages_path, category)
    images, one_norms = load_images(category_path)
    output_folder = os.path.join(output_base_dir, f"{category}_best")
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nSaving deblurred outputs for category: {category} using best hyperparameters")
    
    for idx, (img, imgnorm) in enumerate(zip(images, one_norms)):
        # For each image, use the best hyperparameters found for this category
        k = selected_kernel()  # instantiate the selected kernel (gaussian_kernel here)
        solver = DouglasRachfordPrimal(k=k, shape=(256,256), maxiter=150,
                                       deblurring_objective='l1', **results[category]["best_params"])
        # Create blurred version and deblur it
        b, x = blur_image(image=img, kernel=k, noise_mode='s&p', noise_density=0.1)
        out, loss = solver.solve(b, if_track=True)
        output_path = os.path.join(output_folder, f"{category}_image_{idx}.png")
        plt.imsave(output_path, out, cmap='gray')
        print(f"Saved output image to {output_path}")

# Process all images using the best hyperparameters from the combined study
all_output_folder = os.path.join(output_base_dir, "all_best")
os.makedirs(all_output_folder, exist_ok=True)
print("\nSaving deblurred outputs for all images using combined best hyperparameters")
for idx, (img, imgnorm, label) in enumerate(zip(all_images, all_norms, all_labels)):
    k = selected_kernel()  # instantiate the selected kernel
    solver = DouglasRachfordPrimal(k=k, shape=(256,256), maxiter=150,
                                   deblurring_objective='l1', **results["all"]["best_params"])
    b, x = blur_image(image=img, kernel=k, noise_mode='s&p', noise_density=0.1)
    output_path = os.path.join(all_output_folder, f"{label}_image_{idx}.png")
    plt.imsave(output_path, out, cmap='gray')
    print(f"Saved output image to {output_path}")
