import os
import json
import numpy as np
import optuna
import concurrent.futures
import matplotlib.pyplot as plt
import pandas as pd

from preprocess_image import *         # custom preprocessing functions
from test_util import blur_image         # function that blurs images
from optsolver import ChambollePock
from kernel import gaussian_kernel

# Global configuration parameters
KERNEL = gaussian_kernel()
SHAPE = (256, 256)
MAX_ITER = 100
NOISE_MODE = 's&p'
NOISE_DENSITY = 0.1
DEBLOB = 'l1'
SOLVER=ChambollePock

# ------------------ Data Configuration & Loading ------------------
cur_dir = os.path.dirname(__file__)
base_testimages_path = os.path.join(cur_dir, '../testimages')
image_categories = ['bright', 'dark', 'noisy']

# Create an output directory for the deblurred images and true images.
output_base_dir = os.path.join(cur_dir, f'BrightDarkNoisy_{SOLVER.__name__}_Results')
os.makedirs(output_base_dir, exist_ok=True)

def load_images(category_path):
    """
    Loads images from a given directory. Each image is converted to grayscale, normalized,
    and its one-norm is computed.
    """
    image_paths = [os.path.join(category_path, fname) for fname in os.listdir(category_path)]
    images = [normalize_image(image_to_numpy(rgb2gray(path_to_image=img_path))) for img_path in image_paths]
    norms = [np.linalg.norm(x, ord=1) for x in images]
    return images, norms

def precompute_blurred_data(category):
    """
    Precomputes the blurred image pairs for all images in a given category.
    Returns a tuple (one_norms, blurred_pairs) where:
        - one_norms: a list of one-norm values for each image.
        - blurred_pairs: a list of tuples (b, x); b is the blurred image and x is the true image.
    """
    category_path = os.path.join(base_testimages_path, category)
    images, one_norms = load_images(category_path)
    blurred_pairs = []
    for img in images:
        b, x = blur_image(image=img, kernel=KERNEL, noise_mode=NOISE_MODE, noise_density=NOISE_DENSITY)
        blurred_pairs.append((b, x))
    return one_norms, blurred_pairs

def get_objective(blurred_pairs, one_norms):
    """
    Returns an objective function that computes the average loss (total loss divided by number of images)
    for a given trial. Three hyperparameters ('relax', 'step_size', 'gamma') are tuned, while the 
    solver parameters maxiter (100) and deblurring_objective ('l1') remain fixed.
    """
    def objective(trial):
        params = {
            "relax": trial.suggest_float('relax', 1e-2, 2.0),  # rho
            "step_size": trial.suggest_float('step_size', 1e-2, 2.0),  # t
            "step_size2": trial.suggest_float('step_size2', 1e-2, 2.0),  # s: Chambolle-Pock
            "gamma": trial.suggest_float('gamma', 1e-2, 0.5)
        }
        solver = SOLVER(
            k=KERNEL,
            shape=SHAPE,
            maxiter=MAX_ITER,
            deblurring_objective=DEBLOB,
            **params
        )
        total_loss = 0
        for (b, x), imgnorm in zip(blurred_pairs, one_norms):
            out, loss = solver.solve(b, if_track=True)
            out = out.real
            total_loss += np.linalg.norm((out - x) / imgnorm, ord=1)
        return total_loss / len(blurred_pairs)
    return objective

def run_category_study(category, precomputed_data):
    """ 
    Runs the hyperparameter tuning study for a single image category using precomputed blurred data.
    """
    print(f"Processing category: {category}")
    one_norms, blurred_pairs = precomputed_data[category]
    objective = get_objective(blurred_pairs, one_norms)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    
    print(f"Finished category: {category}")
    return {
        "category": category,
        "best_params": study.best_params,
        "best_value": study.best_value
    }

def run_all_images_study(precomputed_data, image_categories):
    """ 
    Runs the hyperparameter tuning study for all images by combining data from all categories.
    """
    all_one_norms = []
    all_blurred_pairs = []
    # Combine data from all categories.
    for category in image_categories:
        one_norms, blurred_pairs = precomputed_data[category]
        all_one_norms.extend(one_norms)
        all_blurred_pairs.extend(blurred_pairs)
        
    objective_all = get_objective(all_blurred_pairs, all_one_norms)
    study_all = optuna.create_study(direction="minimize")
    study_all.optimize(objective_all, n_trials=50)
    
    print("Finished combined images study")
    return {
        "category": "all",
        "best_params": study_all.best_params,
        "best_value": study_all.best_value
    }

if __name__ == '__main__':
    # ------------------ Precompute Blurred Data for All Categories ------------------
    # This dictionary will map each category to its precomputed tuple (one_norms, blurred_pairs)
    precomputed_data = {}
    for category in image_categories:
        one_norms, blurred_pairs = precompute_blurred_data(category)
        precomputed_data[category] = (one_norms, blurred_pairs)
    
    # ------------------ Parallel Hyperparameter Tuning for Each Category and All Images ------------------
    results = {}  # To store best hyperparameters and corresponding loss for each study

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit studies for individual categories
        futures = [
            executor.submit(run_category_study, category, precomputed_data)
            for category in image_categories
        ]
        # Also submit the combined study for all images
        futures.append(executor.submit(run_all_images_study, precomputed_data, image_categories))
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            cat = result["category"]
            results[cat] = {
                "best_params": result["best_params"],
                "best_value": result["best_value"]
            }
            print(f"Best params for {cat}: {result['best_params']}")
            print(f"Best loss for {cat}: {result['best_value']}\n")
    
    # ------------------ Saving Hyperparameter Tuning Results ------------------
    with open(os.path.join(output_base_dir, "best_hyperparams.json"), "w") as f:
        json.dump(results, f, indent=4)
    print("\nHyperparameter tuning results saved to best_hyperparams.json")
    
    # ------------------ Visualizations ------------------
    # Bar chart showing best loss values across each category and the combined study.
    categories_keys = list(results.keys())
    loss_values = [results[cat]["best_value"] for cat in categories_keys]
    
    plt.figure(figsize=(8, 6))
    plt.bar(categories_keys, loss_values)
    plt.ylabel("Best Loss")
    plt.title("Comparison of Best Loss Across Image Categories")
    plt.savefig(os.path.join(output_base_dir, "best_loss_comparison.png"))
    # plt.show()
    
    # Save best hyperparameters to CSV.
    df_params = pd.DataFrame(results).transpose()
    print("\nBest Hyperparameters per Category:")
    print(df_params)
    df_params.to_csv(os.path.join(output_base_dir, "best_hyperparams.csv"))
    print("Best hyperparameters saved to best_hyperparams.csv")
    
    # ------------------ Saving Deblurred Outputs ------------------
    # Save deblurred outputs per category using each study's best parameters.
    for category in image_categories:
        one_norms, blurred_pairs = precomputed_data[category]
        output_folder = os.path.join(output_base_dir, f"{category}_best")
        os.makedirs(output_folder, exist_ok=True)
        print(f"\nSaving deblurred outputs for category: {category} using best hyperparameters")
    
        for idx, ((b, x), imgnorm) in enumerate(zip(blurred_pairs, one_norms)):
            solver = SOLVER(
                k=KERNEL,
                shape=SHAPE,
                maxiter=MAX_ITER,
                deblurring_objective=DEBLOB,
                **results[category]["best_params"]
            )
            out, loss = solver.solve(b, if_track=True)
            out = out.real
            output_path = os.path.join(output_folder, f"{category}_image_{idx}.png")
            plt.imsave(output_path, out, cmap='gray')
            print(f"Saved deblurred output image to {output_path}")
    
    # Process all images using the combined best hyperparameters.
    all_output_folder = os.path.join(output_base_dir, "all_best")
    os.makedirs(all_output_folder, exist_ok=True)
    print("\nSaving deblurred outputs for all images using combined best hyperparameters")
    # First, recombine data for all images.
    all_blurred_pairs = []
    all_labels = []
    for category in image_categories:
        one_norms, blurred_pairs = precomputed_data[category]
        all_blurred_pairs.extend(blurred_pairs)
        all_labels.extend([category] * len(one_norms))
    
    for idx, ((b, x), label) in enumerate(zip(all_blurred_pairs, all_labels)):
        solver = SOLVER(
            k=KERNEL,
            shape=SHAPE,
            maxiter=MAX_ITER,
            deblurring_objective=DEBLOB,
            **results["all"]["best_params"]
        )
        out, loss = solver.solve(b, if_track=False)
        out = out.real
        output_path = os.path.join(all_output_folder, f"{label}_image_{idx}.png")
        plt.imsave(output_path, out, cmap='gray')
        print(f"Saved deblurred output image to {output_path}")
    
    # ------------------ Saving True Images ------------------
    # Create a subdirectory for the true images.
    true_images_dir = os.path.join(output_base_dir, "true_images")
    os.makedirs(true_images_dir, exist_ok=True)
    
    # For each category, create a folder and save the true images.
    for category in image_categories:
        one_norms, blurred_pairs = precomputed_data[category]
        category_true_dir = os.path.join(true_images_dir, category)
        os.makedirs(category_true_dir, exist_ok=True)
        print(f"\nSaving true images for category: {category}")
        for idx, (b, x) in enumerate(blurred_pairs):
            true_image_path = os.path.join(category_true_dir, f"{category}_true_{idx}.png")
            plt.imsave(true_image_path, x, cmap='gray')
            print(f"Saved true image to {true_image_path}")
