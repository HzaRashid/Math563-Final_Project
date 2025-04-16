import os
import json
import numpy as np
import optuna
import concurrent.futures
import matplotlib.pyplot as plt
import pandas as pd
import inspect

from preprocess_image import *         # custom preprocessing functions
from test_util import blur_image         # function that blurs images
from optsolver import (
    DouglasRachfordPrimal,
    DouglasRachfordPrimalDual,
    ADMM,
    ChambollePock
)
from kernel import gaussian_kernel

# Global configuration parameters
KERSIZE = 9
STDEV=4
KERNEL = gaussian_kernel([KERSIZE, KERSIZE], sigma=STDEV)
SHAPE = (256, 256)
MAX_ITER = 100
NOISE_ARGS = {'mode': 's&p', 'amount': 0.1}
DEBLOB = 'l1'
# List of solver classes to test
SOLVERS = [DouglasRachfordPrimal, DouglasRachfordPrimalDual, ADMM, ChambollePock]

N_TRIALS=30

# ------------------ Data Configuration & Loading ------------------
cur_dir = os.path.dirname(__file__)
base_testimages_path = os.path.join(cur_dir, '../testimages')
image_categories = ['bright', 'dark', 'noisy']

# Base directory for all results
base_output_dir = os.path.join(
    cur_dir,
    f'BrightDarkNoisy_GaussKer{KERSIZE}x{KERSIZE}stdev{STDEV}_maxiter{MAX_ITER}_L1_ntrials{N_TRIALS}_Results'
)
os.makedirs(base_output_dir, exist_ok=True)

# Directory to save blurred images
blurred_dir = os.path.join(base_output_dir, 'blurred_images')
os.makedirs(blurred_dir, exist_ok=True)


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
    Returns a tuple (one_norms, blurred_pairs).
    Also saves each blurred image to disk.
    """
    category_path = os.path.join(base_testimages_path, category)
    images, one_norms = load_images(category_path)
    blurred_pairs = []
    # make category-specific subdir for blurred images
    cat_blur_dir = os.path.join(blurred_dir, category)
    os.makedirs(cat_blur_dir, exist_ok=True)
    for idx, img in enumerate(images):
        b, x = blur_image(image=img, kernel=KERNEL, noise_args=NOISE_ARGS)
        # save blurred image
        plt.imsave(
            os.path.join(cat_blur_dir, f'{category}_blurred_{idx}.png'),
            b,
            cmap='gray'
        )
        blurred_pairs.append((b, x))
    return one_norms, blurred_pairs


def get_objective(blurred_pairs, one_norms, solver_cls):
    """
    Returns an Optuna objective for the given solver class over blurred_pairs.
    """
    def objective(trial):
        params = {
            'relax': trial.suggest_float('relax', 1e-2, 2.0),
            'step_size': trial.suggest_float('step_size', 1e-2, 2.0),
            'gamma': trial.suggest_float('gamma', 1e-2, 0.5)
        }
        if solver_cls.__name__ == 'ChambollePock':
            params['step_size2'] = trial.suggest_float('step_size2', 1e-2, 2.0)
            params.pop('relax', None)
        solver = solver_cls(
            k=KERNEL,
            shape=SHAPE,
            maxiter=MAX_ITER,
            deblurring_objective=DEBLOB,
            **params
        )
        total_loss = 0
        for (b, x), imgnorm in zip(blurred_pairs, one_norms):
            out, _ = solver.solve(b, if_track=False)
            total_loss += np.linalg.norm((out.real - x) / imgnorm, ord=1)
        return total_loss / len(blurred_pairs)
    return objective


def run_category_study(solver_cls, category, precomputed_data):
    """
    Runs hyperparameter tuning for one solver and category.
    """
    solver_name = solver_cls.__name__
    print(f"[{solver_name}] Category: {category}")

    one_norms, blurred_pairs = precomputed_data[category]
    study = optuna.create_study(direction='minimize')
    study.optimize(get_objective(blurred_pairs, one_norms, solver_cls), n_trials=N_TRIALS)

    return {
        'solver': solver_name,
        'category': category,
        'best_params': study.best_params,
        'best_value': study.best_value
    }


def run_all_images_study(solver_cls, precomputed_data):
    """
    Runs hyperparameter tuning combining all categories for one solver.
    """
    solver_name = solver_cls.__name__
    print(f"[{solver_name}] Combined all categories")

    all_one_norms, all_blurred = [], []
    for cat in image_categories:
        norms, pairs = precomputed_data[cat]
        all_one_norms.extend(norms)
        all_blurred.extend(pairs)

    study = optuna.create_study(direction='minimize')
    study.optimize(get_objective(all_blurred, all_one_norms, solver_cls), n_trials=N_TRIALS)

    return {
        'solver': solver_name,
        'category': 'all',
        'best_params': study.best_params,
        'best_value': study.best_value
    }


if __name__ == '__main__':
    # Precompute data (also saves blurred images)
    precomputed_data = {cat: precompute_blurred_data(cat) for cat in image_categories}
    results = {solver.__name__: {} for solver in SOLVERS}

    # Parallel tuning
    tasks = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for solver in SOLVERS:
            for cat in image_categories:
                tasks.append(executor.submit(run_category_study, solver, cat, precomputed_data))
            tasks.append(executor.submit(run_all_images_study, solver, precomputed_data))

        for future in concurrent.futures.as_completed(tasks):
            res = future.result()
            results[res['solver']][res['category']] = {
                'best_params': res['best_params'],
                'best_value': res['best_value']
            }
            print(f"{res['solver']}:{res['category']} -> {res['best_params']} = {res['best_value']}")

    # Save, visualize, and output results
    for solver_name, solver_results in results.items():
        solver_dir = os.path.join(base_output_dir, solver_name)
        os.makedirs(solver_dir, exist_ok=True)
        # JSON and CSV
        with open(os.path.join(solver_dir, 'best_hyperparams.json'), 'w') as f:
            json.dump(solver_results, f, indent=4)
        pd.DataFrame(solver_results).transpose().to_csv(os.path.join(solver_dir, 'best_hyperparams.csv'))

        # Loss bar chart
        cats = list(solver_results.keys())
        losses = [solver_results[c]['best_value'] for c in cats]
        plt.figure(figsize=(8,6))
        plt.bar(cats, losses)
        plt.ylabel('Best Loss')
        plt.title(f'{solver_name} Loss Comparison')
        plt.savefig(os.path.join(solver_dir, 'best_loss_comparison.png'))
        plt.close()

        # Save deblurred outputs filtering init kwargs by signature
        for cat in image_categories:
            norms, pairs = precomputed_data[cat]
            out_dir = os.path.join(solver_dir, f'{cat}_best')
            os.makedirs(out_dir, exist_ok=True)
            best_params = solver_results[cat]['best_params']
            SolverClass = next(s for s in SOLVERS if s.__name__ == solver_name)
            sig = inspect.signature(SolverClass.__init__)
            filtered = {k: v for k, v in best_params.items() if k in sig.parameters}
            for idx, ((b, x), _) in enumerate(zip(pairs, norms)):
                inst = SolverClass(
                    k=KERNEL,
                    shape=SHAPE,
                    maxiter=MAX_ITER,
                    deblurring_objective=DEBLOB,
                    **filtered
                )
                out, _ = inst.solve(b, if_track=False)
                plt.imsave(os.path.join(out_dir, f'{cat}_image_{idx}.png'), out.real, cmap='gray')

    # Save true images once
    true_dir = os.path.join(base_output_dir, 'true_images')
    os.makedirs(true_dir, exist_ok=True)
    for cat in image_categories:
        cat_dir = os.path.join(true_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)
        _, pairs = precomputed_data[cat]
        for idx, (_, x) in enumerate(pairs):
            plt.imsave(os.path.join(cat_dir, f'{cat}_true_{idx}.png'), x, cmap='gray')

    print('All processing complete.')
