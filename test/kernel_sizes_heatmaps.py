import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Load the JSON file containing the evaluation results
cur_dir = os.path.dirname(__file__)
data_path = os.path.join(cur_dir,  "apply_BDN_L1_KernelSizes", "evaluation_results.json")
df = pd.read_json(data_path)

out_dir = os.path.join(cur_dir, "kernelSize_heatmaps")
os.makedirs(out_dir, exist_ok=True)

# Define a function to extract a numeric kernel size from the kernel_params dictionary.
def extract_size(row):
    kernel_type = row['kernel'].lower()
    params = row['kernel_params']
    if kernel_type == 'disk':
        # For disk kernels, use the 'r' parameter.
        return params.get('r', None)
    elif kernel_type == 'gaussian':
        # For gaussian, use the first entry of 'hsize' (assuming it is a square kernel)
        hsize = params.get('hsize', None)
        if isinstance(hsize, list) and hsize:
            return hsize[0]
    elif kernel_type == 'motion':
        # For motion kernels, use the 'len' parameter.
        return params.get('len', None)
    return None

# Apply the extraction function to create a new column "Size"
df['Size'] = df.apply(extract_size, axis=1)

# Define metrics to visualize
metrics = ['time', 'error']

# Loop over each unique kernel type and generate heatmaps for each metric.
for kernel in df['kernel'].unique():
    # Filter for the current kernel type
    df_kernel = df[df['kernel'] == kernel]
    
    for metric in metrics:
        # Create a pivot table with rows as algorithms and columns as kernel sizes,
        # with the metric value as the cell content.
        pivot_table = df_kernel.pivot(index='algorithm', columns='Size', values=metric)
        # Sort columns for a more natural order of kernel sizes.
        pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)
        
        # Create the heatmap using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis")
        plt.title(f"{kernel.capitalize()} Kernel - {metric.capitalize()}")
        plt.xlabel("Kernel Size")
        plt.ylabel("Algorithm")
        plt.tight_layout()
        
        # Save the generated heatmap to a file (e.g., disk_time_heatmap.png)
        filename = f"{kernel.lower()}_{metric.lower()}_heatmap.pdf"
        save_path = os.path.join(out_dir, filename)
        plt.savefig(save_path)
        print(f"Saved heatmap to {save_path}")
        plt.close()
