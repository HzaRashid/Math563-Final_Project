import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the JSON file containing the evaluation results
cur_dir = os.path.dirname(__file__)
data_path = os.path.join(cur_dir, "KernelSizes_results", "evaluation_results_all_objectives.json")
df = pd.read_json(data_path)

out_dir = os.path.join(cur_dir, "kernelSize_heatmaps_by_objective")
os.makedirs(out_dir, exist_ok=True)

# Extract kernel‐size as before
def extract_size(row):
    kt = row['kernel'].lower()
    p  = row['kernel_params']
    if kt == 'disk':
        return p.get('r')
    if kt == 'gaussian':
        h = p.get('hsize')
        return h[0] if isinstance(h, list) and h else None
    if kt == 'motion':
        return p.get('len')
    return None

df['Size'] = df.apply(extract_size, axis=1)

# What to plot
metrics = ['time', 'l1_rel_error', 'l2_rel_error']

# Loop over objectives
for obj in df['objective'].unique():
    df_obj = df[df['objective'] == obj]
    obj_dir = os.path.join(out_dir, obj)
    os.makedirs(obj_dir, exist_ok=True)
    
    for kernel in df_obj['kernel'].unique():
        df_k = df_obj[df_obj['kernel'] == kernel]
        for metric in metrics:
            pivot = df_k.pivot(index='algorithm', columns='Size', values=metric)
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
            
            plt.figure(figsize=(8,6))
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
            
            # Title includes objective
            title_metric = metric.replace('_',' ').title()
            plt.title(f"{kernel.capitalize()} Kernel – {title_metric} ({obj.upper()} Objective)")
            plt.xlabel("Kernel Size")
            plt.ylabel("Algorithm")
            plt.tight_layout()
            
            fname = f"{kernel.lower()}_{metric.lower()}_{obj.lower()}_heatmap.pdf"
            plt.savefig(os.path.join(obj_dir, fname))
            plt.close()
            print(f"Saved {obj}–{metric} heatmap for {kernel} → {os.path.join(obj_dir, fname)}")
