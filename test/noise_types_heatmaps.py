import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the JSON file containing the noise evaluation results.
cur_dir = os.path.dirname(__file__)
data_path = os.path.join(cur_dir,  "apply_BDN_L1_NoiseTypes", "evaluation_results.json")
df = pd.read_json(data_path)

out_dir = os.path.join(cur_dir, "NoiseTypes_heatmaps")
os.makedirs(out_dir, exist_ok=True)

# For Gaussian mode, the relevant parameters are 'mean' and 'var'.
df_gaussian = df[df['mode'] == 'gaussian'].copy()
# Convert the mean and var columns to numeric types (they should be already numeric)
df_gaussian['mean'] = pd.to_numeric(df_gaussian['mean'])
df_gaussian['var'] = pd.to_numeric(df_gaussian['var'])

# For Salt & Pepper mode, the relevant parameter is 'amount'.
df_sp = df[df['mode'] == 's&p'].copy()
df_sp['amount'] = pd.to_numeric(df_sp['amount'])

# Define an aggregator for the pivot table (use mean error in case an algorithm has multiple measurements)
aggfunc = 'mean'

# Generate Gaussian heatmap for the "mean" parameter.
pivot_gauss_mean = df_gaussian.pivot_table(index='algorithm', columns='mean', values='error', aggfunc=aggfunc)
pivot_gauss_mean = pivot_gauss_mean.reindex(sorted(pivot_gauss_mean.columns), axis=1)
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_gauss_mean, annot=True, fmt=".4f", cmap="viridis")
plt.title("Gaussian Noise - Mean Parameter (Error)")
plt.xlabel("Mean Value")
plt.ylabel("Algorithm")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "gaussian_mean_heatmap.pdf"))
plt.close()
print("Saved heatmap to gaussian_mean_heatmap.pdf")

# Generate Gaussian heatmap for the "var" parameter.
pivot_gauss_var = df_gaussian.pivot_table(index='algorithm', columns='var', values='error', aggfunc=aggfunc)
pivot_gauss_var = pivot_gauss_var.reindex(sorted(pivot_gauss_var.columns), axis=1)
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_gauss_var, annot=True, fmt=".4f", cmap="viridis")
plt.title("Gaussian Noise - Variance Parameter (Error)")
plt.xlabel("Variance Value")
plt.ylabel("Algorithm")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "gaussian_var_heatmap.pdf"))
plt.close()
print("Saved heatmap to gaussian_var_heatmap.pdf")

# Generate heatmap for Salt & Pepper mode using the "amount" parameter.
pivot_sp_amount = df_sp.pivot_table(index='algorithm', columns='amount', values='error', aggfunc=aggfunc)
pivot_sp_amount = pivot_sp_amount.reindex(sorted(pivot_sp_amount.columns), axis=1)
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_sp_amount, annot=True, fmt=".4f", cmap="viridis")
plt.title("Salt & Pepper Noise - Amount Parameter (Error)")
plt.xlabel("Amount Value")
plt.ylabel("Algorithm")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "sp_amount_heatmap.pdf"))
plt.close()
print("Saved heatmap to sp_amount_heatmap.pdf")
