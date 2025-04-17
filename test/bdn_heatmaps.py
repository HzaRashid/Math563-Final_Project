# plot_error_heatmaps.py

import os
import numpy as np
import matplotlib.pyplot as plt

# Data for Experiment 1 (Gaussian kernel)
data_exp1 = {
    'Primal DR':      {'noisy': 0.03010195454, 'bright': 0.03219430005, 'dark': 0.06194851327, 'all': 0.04270535497},
    'Primal–Dual DR': {'noisy': 0.03393180007, 'bright': 0.03936545600, 'dark': 0.07633658531, 'all': 0.04751873235},
    'ADMM':            {'noisy': 0.03705244016, 'bright': 0.03560416043, 'dark': 0.06925516651, 'all': 0.04854400493},
    'Chambolle–Pock': {'noisy': 0.03162340243, 'bright': 0.03088727418, 'dark': 0.08429944273, 'all': 0.04745065417},
}

# Data for Experiment 2
data_exp2 = {
    'Primal DR':      {'noisy': 0.1560035587,  'bright': 0.1815820369,  'dark': 0.09288852985, 'all': 0.1588567666},
    'Primal–Dual DR': {'noisy': 0.1604454635,  'bright': 0.1983653992,  'dark': 0.1527499033,  'all': 0.1597308914},
    'ADMM':            {'noisy': 0.1539461930,  'bright': 0.1892943313,  'dark': 0.1075157434,  'all': 0.1533212985},
    'Chambolle–Pock': {'noisy': 0.1513826211,  'bright': 0.1915540715,  'dark': 0.1769365061,  'all': 0.1884900622},
}

algorithms = list(data_exp1.keys())
categories = ['noisy', 'bright', 'dark', 'all']

def plot_heatmap(data, title, filename_base):
    # determine directories
    cur_dir = os.path.dirname(__file__)
    out_dir = os.path.join(cur_dir, 'BDN_heatmaps')
    os.makedirs(out_dir, exist_ok=True)

    # prepare matrix
    matrix = np.array([[data[alg][cat] for cat in categories] for alg in algorithms])

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, aspect='auto')

    # annotate each cell with its value in white text
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.4f}",
                    ha='center', va='center', color='white')

    # set labels and ticks
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_yticklabels(algorithms)
    ax.set_xlabel('Image Category')
    ax.set_ylabel('Algorithm')
    ax.set_title(title)

    # colorbar and layout
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    # save as PDF
    pdf_path = os.path.join(out_dir, filename_base + '.pdf')
    plt.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {pdf_path}")

if __name__ == "__main__":
    plot_heatmap(data_exp1, 'Bright/Dark/Noisy/All Relative L1 Loss \n(Gaussian kernel 15x15 sigma=1.0)', 'heatmap_exp1')
    plot_heatmap(data_exp2, 'Bright/Dark/Noisy/All Relative L1 Loss \n(Gaussian kernel 9x9 sigma=4.0)', 'heatmap_exp2')
