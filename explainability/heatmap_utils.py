import os
import numpy as np
import matplotlib.pyplot as plt


def next_experiment_dir(root: str, experiment_name: str):
    '''Under data/explainability/<experiment_name>/<n>/, get filename for next unused integer after `n`'''
    base = os.path.join(root, 'data', 'explainability', experiment_name)
    os.makedirs(base, exist_ok=True)
    existing = [int(d) for d in os.listdir(base) if d.isdigit()]
    n = max(existing, default=0) + 1
    out_dir = os.path.join(base, str(n))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def per_position_error(orig: np.ndarray, recon: np.ndarray):
    # orig, recon: (N, seq_len, features) -> (N, seq_len) squared error averaged over features
    return ((orig - recon) ** 2).mean(axis=-1)


def plot_error_heatmap(error_map: np.ndarray, out_path: str, title: str,
                        xlabel: str = 'Position', ylabel: str = 'Sample',
                        cbar_label: str = 'Squared error'):
    fig, ax = plt.subplots(figsize=(10, max(4, 0.2 * error_map.shape[0])))
    im = ax.imshow(error_map, aspect='auto', cmap='viridis')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Heatmap saved to {out_path}')


def gini_coefficient(x: np.ndarray) -> float:
    """0 = error spread perfectly evenly across positions, 1 = concentrated in a single position."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    total = x.sum()
    if total == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return (2 * np.sum(i * x)) / (n * total) - (n + 1) / n


def per_window_gini(error_map: np.ndarray) -> np.ndarray:
    # error_map: (N, seq_len) -> (N,) gini coefficient per row
    return np.array([gini_coefficient(row) for row in error_map])


def plot_gini_comparison(gini_a: np.ndarray, gini_b: np.ndarray,
                          label_a: str, label_b: str, out_path: str, title: str):
    """Histogram of per-sample error-concentration (Gini) values, two models overlaid."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 31)
    ax.hist(gini_a, bins=bins, alpha=0.6, label=f'{label_a} (mean={gini_a.mean():.3f})')
    ax.hist(gini_b, bins=bins, alpha=0.6, label=f'{label_b} (mean={gini_b.mean():.3f})')
    ax.set_xlabel('Gini coefficient (per-sample error concentration)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Gini comparison saved to {out_path}')


def plot_value_distribution_comparison(values_a: np.ndarray, values_b: np.ndarray,
                                        label_a: str, label_b: str, out_path: str,
                                        title: str, xlabel: str, bins: int = 30):
    """Generic overlaid histogram comparison of two per-sample scalar distributions
    (e.g. attention centroid lag) -- same visual convention as plot_gini_comparison,
    but for metrics that aren't bounded to [0, 1]."""
    fig, ax = plt.subplots(figsize=(8, 4))
    lo = min(values_a.min(), values_b.min())
    hi = max(values_a.max(), values_b.max())
    bin_edges = np.linspace(lo, hi, bins + 1)
    ax.hist(values_a, bins=bin_edges, alpha=0.6, label=f'{label_a} (mean={values_a.mean():.3f})')
    ax.hist(values_b, bins=bin_edges, alpha=0.6, label=f'{label_b} (mean={values_b.mean():.3f})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Distribution comparison saved to {out_path}')


def plot_paired_window_comparison(error_a: np.ndarray, error_b: np.ndarray,
                                   label_a: str, label_b: str,
                                   out_path: str, title: str, xlabel: str = 'Position'):
    """Per-position error of two models on a single sample, overlaid as line plots."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(error_a, label=label_a, marker='o', markersize=3)
    ax.plot(error_b, label=label_b, marker='o', markersize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Squared error')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Comparison plot saved to {out_path}')
