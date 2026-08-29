import os
import numpy as np
import shap
import matplotlib.pyplot as plt


def next_experiment_dir(root: str, experiment_name: str) -> str:
    base = os.path.join(root, 'data', 'explainability', experiment_name)
    os.makedirs(base, exist_ok=True)
    existing = [int(d) for d in os.listdir(base) if d.isdigit()]
    n = max(existing, default=0) + 1
    out_dir = os.path.join(base, str(n))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def tree_explainer_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


def gradient_explainer_shap_values(error_module, background, X, n_samples=None):
    """
    error_module: torch.nn.Module whose forward(x) returns a per-sample scalar
    (e.g. reconstruction MSE). shap.GradientExplainer explains that scalar output
    directly w.r.t. the input features.
    """
    import torch
    background_t = torch.from_numpy(background)
    X_t = torch.from_numpy(X if n_samples is None else X[:n_samples])
    explainer = shap.GradientExplainer(error_module, background_t)
    shap_values = explainer.shap_values(X_t)
    return shap_values, X_t.numpy()


def _squeeze_singleton_output(shap_values):
    """GradientExplainer returns shape (n, F, 1) for a scalar-output model (a
    trailing singleton output dim) instead of (n, F). Squeeze it so downstream
    consumers (shap.summary_plot, mean-|SHAP| ranking) see a plain 2D array --
    otherwise summary_plot silently misreads the shape and renders garbage."""
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 3 and shap_arr.shape[-1] == 1:
        shap_arr = shap_arr[..., 0]
    return shap_arr


def top_features_by_mean_abs_shap(shap_values, feature_names, top_k=15):
    """Ranks features by mean |SHAP| across samples. Returns a list of
    (feature_name, mean_abs_shap) sorted descending -- used to check whether a
    specific feature family (e.g. ctx_* host-context features) dominates."""
    shap_arr = _squeeze_singleton_output(shap_values)
    mean_abs = np.abs(shap_arr).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_k]
    return [(feature_names[i], float(mean_abs[i])) for i in order]


def save_summary_plot(shap_values, X, feature_names, out_path, title, max_display=15):
    shap_arr = _squeeze_singleton_output(shap_values)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_arr, X, feature_names=feature_names,
                       max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'SHAP summary plot saved to {out_path}')
