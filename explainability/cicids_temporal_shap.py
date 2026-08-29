"""
Tests the "genuinely disjoint feature regime" half of the AE+Context / Temporal
IForest ensemble idea (explainability_plan.md, note after claim #3): does Temporal
IForest's SHAP attribution reflect real cross-timestep structure (a feature that
trends/repeats/builds up across the 16-flow window), or does it just concentrate on
the last flow's own static features — in which case the window would be exposing
the same evidence a flat model already sees, not a genuinely different signal from
AE+Context's ctx_* features.

Temporal IForest is trained on flattened windows: X.reshape(len(X), -1) turns each
(16, F) window into a length-16*F vector in row-major order, so flat index i maps to
timestep = i // F, feature = i % F. TreeExplainer's SHAP values on the flattened
input therefore decompose cleanly back into a (window, timestep, feature) cube.

Two checks on the strong classes (DDoS, DoS GoldenEye, DoS Slowhttptest, DoS
slowloris — confirmed non-artifactual by cicids_temporal_lastflow_check.py):

1. Position-concentration: sum |SHAP| over features, per timestep. If mass
   concentrates at timestep 15 (the flow paired with AE+Context) only, the window
   is functionally inert padding. Spread-out mass supports real temporal use.
2. Repeated-feature: for the top features by total |SHAP|, plot their per-timestep
   SHAP profile across all 16 steps. The same feature carrying same-signed SHAP at
   several timesteps (not just t=15) is the signature of a trend/buildup being read
   across the window, rather than one flow's value copy-pasted into a wider vector.
3. Feature beeswarm: collapse the timestep axis by summing each feature's SHAP
   contribution across the 16 timesteps (its total contribution to the window's
   score), paired with that feature's mean value across the window for
   color-coding, then render a standard SHAP beeswarm. Unlike the two aggregate
   checks above (mean magnitude/direction only), this shows the per-window
   distribution and the sign's dependence on feature value.
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'cicids2017'))

import numpy as np
import joblib
import matplotlib.pyplot as plt

from shap_utils import next_experiment_dir as shap_experiment_dir, tree_explainer_shap_values, save_summary_plot
from cicids_temporal_lastflow_check import build_lastflow_labels

DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_cicids_temporal')
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_cicids')

WINDOW = 16
MAX_EXPLAIN_SAMPLES = 200
TOP_K_FEATURES = 6
CLASSES = ['DDoS', 'DoS GoldenEye', 'DoS Slowhttptest', 'DoS slowloris']


def subset(X_flat, labels, label, max_n=MAX_EXPLAIN_SAMPLES, seed=0):
    idx = np.where(labels == label)[0]
    if len(idx) > max_n:
        idx = np.random.default_rng(seed).choice(idx, size=max_n, replace=False)
    return X_flat[idx], idx


def plot_position_concentration(shap_cube, out_path, title):
    """shap_cube: (n, WINDOW, F). Bar chart of mean |SHAP| summed over features,
    per timestep — where t=WINDOW-1 is the flow paired with AE+Context."""
    per_timestep = np.abs(shap_cube).sum(axis=2).mean(axis=0)  # (WINDOW,)
    frac_last = per_timestep[-1] / per_timestep.sum()

    plt.figure(figsize=(8, 4))
    colors = ['tab:blue'] * (WINDOW - 1) + ['tab:red']
    plt.bar(range(WINDOW), per_timestep, color=colors)
    plt.xlabel('Timestep within window (15 = flow paired with AE+Context)')
    plt.ylabel('Mean |SHAP| summed over features')
    plt.title(f'{title}\nlast-timestep share = {frac_last:.2%}')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Position-concentration plot saved to {out_path} (last-timestep share = {frac_last:.2%})')
    return per_timestep, frac_last


def plot_top_feature_profiles(shap_cube, feature_names, out_path, title, top_k=TOP_K_FEATURES):
    """For the top_k features by total |SHAP| (summed over samples and timesteps),
    plot each feature's mean SHAP value at every timestep — a flat line near t=15
    only means static last-flow exposure, a spread/trending line means the model
    is reading the feature's evolution across the window."""
    per_feature_total = np.abs(shap_cube).sum(axis=(0, 1))  # (F,)
    top_idx = np.argsort(-per_feature_total)[:top_k]

    plt.figure(figsize=(9, 5))
    for fi in top_idx:
        profile = shap_cube[:, :, fi].mean(axis=0)  # (WINDOW,) signed mean SHAP
        plt.plot(range(WINDOW), profile, marker='o', label=feature_names[fi])
    plt.axvline(WINDOW - 1, color='red', linestyle='--', linewidth=0.8, label='flow paired w/ AE+Context')
    plt.xlabel('Timestep within window')
    plt.ylabel('Mean signed SHAP value')
    plt.title(title)
    plt.legend(fontsize=8, loc='best')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Top-feature timestep profiles saved to {out_path}')
    return [feature_names[fi] for fi in top_idx]


def plot_feature_beeswarm(shap_cube, X_window, feature_names, out_path, title, max_display=15):
    """shap_cube, X_window: (n, WINDOW, F). Sums each feature's SHAP contribution
    across the window's 16 timesteps (its total pull on that window's score) and
    pairs it with the feature's mean value across the window for color-coding,
    then renders a standard SHAP beeswarm over the resulting (n, F) matrix."""
    shap_summed = shap_cube.sum(axis=1)      # (n, F)
    feature_mean = X_window.mean(axis=1)     # (n, F)
    save_summary_plot(shap_summed, feature_mean, feature_names, out_path, title, max_display=max_display)


if __name__ == '__main__':
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy')).astype(np.float32)
    feature_names = list(np.load(os.path.join(DATA_DIR, 'feature_names.npy'), allow_pickle=True))
    F = X_test.shape[2]
    assert X_test.shape[1] == WINDOW

    print('Rebuilding last-flow labels (ensemble-relevant, matches cicids_temporal_lastflow_check.py) …')
    _, attack_types_last = build_lastflow_labels()
    assert len(attack_types_last) == len(X_test)

    X_test_flat = X_test.reshape(len(X_test), -1)
    iforest = joblib.load(os.path.join(MODELS_DIR, 'iforest_cicids_temporal.pkl'))

    out_dir = shap_experiment_dir(_ROOT, 'cicids_temporal_shap')
    summary_lines = ['Temporal IForest — position-concentration summary (last-flow labeling):']

    for cls in CLASSES:
        X_c, idx = subset(X_test_flat, attack_types_last, cls)
        if len(X_c) == 0:
            print(f'No samples for {cls}, skipping.')
            continue
        print(f'\n{cls}: {len(X_c)} samples')

        sv = tree_explainer_shap_values(iforest, X_c)
        shap_cube = np.asarray(sv).reshape(len(X_c), WINDOW, F)
        X_c_window = X_c.reshape(len(X_c), WINDOW, F)

        safe_name = cls.lower().replace(' ', '_')
        per_timestep, frac_last = plot_position_concentration(
            shap_cube, os.path.join(out_dir, f'{safe_name}_position_concentration.png'),
            title=f'Temporal IForest — SHAP position concentration, {cls}',
        )
        top_features = plot_top_feature_profiles(
            shap_cube, feature_names, os.path.join(out_dir, f'{safe_name}_top_feature_profiles.png'),
            title=f'Temporal IForest — top-feature SHAP profiles across window, {cls}',
        )
        plot_feature_beeswarm(
            shap_cube, X_c_window, feature_names, os.path.join(out_dir, f'{safe_name}_beeswarm.png'),
            title=f'Temporal IForest — SHAP beeswarm (summed over window), {cls}',
        )
        summary_lines.append(f'  {cls:<20} n={len(X_c):<4} last-timestep share={frac_last:.2%}  top features={top_features}')

    summary = '\n'.join(summary_lines)
    print('\n' + summary)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')
    print(f'\nAll outputs saved under {out_dir}')
