"""
Tests explainability_plan.md claims #1 and #2 (CIC-IDS2017 host-context features).

Claim #1 — Host-context features / PortScan blind spot (results.tex "Host-context
features dominate" / per-attack breakdown): ctx_max_port_cnt / ctx_max_port_rate are
asserted to explain why context models jump from AP 0.07-0.28 to >0.60 on PortScan.
Tested with a SHAP summary plot on IForest+Context and AE+Context, PortScan samples only.

Claim #2 — DoS GoldenEye split between IForest+Context (AP 0.655) and AE+Context (~0.22):
IForest+Context is claimed to isolate "high ctx_max_flow_cnt, low ctx_max_port_rate"
cleanly, while AE+Context spreads its reconstruction error across many dimensions
instead. Tested with a SHAP summary plot on IForest+Context (TreeExplainer) for
GoldenEye samples, compared directly against the AE+Context per-feature
reconstruction-error heatmap on the same samples (not a SHAP explainer — the AE side
of this claim is about where reconstruction error concentrates, which the heatmap
already answers; forcing a SHAP explainer onto it would test a different question).

Claim #3 (AE+Context half) — disjoint-feature-regime premise behind the proposed
AE+Context / Temporal IForest ensemble (results.tex, CIC Discussion, "A practical
ensemble"; explainability_plan.md note after claim #3). Temporal IForest's half was
already confirmed in explainability.tex §"Does Temporal IForest Use the Whole
Window?" — its strong-class SHAP attribution is spread across timesteps and keys
off temporal features (packet-length stats for DDoS, IAT stats for slowloris), not
one static flow. This tests the mirror claim for AE+Context on ITS strong classes
(DDoS, PortScan): that its attribution is dominated by ctx_* host-context features —
the 9 features that exist only in AE+Context's feature set, not in Temporal
IForest's — so the two models' advantages trace to genuinely non-overlapping
feature evidence, not just uncorrelated noise that happens to cover different
classes. PortScan reuses the claim #1 run below; DDoS is the new subset.
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'cicids2017'))

import numpy as np
import joblib
import torch
import torch.nn as nn

from shap_utils import (
    next_experiment_dir as shap_experiment_dir,
    tree_explainer_shap_values, gradient_explainer_shap_values, save_summary_plot,
    top_features_by_mean_abs_shap,
)
from heatmap_utils import plot_error_heatmap

import train_autoencoder_context as ae_mod

DATA_DIR = ae_mod.DATA_DIR
MODELS_DIR = ae_mod.MODELS_DIR
BACKGROUND_SIZE = 100
MAX_EXPLAIN_SAMPLES = 200 


class ReconstructionError(nn.Module):
    """Wraps an MLP_AE so its forward() returns the scalar anomaly score directly,
    which is what shap.GradientExplainer needs to attribute."""

    def __init__(self, ae: nn.Module):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        recon = self.ae(x)
        return ((recon - x) ** 2).mean(dim=1, keepdim=True)


def load_iforest():
    return joblib.load(os.path.join(MODELS_DIR, 'iforest_context.pkl'))


def load_autoencoder(device):
    model = ae_mod.MLP_AE(input_dim=len(feature_names), hidden=ae_mod.HIDDEN).to(device)
    state = torch.load(os.path.join(MODELS_DIR, 'autoencoder_context.pt'), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def subset(X, attack_types, label, max_n=MAX_EXPLAIN_SAMPLES, seed=0):
    idx = np.where(attack_types == label)[0]
    if len(idx) > max_n:
        idx = np.random.default_rng(seed).choice(idx, size=max_n, replace=False)
    return X[idx], idx


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy')).astype(np.float32)
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy')).astype(np.float32)
    attack_types = np.load(os.path.join(DATA_DIR, 'attack_types_test.npy'), allow_pickle=True)
    feature_names = list(np.load(os.path.join(DATA_DIR, 'feature_names.npy'), allow_pickle=True))

    print('Attack types present:', sorted(set(attack_types.tolist())))

    iforest = load_iforest()
    ae = load_autoencoder(device)
    error_model = ReconstructionError(ae)
    error_model.eval()

    rng = np.random.default_rng(0)
    background = X_train[rng.choice(len(X_train), size=BACKGROUND_SIZE, replace=False)]

    # Claim #1 — PortScan
    out_dir = shap_experiment_dir(_ROOT, 'cicids_shap_portscan')

    X_ps, _ = subset(X_test, attack_types, 'PortScan')
    print(f'PortScan subset: {len(X_ps)} samples')

    sv_iforest_ps = tree_explainer_shap_values(iforest, X_ps)
    save_summary_plot(
        sv_iforest_ps, X_ps, feature_names,
        os.path.join(out_dir, 'iforest_context_portscan_shap.png'),
        title='IForest+Context — SHAP summary, PortScan samples',
    )

    sv_ae_ps, X_ae_ps = gradient_explainer_shap_values(error_model, background, X_ps)
    save_summary_plot(
        sv_ae_ps, X_ae_ps, feature_names,
        os.path.join(out_dir, 'autoencoder_context_portscan_shap.png'),
        title='AE+Context — SHAP summary, PortScan samples',
    )
    print(f'Claim #1 outputs saved under {out_dir}')

    # Claim #2 — DoS GoldenEye
    out_dir = shap_experiment_dir(_ROOT, 'cicids_shap_goldeneye')

    X_ge, _ = subset(X_test, attack_types, 'DoS GoldenEye')
    print(f'DoS GoldenEye subset: {len(X_ge)} samples')

    sv_iforest_ge = tree_explainer_shap_values(iforest, X_ge)
    save_summary_plot(
        sv_iforest_ge, X_ge, feature_names,
        os.path.join(out_dir, 'iforest_context_goldeneye_shap.png'),
        title='IForest+Context — SHAP summary, DoS GoldenEye samples',
    )

    # AE+Context side of this claim: reconstruction-error heatmap, not SHAP — this
    # answers "where does the error concentrate" directly, matching what the claim
    # in results.tex is actually about.
    with torch.no_grad():
        recon = ae(torch.from_numpy(X_ge)).numpy()
    per_feature_error = (X_ge - recon) ** 2  # (n_goldeneye, 86)

    plot_error_heatmap(
        per_feature_error, os.path.join(out_dir, 'autoencoder_context_goldeneye_error_heatmap.png'),
        title='AE+Context — per-feature reconstruction error, DoS GoldenEye samples',
        xlabel=f'Feature ({feature_names[0]}, {feature_names[1]}, ...)', ylabel='GoldenEye sample',
    )
    print(f'Claim #2 outputs saved under {out_dir}')

    # Claim #3 (AE+Context half) — disjoint-feature-regime check on DDoS and PortScan
    out_dir = shap_experiment_dir(_ROOT, 'cicids_context_disjoint_regime')
    ctx_features = {f for f in feature_names if f.startswith('ctx_')}
    print(f'\nctx_* host-context features ({len(ctx_features)}): {sorted(ctx_features)}')

    summary_lines = ['AE+Context — top-15 features by mean |SHAP|, disjoint-feature-regime check:']

    for label, X_subset in [('PortScan', X_ps), ('DDoS', None)]:
        if label == 'DDoS':
            X_subset, _ = subset(X_test, attack_types, 'DDoS')
            print(f'\nDDoS subset: {len(X_subset)} samples')
            sv_ae, X_ae = gradient_explainer_shap_values(error_model, background, X_subset)
            save_summary_plot(
                sv_ae, X_ae, feature_names,
                os.path.join(out_dir, 'autoencoder_context_ddos_shap.png'),
                title='AE+Context — SHAP summary, DDoS samples',
            )
        else:
            sv_ae = sv_ae_ps  # reuse claim #1's PortScan SHAP run, no need to recompute

        top = top_features_by_mean_abs_shap(sv_ae, feature_names, top_k=15)
        n_ctx_in_top = sum(1 for f, _ in top if f in ctx_features)
        summary_lines.append(f'\n{label} (n_ctx_in_top15 = {n_ctx_in_top}/15):')
        for f, v in top:
            marker = ' [ctx]' if f in ctx_features else ''
            summary_lines.append(f'  {f:<30} mean|SHAP|={v:.5f}{marker}')

    summary = '\n'.join(summary_lines)
    print('\n' + summary)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')
    print(f'\nClaim #3 (AE+Context) outputs saved under {out_dir}')
