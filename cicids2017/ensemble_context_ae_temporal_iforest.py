"""
Practical ensemble: AE+Context (per-flow score) + Temporal IForest (last-flow-window
score), row-aligned by flow identity.

Motivation (see mds/handoff.md "Practical ensemble" item, results.tex CIC Discussion
"A practical ensemble" paragraph): AE+Context and Temporal IForest cover disjoint,
non-overlapping attack classes (confirmed via SHAP in the "Disjoint Feature Regimes"
explainability section) — AE+Context is strong on DDoS/PortScan, Temporal IForest is
strong on DoS GoldenEye/Slowhttptest/slowloris. This script builds the combined score
and tests whether it preserves both models' specialties, under two fusion rules:

  1. Weighted soft voting: s = w * s_AE_norm + (1-w) * s_IF_norm, swept over several w.
  2. Max-rule fusion: s = max(s_AE_norm, s_IF_norm) — an OR-like combiner, matching the
     "either model can independently flag an anomaly" story from the disjoint-feature
     finding better than an average would.

Row alignment: preprocess_context.py and preprocess_temporal.py both start from the same
5 raw CSVs, concatenated in the same FILES order, with the same "- Attempted" filter
applied before any reset_index — so the row position in that shared, freshly-concatenated
dataframe (called `base_id` below) is a stable flow identifier shared by both pipelines,
even though each pipeline subsequently reorders rows differently (context: one global
chronological sort; temporal: per-host chronological sort). This script replays both
pipelines' exact row-ordering logic (without recomputing features — the already-saved
X_test/model files are reused) purely to recover, for every temporal test window, which
position in AE+Context's X_test corresponds to that window's last flow.

Score normalization: each model's test scores are standardized (z-score) using mean/std
computed on that SAME model's own benign TRAINING scores, never on test data — avoids
leaking test-set anomaly statistics into the fusion.
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'cicids2017'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from plot_results import sweep_threshold, evaluate, evaluate_per_class
from preprocess_temporal import (
    FILES, ATTEMPTED_SUFFIX, GROUP_KEY, WINDOW, STRIDE_TEST, TRAIN_DAYS,
)

CONTEXT_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_cicids_context')
TEMPORAL_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_cicids_temporal')
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_cicids')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_cicids')

SKIP_LABELS = {'DoS Hulk'}
SOFT_VOTE_WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]  # weight on AE+Context

AE_HIDDEN = [64, 32, 16]
AE_DROPOUT = 0.1


class MLP_AE(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        enc_dims = [input_dim] + hidden
        dec_dims = hidden[::-1] + [input_dim]

        enc = []
        for i in range(len(enc_dims) - 1):
            enc += [nn.Linear(enc_dims[i], enc_dims[i + 1]), nn.ReLU()]
            if i < len(enc_dims) - 2:
                enc.append(nn.Dropout(AE_DROPOUT))
        self.encoder = nn.Sequential(*enc)

        dec = []
        for i in range(len(dec_dims) - 1):
            if i < len(dec_dims) - 2:
                dec += [nn.Linear(dec_dims[i], dec_dims[i + 1]), nn.ReLU(),
                        nn.Dropout(AE_DROPOUT)]
            else:
                dec += [nn.Linear(dec_dims[i], dec_dims[i + 1]), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


@torch.no_grad()
def ae_reconstruction_scores(model, X, device, batch_size=2048):
    model.eval()
    scores = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).to(device)
        recon = model(batch)
        scores.append(((recon - batch) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(scores)


def load_shared_base_raw():
    """Replays the shared prefix of both preprocess_context.py and
    preprocess_temporal.py: concat FILES in order, drop '- Attempted' rows,
    reset_index(drop=True). The resulting row position is `base_id`, a flow
    identifier shared by both pipelines (verified identical FILES/filter in both)."""
    dfs = []
    for file_name in FILES:
        d = pd.read_csv(os.path.join(_ROOT, 'data', 'raw', 'cicids2017', file_name))
        d['Day'] = file_name.split('-')[0]
        dfs.append(d)
    raw = pd.concat(dfs, ignore_index=True)
    raw = raw[~raw['Label'].str.endswith(ATTEMPTED_SUFFIX, na=False)].reset_index(drop=True)
    raw['ts_float'] = pd.to_datetime(raw['Timestamp'], format='mixed').astype(np.int64) / 1e9
    return raw


def context_test_base_id_order(raw):
    """Reproduces preprocess_context.py's row ordering: global chronological sort,
    then benign-test flows (chronological) followed by attack flows (chronological).
    Returns the base_id (position in `raw`) for every row of AE+Context's X_test, in
    X_test's own order."""
    order = np.argsort(raw['ts_float'].values, kind='stable')
    sorted_base_ids = order  # position i in sorted view -> base_id order[i]

    labels = raw['Label'].values[order]
    days = raw['Day'].values[order]

    is_benign = labels == 'BENIGN'
    is_attack = ~is_benign
    train_mask = is_benign & np.isin(days, list(TRAIN_DAYS))
    benign_test_mask = is_benign & ~np.isin(days, list(TRAIN_DAYS))

    return np.concatenate([sorted_base_ids[benign_test_mask], sorted_base_ids[is_attack]])


def temporal_test_window_lastflow(raw):
    """Reproduces preprocess_temporal.py's per-host window loop (same host iteration
    order, same WINDOW/STRIDE_TEST), returning, for every test window, the base_id and
    the true label ('BENIGN' or attack type) of its LAST flow — the ensemble-relevant
    labeling convention validated in explainability/cicids_temporal_lastflow_check.py,
    used here instead of the lenient any-of-16 window label saved in y_test.npy."""
    base_ids = raw.index.values  # base_id == row position in the shared raw frame
    src_ips = raw[GROUP_KEY].values
    days = raw['Day'].values
    labels = raw['Label'].values
    timestamps = pd.to_datetime(raw['Timestamp'], format='mixed')

    lastflow_base_ids, lastflow_types = [], []
    unique_hosts = np.unique(src_ips)

    for host in unique_hosts:
        mask = src_ips == host
        ts = timestamps[mask]
        day_host = days[mask]
        label_host = labels[mask]
        order = np.argsort(ts)

        ids_host = base_ids[mask][order]
        day_host = day_host[order]
        label_host = label_host[order]

        train_mask = np.isin(day_host, list(TRAIN_DAYS))
        test_mask = ~train_mask
        ids_test_host = ids_host[test_mask]
        label_test_host = label_host[test_mask]

        if len(ids_test_host) >= WINDOW:
            for start in range(0, len(ids_test_host) - WINDOW + 1, STRIDE_TEST):
                lastflow_base_ids.append(ids_test_host[start + WINDOW - 1])
                lastflow_types.append(label_test_host[start + WINDOW - 1])

    return np.array(lastflow_base_ids), np.array(lastflow_types)


def zscore_normalize(scores_test, scores_train_benign):
    mu, sigma = scores_train_benign.mean(), scores_train_benign.std()
    return (scores_test - mu) / sigma


def rank_normalize(scores_test, scores_train_benign):
    """Map each test score to its percentile (0-1) within the benign TRAINING score
    distribution (empirical CDF via searchsorted). Unlike z-score, this is immune to
    heavy tails / outlier magnitude — a score's position is bounded to [0, 1] regardless
    of how extreme the raw value is, so one model's heavy-tailed outliers can no longer
    dominate a linear/max combination with another model's more tightly-bounded scores."""
    sorted_train = np.sort(scores_train_benign)
    ranks = np.searchsorted(sorted_train, scores_test, side='right')
    return ranks / len(sorted_train)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=== Loading AE+Context ===')
    X_train_ae = np.load(os.path.join(CONTEXT_DIR, 'X_train.npy'))
    X_test_ae = np.load(os.path.join(CONTEXT_DIR, 'X_test.npy'))
    ae_model = MLP_AE(X_test_ae.shape[1], AE_HIDDEN).to(device)
    ae_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'autoencoder_context.pt'),
                                         map_location=device))
    s_ae_test_all = ae_reconstruction_scores(ae_model, X_test_ae, device)
    s_ae_train = ae_reconstruction_scores(ae_model, X_train_ae, device)

    print('=== Loading Temporal IForest ===')
    X_train_if = np.load(os.path.join(TEMPORAL_DIR, 'X_train.npy'))
    X_train_if = X_train_if.reshape(len(X_train_if), -1)
    X_test_if = np.load(os.path.join(TEMPORAL_DIR, 'X_test.npy'))
    X_test_if = X_test_if.reshape(len(X_test_if), -1)
    if_model = joblib.load(os.path.join(MODELS_DIR, 'iforest_cicids_temporal.pkl'))
    s_if_test_all = -if_model.score_samples(X_test_if)
    s_if_train = -if_model.score_samples(X_train_if)

    print('=== Recovering row alignment (shared base_id) ===')
    raw = load_shared_base_raw()
    context_base_ids = context_test_base_id_order(raw)
    assert len(context_base_ids) == len(s_ae_test_all), (
        f'context alignment length mismatch: {len(context_base_ids)} vs {len(s_ae_test_all)}')
    base_id_to_context_pos = {bid: i for i, bid in enumerate(context_base_ids)}

    lastflow_base_ids, attack_types_if = temporal_test_window_lastflow(raw)
    assert len(lastflow_base_ids) == len(s_if_test_all), (
        f'temporal alignment length mismatch: {len(lastflow_base_ids)} vs {len(s_if_test_all)}')

    # Every temporal test window's last flow is a non-Monday, non-Attempted flow,
    # so it must appear in AE+Context's test set (benign-test ∪ attack).
    context_pos = np.array([base_id_to_context_pos[bid] for bid in lastflow_base_ids])
    n_matched = len(context_pos)
    print(f'Matched {n_matched} temporal test windows to AE+Context rows by flow identity.')

    y_row = (attack_types_if != 'BENIGN').astype(int)
    print(f'Row-aligned ensemble set size: {len(y_row)}   positive rate: {y_row.mean():.4f}')

    from sklearn.metrics import average_precision_score
    overall_summary = []
    perclass_summary = {}  # (norm_name, variant_name) -> {class: ap}

    def perclass_ap_dict(scores):
        d = {}
        for cls in sorted(np.unique(attack_types_if)):
            if cls == 'BENIGN' or cls in SKIP_LABELS:
                continue
            mask = (attack_types_if == cls) | (attack_types_if == 'BENIGN')
            d[cls] = average_precision_score((attack_types_if[mask] != 'BENIGN').astype(int), scores[mask])
        return d

    for norm_name, norm_fn in [('zscore', zscore_normalize), ('rank', rank_normalize)]:
        print(f'\n\n########## Normalization: {norm_name} ##########')
        s_ae_row = norm_fn(s_ae_test_all[context_pos], s_ae_train)
        s_if_row = norm_fn(s_if_test_all, s_if_train)

        print(f'\n=== Standalone AE+Context ({norm_name}-normalized) ===')
        evaluate_per_class(attack_types_if, s_ae_row, skip_labels=SKIP_LABELS,
                           out_path=os.path.join(RESULTS_DIR, f'ensemble_standalone_ae_context_{norm_name}_perclass_ap.txt'))
        print(f'\n=== Standalone Temporal IForest ({norm_name}-normalized, last-flow label) ===')
        evaluate_per_class(attack_types_if, s_if_row, skip_labels=SKIP_LABELS,
                           out_path=os.path.join(RESULTS_DIR, f'ensemble_standalone_temporal_iforest_{norm_name}_perclass_ap.txt'))
        overall_summary.append((f'{norm_name}: AE+Context (standalone)', average_precision_score(y_row, s_ae_row)))
        overall_summary.append((f'{norm_name}: Temporal IForest (standalone)', average_precision_score(y_row, s_if_row)))
        perclass_summary[(norm_name, 'AE+Context (standalone)')] = perclass_ap_dict(s_ae_row)
        perclass_summary[(norm_name, 'Temporal IForest (standalone)')] = perclass_ap_dict(s_if_row)

        print(f'\n=== [{norm_name}] Max-rule fusion ===')
        s_max = np.maximum(s_ae_row, s_if_row)
        threshold = sweep_threshold(y_row, s_max, beta=2.0)
        evaluate(y_row, s_max, threshold, f'Ensemble (max-rule, {norm_name})',
                 out_path=os.path.join(RESULTS_DIR, f'ensemble_maxrule_{norm_name}_report.txt'))
        evaluate_per_class(attack_types_if, s_max, skip_labels=SKIP_LABELS,
                           out_path=os.path.join(RESULTS_DIR, f'ensemble_maxrule_{norm_name}_perclass_ap.txt'))
        overall_summary.append((f'{norm_name}: max-rule', average_precision_score(y_row, s_max)))
        perclass_summary[(norm_name, 'max-rule')] = perclass_ap_dict(s_max)

        for w in SOFT_VOTE_WEIGHTS:
            print(f'\n=== [{norm_name}] Soft voting (w_AE={w}) ===')
            s_soft = w * s_ae_row + (1 - w) * s_if_row
            threshold = sweep_threshold(y_row, s_soft, beta=2.0)
            evaluate(y_row, s_soft, threshold, f'Ensemble (soft-vote w={w}, {norm_name})',
                     out_path=os.path.join(RESULTS_DIR, f'ensemble_softvote_w{w}_{norm_name}_report.txt'))
            evaluate_per_class(attack_types_if, s_soft, skip_labels=SKIP_LABELS,
                               out_path=os.path.join(RESULTS_DIR, f'ensemble_softvote_w{w}_{norm_name}_perclass_ap.txt'))
            overall_summary.append((f'{norm_name}: soft-vote w={w}', average_precision_score(y_row, s_soft)))
            perclass_summary[(norm_name, f'soft-vote w={w}')] = perclass_ap_dict(s_soft)

    print('\n\n=== Overall AP summary (all normalizations/variants) ===')
    summary_lines = ['Overall AP (row-aligned ensemble set):']
    for name, ap in overall_summary:
        summary_lines.append(f'  {name:<45} AP={ap:.4f}')
    summary = '\n'.join(summary_lines)
    print(summary)
    with open(os.path.join(RESULTS_DIR, 'ensemble_overall_ap_summary.txt'), 'w') as f:
        f.write(summary + '\n')

    # Focused table: the 5 classes the ensemble is meant to cover
    focus_classes = ['DDoS', 'PortScan', 'DoS GoldenEye', 'DoS Slowhttptest', 'DoS slowloris']
    pc_lines = ['Per-class AP, focus classes only (rows=variant, cols=class):', '']
    header = f'{"variant":<45}' + ''.join(f'{c:>18}' for c in focus_classes)
    pc_lines.append(header)
    for (norm_name, variant), d in perclass_summary.items():
        row = f'{norm_name + ": " + variant:<45}' + ''.join(f'{d.get(c, float("nan")):>18.4f}' for c in focus_classes)
        pc_lines.append(row)
    pc_summary = '\n'.join(pc_lines)
    print('\n' + pc_summary)
    with open(os.path.join(RESULTS_DIR, 'ensemble_focus_classes_ap_summary.txt'), 'w') as f:
        f.write(pc_summary + '\n')


if __name__ == '__main__':
    main()
