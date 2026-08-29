"""
Sanity check before building the AE+Context / Temporal IForest ensemble.

The saved Temporal IForest per-class AP numbers (results_analysis.md) label a
16-flow test window positive if ANY flow inside it is an attack ("any-of-16"
rule), and tag it with the majority attack type across the window. That is a
more lenient label than the one AE+Context is scored against, where every
flow carries its own true label.

If the ensemble is meant to pair "window ending at flow t" with "AE+Context's
score for flow t", the relevant question is whether Temporal IForest's score
is actually informative about flow t's OWN label — not about whether *some*
flow in the window is an attack. This script re-labels every existing test
window by the last flow's own label/type instead of the any-of-16 rule,
reuses the already-computed model + scores (no retraining), and compares
per-class AP under both labeling conventions.
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'cicids2017'))

import numpy as np
import joblib

from preprocess_temporal import load_all, GROUP_KEY, WINDOW, STRIDE_TEST, TRAIN_DAYS
from plot_results import evaluate_per_class

DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_cicids_temporal')
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_cicids')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_cicids')

SKIP_LABELS = {'DoS Hulk'}


def build_lastflow_labels():
    """Replays preprocess_temporal.py's exact host/window loop order, but tags
    each test window with the LAST flow's own label instead of any-of-16."""
    raw = load_all()
    src_ips = raw[GROUP_KEY].values
    days = raw['Day'].values
    import pandas as pd
    timestamps = pd.to_datetime(raw['Timestamp'], format='mixed')
    labels = raw['Label'].values

    y_last, types_last = [], []
    unique_hosts = np.unique(src_ips)

    for host in unique_hosts:
        mask = src_ips == host
        ts = timestamps[mask]
        day_host = days[mask]
        order = np.argsort(ts)

        y_host = labels[mask][order]
        day_host = day_host[order]

        train_mask = np.isin(day_host, list(TRAIN_DAYS))
        test_mask = ~train_mask
        y_test_host = y_host[test_mask]

        if len(y_test_host) >= WINDOW:
            for start in range(0, len(y_test_host) - WINDOW + 1, STRIDE_TEST):
                last_label = y_test_host[start + WINDOW - 1]
                y_last.append(int(last_label != 'BENIGN'))
                types_last.append(last_label)

    return np.array(y_last, dtype=np.int32), np.array(types_last)


if __name__ == '__main__':
    print('Rebuilding last-flow labels for every existing test window …')
    y_last, attack_types_last = build_lastflow_labels()

    y_any = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    attack_types_any = np.load(os.path.join(DATA_DIR, 'attack_types_test.npy'), allow_pickle=True)

    assert len(y_last) == len(y_any), (
        f'Length mismatch: rebuilt {len(y_last)} vs saved {len(y_any)} — '
        'host/window loop did not reproduce the original ordering.'
    )

    print(f'Windows: {len(y_last)}')
    print(f'Positive rate — any-of-16: {y_any.mean():.4f}   last-flow-only: {y_last.mean():.4f}')

    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    X_test = X_test.reshape(len(X_test), -1)
    model = joblib.load(os.path.join(MODELS_DIR, 'iforest_cicids_temporal.pkl'))
    scores = -model.score_samples(X_test)

    print('\n=== Per-class AP — ANY-OF-16 window label (existing convention) ===')
    evaluate_per_class(attack_types_any, scores, skip_labels=SKIP_LABELS,
                       out_path=os.path.join(RESULTS_DIR, 'iforest_cicids_temporal_perclass_ap_anyof16.txt'))

    print('\n=== Per-class AP — LAST-FLOW-ONLY window label (ensemble-relevant) ===')
    evaluate_per_class(attack_types_last, scores, skip_labels=SKIP_LABELS,
                       out_path=os.path.join(RESULTS_DIR, 'iforest_cicids_temporal_perclass_ap_lastflow.txt'))
