"""
Trivial baseline from Kim et al. (2022): anomaly score = ||w||_2 (L2 norm of the window).

This corresponds to Case 2 in their paper — assuming the model outputs zero regardless
of input, so reconstruction error equals the input magnitude. No training required.

Any trained model should exceed this baseline on PR-AUC; if it does not, the window size
or model is not adding value over raw signal amplitude.
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np

from plot_results import sweep_threshold, evaluate, plot_results

BENCHMARK = 'A3Benchmark'
DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_yahoo', BENCHMARK)
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_yahoo')
GRAPHS_DIR = os.path.join(_ROOT, 'data', 'graphs', 'results', 'yahoo')

MODEL_NAME = f'Trivial Baseline — ||w||_2 ({BENCHMARK})'


def load_data():
    X_test = np.load(f'{DATA_DIR}/X_test.npy')       # (N, 64, 1)
    y_test = np.load(f'{DATA_DIR}/y_test.npy')        # (N,)
    return X_test, y_test


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    X_test, y_test = load_data()

    # Score = L2 norm of each window (flattened), matching Kim et al. Case 2.
    # Shape: (N, 64, 1) -> (N, 64) -> scalar per window.
    scores = np.linalg.norm(X_test.reshape(len(X_test), -1), axis=1)

    print(f'Benchmark      : {BENCHMARK}')
    print(f'Test windows   : {len(X_test)}')
    print(f'Anomaly rate   : {y_test.mean():.4f}')
    print(f'Score range    : [{scores.min():.4f}, {scores.max():.4f}]')

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(
        y_test, scores, threshold, MODEL_NAME,
        out_path=os.path.join(RESULTS_DIR, f'trivial_baseline_{BENCHMARK}_report.txt'),
    )
    plot_results(
        y_test, y_pred, scores, MODEL_NAME,
        out_path=os.path.join(GRAPHS_DIR, f'trivial_baseline_{BENCHMARK}_results.png'),
    )
