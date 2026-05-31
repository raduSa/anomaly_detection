import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

from plot_results import sweep_threshold, evaluate, plot_results

BENCHMARK = 'A4Benchmark'
DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_yahoo', BENCHMARK)
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_yahoo')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_yahoo')

N_ESTIMATORS = 150
RANDOM_SEED = 123


def load_data():
    X_train = np.load(f'{DATA_DIR}/X_train_normal.npy')
    X_test = np.load(f'{DATA_DIR}/X_test.npy')
    y_test = np.load(f'{DATA_DIR}/y_test.npy')
    return X_train.reshape(len(X_train), -1), X_test.reshape(len(X_test), -1), y_test


def train(X_train):
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=min(len(X_train), 50_000),
        contamination='auto',
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    print(f'Training Isolation Forest on {X_train.shape[0]} windows …')
    model.fit(X_train)
    return model


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, X_test, y_test = load_data()
    model = train(X_train)

    joblib.dump(model, f'{MODELS_DIR}/iforest_{BENCHMARK}.pkl')

    scores = -model.score_samples(X_test)

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(y_test, scores, threshold, f'Isolation Forest ({BENCHMARK})',
                    out_path=f'{RESULTS_DIR}/iforest_{BENCHMARK}_report.txt')
    plot_results(y_test, y_pred, scores, f'Isolation Forest ({BENCHMARK})',
                out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'yahoo', f'iforest_{BENCHMARK}_results.png'))
