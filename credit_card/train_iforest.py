import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

from plot_results import sweep_threshold, evaluate, plot_results

DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_credit_card')
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_credit_card')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_credit_card')

N_ESTIMATORS = 150
MAX_SAMPLES = 200000
RANDOM_SEED = 123


def load_data():
    X_train = np.load(f'{DATA_DIR}/X_train_normal.npy')
    X_test = np.load(f'{DATA_DIR}/X_test.npy')
    y_test = np.load(f'{DATA_DIR}/y_test.npy')
    return X_train, X_test, y_test


def train(X_train):
    # contamination='auto' -> sweep for the best threshold ourselves via the PR curve
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination='auto',
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    print(f'Training Isolation Forest on {X_train.shape[0]} samples …')
    model.fit(X_train)
    return model


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, X_test, y_test = load_data()
    model = train(X_train)

    joblib.dump(model, f'{MODELS_DIR}/iforest.pkl')

    # score_samples returns the negative mean depth; negate so higher = more anomalous
    scores = -model.score_samples(X_test)

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(y_test, scores, threshold, 'Isolation Forest',
                        out_path=f'{RESULTS_DIR}/iforest_report.txt')
    plot_results(y_test, y_pred, scores, 'Isolation Forest',
                 out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'credit_card', 'iforest_results.png'))
