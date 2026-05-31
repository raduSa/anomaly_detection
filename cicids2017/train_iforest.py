import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from sklearn.ensemble import IsolationForest
from plot_results import sweep_threshold, evaluate, plot_results, evaluate_per_class

SKIP_LABELS = {'DoS Hulk'}

DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_cicids')
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_cicids')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_cicids')

N_ESTIMATORS = 150
MAX_SAMPLES = 300000
CONTAMINATION = 'auto'
N_JOBS = -1
RANDOM_STATE = 123


def load_data():
    X_train = np.load(f'{DATA_DIR}/X_train.npy')
    X_test = np.load(f'{DATA_DIR}/X_test.npy')
    y_test = np.load(f'{DATA_DIR}/y_test.npy')
    return X_train, X_test, y_test


def train(X_train):
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train)
    return model


def anomaly_scores(model, X):
    # decision_function returns higher values for normal samples.
    # Negate so that higher score = more anomalous (matches evaluate / sweep_threshold).
    return -model.decision_function(X)


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, X_test, y_test = load_data()
    print(f'Training IsolationForest on {X_train.shape[0]} samples, {X_train.shape[1]} features …')
    model = train(X_train)

    import joblib
    joblib.dump(model, f'{MODELS_DIR}/iforest.pkl')

    scores = anomaly_scores(model, X_test)

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(y_test, scores, threshold, 'IsolationForest (CIC-IDS2017)',
                      out_path=f'{RESULTS_DIR}/iforest_report.txt')
    plot_results(y_test, y_pred, scores, 'IsolationForest (CIC-IDS2017)',
                 out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'cicids', 'iforest_cicids_results.png'))

    attack_types = np.load(f'{DATA_DIR}/attack_types_test.npy', allow_pickle=True)
    evaluate_per_class(attack_types, scores, skip_labels=SKIP_LABELS,
                       out_path=f'{RESULTS_DIR}/iforest_perclass_ap.txt')
