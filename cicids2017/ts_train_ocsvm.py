import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from sklearn.svm import OneClassSVM
import joblib

from plot_results import sweep_threshold, evaluate, plot_results, evaluate_per_class

DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_cicids_temporal')
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_cicids')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_cicids')

NU = 0.005
GAMMA = 'scale'
SUBSAMPLE = 20_000   # 1232-dim windows make full-set SVM infeasible
RANDOM_SEED = 123
SKIP_LABELS = {'DoS Hulk'}


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    return X_train.reshape(len(X_train), -1), X_test.reshape(len(X_test), -1), y_test


def train(X_train):
    if len(X_train) > SUBSAMPLE:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(X_train), SUBSAMPLE, replace=False)
        X_train = X_train[idx]
    model = OneClassSVM(kernel='rbf', nu=NU, gamma=GAMMA)
    print(f'Training OC-SVM on {X_train.shape[0]} windows of {X_train.shape[1]} features …')
    model.fit(X_train)
    return model


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, X_test, y_test = load_data()
    model = train(X_train)

    joblib.dump(model, os.path.join(MODELS_DIR, 'ocsvm_cicids_temporal.pkl'))

    scores = -model.decision_function(X_test)

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(y_test, scores, threshold, 'OC-SVM (CIC-IDS2017 temporal)',
                      out_path=os.path.join(RESULTS_DIR, 'ocsvm_cicids_temporal_report.txt'))
    plot_results(y_test, y_pred, scores, 'OC-SVM (CIC-IDS2017 temporal)',
                 out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'cicids', 'ocsvm_cicids_temporal_results.png'))

    attack_types = np.load(os.path.join(DATA_DIR, 'attack_types_test.npy'), allow_pickle=True)
    evaluate_per_class(attack_types, scores, skip_labels=SKIP_LABELS,
                       out_path=os.path.join(RESULTS_DIR, 'ocsvm_cicids_temporal_perclass_ap.txt'))
