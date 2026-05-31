import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import joblib
from sklearn.svm import OneClassSVM
from plot_results import sweep_threshold, evaluate, plot_results, evaluate_per_class

SKIP_LABELS = {'DoS Hulk'}

DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_cicids_context')
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_cicids')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_cicids')

NU = 0.01
KERNEL = 'rbf'
GAMMA = 'scale'
SUBSAMPLE = 50_000
RANDOM_STATE = 42


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    return X_train, X_test, y_test


def train(X_train):
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_train), size=min(SUBSAMPLE, len(X_train)), replace=False)
    X_sub = X_train[idx]
    print(f'  Subsampled to {len(X_sub)} rows for OC-SVM training')
    model = OneClassSVM(nu=NU, kernel=KERNEL, gamma=GAMMA)
    model.fit(X_sub)
    return model


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, X_test, y_test = load_data()
    print('Training OC-SVM+Context on subsampled BENIGN flows …')
    model = train(X_train)

    joblib.dump(model, os.path.join(MODELS_DIR, 'ocsvm_context.pkl'))

    scores = -model.decision_function(X_test)

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(y_test, scores, threshold, 'OC-SVM+Context (CIC-IDS2017)',
                        out_path=os.path.join(RESULTS_DIR, 'ocsvm_context_report.txt'))
    plot_results(y_test, y_pred, scores, 'OC-SVM+Context (CIC-IDS2017)',
                 out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'cicids', 'ocsvm_context_results.png'))

    attack_types = np.load(os.path.join(DATA_DIR, 'attack_types_test.npy'), allow_pickle=True)
    evaluate_per_class(attack_types, scores, skip_labels=SKIP_LABELS,
                       out_path=os.path.join(RESULTS_DIR, 'ocsvm_context_perclass_ap.txt'))
