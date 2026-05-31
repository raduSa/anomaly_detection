import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
import joblib

RAW_PATH = os.path.join(_ROOT, 'data', 'creditcard.csv')
OUT_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_credit_card')
TEST_SIZE = 0.2
RANDOM_SEED = 123

# Controls preprocessing applied to V1-V28 features:
#   'none'     — pass through raw PCA values unchanged
#   'standard' — zero mean, unit variance
#   'quantile' — maps each feature to a Gaussian via empirical quantiles
V_SCALING = 'none'


def load_and_split(path: str):
    df = pd.read_csv(path).drop(columns=['Time'])

    X = df.drop(columns=['Class']).values
    y = df['Class'].values

    X_normal = X[y == 0]
    X_fraud = X[y == 1]

    # only normal in train split
    X_train, X_test_normal = train_test_split(
        X_normal, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    X_test = np.vstack([X_test_normal, X_fraud])
    y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_fraud))])

    return X_train, X_test, y_test


def fit_scalers(X_train):
    if V_SCALING == 'standard':
        v_scaler = StandardScaler()
    elif V_SCALING == 'quantile':
        v_scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_SEED)
    elif V_SCALING == 'none':
        v_scaler = None
    else:
        raise ValueError(f"V_SCALING must be 'none', 'standard', or 'quantile', got '{V_SCALING}'")

    amount_scaler = RobustScaler()

    if v_scaler is not None:
        v_scaler.fit(X_train[:, :-1])
    amount_scaler.fit(X_train[:, -1:])

    return v_scaler, amount_scaler


def apply_scalers(X, v_scaler, amount_scaler):
    X_v = v_scaler.transform(X[:, :-1]) if v_scaler is not None else X[:, :-1]
    X_amount = amount_scaler.transform(X[:, -1:])
    return np.hstack([X_v, X_amount])


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    X_train, X_test, y_test = load_and_split(RAW_PATH)

    v_scaler, am_scaler = fit_scalers(X_train)
    X_train_scaled = apply_scalers(X_train, v_scaler, am_scaler)
    X_test_scaled = apply_scalers(X_test,  v_scaler, am_scaler)

    np.save(f'{OUT_DIR}/X_train_normal.npy', X_train_scaled)
    np.save(f'{OUT_DIR}/X_test.npy', X_test_scaled)
    np.save(f'{OUT_DIR}/y_test.npy', y_test)

    joblib.dump(v_scaler,  f'{OUT_DIR}/v_scaler.pkl')
    joblib.dump(am_scaler, f'{OUT_DIR}/am_scaler.pkl')

    print(f'Train samples: {X_train_scaled.shape[0]}')
    print(f'Test samples: {X_test_scaled.shape[0]}')
    print(f'Test fraud samples: {(y_test == 1).sum()}')
    print(f'Saved to {OUT_DIR}/')
