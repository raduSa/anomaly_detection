import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

YAHOO_DIR = os.path.join(_ROOT, 'data', 'ydata-labeled-time-series-anomalies-v1_0')

BENCHMARK = 'A4Benchmark'
OUT_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_yahoo', BENCHMARK)

WINDOW = 64
STRIDE_TRAIN = 4
STRIDE_TEST = 1
TRAIN_FRAC = 0.70


def load_series(benchmark: str) -> list[pd.DataFrame]:
    path = os.path.join(YAHOO_DIR, benchmark)
    files = sorted(f for f in glob.glob(os.path.join(path, '*.csv')) if '_all' not in f)
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df = df.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'})
        dfs.append(df[['timestamp', 'value', 'is_anomaly']].reset_index(drop=True))
    return dfs


def make_windows(values: np.ndarray, labels: np.ndarray, stride: int):
    X, y = [], []
    for start in range(0, len(values) - WINDOW + 1, stride):
        X.append(values[start : start + WINDOW])
        y.append(int(labels[start : start + WINDOW].any()))
    return (
        np.array(X, dtype=np.float32)[:, :, np.newaxis],
        np.array(y, dtype=np.int32),
    )


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    dfs = load_series(BENCHMARK)
    print(f'Loaded {len(dfs)} series from {BENCHMARK}')

    train_X_all, test_X_all, test_y_all = [], [], []
    scalers = []

    for df in dfs:
        values = df['value'].values.astype(np.float64)
        labels = df['is_anomaly'].values.astype(bool)

        split = int(len(values) * TRAIN_FRAC)
        train_vals, train_labels = values[:split], labels[:split]
        test_vals, test_labels = values[split:], labels[split:]

        scaler = MinMaxScaler()
        scaler.fit(train_vals[~train_labels].reshape(-1, 1))
        scalers.append(scaler)

        train_scaled = scaler.transform(train_vals.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_vals.reshape(-1, 1)).flatten()

        X_tr, y_tr = make_windows(train_scaled, train_labels, STRIDE_TRAIN)
        train_X_all.append(X_tr[y_tr == 0])

        X_te, y_te = make_windows(test_scaled, test_labels, STRIDE_TEST)
        test_X_all.append(X_te)
        test_y_all.append(y_te)

    X_train = np.concatenate(train_X_all)
    X_test = np.concatenate(test_X_all)
    y_test = np.concatenate(test_y_all)

    np.save(f'{OUT_DIR}/X_train_normal.npy', X_train)
    np.save(f'{OUT_DIR}/X_test.npy', X_test)
    np.save(f'{OUT_DIR}/y_test.npy', y_test)
    joblib.dump(scalers, f'{OUT_DIR}/scalers.pkl')

    print(f'Window: {WINDOW}  stride_train={STRIDE_TRAIN}  stride_test={STRIDE_TEST}')
    print(f'Train windows: {X_train.shape[0]}  shape={X_train.shape}')
    print(f'Test  windows: {X_test.shape[0]}   shape={X_test.shape}')
    print(f'Anomalous test windows: {y_test.sum()} ({100 * y_test.mean():.2f}%)')
    print(f'Saved to {OUT_DIR}/')
