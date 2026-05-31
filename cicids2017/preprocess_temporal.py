import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.join(_ROOT, 'data', 'raw', 'cicids2017')
OUT_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_cicids_temporal')

FILES = [
    'Monday-WorkingHours.csv',
    'Tuesday-WorkingHours.csv',
    'Wednesday-WorkingHours.csv',
    'Thursday-WorkingHours.csv',
    'Friday-WorkingHours.csv',
]

DROP_COLS = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Src Port']

# Each stream is one source IP's ordered flow history.
# Grouping key: Src IP only 
#
# Why Src IP streams work:
#   - The testbed has ~48 machines. Each Src IP has tens of thousands of flows.
#   - A host doing a PortScan generates 159k flows in sequence — the temporal
#     pattern of repeated short connections is exactly what the model should see.
#   - At W=16, Src IP streams cover 100% of flows for 12/14 attack classes
#     (only Heartbleed [11 flows] and SQL Injection [12 flows] fall below W).
#   - Analogous to Yahoo: one "series" per entity (host rather than sensor).
GROUP_KEY = 'Src IP'

# Window / stride settings.
# W=16 instead of Yahoo's 64 — multivariate (80 features) windows need fewer
# steps to encode a behavioral episode.  Shorter windows also keep the shapes
# of DoS bursts and PortScan sweeps within a single window
WINDOW = 16
STRIDE_TRAIN = 4
STRIDE_TEST = 1

TRAIN_DAYS = {'Monday'}

ATTEMPTED_SUFFIX = '- Attempted'
RANDOM_SEED = 42

def load_all():
    dfs = list()
    for file_name in FILES:
        df = pd.read_csv(os.path.join(DATA_DIR, file_name))
        df['Day'] = file_name.split('-')[0]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df[~df['Label'].str.endswith(ATTEMPTED_SUFFIX, na=False)].reset_index(drop=True)
    return df


def build_feature_matrix(df):
    # Keep Src IP, Timestamp, and Day as separate arrays for grouping/splitting
    # before they are dropped from the feature matrix
    src_ips = df[GROUP_KEY].values
    days = df['Day'].values
    timestamps = pd.to_datetime(df['Timestamp'], format='mixed')

    feat = df.drop(columns=DROP_COLS + ['Label', 'Day'], errors='ignore')

    feat = pd.get_dummies(feat, columns=['Protocol'], prefix='Proto', dtype=np.float32)

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feat, src_ips, days, timestamps


def make_windows(values: np.ndarray, labels: np.ndarray, stride: int):
    X, y = list(), list()
    for start in range(0, len(values) - WINDOW + 1, stride):
        X.append(values[start:start + WINDOW])
        y.append(int(labels[start:start + WINDOW].any()))
    if not X:
        return np.empty((0, WINDOW, values.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading and filtering CSVs …')
    raw = load_all()
    print(f'Total flows (excluding Attempted): {len(raw)}')

    feat, src_ips, days, timestamps = build_feature_matrix(raw)
    labels = raw['Label'].values
    feature_cols = feat.columns.tolist()
    X_all = feat.values.astype(np.float32)

    # Training rows - Monday only
    train_mask = np.isin(days, list(TRAIN_DAYS))
    print(f'Training rows (Monday): {train_mask.sum()}')

    # Drop zero-variance columns measured on the training set
    zero_var_mask = np.std(X_all[train_mask], axis=0) == 0
    if zero_var_mask.any():
        print(f'Dropping zero-variance cols: {[feature_cols[i] for i in range(len(feature_cols)) if zero_var_mask[i]]}')
        X_all = X_all[:, ~zero_var_mask]
        feature_cols = [c for c, drop in zip(feature_cols, zero_var_mask) if not drop]
    print(f'Feature count: {X_all.shape[1]}')

    # Impute NaN/Inf with training-set (Monday) medians — no leakage from test days
    print('Imputing NaN/Inf with training medians …')
    train_medians = np.nanmedian(X_all[train_mask], axis=0)
    for j in range(X_all.shape[1]):
        mask = np.isnan(X_all[:, j])
        if mask.any():
            X_all[mask, j] = train_medians[j]

    # MinMaxScaler fit on training set only -> no leakage into test
    print('Scaling ...')
    scaler = MinMaxScaler()
    scaler.fit(X_all[train_mask])
    X_all = scaler.transform(X_all).astype(np.float32)

    print('\n--- Sanity check: sample attack windows ---')
    inspect_attack_windows(src_ips, days, timestamps, labels)

    # Build per-host streams sorted by Timestamp
    # window formation is over the full sorted sub-stream for each day    
    train_windows, test_windows, test_labels_win, test_attack_types = list(), list(), list(), list()
    unique_hosts = np.unique(src_ips)

    for host in unique_hosts:
        mask = src_ips == host
        ts = timestamps[mask]
        day_host = days[mask]
        order = np.argsort(ts)

        X_host = X_all[mask][order]
        y_host = labels[mask][order]
        day_host = day_host[order]
        is_attack_host = y_host != 'BENIGN'

        # Training - monday only (benign)        
        train_mask = np.isin(day_host, list(TRAIN_DAYS))
        X_train = X_host[train_mask]
        is_att_tr = is_attack_host[train_mask]
        if len(X_train) >= WINDOW:
            Xw, yw = make_windows(X_train, is_att_tr, STRIDE_TRAIN)
            train_windows.append(Xw[yw == 0])

        # Test - other days
        # Slide over the full sub-stream; windows inherit label from any-attack rule.        
        test_mask = ~train_mask
        X_test = X_host[test_mask]
        y_test = y_host[test_mask]
        is_att_te = is_attack_host[test_mask]
        if len(X_test) >= WINDOW:
            Xw, yw = make_windows(X_test, is_att_te, STRIDE_TEST)
            attack_types = list()
            for start in range(0, len(X_test) - WINDOW + 1, STRIDE_TEST):
                window_labels = y_test[start:start + WINDOW]
                attack_labels = window_labels[window_labels != 'BENIGN']
                if len(attack_labels):
                    uniq, cnts = np.unique(attack_labels, return_counts=True)
                    attack_types.append(uniq[np.argmax(cnts)])
                else:
                    attack_types.append('BENIGN')
            attack_types = attack_types[:len(Xw)]
            test_windows.append(Xw)
            test_labels_win.append(yw)
            test_attack_types.append(np.array(attack_types))

    X_train = np.concatenate(train_windows) if train_windows else np.empty((0, WINDOW, X_all.shape[1]))
    X_test = np.concatenate(test_windows) if test_windows else np.empty((0, WINDOW, X_all.shape[1]))
    y_test = np.concatenate(test_labels_win) if test_labels_win else np.empty(0, dtype=np.int32)
    attack_types_test = np.concatenate(test_attack_types) if test_attack_types else np.empty(0)

    print(f'\nX_train shape : {X_train.shape}  (W={WINDOW}, F={X_all.shape[1]})')
    print(f'X_test shape : {X_test.shape}')
    print(f'Attack rate : {y_test.mean():.4f}')

    np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test)
    np.save(os.path.join(OUT_DIR, 'attack_types_test.npy'), attack_types_test)
    np.save(os.path.join(OUT_DIR, 'feature_names.npy'), np.array(feature_cols))
    
    print('\nAttack type breakdown in test windows:')
    unique, counts = np.unique(attack_types_test, return_counts=True)
    for lbl, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f' {lbl:<50} {cnt}')


def inspect_attack_windows(src_ips, days, timestamps, labels, n_hosts=3):
    """
    For a few hosts that have attack flows, print the first attack-containing
    test window: Verify that a) all flows in a
    window belong to the same host, b) no Monday flows leak into test windows,
    and c) the attack-type label matches the majority label.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    is_attack = labels != 'BENIGN'
    test_mask_global = ~np.isin(days, list(TRAIN_DAYS))
    attack_hosts = np.unique(src_ips[is_attack & test_mask_global])

    if not len(attack_hosts):
        print('No attack hosts found — nothing to inspect.')
        return

    chosen = rng.choice(attack_hosts, size=min(n_hosts, len(attack_hosts)), replace=False)

    for host in chosen:
        host_mask = src_ips == host
        order = np.argsort(timestamps[host_mask])
        y_host  = labels[host_mask][order]
        day_host = days[host_mask][order]

        te = ~np.isin(day_host, list(TRAIN_DAYS))
        y_te = y_host[te]
        day_te = day_host[te]

        if len(y_te) < WINDOW:
            print(f'\nHost {host}: fewer than {WINDOW} test flows — skipping.')
            continue

        for start in range(0, len(y_te) - WINDOW + 1):
            win_labels = y_te[start : start + WINDOW]
            if (win_labels != 'BENIGN').any():
                win_days = day_te[start : start + WINDOW]
                attack_labels = win_labels[win_labels != 'BENIGN']
                uniq, cnts = np.unique(attack_labels, return_counts=True)
                computed_type = uniq[np.argmax(cnts)]

                print(f'\nHost : {host}')
                print(f'Window [{start}:{start+WINDOW}] of {len(y_te)} test flows')
                print(f'Days in window : {np.unique(win_days).tolist()}')
                print(f'Flow labels : {win_labels.tolist()}')
                print(f'Computed type : {computed_type}')
                break


if __name__ == '__main__':
    main()
