import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.join(_ROOT, 'data', 'raw', 'cicids2017')
OUT_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_cicids')

FILES = [
    'Monday-WorkingHours.csv',
    'Tuesday-WorkingHours.csv',
    'Wednesday-WorkingHours.csv',
    'Thursday-WorkingHours.csv',
    'Friday-WorkingHours.csv',
]

# Drop columns that encode identity, not relevant traffic behaviour
# keeping these causes shortcut learning -> Engelen et al. 2021
DROP_COLS = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Src Port']

# keep train split same as for the temporal models
TRAIN_DAYS = {'Monday'}

ATTEMPTED_SUFFIX = '- Attempted'

def load_all():
    dfs = list()
    for file_name in FILES:
        df = pd.read_csv(os.path.join(DATA_DIR, file_name))
        df['Day'] = file_name.split('-')[0]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_features(df):
    df = df.drop(columns=DROP_COLS + ['Day', 'Label'])

    # One-hot encode protocol feature    
    df = pd.get_dummies(df, columns=['Protocol'], prefix='Proto', dtype=np.float32)

    # handle inf uniformly
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading CSVs ...')
    raw = load_all()
    print(f'Total rows: {len(raw)}\n')

    labels = raw['Label'].copy()
    day_col = raw['Day'].copy()
    df = build_features(raw)

    is_benign = labels == 'BENIGN'
    is_attempted = labels.str.endswith(ATTEMPTED_SUFFIX)
    is_attack = ~is_benign & ~is_attempted

    # Train set - monday (benign)
    train_mask = is_benign & day_col.isin(TRAIN_DAYS)
    train_df = df[train_mask].copy()
    print(f'Training rows (Monday): {len(train_df)}')

    # Test set - everything else apart from flows marked 'ATTEMPTED'
    benign_test_mask = is_benign & ~day_col.isin(TRAIN_DAYS)
    test_df = pd.concat([
        df[benign_test_mask],
        df[is_attack],
    ], ignore_index=True)
    y_test = np.concatenate([
        np.zeros(benign_test_mask.sum(), dtype=np.int32),
        np.ones(is_attack.sum(), dtype=np.int32),
    ])
    attack_types_test = np.concatenate([
        np.array(['BENIGN'] * benign_test_mask.sum()),
        labels[is_attack].values,
    ])
    print(f'Test rows:     {len(test_df)} (BENIGN: {benign_test_mask.sum()}, attacks: {is_attack.sum()})')

    feature_cols = train_df.columns.tolist()
    print(f'Feature count before zero-variance check: {len(feature_cols)}')

    # Drop zero-variance columns measured on the training set
    X_train_raw = train_df.values.astype(np.float32)
    X_test_raw = test_df.values.astype(np.float32)

    zero_var_mask = np.std(X_train_raw, axis=0) == 0
    if zero_var_mask.any():
        zero_var_cols = [feature_cols[i] for i in range(len(feature_cols)) if zero_var_mask[i]]
        print(f'Dropping zero-variance cols: {zero_var_cols}')
        X_train_raw = X_train_raw[:, ~zero_var_mask]
        X_test_raw = X_test_raw[:, ~zero_var_mask]
        feature_cols = [c for c, drop in zip(feature_cols, zero_var_mask) if not drop]
    print(f'Feature count: {len(feature_cols)}')

    # Impute NaN with per-column median of the training set
    # Affected: Flow IAT Mean/Std/Max/Min (zero-duration flows have undefined IAT)
    # and Flow Bytes/s, Flow Packets/s (after Inf->NaN replacement above)
    print('Imputing NaN with training medians ...')
    train_medians = np.nanmedian(X_train_raw, axis=0)
    for i in range(X_train_raw.shape[1]):
        m_tr = np.isnan(X_train_raw[:, i])
        m_te = np.isnan(X_test_raw[:, i])
        if m_tr.any():
            X_train_raw[m_tr, i] = train_medians[i]
        if m_te.any():
            X_test_raw[m_te, i] = train_medians[i]

    # MinMaxScaler fit on training set only -> no leakage into test
    print('Scaling ...')
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    print(f'\nX_train shape : {X_train.shape}')
    print(f'X_test  shape : {X_test.shape}')
    print(f'Attack rate : {y_test.mean()}')

    np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test)
    np.save(os.path.join(OUT_DIR, 'attack_types_test.npy'), attack_types_test)
    np.save(os.path.join(OUT_DIR, 'feature_names.npy'), np.array(feature_cols))

    print('\nAttack type breakdown in test set:')
    unique, counts = np.unique(attack_types_test, return_counts=True)
    for lbl, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f' {lbl:<50} {cnt}')


if __name__ == '__main__':
    main()
