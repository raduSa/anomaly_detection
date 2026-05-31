import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.join(_ROOT, 'data', 'raw', 'cicids2017')
OUT_DIR  = os.path.join(_ROOT, 'data', 'processed_data', 'processed_cicids_context')

FILES = [
    'Monday-WorkingHours.csv',
    'Tuesday-WorkingHours.csv',
    'Wednesday-WorkingHours.csv',
    'Thursday-WorkingHours.csv',
    'Friday-WorkingHours.csv',
]

DROP_COLS = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Src Port']
TRAIN_DAYS = {'Monday'}
ATTEMPTED_SUFFIX = '- Attempted'

# Wall-clock time window length (seconds) — same concept as Raskovalov et al. 2024.
# 60 s captures short scan/flood bursts while ignoring idle periods between sessions.
TIME_WINDOW = 60.0


class _HostState:    
    __slots__ = ('_dq', '_port_cnt')

    def __init__(self):
        self._dq = deque() # (ts_float, port)
        self._port_cnt = defaultdict(int)

    def expire(self, t: float) -> None:
        cutoff = t - TIME_WINDOW
        while self._dq and self._dq[0][0] < cutoff:
            _, p = self._dq.popleft()
            self._port_cnt[p] -= 1
            if self._port_cnt[p] == 0:
                del self._port_cnt[p]

    def add(self, t: float, port: int) -> None:
        self._dq.append((t, port))
        self._port_cnt[port] += 1

    @property
    def flow_count(self) -> int:
        return len(self._dq)

    @property
    def port_count(self) -> int:
        return len(self._port_cnt)

    def has_port(self, port: int) -> bool:
        return self._port_cnt.get(port, 0) > 0


def _build_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate flows in chronological order and compute 9 host-context features per flow,
    following Raskovalov et al. 2024 (Table 3, features 12-20).

    Features capture how active each endpoint has been in the last TIME_WINDOW seconds:
      - flow count (how many flows from/to this host)
      - unique port count (how many distinct ports contacted/received)
      - port-diversity rate (ports / flows)
      - new-port flag (is the current flow's port unseen in the window?)

    These are the signals that distinguish DDoS (high flow count) and PortScan (high
    port diversity) from benign traffic without requiring any attack labels.

    df must be sorted by ts_float and contain: ts_float, Src IP, Dst IP, Src Port, Dst Port.
    """
    src_states: dict = defaultdict(_HostState)
    dst_states: dict = defaultdict(_HostState)

    EPS = 1e-4
    rows = list()

    for ts, sip, dip, sport, dport in zip(
        df['ts_float'].values,
        df['Src IP'].values,
        df['Dst IP'].values,
        df['Src Port'].values,
        df['Dst Port'].values,
    ):
        ss = src_states[sip]
        ds = dst_states[dip]

        ss.expire(ts)
        ds.expire(ts)

        sfc = ss.flow_count
        dfc = ds.flow_count
        spc = ss.port_count
        dpc = ds.port_count

        src_new = 0.0 if ss.has_port(dport) else 1.0
        dst_new = 0.0 if ds.has_port(sport) else 1.0

        rows.append((
            0.5 * (src_new + dst_new), # ctx_new_port
            float(max(sfc, dfc)), # ctx_max_flow_cnt
            float(min(sfc, dfc)), # ctx_min_flow_cnt
            float(max(spc, dpc)), # ctx_max_port_cnt
            float(min(spc, dpc)), # ctx_min_port_cnt
            float(abs(sfc - dfc)), # ctx_abs_flow_diff
            float(abs(spc - dpc)), # ctx_abs_port_diff
            max(spc / (sfc + EPS), dpc / (dfc + EPS)), # ctx_max_port_rate
            min(spc / (sfc + EPS), dpc / (dfc + EPS)), # ctx_min_port_rate
        ))

        # Add after computing — context reflects the host's prior activity
        ss.add(ts, dport)
        ds.add(ts, sport)

    ctx_cols = [
        'ctx_new_port',
        'ctx_max_flow_cnt', 'ctx_min_flow_cnt',
        'ctx_max_port_cnt', 'ctx_min_port_cnt',
        'ctx_abs_flow_diff', 'ctx_abs_port_diff',
        'ctx_max_port_rate', 'ctx_min_port_rate',
    ]
    return pd.DataFrame(rows, columns=ctx_cols, index=df.index)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading CSVs …')
    dfs = list()
    for file_name in FILES:
        d = pd.read_csv(os.path.join(DATA_DIR, file_name))
        d['Day'] = file_name.split('-')[0]
        dfs.append(d)
    raw = pd.concat(dfs, ignore_index=True)

    raw = raw[~raw['Label'].str.endswith(ATTEMPTED_SUFFIX, na=False)].reset_index(drop=True)
    print(f'Total flows (excluding Attempted): {len(raw)}')

    # Parse timestamps to float seconds for window arithmetic
    ts_parsed = pd.to_datetime(raw['Timestamp'], format='mixed')
    raw['ts_float'] = ts_parsed.astype(np.int64) / 1e9

    # Global chronological sort — required for correct window state
    order = np.argsort(raw['ts_float'].values, kind='stable')
    raw = raw.iloc[order].reset_index(drop=True)

    print(f'Computing host-context features (T={TIME_WINDOW}s) …')
    ctx_df = _build_context(raw)

    # Standard feature matrix — identical pipeline to preprocess.py
    feat = raw.drop(columns=DROP_COLS + ['Day', 'Label', 'ts_float'], errors='ignore')
    feat = pd.get_dummies(feat, columns=['Protocol'], prefix='Proto', dtype=np.float32)
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Append context columns
    feat = pd.concat([feat, ctx_df], axis=1)

    labels = raw['Label'].values
    days = raw['Day'].values

    is_benign = labels == 'BENIGN'
    is_attack = ~is_benign & ~np.array([str(l).endswith(ATTEMPTED_SUFFIX) for l in labels])
    train_mask = is_benign & np.isin(days, list(TRAIN_DAYS))
    benign_test_mask = is_benign & ~np.isin(days, list(TRAIN_DAYS))

    feature_cols = feat.columns.tolist()
    print(f'Feature count before zero-variance check: {len(feature_cols)}')

    X_all = feat.values.astype(np.float32)

    # Drop zero-variance columns (measured on Monday training set only)
    zero_var_mask = np.std(X_all[train_mask], axis=0) == 0
    if zero_var_mask.any():
        drop_names = [feature_cols[i] for i in range(len(feature_cols)) if zero_var_mask[i]]
        print(f'Dropping zero-variance cols: {drop_names}')
        X_all = X_all[:, ~zero_var_mask]
        feature_cols = [c for c, drop in zip(feature_cols, zero_var_mask) if not drop]
    print(f'Feature count: {len(feature_cols)}')

    # Impute NaN with per-column training median — no leakage
    print('Imputing NaN with training medians …')
    train_medians = np.nanmedian(X_all[train_mask], axis=0)
    for j in range(X_all.shape[1]):
        m = np.isnan(X_all[:, j])
        if m.any():
            X_all[m, j] = train_medians[j]

    # MinMaxScaler fit on training set only
    print('Scaling ...')
    scaler = MinMaxScaler()
    scaler.fit(X_all[train_mask])
    X_all = scaler.transform(X_all).astype(np.float32)

    X_train = X_all[train_mask]
    X_test = np.concatenate([X_all[benign_test_mask], X_all[is_attack]], axis=0)
    y_test = np.concatenate([
        np.zeros(benign_test_mask.sum(), dtype=np.int32),
        np.ones(is_attack.sum(),         dtype=np.int32),
    ])
    attack_types_test = np.concatenate([
        np.array(['BENIGN'] * int(benign_test_mask.sum())),
        labels[is_attack],
    ])

    print(f'\nX_train : {X_train.shape}')
    print(f'X_test  : {X_test.shape}')
    print(f'Attack rate : {y_test.mean():.4f}')

    np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test)
    np.save(os.path.join(OUT_DIR, 'attack_types_test.npy'), attack_types_test)
    np.save(os.path.join(OUT_DIR, 'feature_names.npy'), np.array(feature_cols))

    print('\nContext features appended:')
    for c in feature_cols:
        if c.startswith('ctx_'):
            print(f'{c}')

    print('\nAttack type breakdown in test set:')
    unique, counts = np.unique(attack_types_test, return_counts=True)
    for lbl, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f'{lbl:<50} {cnt}')


if __name__ == '__main__':
    main()
