import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

DATA_DIR = os.path.join(_ROOT, 'data', 'raw', 'cicids2017')

FILES = [
    'Monday-WorkingHours.csv',
    'Tuesday-WorkingHours.csv',
    'Wednesday-WorkingHours.csv',
    'Thursday-WorkingHours.csv',
    'Friday-WorkingHours.csv',
]

ATTEMPTED_SUFFIX = '- Attempted'

ATTACK_COLORS = {
    'BENIGN': 'steelblue',
    'PortScan': '#e377c2',
    'DoS Hulk': '#ff7f0e',
    'DDoS': '#d62728',
    'DoS GoldenEye': '#9467bd',
    'DoS slowloris': '#8c564b',
    'DoS Slowhttptest': '#bcbd22',
    'FTP-Patator': '#e6ab02',
    'SSH-Patator': '#7b3294',
    'Bot': '#2ca02c',
    'Web Attack - Brute Force': '#7f7f7f',
    'Infiltration': '#fc8d59',
    'Web Attack - XSS': '#ffbb78',
    'Web Attack - Sql Injection':'#98df8a',
    'Heartbleed': '#ff9896',
}

def load_all(include_attempted=False):
    dfs = list()
    for file_name in FILES:
        df = pd.read_csv(os.path.join(DATA_DIR, file_name))
        df['Day'] = file_name.split('-')[0]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    if not include_attempted:
        df = df[~df['Label'].str.endswith(ATTEMPTED_SUFFIX, na=False)].reset_index(drop=True)
    return df


def print_summary(df):
    print('Dataset Summary:')
    print(f'Total flows (excluding Attempted): {len(df)}')
    print()

    print('Flows per day:')
    for day, grp in df.groupby('Day'):
        n_att = (grp['Label'] != 'BENIGN').sum()
        print(f'{day} {len(grp)} attacks: {n_att}  ({100*n_att/len(grp)}%)')
    print()

    print('Label distribution:')
    vc = df['Label'].value_counts()
    for label, cnt in vc.items():
        print(f'{label} {cnt}  ({100*cnt/len(df)}%)')
    print()

    num_cols = df.select_dtypes(include=[np.number]).columns.difference(['Src Port', 'Dst Port'])
    inf_counts = {c: np.isinf(df[c]).sum() for c in num_cols if df[c].dtype == float}
    nan_counts = df[num_cols].isna().sum()
    print('Columns with Inf values:')
    for c, n in sorted(inf_counts.items(), key=lambda x: -x[1]):
        if n > 0:
            print(f'{c} {n}')
    print('Columns with NaN values:')
    for c in nan_counts[nan_counts > 0].index:
        print(f'{c} {nan_counts[c]}')
    print()

    benign = df[df['Label'] == 'BENIGN']
    num_cols2 = df.select_dtypes(include=[np.number]).columns
    zero_var = [c for c in num_cols2 if benign[c].nunique() <= 1]
    print(f'Zero-variance cols on BENIGN: {zero_var if zero_var else 'none'}')
    print(f'\nUnique Src IPs: {df['Src IP'].nunique()}')
    print(f'Unique Dst IPs: {df['Dst IP'].nunique()}')
    print(f'Protocol values: {sorted(df["Protocol"].unique())}')


# All-class stacked bar

def plot_temporal_all(df, out_path):
    fig, ax = plt.subplots(figsize=(18, 6))
    fig.suptitle('CIC-IDS2017 — Flow Volume Over Time (All Classes)', fontsize=13, fontweight='bold')

    df_copy = df.copy()
    df_copy['ts'] = pd.to_datetime(df_copy['Timestamp'], format='mixed', errors='coerce')
    df_copy = df_copy.dropna(subset=['ts'])
    df_copy['hour'] = df_copy['ts'].dt.floor('30min')

    # plot order - BENIGN first (largest, base of stack), then attacks by total count descending
    label_order = ['BENIGN'] + [
        label for label in df_copy['Label'].value_counts().index if label != 'BENIGN'
    ]
    pivot = df_copy.groupby(['hour', 'Label']).size().unstack(fill_value=0)
    # Ensure all labels present as columns even if count is 0 in some buckets
    for label in label_order:
        if label not in pivot.columns:
            pivot[label] = 0
    pivot = pivot[label_order]

    bottom = np.zeros(len(pivot))
    for label in label_order:
        vals = pivot[label].values
        ax.bar(range(len(pivot)), vals, bottom=bottom,
               label=label, color=ATTACK_COLORS.get(label, 'grey'), width=1.0, alpha=0.85)
        bottom += vals

    ax.set_xticks(range(0, len(pivot), 4))
    ax.set_xticklabels([str(t)[:13] for t in pivot.index[::4]], rotation=40, ha='right', fontsize=7)
    ax.set_ylabel('Flows per 30 min')
    ax.set_title('Flow volume over time (stacked by class, all classes)')
    ax.legend(fontsize=7, ncol=4, loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()


# Attack-only line traces

def plot_temporal_attacks(df, out_path):
    fig, ax = plt.subplots(figsize=(18, 6))
    fig.suptitle('CIC-IDS2017 — Attack Classes Over Time', fontsize=13, fontweight='bold')

    df_copy = df.copy()
    df_copy['ts'] = pd.to_datetime(df_copy['Timestamp'], format='mixed', errors='coerce')
    df_copy = df_copy.dropna(subset=['ts'])
    df_copy['hour'] = df_copy['ts'].dt.floor('30min')

    attack_df = df_copy[df_copy['Label'] != 'BENIGN']

    # Shared integer x-axis over all 30-min buckets in the attack data
    all_buckets = sorted(attack_df['hour'].unique())
    bucket_to_idx = {b: i for i, b in enumerate(all_buckets)}

    for label in attack_df['Label'].unique():
        sub = attack_df[attack_df['Label'] == label].groupby('hour').size()
        x = [bucket_to_idx[b] for b in sub.index]
        ax.plot(x, sub.values, label=label, color=ATTACK_COLORS.get(label, 'grey'),
                linewidth=1.2, marker='.')

    ax.set_xticks(range(0, len(all_buckets), 4))
    ax.set_xticklabels([str(t)[:13] for t in all_buckets[::4]], rotation=40, ha='right', fontsize=7)
    ax.set_ylabel('Attack flows per 30 min')
    ax.set_title('Attack classes over time (individual traces)')
    ax.legend(fontsize=7, ncol=3, loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()


# Per-class feature profiles

def plot_feature_profiles(df, out_path):
    # Six features chosen to reveal different attack characters
    features = [
        ('Flow Duration', 'Flow duration (µs)', True),
        ('Total Fwd Packet', 'Fwd packet count', True),
        ('Flow Bytes/s', 'Flow bytes/s', True),
        ('SYN Flag Count', 'SYN flag count', False),
        ('Fwd Packet Length Mean', 'Fwd pkt length mean (B)', False),
        ('Flow IAT Mean', 'Inter-arrival mean (µs)', True),
    ]
    classes = ['BENIGN', 'PortScan', 'DDoS',
               'FTP-Patator', 'SSH-Patator', 'Bot']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('CIC-IDS2017 — Feature Distributions by Class', fontsize=13, fontweight='bold')

    for ax, (feat, xlabel, use_log) in zip(axes.flat, features):
        plotted = []
        for label in classes:
            sub = df[df['Label'] == label][feat].replace([np.inf, -np.inf], np.nan).dropna()
            if use_log:
                sub = sub[sub > 0]
            if len(sub) == 0:
                continue
            ax.hist(sub, bins=60, alpha=0.45, density=True,
                    label=label, color=ATTACK_COLORS.get(label, 'grey'),
                    log=use_log)
            plotted.append(sub)
        if not use_log and plotted:
            combined = np.concatenate([s.values for s in plotted])
            ax.set_xlim(*np.percentile(combined, [1, 99]))
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        if use_log:
            ax.set_xscale('log')
        ax.set_title(feat)
        ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()


# Figure 4: Correlation and feature variance

def plot_feature_structure(df, out_path):
    benign = df[df['Label'] == 'BENIGN']
    attack = df[df['Label'] != 'BENIGN']
    drop = ['Src Port', 'Dst Port', 'Day']
    num_cols = benign.select_dtypes(include=[np.number]).columns.difference(drop).tolist()

    benign = benign[num_cols].replace([np.inf, -np.inf], np.nan)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('CIC-IDS2017 — Feature Structure (BENIGN rows)', fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Correlation heatmap
    ax = fig.add_subplot(gs[0, 0])
    corr = benign.corr().values
    n = len(corr)
    
    corr_masked = np.where(np.triu(np.ones((n, n), dtype=bool), k=1), np.nan, corr)
    im = ax.imshow(corr_masked, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto',
                   interpolation='nearest')
    
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_xticks(range(n))
    ax.set_xticklabels(benign.columns, rotation=90, fontsize=4.5, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(benign.columns, fontsize=4.5)
    ax.set_title(f'Feature correlation (BENIGN, {n}x{n}, lower triangle)')

    # Top-20 features by standardised mean separation between BENIGN and all attacks
    ax = fig.add_subplot(gs[0, 1])
    ben_means = benign.mean()
    att_means = attack[num_cols].replace([np.inf, -np.inf], np.nan).mean()
    ben_std = benign.std().replace(0, 1e-9)
    signed_sep = (att_means - ben_means) / ben_std
    separation = signed_sep.abs().sort_values(ascending=False)
    top20 = separation.head(20)
    bar_colors = ['crimson' if signed_sep[c] > 0 else 'steelblue' for c in top20.index]
    ax.barh(range(len(top20)), top20.values, color=bar_colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.6)
    ax.set_xlabel('|Standardised mean difference|  (red = attacks higher, blue = attacks lower)')
    ax.set_title('Top 20 discriminating features (attack vs BENIGN)')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    print('Loading dataset ...')
    df = load_all(include_attempted=False)

    print_summary(df)

    out = os.path.join(_ROOT, 'data', 'graphs', 'EDA', 'cicids')
    plot_temporal_all(df, os.path.join(out, 'cicids_temporal_all.png'))
    plot_temporal_attacks(df, os.path.join(out, 'cicids_temporal_attacks.png'))
    plot_feature_profiles(df, os.path.join(out, 'cicids_feature_profiles.png'))
    plot_feature_structure(df, os.path.join(out, 'cicids_feature_structure.png'))
