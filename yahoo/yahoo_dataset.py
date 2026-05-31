import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.graphics.tsaplots import plot_acf

YAHOO_DIR = os.path.join(_ROOT, 'data', 'ydata-labeled-time-series-anomalies-v1_0')

def load_series(benchmark: str):
    path = os.path.join(YAHOO_DIR, benchmark)
    files = sorted(f for f in glob.glob(os.path.join(path, '*.csv')) if '_all' not in f)
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df = df.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'})
        dfs.append(df[['timestamp', 'value', 'is_anomaly']])
    return dfs

def benchmark_stats(name, dfs):
    lengths = [len(d) for d in dfs]
    anom_pts = sum(d['is_anomaly'].sum() for d in dfs)
    total = sum(len(d) for d in dfs)
    return {
        'benchmark': name,
        'series': len(dfs),
        'total_pts': total,
        'min_len': min(lengths),
        'max_len': max(lengths),
        'median_len': int(np.median(lengths)),
        'anom_pts': int(anom_pts),
        'anom_pct': 100 * anom_pts / total,
    }


benchmarks = {
    'A1 (real)': load_series('A1Benchmark'),
    'A2 (synthetic)': load_series('A2Benchmark'),
    'A3 (synth+decomp)': load_series('A3Benchmark'),
    'A4 (synth+decomp)': load_series('A4Benchmark'),
}

stats_df = pd.DataFrame([benchmark_stats(k, v) for k, v in benchmarks.items()]).set_index('benchmark')
print('\nBenchmark overview')
print(stats_df.to_string())

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Yahoo S5 — Dataset Analysis', fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

for col, (name, idx) in enumerate([('A1 (real)', 0), ('A2 (synthetic)', 2), ('A4 (synth+decomp)', 4)]):
    ax = fig.add_subplot(gs[0, col])
    df = benchmarks[name][idx]
    anom = df['is_anomaly'].astype(bool)
    ax.plot(df['value'], color='steelblue', linewidth=0.7, label='Normal')
    ax.scatter(df.index[anom], df['value'][anom], color='crimson', s=18, zorder=5, label='Anomaly')
    ax.set_title(f'{name} — series {idx + 1}', fontsize=9)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value')
    ax.legend(fontsize=7)

ax = fig.add_subplot(gs[1, :2])
for name, dfs in benchmarks.items():
    rates = sorted(100 * d['is_anomaly'].mean() for d in dfs)
    ax.plot(rates, marker='.', markersize=3, linewidth=0.8, label=name)
ax.set_title('Anomaly rate per series (sorted)')
ax.set_xlabel('Series rank')
ax.set_ylabel('Anomaly %')
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 2])
ax.hist([len(d) for d in benchmarks['A1 (real)']], bins=20, color='steelblue', edgecolor='white')
ax.set_title('A1 series length distribution')
ax.set_xlabel('Length (timesteps)')
ax.set_ylabel('Count')

ax = fig.add_subplot(gs[2, 0])
normal_vals = np.concatenate([d.loc[d['is_anomaly'] == 0, 'value'].values for d in benchmarks['A1 (real)']])
anomaly_vals = np.concatenate([d.loc[d['is_anomaly'] == 1, 'value'].values for d in benchmarks['A1 (real)']])
ax.hist(normal_vals,  bins=80, alpha=0.6, color='steelblue', label='Normal',  density=True, log=True)
ax.hist(anomaly_vals, bins=80, alpha=0.6, color='crimson',   label='Anomaly', density=True, log=True)
ax.set_title('A1 value distribution — normal vs anomaly (log y)')
ax.set_xlabel('Value')
ax.set_ylabel('Density (log)')
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 1])
plot_acf(benchmarks['A1 (real)'][0]['value'].values, lags=60, ax=ax, color='steelblue', alpha=0.05)
ax.set_title('A1 series 1 — Autocorrelation (ACF)\njustifies temporal models over tabular')
ax.set_xlabel('Lag')

ax = fig.add_subplot(gs[2, 2])
counts = sorted(int(d['is_anomaly'].sum()) for d in benchmarks['A1 (real)'])
ax.bar(range(len(counts)), counts, color='steelblue')
ax.set_title('A1 anomaly count per series (sorted)')
ax.set_xlabel('Series rank')
ax.set_ylabel('Anomaly points')

out = os.path.join(_ROOT, 'data', 'graphs', 'EDA', 'yahoo', 'yahoo_analysis.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'\nPlot saved to {out}')
