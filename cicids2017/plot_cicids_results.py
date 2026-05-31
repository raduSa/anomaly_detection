import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_cicids')
OUT_DIR = os.path.join(_ROOT, 'data', 'graphs', 'EDA', 'cicids')

# (display name, report stem, perclass stem)  — order determines row order in tables
MODELS = [
    ('Flat IForest',          'iforest_report',                                'iforest_perclass_ap'),
    ('Flat OC-SVM',           'ocsvm_report',                                  'ocsvm_perclass_ap'),
    ('Flat AE',               'autoencoder_report',                            'autoencoder_perclass_ap'),
    ('IForest+Ctx',           'iforest_context_report',                        'iforest_context_perclass_ap'),
    ('OC-SVM+Ctx',            'ocsvm_context_report',                          'ocsvm_context_perclass_ap'),
    ('AE+Ctx',                'autoencoder_context_report',                    'autoencoder_context_perclass_ap'),
    ('Temporal IForest',      'iforest_cicids_temporal_report',                'iforest_cicids_temporal_perclass_ap'),
    ('Temporal OC-SVM',       'ocsvm_cicids_temporal_report',                  'ocsvm_cicids_temporal_perclass_ap'),
    ('Temporal AE',           'autoencoder_cicids_temporal_report',            'autoencoder_cicids_temporal_perclass_ap'),
    ('LSTM AE',               'lstm_ae_cicids_temporal_report',                'lstm_ae_cicids_temporal_perclass_ap'),
    ('Predictive LSTM',       'lstm_cicids_temporal_report',                   'lstm_cicids_temporal_perclass_ap'),
    ('Bottleneck Transformer','transformer_bottleneck_cicids_temporal_report', 'transformer_bottleneck_cicids_temporal_perclass_ap'),
    ('Causal Transformer',    'transformer_causal_cicids_temporal_report',     'transformer_causal_cicids_temporal_perclass_ap'),
]

SHORT_NAMES = {
    'Flat IForest':           'Flat IF',
    'Flat OC-SVM':            'Flat SVM',
    'Flat AE':                'Flat AE',
    'IForest+Ctx':            'IF+Ctx',
    'OC-SVM+Ctx':             'SVM+Ctx',
    'AE+Ctx':                 'AE+Ctx',
    'Temporal IForest':       'Tmp IF',
    'Temporal OC-SVM':        'Tmp SVM',
    'Temporal AE':            'Tmp AE',
    'LSTM AE':                'LSTM AE',
    'Predictive LSTM':        'Pred LSTM',
    'Bottleneck Transformer': 'Bottnck',
    'Causal Transformer':     'Causal',
}

METRIC_COLS = ['ROC-AUC', 'Avg Precision', 'Balanced Acc', 'F1 (attack)']
GREEN_MAP = LinearSegmentedColormap.from_list('wg', ['#ffffff', '#1a7a3c'])


# parsers

def parse_report(stem):
    path = os.path.join(RESULTS_DIR, stem + '.txt')
    with open(path) as f:
        text = f.read()
    row = {}
    for col, pat in zip(METRIC_COLS, [
        r'ROC-AUC\s*:\s*([\d.]+)',
        r'Avg Precision\s*:\s*([\d.]+)',
        r'Balanced Accuracy\s*:\s*([\d.]+)',
        r'F1 \(fraud\)\s*:\s*([\d.]+)',
    ]):
        m = re.search(pat, text)
        row[col] = float(m.group(1)) if m else float('nan')
    return row


def parse_perclass(stem):
    path = os.path.join(RESULTS_DIR, stem + '.txt')
    out = {}
    with open(path) as f:
        for line in f:
            m = re.search(r'^\s+(.+?)\s+AP=([\d.]+)\s+n=\d+', line)
            if m:
                out[m.group(1).strip()] = float(m.group(2))
    return out


# data loading

def load_all():
    overall_rows, perclass_rows = {}, {}
    for name, rep_stem, pc_stem in MODELS:
        rep_path = os.path.join(RESULTS_DIR, rep_stem + '.txt')
        pc_path  = os.path.join(RESULTS_DIR, pc_stem  + '.txt')
        if os.path.exists(rep_path):
            overall_rows[name] = parse_report(rep_stem)
        if os.path.exists(pc_path):
            perclass_rows[name] = parse_perclass(pc_stem)

    overall_df = pd.DataFrame(overall_rows).T[METRIC_COLS]
    perclass_df = pd.DataFrame(perclass_rows).T.fillna(0.0)
    # keep only models present in MODELS order
    model_order = [n for n, *_ in MODELS if n in overall_df.index]
    return overall_df.loc[model_order], perclass_df.loc[model_order]


# plot helpers

def _render_table(ax, df, fmt='.4f', highlight_col=True,
                  col_header_color='#2c5f8a', row_header_color='#3a3a3a',
                  best_color='#d4edda', best_edge='#1a7a3c'):
    """
    Render df as a coloured matplotlib table.
    highlight_col=True: highlight max per column (overall metrics).
    highlight_col=False: highlight max per row (per-class AP).
    """
    nrows, ncols = df.shape
    vals = df.values.astype(float)
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows + 1)
    ax.axis('off')

    col_w = 1.0
    row_h = 1.0

    # column headers
    for j, col in enumerate(df.columns):
        ax.add_patch(plt.Rectangle((j, nrows), col_w, row_h,
                    color=col_header_color, zorder=1))
        ax.text(j + 0.5, nrows + 0.5, col,
                ha='center', va='center', fontsize=8,
                color='white', fontweight='bold', zorder=2)

    # row headers
    for i, idx in enumerate(df.index):
        row_y = nrows - i - 1
        ax.add_patch(plt.Rectangle((-1.8, row_y), 1.8, row_h,
                    color=row_header_color, zorder=1))
        ax.text(-0.9, row_y + 0.5, idx,
                ha='center', va='center', fontsize=7.5,
                color='white', fontweight='bold', zorder=2)

    # best-value mask
    if highlight_col:
        best_mask = (vals == np.nanmax(vals, axis=0, keepdims=True))
    else:
        best_mask = (vals == np.nanmax(vals, axis=1, keepdims=True))

    # cells
    for i in range(nrows):
        for j in range(ncols):
            row_y = nrows - i - 1
            v = vals[i, j]
            is_best = best_mask[i, j]
            bg = best_color if is_best else '#f9f9f9' if i % 2 == 0 else 'white'
            ec = best_edge  if is_best else '#cccccc'
            lw = 1.5        if is_best else 0.5
            ax.add_patch(plt.Rectangle((j, row_y), col_w, row_h,
                        color=bg, ec=ec, lw=lw, zorder=1))
            ax.text(j + 0.5, row_y + 0.5, f'{v:{fmt}}',
                    ha='center', va='center', fontsize=8,
                    fontweight='bold' if is_best else 'normal', zorder=2)

    ax.set_xlim(-1.8, ncols)
    ax.set_ylim(0, nrows + 1)


# figure 1 : overall metrics

def plot_overall(overall_df, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('#f0f4f8')
    ax.set_facecolor('#f0f4f8')

    _render_table(ax, overall_df, fmt='.4f', highlight_col=True)

    ax.set_title('CIC-IDS2017 — Overall Model Metrics\n'
                 '(green = best per metric)',
                 fontsize=12, fontweight='bold', pad=12)

    legend_patch = mpatches.Patch(facecolor='#d4edda', edgecolor='#1a7a3c',
                                   linewidth=1.5, label='Best per metric')
    ax.legend(handles=[legend_patch], loc='lower right', fontsize=8,
              framealpha=0.9, bbox_to_anchor=(1.0, -0.02))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# figure 2 : per-class AP heatmap

def plot_perclass(perclass_df, out_path):
    # transpose: rows = attack class, cols = model
    df = perclass_df.T
    df.columns = [SHORT_NAMES.get(c, c) for c in df.columns]
    df = df.sort_index() # alphabetical attack order

    nrows, ncols = df.shape
    vals = df.values.astype(float)

    fig_h = max(6, nrows * 0.55 + 1.5)
    fig_w = max(10, ncols * 1.1 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#f0f4f8')

    # heatmap background
    im = ax.imshow(vals, cmap=GREEN_MAP, vmin=0.0, vmax=1.0,
                   aspect='auto', origin='upper')

    # cell text + best-per-row border
    best_mask = (vals == np.nanmax(vals, axis=1, keepdims=True))
    for i in range(nrows):
        for j in range(ncols):
            v = vals[i, j]
            text_color = 'white' if v > 0.55 else 'black'
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=7.5,
                    fontweight='bold' if best_mask[i, j] else 'normal',
                    color=text_color)
            if best_mask[i, j]:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    boxstyle='round,pad=0.02',
                    linewidth=2, edgecolor='#e8a000',
                    facecolor='none', zorder=3))

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(df.columns, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(df.index, fontsize=9)
    ax.set_title('CIC-IDS2017 — Per-Class Average Precision by Model\n'
                 '(gold border = best model per attack class)',
                 fontsize=12, fontweight='bold', pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Average Precision', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    overall_df, perclass_df = load_all()

    print('\nOverall metrics:')
    print(overall_df.to_string(float_format=lambda x: f'{x:.4f}'))

    print('\nPer-class AP (models as rows):')
    print(perclass_df.to_string(float_format=lambda x: f'{x:.4f}'))

    plot_overall(overall_df,
                 os.path.join(OUT_DIR, 'cicids_overall_metrics.png'))
    plot_perclass(perclass_df,
                  os.path.join(OUT_DIR, 'cicids_perclass_ap.png'))
