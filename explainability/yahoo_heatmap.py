"""
Tests explainability_plan.md claim #3 (Yahoo A3): does the LSTM AE's single pooled
hidden state destroy seasonal phase information (diffuse/flat per-timestep error),
while the Transformer Bottleneck AE's global self-attention localizes the error at
the phase-violation point?

Loads both trained models, reconstructs every anomalous A3 test window, and plots:
  1. a per-timestep error heatmap (all anomalous windows) for each model
  2. a side-by-side per-timestep error curve for one representative anomalous window
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'yahoo'))

import numpy as np
import torch

from heatmap_utils import (
    next_experiment_dir, per_position_error, plot_error_heatmap, plot_paired_window_comparison,
    per_window_gini, plot_gini_comparison,
)

import ts_train_lstm_ae as lstm_ae_mod
import ts_train_transformer_bottleneck as transformer_mod

BENCHMARK = lstm_ae_mod.BENCHMARK
DATA_DIR = lstm_ae_mod.DATA_DIR
MODELS_DIR = lstm_ae_mod.MODELS_DIR
EXPERIMENT_NAME = f'yahoo_{BENCHMARK}_lstm_vs_transformer'
MAX_HEATMAP_SAMPLES = 300  # random subsample so the heatmap stays a legible, decodable image


def load_lstm_ae(device):
    model = lstm_ae_mod.LSTMAutoencoder(
        lstm_ae_mod.INPUT_SIZE, lstm_ae_mod.HIDDEN_DIM, lstm_ae_mod.NUM_LAYERS, lstm_ae_mod.WINDOW
    ).to(device)
    state = torch.load(f'{MODELS_DIR}/lstm_ae_{BENCHMARK}.pt', map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_transformer_bottleneck(device):
    model = transformer_mod.BottleneckTransformerAE(
        transformer_mod.WINDOW, transformer_mod.INPUT_DIM, transformer_mod.D_MODEL,
        transformer_mod.NHEAD, transformer_mod.NUM_LAYERS, transformer_mod.DIM_FF,
        transformer_mod.DROPOUT, transformer_mod.LATENT_DIM,
    ).to(device)
    state = torch.load(f'{MODELS_DIR}/transformer_bottleneck_{BENCHMARK}.pt', map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def reconstruct(model, X, device, batch_size=256):
    recons = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).to(device)
        recons.append(model(batch).cpu().numpy())
    return np.concatenate(recons)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    X_test = np.load(f'{DATA_DIR}/X_test.npy').astype(np.float32)
    y_test = np.load(f'{DATA_DIR}/y_test.npy')

    anom_idx = np.where(y_test == 1)[0]
    print(f'{len(anom_idx)} anomalous windows out of {len(y_test)}')

    lstm_model = load_lstm_ae(device)
    transformer_model = load_transformer_bottleneck(device)

    # Reconstruct the full anomalous population — cheap for these models — so the
    # Gini distribution reflects the true population, not just the heatmap subsample.
    X_anom = X_test[anom_idx]
    lstm_recon = reconstruct(lstm_model, X_anom, device)
    transformer_recon = reconstruct(transformer_model, X_anom, device)

    lstm_error = per_position_error(X_anom, lstm_recon)               # (n_anom, WINDOW)
    transformer_error = per_position_error(X_anom, transformer_recon)  # (n_anom, WINDOW)

    out_dir = next_experiment_dir(_ROOT, EXPERIMENT_NAME)

    # Per-window error-concentration (Gini) distribution: 0 = spread evenly across the
    # window (destroyed phase), 1 = concentrated at a single timestep (localized spike).
    lstm_gini = per_window_gini(lstm_error)
    transformer_gini = per_window_gini(transformer_error)
    print(f'LSTM AE Gini:               mean={lstm_gini.mean():.4f}  median={np.median(lstm_gini):.4f}')
    print(f'Transformer Bottleneck Gini: mean={transformer_gini.mean():.4f}  median={np.median(transformer_gini):.4f}')
    plot_gini_comparison(
        lstm_gini, transformer_gini, label_a='LSTM AE', label_b='Transformer Bottleneck AE',
        out_path=os.path.join(out_dir, 'gini_comparison.png'),
        title=f'Per-window error concentration (Gini), anomalous {BENCHMARK} windows',
    )

    # Heatmap: random subsample so the image stays a legible, decodable size.
    if len(anom_idx) > MAX_HEATMAP_SAMPLES:
        heatmap_pos = np.sort(np.random.default_rng(0).choice(len(anom_idx), size=MAX_HEATMAP_SAMPLES, replace=False))
        print(f'Subsampled to {len(heatmap_pos)} anomalous windows for the heatmap')
    else:
        heatmap_pos = np.arange(len(anom_idx))

    plot_error_heatmap(
        lstm_error[heatmap_pos], os.path.join(out_dir, 'lstm_ae_error_heatmap.png'),
        title=f'LSTM AE — per-timestep error on anomalous {BENCHMARK} windows',
        xlabel='Timestep', ylabel='Anomalous window',
    )
    plot_error_heatmap(
        transformer_error[heatmap_pos], os.path.join(out_dir, 'transformer_bottleneck_error_heatmap.png'),
        title=f'Transformer Bottleneck AE — per-timestep error on anomalous {BENCHMARK} windows',
        xlabel='Timestep', ylabel='Anomalous window',
    )

    # One representative window: the one with the highest LSTM AE error, for a direct
    # side-by-side curve comparison.
    example = int(np.argmax(lstm_error.sum(axis=1)))
    plot_paired_window_comparison(
        lstm_error[example], transformer_error[example],
        label_a='LSTM AE', label_b='Transformer Bottleneck AE',
        out_path=os.path.join(out_dir, f'window_{example}_comparison.png'),
        title=f'Per-timestep error, anomalous window #{anom_idx[example]} ({BENCHMARK})',
        xlabel='Timestep',
    )

    print(f'All outputs saved under {out_dir}')
