"""
Tests explainability_plan.md claim #4 (Credit Card): the three "temporal" architectures
reshape 29 independent PCA features into an artificial sequence with no real ordering.
If that's true, per-"timestep" (per-V-feature-position) reconstruction error on fraud
samples should show no coherent positional pattern for LSTM AE or Transformer Bottleneck
AE — just noise. A structured pattern would complicate that story.

Loads both trained models, reconstructs every fraud test sample, and plots:
  1. a per-position error heatmap (all fraud samples) for each model
  2. a side-by-side per-position error curve for one representative fraud sample

A first pass at running this (see mds/explainability_plan.md discussion) found a
*structured*, repeated pattern rather than noise -- specific column positions light up
consistently across many fraud samples. That's still consistent with "no real sequence
order", but only if the pattern is driven by feature *identity* (some V-features are
just harder to reconstruct, e.g. because they're strongly fraud-discriminative) rather
than by *position* (i.e. an artifact of adjacency/order). To tell these apart, this
script adds a column-shuffle experiment: the same fraud rows are re-run through both
models with a fixed random permutation of the 29 columns applied first. If the same
underlying features dominate the top-error positions in both the original and the
shuffled run, the effect is feature-identity-driven, not order-driven.
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'credit_card'))

import numpy as np
import torch

from heatmap_utils import next_experiment_dir, per_position_error, plot_error_heatmap, plot_paired_window_comparison

import train_lstm_ae as lstm_ae_mod
import train_transformer_bottleneck as transformer_mod

DATA_DIR = lstm_ae_mod.DATA_DIR
MODELS_DIR = lstm_ae_mod.MODELS_DIR
EXPERIMENT_NAME = 'credit_card_lstm_vs_transformer'
MAX_HEATMAP_SAMPLES = 300  # random subsample so the heatmap stays a legible, decodable image
PERMUTE_SEED = 42
TOP_K = 10


def load_lstm_ae(device):
    model = lstm_ae_mod.LSTMAutoencoder(
        lstm_ae_mod.INPUT_SIZE, lstm_ae_mod.HIDDEN_DIM, lstm_ae_mod.NUM_LAYERS, lstm_ae_mod.SEQ_LEN
    ).to(device)
    state = torch.load(f'{MODELS_DIR}/lstm_ae.pt', map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_transformer_bottleneck(device):
    model = transformer_mod.BottleneckTransformerAE(
        transformer_mod.SEQ_LEN, transformer_mod.INPUT_DIM, transformer_mod.D_MODEL,
        transformer_mod.NHEAD, transformer_mod.NUM_LAYERS, transformer_mod.DIM_FF,
        transformer_mod.DROPOUT, transformer_mod.LATENT_DIM,
    ).to(device)
    state = torch.load(f'{MODELS_DIR}/transformer_bottleneck.pt', map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def reconstruct(model, X, device, batch_size=512):
    recons = list()
    for start in range(0, len(X), batch_size):
        batch = lstm_ae_mod.as_sequences(X[start:start + batch_size]).to(device)
        recons.append(model(batch).cpu().numpy())
    return np.concatenate(recons)


def top_k_features(error_map, names, k=TOP_K):
    """Mean error per column position, ranked descending, paired with the feature
    name occupying that position (which differs between the original and shuffled runs)."""
    mean_error = error_map.mean(axis=0)  # (n_features,)
    order = np.argsort(mean_error)[::-1][:k]
    return [(names[i], float(mean_error[i])) for i in order]


def report_shuffle_comparison(model_name, orig_top, shuf_top, out_lines):
    orig_names = [name for name, _ in orig_top]
    shuf_names = [name for name, _ in shuf_top]
    overlap = sorted(set(orig_names) & set(shuf_names))

    out_lines.append(f'\n{model_name}')
    out_lines.append(f'  Top {len(orig_top)} (original column order): ' +
                      ', '.join(f'{n} ({e:.4f})' for n, e in orig_top))
    out_lines.append(f'  Top {len(shuf_top)} (shuffled column order):  ' +
                      ', '.join(f'{n} ({e:.4f})' for n, e in shuf_top))
    out_lines.append(f'  Overlap: {len(overlap)}/{len(orig_top)} features in common -> {overlap}')
    return len(overlap)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    X_test = np.load(f'{DATA_DIR}/X_test.npy').astype(np.float32)
    y_test = np.load(f'{DATA_DIR}/y_test.npy')

    fraud_idx = np.where(y_test == 1)[0]
    print(f'{len(fraud_idx)} fraud samples out of {len(y_test)}')

    lstm_model = load_lstm_ae(device)
    transformer_model = load_transformer_bottleneck(device)

    feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']

    # Reconstruct the full fraud population — cheap for these models — so the top-K
    # feature comparison reflects the true population, not just the heatmap subsample.
    X_fraud = X_test[fraud_idx]
    X_fraud_seq = X_fraud[:, :, np.newaxis]  # (N, 29, 1) to match reconstruct() output shape

    lstm_recon = reconstruct(lstm_model, X_fraud, device)
    transformer_recon = reconstruct(transformer_model, X_fraud, device)

    lstm_error = per_position_error(X_fraud_seq, lstm_recon)               # (n_fraud, 29)
    transformer_error = per_position_error(X_fraud_seq, transformer_recon)  # (n_fraud, 29)

    out_dir = next_experiment_dir(_ROOT, EXPERIMENT_NAME)

    # ------------------------------------------------------------------
    # Column-shuffle experiment: does the same handful of *features* dominate
    # reconstruction error regardless of which column position they sit at, or is
    # the pattern seen in the original-order heatmap an artifact of position/adjacency?
    # ------------------------------------------------------------------
    perm = np.random.default_rng(PERMUTE_SEED).permutation(len(feature_names))
    feature_names_shuffled = [feature_names[i] for i in perm]
    X_fraud_shuffled = X_fraud[:, perm]

    lstm_recon_shuf = reconstruct(lstm_model, X_fraud_shuffled, device)
    transformer_recon_shuf = reconstruct(transformer_model, X_fraud_shuffled, device)

    X_fraud_shuffled_seq = X_fraud_shuffled[:, :, np.newaxis]
    lstm_error_shuf = per_position_error(X_fraud_shuffled_seq, lstm_recon_shuf)
    transformer_error_shuf = per_position_error(X_fraud_shuffled_seq, transformer_recon_shuf)

    report_lines = [
        f'Column-shuffle feature-identity check (permutation seed={PERMUTE_SEED}, top {TOP_K}, '
        f'{len(fraud_idx)} fraud samples)',
        f'Shuffled column order: {feature_names_shuffled}',
    ]
    for model_name, err_orig, err_shuf in [
        ('LSTM AE', lstm_error, lstm_error_shuf),
        ('Transformer Bottleneck AE', transformer_error, transformer_error_shuf),
    ]:
        top_orig = top_k_features(err_orig, feature_names)
        top_shuf = top_k_features(err_shuf, feature_names_shuffled)
        report_shuffle_comparison(model_name, top_orig, top_shuf, report_lines)

    report_text = '\n'.join(report_lines)
    print('\n' + report_text)
    report_path = os.path.join(out_dir, 'column_shuffle_top_features.txt')
    with open(report_path, 'w') as f:
        f.write(report_text + '\n')
    print(f'\nTop-feature shuffle comparison saved to {report_path}')

    # ------------------------------------------------------------------
    # Heatmaps: random subsample of fraud rows so the image stays a legible size.
    # ------------------------------------------------------------------
    if len(fraud_idx) > MAX_HEATMAP_SAMPLES:
        heatmap_pos = np.sort(np.random.default_rng(0).choice(len(fraud_idx), size=MAX_HEATMAP_SAMPLES, replace=False))
        print(f'Subsampled to {len(heatmap_pos)} fraud samples for the heatmap')
    else:
        heatmap_pos = np.arange(len(fraud_idx))

    plot_error_heatmap(
        lstm_error[heatmap_pos], os.path.join(out_dir, 'lstm_ae_error_heatmap.png'),
        title='LSTM AE — per-feature-position error on fraud samples',
        xlabel='Feature position (' + ', '.join(feature_names[:5]) + ', ...)', ylabel='Fraud sample',
    )
    plot_error_heatmap(
        transformer_error[heatmap_pos], os.path.join(out_dir, 'transformer_bottleneck_error_heatmap.png'),
        title='Transformer Bottleneck AE — per-feature-position error on fraud samples',
        xlabel='Feature position (' + ', '.join(feature_names[:5]) + ', ...)', ylabel='Fraud sample',
    )

    example = int(np.argmax(lstm_error.sum(axis=1)))
    plot_paired_window_comparison(
        lstm_error[example], transformer_error[example],
        label_a='LSTM AE', label_b='Transformer Bottleneck AE',
        out_path=os.path.join(out_dir, f'sample_{example}_comparison.png'),
        title=f'Per-feature-position error, fraud sample #{fraud_idx[example]}',
        xlabel='Feature position (V1..V28, Amount)',
    )

    print(f'All outputs saved under {out_dir}')
