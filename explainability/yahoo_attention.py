"""
Tests the "removing seasonality helps the Causal Transformer" reading of the A3/A4
architectural-gap paragraph in results.tex: on A3 (seasonal, phase-violation anomalies)
the Causal Transformer must learn an explicit attention pattern to recover phase; on A4
(same structure, no seasonality, level-shift anomalies) no phase accumulation is needed,
so the task collapses to comparing against a short local window.

Claim tested: the Causal Transformer's self-attention should be long-range / spread out
on A3 (attempting phase recovery) and short-range / recency-dominated on A4 (no
periodicity to look back for). This is a descriptive/correlational check -- it shows
attention pattern co-varies with the benchmark switch in the predicted direction, not
that the attention pattern causes the narrower A3-vs-A4 gap to Predictive LSTM.

Two diagnostics, mirroring the qualitative-then-quantitative structure used elsewhere
in this chapter:
  1. Qualitative: mean attention matrix (T x T), averaged over all anomalous test
     windows, A3 vs A4 side by side.
  2. Quantitative: per-window attention centroid lag (attention-weighted average
     distance back into the past), restricted to query positions with enough history
     to make a real long-vs-short choice, compared A3 vs A4.
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'yahoo'))

import numpy as np
import torch
import torch.nn as nn

from heatmap_utils import next_experiment_dir, plot_error_heatmap, plot_value_distribution_comparison

import ts_train_transformer_causal as ct_mod

MODELS_DIR = ct_mod.MODELS_DIR
WINDOW = ct_mod.WINDOW           # 64 -- raw window length
BATCH_SIZE = ct_mod.BATCH_SIZE
EXPERIMENT_NAME = 'yahoo_causal_transformer_attention_A3_vs_A4'

# Query positions need enough available history to show a real long-vs-short-range
# choice; the first few positions are structurally forced to attend locally regardless
# of what the model "wants", so they're excluded from the centroid-lag statistic.
MIN_QUERY_POS = (WINDOW - 1) // 2


def data_dir(benchmark):
    return os.path.join(_ROOT, 'data', 'processed_data', 'processed_yahoo', benchmark)


def load_causal_transformer(benchmark, device):
    model = ct_mod.CausalTransformer(
        ct_mod.INPUT_SIZE, ct_mod.D_MODEL, ct_mod.NHEAD, ct_mod.NUM_LAYERS,
        ct_mod.DIM_FF, ct_mod.DROPOUT,
    ).to(device)
    state = torch.load(os.path.join(MODELS_DIR, f'transformer_causal_{benchmark}.pt'), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def extract_attention(model, x):
    """Manually replays CausalTransformer.forward, forcing each layer's self_attn to
    return attention weights. nn.TransformerEncoderLayer normally routes through a
    fused fast path that never materializes attention weights (a plain forward hook
    would just see None) -- need_weights=True here disables that path. Returns the
    per-layer attention weights averaged into one (batch, T, T) matrix (heads already
    averaged per layer via average_attn_weights=True)."""
    T = x.size(1)
    causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
    h = model.pos_enc(model.input_proj(x))

    layer_weights = []
    for layer in model.encoder.layers:
        normed = layer.norm1(h)
        attn_out, weights = layer.self_attn(
            normed, normed, normed, attn_mask=causal_mask,
            need_weights=True, average_attn_weights=True,
        )
        layer_weights.append(weights)  # (batch, T, T)
        h = h + layer.dropout1(attn_out)
        normed2 = layer.norm2(h)
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(normed2))))
        h = h + layer.dropout2(ff)

    return torch.stack(layer_weights, dim=0).mean(dim=0)  # (batch, T, T)


@torch.no_grad()
def mean_attention_and_centroid_lag(model, X_anom, device, batch_size=BATCH_SIZE):
    """Streams over all anomalous windows and returns:
      - mean_attn: (T, T) attention matrix averaged over the full anomalous population
      - per_window_lag: (n_anom,) attention centroid lag per window, averaged over
        query positions >= MIN_QUERY_POS
    """
    T = X_anom.shape[1] - 1  # model consumes windows[:, :-1, :]
    sum_attn = torch.zeros(T, T, device=device)
    n_seen = 0
    lags = []

    lag_matrix = (torch.arange(T, device=device).unsqueeze(1)
                  - torch.arange(T, device=device).unsqueeze(0)).float()  # lag[t, k] = t - k

    for start in range(0, len(X_anom), batch_size):
        batch = torch.from_numpy(X_anom[start:start + batch_size]).to(device)
        inputs = batch[:, :-1, :]
        attn = extract_attention(model, inputs)  # (b, T, T)

        sum_attn += attn.sum(dim=0)
        n_seen += attn.size(0)

        centroid = (attn * lag_matrix.unsqueeze(0)).sum(dim=-1)  # (b, T)
        per_window = centroid[:, MIN_QUERY_POS:].mean(dim=-1)   # (b,)
        lags.append(per_window.cpu().numpy())

    mean_attn = (sum_attn / n_seen).cpu().numpy()
    per_window_lag = np.concatenate(lags)
    return mean_attn, per_window_lag


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    results = {}
    for benchmark in ['A3Benchmark', 'A4Benchmark']:
        X_test = np.load(os.path.join(data_dir(benchmark), 'X_test.npy')).astype(np.float32)
        y_test = np.load(os.path.join(data_dir(benchmark), 'y_test.npy'))
        X_anom = X_test[y_test == 1]
        print(f'{benchmark}: {len(X_anom)} anomalous windows out of {len(y_test)}')

        model = load_causal_transformer(benchmark, device)
        mean_attn, per_window_lag = mean_attention_and_centroid_lag(model, X_anom, device)
        results[benchmark] = (mean_attn, per_window_lag)
        print(f'{benchmark}: centroid lag mean={per_window_lag.mean():.3f}  median={np.median(per_window_lag):.3f}')

    out_dir = next_experiment_dir(_ROOT, EXPERIMENT_NAME)

    plot_error_heatmap(
        results['A3Benchmark'][0], os.path.join(out_dir, 'causal_transformer_a3_mean_attention.png'),
        title='Causal Transformer -- mean attention matrix, A3 anomalous windows',
        xlabel='Key position (attended-to)', ylabel='Query position', cbar_label='Attention weight',
    )
    plot_error_heatmap(
        results['A4Benchmark'][0], os.path.join(out_dir, 'causal_transformer_a4_mean_attention.png'),
        title='Causal Transformer -- mean attention matrix, A4 anomalous windows',
        xlabel='Key position (attended-to)', ylabel='Query position', cbar_label='Attention weight',
    )

    plot_value_distribution_comparison(
        results['A3Benchmark'][1], results['A4Benchmark'][1],
        label_a='A3 (seasonal)', label_b='A4 (no seasonality)',
        out_path=os.path.join(out_dir, 'centroid_lag_comparison.png'),
        title=f'Attention centroid lag, anomalous windows (query pos >= {MIN_QUERY_POS})',
        xlabel='Attention-weighted mean lag (timesteps back)',
    )

    print(f'All outputs saved under {out_dir}')
