import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from plot_results import sweep_threshold, evaluate, plot_results, plot_loss_curves

DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_credit_card')
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_credit_card')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_credit_card')

EPOCHS = 20
BATCH_SIZE = 512
LR = 1e-3
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 256
DROPOUT = 0.1
MASK_RATIO = 0.40
SCORE_PASSES = 10  # random mask passes averaged at test time

# Each of the 29 features is one token with scalar value (input_size=1).
SEQ_LEN = 29


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class MaskedTransformerAE(nn.Module):
    def __init__(self, seq_len, d_model, nhead, num_layers, dim_ff, dropout, mask_ratio):
        super().__init__()
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio

        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # Learned token substituted at masked positions
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def _make_mask(self, batch_size, device):
        """Boolean mask (True = masked) with a fixed fraction of positions."""
        n_masked = max(1, int(self.seq_len * self.mask_ratio))
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            mask[i, torch.randperm(self.seq_len, device=device)[:n_masked]] = True
        return mask

    def forward(self, x, mask=None):
        # x: (batch, seq_len, 1)
        tokens = self.input_proj(x) # (batch, seq_len, d_model)
        tokens = self.pos_enc(tokens)

        if mask is not None:
            # Soft replacement â€” avoids in-place ops on the computation graph
            mask_f = mask.unsqueeze(-1).float()                        # (B, T, 1)
            mask_emb = self.mask_token.view(1, 1, -1).expand_as(tokens) # (B, T, d)
            tokens = tokens * (1.0 - mask_f) + mask_emb * mask_f

        encoded = self.encoder(tokens)       # (batch, seq_len, d_model)
        return self.output_proj(encoded)     # (batch, seq_len, 1)


def load_data():
    X_train = np.load(f'{DATA_DIR}/X_train_normal.npy').astype(np.float32)
    X_test = np.load(f'{DATA_DIR}/X_test.npy').astype(np.float32)
    y_test = np.load(f'{DATA_DIR}/y_test.npy')
    return X_train, X_test, y_test


def as_sequences(X):
    # (N, 29) â†’ (N, 29, 1)
    return torch.from_numpy(X).unsqueeze(-1)


def train(X_train, device):
    n_val = max(1, int(len(X_train) * 0.1))
    idx   = np.random.default_rng(0).permutation(len(X_train))
    X_val = X_train[idx[:n_val]]
    X_tr  = X_train[idx[n_val:]]

    seqs_tr  = as_sequences(X_tr)
    seqs_val = as_sequences(X_val)
    loader     = DataLoader(TensorDataset(seqs_tr),  batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(seqs_val), batch_size=BATCH_SIZE, shuffle=False)

    model = MaskedTransformerAE(
        SEQ_LEN, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF, DROPOUT, MASK_RATIO
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)                        # (B, 29, 1)
            mask = model._make_mask(batch.size(0), device) # (B, 29)
            recon = model(batch, mask)                      # (B, 29, 1)

            # Loss only on masked positions
            mask_3d = mask.unsqueeze(-1).float() # (B, 29, 1)
            loss = ((recon - batch) ** 2 * mask_3d).sum() / mask_3d.sum()

            optimiser.zero_grad(); loss.backward(); optimiser.step()
            total_loss += loss.item() * batch.size(0)
        tr_loss = total_loss / len(X_tr)

        model.eval()
        with torch.no_grad():
            v_total = 0.0
            for (batch,) in val_loader:
                batch   = batch.to(device)
                mask    = model._make_mask(batch.size(0), device)
                recon   = model(batch, mask)
                mask_3d = mask.unsqueeze(-1).float()
                loss    = ((recon - batch) ** 2 * mask_3d).sum() / mask_3d.sum()
                v_total += loss.item() * batch.size(0)
        val_loss = v_total / len(X_val)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}/{EPOCHS}  train={tr_loss:.6f}  val={val_loss:.6f}')

    return model, train_losses, val_losses


@torch.no_grad()
def reconstruction_scores(model, X, device, n_passes=SCORE_PASSES):
    """
    Average per-sample MSE over n_passes different random masks.
    Each pass scores only the masked positions, matching the training objective.
    Batched to avoid OOM on large test sets.
    """
    model.eval()
    scores = np.zeros(len(X), dtype=np.float32)

    for _ in range(n_passes):
        pass_scores = []
        for start in range(0, len(X), BATCH_SIZE):
            batch = as_sequences(X[start:start + BATCH_SIZE]).to(device)
            mask = model._make_mask(batch.size(0), device)
            recon = model(batch, mask)

            mask_3d = mask.unsqueeze(-1).float()
            ps = ((recon - batch) ** 2 * mask_3d).sum(dim=(1, 2)) / mask_3d.sum(dim=(1, 2))
            pass_scores.append(ps.cpu().numpy())
        scores += np.concatenate(pass_scores)
    return scores / n_passes


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    X_train, X_test, y_test = load_data()
    print(f'Training Masked Transformer AE on {X_train.shape[0]} samples â€¦')
    model, train_losses, val_losses = train(X_train, device)

    torch.save(model.state_dict(), f'{MODELS_DIR}/transformer_ae.pt')

    print(f'Scoring ({SCORE_PASSES} passes) â€¦')
    scores = reconstruction_scores(model, X_test, device)

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(y_test, scores, threshold, 'Transformer AE',
                        out_path=f'{RESULTS_DIR}/transformer_ae_report.txt')
    plot_results(y_test, y_pred, scores, 'Transformer AE',
                out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'credit_card', 'transformer_ae_results.png'))
    plot_loss_curves(train_losses, val_losses,
                     out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'credit_card', 'transformer_ae_loss_curve.png'))
