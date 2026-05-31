import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from plot_results import sweep_threshold, evaluate, plot_results, plot_loss_curves

BENCHMARK = 'A3Benchmark'
DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_yahoo', BENCHMARK)
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_yahoo')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_yahoo')

EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 256
DROPOUT = 0.1
LATENT_DIM = 32

WINDOW = 64
INPUT_DIM = 1


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class BottleneckTransformerAE(nn.Module):
    def __init__(self, seq_len, input_dim, d_model, nhead, num_layers, dim_ff, dropout, latent_dim):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.squeeze = nn.Linear(d_model, latent_dim)
        self.expand = nn.Linear(latent_dim, seq_len * input_dim)

    def forward(self, x):
        tokens = self.pos_enc(self.input_proj(x))
        encoded = self.encoder(tokens)
        latent = self.squeeze(encoded.mean(dim=1))
        return self.expand(latent).view(-1, self.seq_len, self.input_dim)


def load_data():
    X_train = np.load(f'{DATA_DIR}/X_train_normal.npy').astype(np.float32)
    X_test = np.load(f'{DATA_DIR}/X_test.npy').astype(np.float32)
    y_test = np.load(f'{DATA_DIR}/y_test.npy')
    return X_train, X_test, y_test


def train(X_train, device):
    n_val = max(1, int(len(X_train) * 0.1))
    idx   = np.random.default_rng(0).permutation(len(X_train))
    X_val = X_train[idx[:n_val]]
    X_tr  = X_train[idx[n_val:]]

    loader     = DataLoader(TensorDataset(torch.from_numpy(X_tr)),  batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=BATCH_SIZE, shuffle=False)

    model = BottleneckTransformerAE(
        WINDOW, INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF, DROPOUT, LATENT_DIM
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = criterion(model(batch), batch)
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            total += loss.item() * len(batch)
        tr_loss = total / len(X_tr)

        model.eval()
        with torch.no_grad():
            v_total = 0.0
            for (batch,) in val_loader:
                batch = batch.to(device)
                v_total += criterion(model(batch), batch).item() * len(batch)
        val_loss = v_total / len(X_val)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}/{EPOCHS}  train={tr_loss:.6f}  val={val_loss:.6f}')
    return model, train_losses, val_losses


@torch.no_grad()
def reconstruction_scores(model, X, device):
    model.eval()
    scores = []
    for start in range(0, len(X), BATCH_SIZE):
        batch = torch.from_numpy(X[start:start + BATCH_SIZE]).to(device)
        recon = model(batch)
        scores.append(((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(scores)


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    X_train, X_test, y_test = load_data()
    print(f'Training Bottleneck Transformer AE on {X_train.shape[0]} windows â€¦')
    model, train_losses, val_losses = train(X_train, device)

    torch.save(model.state_dict(), f'{MODELS_DIR}/transformer_bottleneck_{BENCHMARK}.pt')

    scores = reconstruction_scores(model, X_test, device)

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(y_test, scores, threshold, f'Transformer Bottleneck AE ({BENCHMARK})',
                    out_path=f'{RESULTS_DIR}/transformer_bottleneck_{BENCHMARK}_report.txt')
    plot_results(y_test, y_pred, scores, f'Transformer Bottleneck AE ({BENCHMARK})',
                out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'yahoo', f'transformer_bottleneck_{BENCHMARK}_results.png'))
    plot_loss_curves(train_losses, val_losses,
                     out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'yahoo', f'transformer_bottleneck_{BENCHMARK}_loss_curve.png'))
