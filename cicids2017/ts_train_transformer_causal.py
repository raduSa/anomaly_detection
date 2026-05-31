import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from plot_results import sweep_threshold, evaluate, plot_results, evaluate_per_class, plot_loss_curves

DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_cicids_temporal')
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_cicids')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_cicids')

EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 256
DROPOUT = 0.1

WINDOW = 16
INPUT_SIZE = 77
SKIP_LABELS = {'DoS Hulk'}


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


class CausalTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_ff, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=WINDOW, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_size)

    def forward(self, x):
        T = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        tokens = self.pos_enc(self.input_proj(x))
        return self.output_proj(self.encoder(tokens, mask=causal_mask, is_causal=True))


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy')).astype(np.float32)
    X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy')).astype(np.float32)
    y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    return X_train, X_test, y_test


def train(X_train, device):
    n_val = max(1, int(len(X_train) * 0.1))
    idx   = np.random.default_rng(0).permutation(len(X_train))
    X_val = X_train[idx[:n_val]]
    X_tr  = X_train[idx[n_val:]]

    seqs_tr  = torch.from_numpy(X_tr)
    seqs_val = torch.from_numpy(X_val)
    inputs_tr,  targets_tr  = seqs_tr[:, :-1, :],  seqs_tr[:, 1:, :]
    inputs_val, targets_val = seqs_val[:, :-1, :], seqs_val[:, 1:, :]

    loader     = DataLoader(TensorDataset(inputs_tr,  targets_tr),  batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(inputs_val, targets_val), batch_size=BATCH_SIZE, shuffle=False)

    model  = CausalTransformer(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF, DROPOUT).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            loss = criterion(model(x_b), y_b)
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            total += loss.item() * len(x_b)
        tr_loss = total / len(X_tr)

        model.eval()
        with torch.no_grad():
            v_total = 0.0
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                v_total += criterion(model(x_b), y_b).item() * len(x_b)
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
        inputs = batch[:, :-1, :]
        targets = batch[:, 1:, :].cpu().numpy()
        preds = model(inputs).cpu().numpy()
        scores.append(np.mean((preds - targets) ** 2, axis=(1, 2)))
    return np.concatenate(scores)


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    X_train, X_test, y_test = load_data()
    print(f'Training Causal Transformer on {X_train.shape[0]} windows â€¦')
    model, train_losses, val_losses = train(X_train, device)

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'transformer_causal_cicids_temporal.pt'))

    scores = reconstruction_scores(model, X_test, device)

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(y_test, scores, threshold, 'Causal Transformer (CIC-IDS2017 temporal)',
                      out_path=os.path.join(RESULTS_DIR, 'transformer_causal_cicids_temporal_report.txt'))
    plot_results(y_test, y_pred, scores, 'Causal Transformer (CIC-IDS2017 temporal)',
                 out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'cicids', 'transformer_causal_cicids_temporal_results.png'))
    plot_loss_curves(train_losses, val_losses,
                     out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'cicids', 'transformer_causal_loss_curve.png'))

    attack_types = np.load(os.path.join(DATA_DIR, 'attack_types_test.npy'), allow_pickle=True)
    evaluate_per_class(attack_types, scores, skip_labels=SKIP_LABELS,
                       out_path=os.path.join(RESULTS_DIR, 'transformer_causal_cicids_temporal_perclass_ap.txt'))
