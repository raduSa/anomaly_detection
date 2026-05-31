import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from plot_results import sweep_threshold, evaluate, plot_results, plot_loss_curves

DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_credit_card')
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_credit_card')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_credit_card')

EPOCHS = 50
BATCH_SIZE = 512
LR = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 1

# Each of the 29 features is treated as one time step (scalar value).
# The LSTM learns inter-feature dependencies in feature order.
SEQ_LEN = 29
INPUT_SIZE = 1


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, num_layers: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len

        self.encoder = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (h, c) = self.encoder(x)

        # Repeat final hidden state as the decoder input at each time step
        dec_input = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder(dec_input, (h, c))
        return self.output(dec_out)


def load_data():
    X_train = np.load(f'{DATA_DIR}/X_train_normal.npy').astype(np.float32)
    X_test = np.load(f'{DATA_DIR}/X_test.npy').astype(np.float32)
    y_test = np.load(f'{DATA_DIR}/y_test.npy')
    return X_train, X_test, y_test


def as_sequences(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(X).unsqueeze(-1)  # (N, 29, 1)


def train(X_train: np.ndarray, device: torch.device):
    n_val = max(1, int(len(X_train) * 0.1))
    idx   = np.random.default_rng(0).permutation(len(X_train))
    X_val = X_train[idx[:n_val]]
    X_tr  = X_train[idx[n_val:]]

    seqs_tr  = as_sequences(X_tr)
    seqs_val = as_sequences(X_val)
    loader     = DataLoader(TensorDataset(seqs_tr),  batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(seqs_val), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMAutoencoder(INPUT_SIZE, HIDDEN_DIM, NUM_LAYERS, SEQ_LEN).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = criterion(model(batch), batch)
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            total_loss += loss.item() * len(batch)
        tr_loss = total_loss / len(X_tr)

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
def reconstruction_scores(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    for start in range(0, len(X), BATCH_SIZE):
        batch = as_sequences(X[start:start + BATCH_SIZE]).to(device)
        recon = model(batch).cpu().numpy()
        orig  = X[start:start + BATCH_SIZE, :, np.newaxis]
        scores.append(np.mean((orig - recon) ** 2, axis=(1, 2)))
    return np.concatenate(scores)


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    X_train, X_test, y_test = load_data()
    print(f'Training LSTM Autoencoder on {X_train.shape[0]} samples â€¦')
    model, train_losses, val_losses = train(X_train, device)

    torch.save(model.state_dict(), f'{MODELS_DIR}/lstm_ae.pt')

    scores = reconstruction_scores(model, X_test, device)

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(y_test, scores, threshold, 'LSTM Autoencoder',
                        out_path=f'{RESULTS_DIR}/lstm_ae_report.txt')
    plot_results(y_test, y_pred, scores, 'LSTM Autoencoder',
                out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'credit_card', 'lstm_ae_results.png'))
    plot_loss_curves(train_losses, val_losses,
                     out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'credit_card', 'lstm_ae_loss_curve.png'))
