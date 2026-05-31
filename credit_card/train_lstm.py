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

SEQ_LEN = 29
INPUT_SIZE = 1


class PredictiveLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_size)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        # Predict each next step from all previous steps
        out, _ = self.lstm(x)   # (batch, seq_len, hidden_dim)
        return self.output(out) # (batch, seq_len, 1)


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
    inputs_tr,  targets_tr  = seqs_tr[:, :-1, :],  seqs_tr[:, 1:, :]
    inputs_val, targets_val = seqs_val[:, :-1, :], seqs_val[:, 1:, :]

    loader     = DataLoader(TensorDataset(inputs_tr,  targets_tr),  batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(inputs_val, targets_val), batch_size=BATCH_SIZE, shuffle=False)

    model = PredictiveLSTM(INPUT_SIZE, HIDDEN_DIM, NUM_LAYERS).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            loss = criterion(model(x_b), y_b)
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            total_loss += loss.item() * len(x_b)
        tr_loss = total_loss / len(X_tr)

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
def reconstruction_scores(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    for start in range(0, len(X), BATCH_SIZE):
        batch = as_sequences(X[start:start + BATCH_SIZE]).to(device)
        inputs  = batch[:, :-1, :]
        targets = batch[:, 1:,  :].cpu().numpy()
        preds   = model(inputs).cpu().numpy()
        scores.append(np.mean((preds - targets) ** 2, axis=(1, 2)))
    return np.concatenate(scores)


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    X_train, X_test, y_test = load_data()
    print(f'Training Predictive LSTM on {X_train.shape[0]} samples â€¦')
    model, train_losses, val_losses = train(X_train, device)

    torch.save(model.state_dict(), f'{MODELS_DIR}/lstm.pt')

    scores = reconstruction_scores(model, X_test, device)

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(y_test, scores, threshold, 'Predictive LSTM',
                        out_path=f'{RESULTS_DIR}/lstm_report.txt')
    plot_results(y_test, y_pred, scores, 'Predictive LSTM',
                out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'credit_card', 'lstm_results.png'))
    plot_loss_curves(train_losses, val_losses,
                     out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'credit_card', 'lstm_loss_curve.png'))
