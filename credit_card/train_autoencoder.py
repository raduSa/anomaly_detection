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

EPOCHS = 20
BATCH_SIZE = 512
LR = 1e-3
HIDDEN = [16, 8]
DROPOUT = 0.1


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int]):
        super().__init__()

        enc_layers, dec_layers = [], []
        dims = [input_dim] + hidden
        for i in range(len(dims) - 1):
            enc_layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
            if i < len(dims) - 2:
                enc_layers.append(nn.Dropout(DROPOUT))

        dims_rev = list(reversed(dims))
        for i in range(len(dims_rev) - 1):
            dec_layers += [nn.Linear(dims_rev[i], dims_rev[i + 1])]
            if i < len(dims_rev) - 2:
                dec_layers.append(nn.ReLU())
                dec_layers.append(nn.Dropout(DROPOUT))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_data():
    X_train = np.load(f'{DATA_DIR}/X_train_normal.npy').astype(np.float32)
    X_test = np.load(f'{DATA_DIR}/X_test.npy').astype(np.float32)
    y_test = np.load(f'{DATA_DIR}/y_test.npy')
    return X_train, X_test, y_test


def train(X_train: np.ndarray, device: torch.device):
    n_val = max(1, int(len(X_train) * 0.1))
    idx   = np.random.default_rng(0).permutation(len(X_train))
    X_val = X_train[idx[:n_val]]
    X_tr  = X_train[idx[n_val:]]

    loader     = DataLoader(TensorDataset(torch.from_numpy(X_tr)),  batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=BATCH_SIZE, shuffle=False)

    model = Autoencoder(X_tr.shape[1], HIDDEN).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss  = criterion(model(batch), batch)
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
    tensor = torch.from_numpy(X).to(device)
    recon  = model(tensor).cpu().numpy()
    # Per-sample mean squared reconstruction error
    return np.mean((X - recon) ** 2, axis=1)


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    X_train, X_test, y_test = load_data()
    print(f'Training Autoencoder on {X_train.shape[0]} samples â€¦')
    model, train_losses, val_losses = train(X_train, device)

    torch.save(model.state_dict(), f'{MODELS_DIR}/autoencoder.pt')

    scores = reconstruction_scores(model, X_test, device)

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(y_test, scores, threshold, 'Autoencoder',
                        out_path=f'{RESULTS_DIR}/autoencoder_report.txt')
    plot_results(y_test, y_pred, scores, 'Autoencoder',
                out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'credit_card', 'autoencoder_results.png'))
    plot_loss_curves(train_losses, val_losses,
                     out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'credit_card', 'autoencoder_loss_curve.png'))
