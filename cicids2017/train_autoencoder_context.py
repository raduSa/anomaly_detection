import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from plot_results import sweep_threshold, evaluate, plot_results, evaluate_per_class, plot_loss_curves

SKIP_LABELS = {'DoS Hulk'}

DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_cicids_context')
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_cicids')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_cicids')

EPOCHS = 20
BATCH_SIZE = 2048
LR = 1e-3
HIDDEN = [64, 32, 16]
DROPOUT = 0.1


class MLP_AE(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        enc_dims = [input_dim] + hidden
        dec_dims = hidden[::-1] + [input_dim]

        enc = []
        for i in range(len(enc_dims) - 1):
            enc += [nn.Linear(enc_dims[i], enc_dims[i + 1]), nn.ReLU()]
            if i < len(enc_dims) - 2:
                enc.append(nn.Dropout(DROPOUT))
        self.encoder = nn.Sequential(*enc)

        dec = []
        for i in range(len(dec_dims) - 1):
            if i < len(dec_dims) - 2:
                dec += [nn.Linear(dec_dims[i], dec_dims[i + 1]), nn.ReLU(),
                        nn.Dropout(DROPOUT)]
            else:
                dec += [nn.Linear(dec_dims[i], dec_dims[i + 1]), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    return X_train, X_test, y_test


def train(X_train, device):
    n_val = max(1, int(len(X_train) * 0.1))
    idx   = np.random.default_rng(0).permutation(len(X_train))
    X_val = X_train[idx[:n_val]]
    X_tr  = X_train[idx[n_val:]]

    loader     = DataLoader(TensorDataset(torch.from_numpy(X_tr)),  batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=BATCH_SIZE, shuffle=False)

    model = MLP_AE(X_tr.shape[1], HIDDEN).to(device)
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
        scores.append(((recon - batch) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(scores)


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    X_train, X_test, y_test = load_data()
    print(f'Training MLP AE+Context on {X_train.shape[0]} samples, {X_train.shape[1]} features â€¦')
    model, train_losses, val_losses = train(X_train, device)

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'autoencoder_context.pt'))

    scores = reconstruction_scores(model, X_test, device)

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(y_test, scores, threshold, 'MLP AE+Context (CIC-IDS2017)',
                        out_path=os.path.join(RESULTS_DIR, 'autoencoder_context_report.txt'))
    plot_results(y_test, y_pred, scores, 'MLP AE+Context (CIC-IDS2017)',
                out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'cicids', 'autoencoder_context_results.png'))
    plot_loss_curves(train_losses, val_losses,
                     out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'cicids', 'autoencoder_context_loss_curve.png'))

    attack_types = np.load(os.path.join(DATA_DIR, 'attack_types_test.npy'), allow_pickle=True)
    evaluate_per_class(attack_types, scores, skip_labels=SKIP_LABELS,
                       out_path=os.path.join(RESULTS_DIR, 'autoencoder_context_perclass_ap.txt'))
