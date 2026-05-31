import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from plot_results import sweep_threshold, evaluate, plot_results, evaluate_per_class

DATA_DIR = os.path.join(_ROOT, 'data', 'processed_data', 'processed_cicids_temporal')
MODELS_DIR = os.path.join(_ROOT, 'data', 'models', 'models_cicids')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'results', 'results_cicids')

EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
WINDOW = 16
INPUT_DIM = 77
HIDDEN = [512, 256, 64]   # flattened input: 16 * 77 = 1232
SKIP_LABELS = {'DoS Hulk'}


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int]):
        super().__init__()
        enc, dec = [], []
        dims = [input_dim] + hidden
        for i in range(len(dims) - 1):
            enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        dims_rev = list(reversed(dims))
        for i in range(len(dims_rev) - 1):
            dec += [nn.Linear(dims_rev[i], dims_rev[i + 1])]
            if i < len(dims_rev) - 2:
                dec.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy')).astype(np.float32)
    X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy')).astype(np.float32)
    y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    return X_train.reshape(len(X_train), -1), X_test.reshape(len(X_test), -1), y_test


def train(X_train, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=BATCH_SIZE, shuffle=True)
    model = Autoencoder(WINDOW * INPUT_DIM, HIDDEN).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = criterion(model(batch), batch)
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            total += loss.item() * len(batch)
        if epoch % 5 == 0 or epoch == 1:
            print(f'  Epoch {epoch}/{EPOCHS}  loss={total / len(X_train)}')
    return model


@torch.no_grad()
def reconstruction_scores(model, X, device):
    model.eval()
    t = torch.from_numpy(X).to(device)
    recon = model(t).cpu().numpy()
    return np.mean((X - recon) ** 2, axis=1)


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    X_train, X_test, y_test = load_data()
    print(f'Training Autoencoder on {X_train.shape[0]} windows â€¦')
    model = train(X_train, device)

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'autoencoder_cicids_temporal.pt'))

    scores = reconstruction_scores(model, X_test, device)

    threshold = sweep_threshold(y_test, scores, beta=2.0)
    y_pred = evaluate(y_test, scores, threshold, 'Autoencoder (CIC-IDS2017 temporal)',
                      out_path=os.path.join(RESULTS_DIR, 'autoencoder_cicids_temporal_report.txt'))
    plot_results(y_test, y_pred, scores, 'Autoencoder (CIC-IDS2017 temporal)',
                 out_path=os.path.join(_ROOT, 'data', 'graphs', 'results', 'cicids', 'autoencoder_cicids_temporal_results.png'))

    attack_types = np.load(os.path.join(DATA_DIR, 'attack_types_test.npy'), allow_pickle=True)
    evaluate_per_class(attack_types, scores, skip_labels=SKIP_LABELS,
                       out_path=os.path.join(RESULTS_DIR, 'autoencoder_cicids_temporal_perclass_ap.txt'))
