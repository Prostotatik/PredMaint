import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze(-1)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    unit_train: np.ndarray,
    cycle_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    lambda_monotonic: float = 0.2,
    lambda_smooth: float = 0.05,
    progress_cb=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[2]

    model = LSTMNet(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(unit_train),
        torch.from_numpy(cycle_train),
    )
    # Важно: для монотонного штрафа нам нужны соседние по циклу элементы в батче.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    val_X = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb, ub, cb in train_loader:
            xb, yb, ub, cb = xb.to(device), yb.to(device), ub.to(device), cb.to(device)
            pred = model(xb)

            loss = criterion(pred, yb)

            # Monotonic penalty: for same unit, consecutive cycles, RUL must NOT increase.
            # Sort within batch by (unit, cycle) to detect consecutive points.
            if lambda_monotonic > 0 or lambda_smooth > 0:
                sort_idx = torch.argsort(ub * 1_000_000 + cb)
                ub_s = ub[sort_idx]
                cb_s = cb[sort_idx]
                pred_s = pred[sort_idx]

                same_unit = ub_s[1:] == ub_s[:-1]
                consecutive = (cb_s[1:] - cb_s[:-1]) == 1
                mask = same_unit & consecutive
                if torch.any(mask):
                    diffs = pred_s[1:] - pred_s[:-1]
                    diffs = diffs[mask]
                    if lambda_monotonic > 0:
                        loss = loss + lambda_monotonic * torch.relu(diffs).mean()
                    if lambda_smooth > 0:
                        loss = loss + lambda_smooth * torch.abs(diffs).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            val_loss = criterion(val_pred, val_y).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if progress_cb:
            progress_cb(epoch + 1, epochs, val_loss ** 0.5)

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model.cpu()


def predict_lstm(model: LSTMNet, sequence: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(sequence).unsqueeze(0)
        pred = model(x)
    return max(0.0, float(pred.item()))


def predict_lstm_batch(model: LSTMNet, sequences: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(sequences)
        preds = model(x).numpy()
    return np.clip(preds, 0, None)


def save_lstm(model: LSTMNet, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": model.lstm.input_size,
        "hidden_dim": model.lstm.hidden_size,
        "num_layers": model.lstm.num_layers,
    }, path)


def load_lstm(path: str) -> LSTMNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LSTMNet(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
