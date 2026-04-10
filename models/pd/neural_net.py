"""
models/pd/neural_net.py

PyTorch feedforward neural network for credit default prediction.

This implementation keeps the existing challenger logic, but simplifies the
training and calibration flow so the file is easier to maintain and review.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.pd.xgboost_model import XGB_FEATURES
from models.scorecard.evaluate import evaluate_model, plot_calibration

PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CreditDefaultNet(nn.Module):
    """Feedforward network for binary default prediction."""

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def predict_proba(model: nn.Module, scaler: StandardScaler,
                  X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    X_scaled = scaler.transform(X)
    X_tensor = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    return np.vstack([1 - probs, probs]).T


def train_neural_net(df: pd.DataFrame) -> dict:
    set_seed(42)

    df_train = df[df["issue_year"] <= 2016].copy()
    print(f"Training on {len(df_train):,} loans")

    features = [f for f in XGB_FEATURES if f in df_train.columns]
    X = df_train[features].fillna(0).astype(np.float32).values
    y = df_train["default_flag"].astype(np.float32).values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train):,} | Cal: {len(X_cal):,} | Test: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = CreditDefaultNet(input_dim=X_train_s.shape[1], dropout=0.3).to(device)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_dataset = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    train_losses = []
    for epoch in range(50):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{50} loss: {avg_loss:.4f}")

    raw_probs_cal = predict_proba(model, scaler, X_cal, device)[:, 1]
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(raw_probs_cal, y_cal)

    y_prob = iso_reg.predict(predict_proba(model, scaler, X_test, device)[:, 1])
    metrics = evaluate_model(y_test, y_prob, "PyTorch Neural Net")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, color="#2563EB")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Neural Net Training Loss")
    ax.grid(True, alpha=0.3)
    fig.savefig(PLOTS_DIR / "nn_loss.png", dpi=100, bbox_inches="tight")
    mlflow.log_artifact(str(PLOTS_DIR / "nn_loss.png"))
    plt.close()

    fig_cal = plot_calibration(y_test, y_prob, "PyTorch NN")
    fig_cal.savefig(PLOTS_DIR / "calibration_nn.png", dpi=100, bbox_inches="tight")
    mlflow.log_artifact(str(PLOTS_DIR / "calibration_nn.png"))
    plt.close()

    mlflow.log_metric("final_train_loss", train_losses[-1])
    return metrics
