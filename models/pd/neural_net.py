"""
models/pd/neural_net.py

PyTorch feedforward neural network for credit default prediction.

Architecture: 3-layer network with batch normalisation and dropout.
Included to demonstrate deep learning range alongside traditional
credit risk methods. Not expected to significantly outperform
XGBoost on tabular data — included for completeness and to show
PyTorch proficiency.

Reference: Goodfellow et al. (2016) Deep Learning, chapter 6
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import mlflow
import matplotlib.pyplot as plt

from models.scorecard.evaluate import evaluate_model, plot_calibration, plot_ks

# Same features as XGBoost — ensures fair comparison
from models.pd.xgboost_model import XGB_FEATURES


class LoanDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CreditDefaultNet(nn.Module):
    """
    3-layer feedforward network.
    BatchNorm stabilises training on tabular data.
    Dropout prevents overfitting on moderate dataset sizes.
    """
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Output
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


class PyTorchWrapper:
    """
    Sklearn-compatible wrapper around the PyTorch model.
    Allows use with CalibratedClassifierCV and evaluate_model.
    """
    def __init__(self, model: CreditDefaultNet,
                 scaler: StandardScaler):
        self.model  = model
        self.scaler = scaler

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(next(self.model.parameters()).device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs  = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - probs, probs])

def train_neural_net(df: pd.DataFrame) -> dict:
    # Restrict to mature vintages
    df_train = df[df["issue_year"] <= 2016].copy()
    print(f"Training on {len(df_train):,} loans")

    features = [f for f in XGB_FEATURES if f in df_train.columns]
    X = df_train[features].fillna(0).values
    y = df_train["default_flag"].values

    # Three-way split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )

    # Scale features — critical for neural networks
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Class weights for imbalanced data
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    with mlflow.start_run(run_name="pytorch_nn_v1"):
        mlflow.set_tag("model_type", "pytorch_feedforward")
        mlflow.set_tag("challenger", "true")

        # Hyperparameters
        hparams = {
            "epochs":        50,
            "batch_size":    256,
            "learning_rate": 0.001,
            "dropout":       0.3,
            "hidden_layers": "128-64-32",
        }
        mlflow.log_params(hparams)

        # Build model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")

        model = CreditDefaultNet(
            input_dim=X_train_s.shape[1],
            dropout=hparams["dropout"]
        ).to(device)

        # Loss — BCELoss with LogitLoss is standard for binary classification.
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hparams["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        # Data loaders
        train_loader = DataLoader(
            LoanDataset(X_train_s, y_train),
            batch_size=hparams["batch_size"],
            shuffle=True
        )

        # Training loop
        train_losses = []
        for epoch in range(hparams["epochs"]):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = model(X_batch)
                loss  = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{hparams['epochs']} "
                      f"loss: {avg_loss:.4f}")

        # Wrap for sklearn compatibility
        wrapper = PyTorchWrapper(model, scaler)

        # Calibrate
        print("Calibrating...")

        # Get raw probabilities on calibration set
        raw_probs_cal = wrapper.predict_proba(X_cal)[:, 1]

        # Fit isotonic regression to correct probabilities
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(raw_probs_cal, y_cal)

        # Wrap everything into a final predictor
        class CalibratedWrapper:
            def __init__(self, base, calibrator):
                self.base       = base
                self.calibrator = calibrator

            def predict_proba(self, X):
                raw = self.base.predict_proba(X)[:, 1]
                cal = self.calibrator.predict(raw)
                return np.column_stack([1 - cal, cal])

        calibrated_model = CalibratedWrapper(wrapper, iso_reg)

        # Evaluate
        y_prob = calibrated_model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(
            y_test, y_prob, "PyTorch Neural Net"
        )

        # Loss curve
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_losses, color="#2563EB")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.set_title("Neural Net Training Loss")
        ax.grid(True, alpha=0.3)
        fig.savefig("nn_loss.png", dpi=100, bbox_inches="tight")
        mlflow.log_artifact("nn_loss.png")
        plt.close()

        # Calibration plot
        fig_cal = plot_calibration(y_test, y_prob, "PyTorch NN")
        fig_cal.savefig("calibration_nn.png", dpi=100, bbox_inches="tight")
        mlflow.log_artifact("calibration_nn.png")
        plt.close()

        mlflow.log_metric("final_train_loss", train_losses[-1])

    return metrics