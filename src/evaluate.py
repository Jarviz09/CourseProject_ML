"""Evaluate trained LSTM model and save regression metrics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import inverse_target, load_or_create_dataset, prepare_datasets
from src.model import EnergyLSTM


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE in percent."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100)


def save_training_plots(log_path: Path, output_dir: Path) -> None:
    """Save training curves from CSV log."""
    log = pd.read_csv(log_path)

    plt.figure(figsize=(8, 4.5))
    plt.plot(log["epoch"], log["train_loss"], marker="o", label="Train")
    plt.plot(log["epoch"], log["val_loss"], marker="o", label="Validation")
    plt.title("Training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(log["epoch"], log["val_rmse_scaled"], marker="o")
    plt.title("Validation RMSE in scaled units")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(output_dir / "validation_rmse.png", dpi=180)
    plt.close()


def main() -> None:
    """Evaluate model on the test split."""
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    output_dir = root / config["outputs"]["dir"]
    df = load_or_create_dataset(root / config["data"]["csv_path"], periods=config["data"]["periods"], seed=config["project"]["seed"])
    data = prepare_datasets(
        df,
        window_size=config["data"]["window_size"],
        horizon=config["data"]["horizon"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
    )

    checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu")
    model = EnergyLSTM(**config["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        scaled_pred = model(torch.tensor(data["x_test"])).numpy().reshape(-1)

    scaled_true = data["y_test"].reshape(-1)
    y_pred = inverse_target(scaled_pred, data["scaler"], data["target_index"])
    y_true = inverse_target(scaled_true, data["scaler"], data["target_index"])

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape, "R2": r2}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    errors = pd.DataFrame({"actual": y_true, "predicted": y_pred, "absolute_error": np.abs(y_true - y_pred)})
    errors.to_csv(output_dir / "prediction_errors.csv", index=False)

    plt.figure(figsize=(12, 4.5))
    plt.plot(y_true[:240], label="Actual")
    plt.plot(y_pred[:240], label="Predicted")
    plt.title("Energy consumption forecast: first 240 test hours")
    plt.xlabel("Test hour")
    plt.ylabel("Global active power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_plot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_true, y_pred, alpha=0.45)
    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value])
    plt.title("Actual vs predicted values")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=180)
    plt.close()

    save_training_plots(output_dir / "training_log.csv", output_dir)
    print(metrics)


if __name__ == "__main__":
    main()
