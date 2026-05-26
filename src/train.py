"""Train LSTM model for energy consumption forecasting."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_or_create_dataset, prepare_datasets
from src.eda import save_eda_charts
from src.model import EnergyLSTM


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Return RMSE in scaled units."""
    return float(torch.sqrt(nn.functional.mse_loss(pred, target)).detach().cpu().item())


def main() -> None:
    """Run the full training pipeline."""
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    set_seed(config["project"]["seed"])
    output_dir = root / config["outputs"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_create_dataset(
        root / config["data"]["csv_path"],
        periods=config["data"]["periods"],
        seed=config["project"]["seed"],
    )
    save_eda_charts(df, output_dir)

    data = prepare_datasets(
        df,
        window_size=config["data"]["window_size"],
        horizon=config["data"]["horizon"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(data["x_train"]), torch.tensor(data["y_train"])),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_x = torch.tensor(data["x_val"])
    val_y = torch.tensor(data["y_val"])

    model = EnergyLSTM(**config["model"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_val = float("inf")
    history = []
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = float(criterion(val_pred, val_y).item())
            val_rmse = rmse_loss(val_pred, val_y)

        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_rmse_scaled": val_rmse})
        print(f"epoch={epoch:02d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_rmse={val_rmse:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "config": config}, output_dir / "best_model.pt")

    with open(output_dir / "training_log.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "rows": int(len(df)),
                "features": data["feature_columns"],
                "window_size": config["data"]["window_size"],
                "horizon": config["data"]["horizon"],
                "train_windows": int(len(data["x_train"])),
                "val_windows": int(len(data["x_val"])),
                "test_windows": int(len(data["x_test"])),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
