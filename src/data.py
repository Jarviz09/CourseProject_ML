"""Data generation and preprocessing for energy consumption forecasting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "Global_active_power",
    "temperature",
    "humidity",
    "hour_sin",
    "hour_cos",
    "is_weekend",
]
TARGET_COLUMN = "Global_active_power"


def set_seed(seed: int = 42) -> None:
    """Fix NumPy random seed."""
    np.random.seed(seed)


def generate_energy_dataset(csv_path: str | Path, periods: int = 4380, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible synthetic hourly energy consumption dataset.

    The generated time series includes daily seasonality, weekly seasonality,
    temperature influence and random noise. This makes the task close to a
    real educational forecasting problem while keeping the project fully
    reproducible without external downloads.
    """
    set_seed(seed)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = pd.date_range("2024-01-01", periods=periods, freq="h")
    hour = timestamp.hour.to_numpy()
    day_of_week = timestamp.dayofweek.to_numpy()
    is_weekend = (day_of_week >= 5).astype(float)

    daily = 0.9 + 0.45 * np.sin(2 * np.pi * (hour - 7) / 24)
    evening_peak = 0.55 * np.exp(-((hour - 20) ** 2) / 12)
    weekly = np.where(is_weekend == 1, -0.22, 0.1)

    seasonal_temperature = 8 + 11 * np.sin(2 * np.pi * np.arange(periods) / (24 * 180))
    daily_temperature = 3 * np.sin(2 * np.pi * (hour - 14) / 24)
    temperature = seasonal_temperature + daily_temperature + np.random.normal(0, 1.2, periods)
    humidity = 64 - 0.7 * temperature + np.random.normal(0, 4.5, periods)
    humidity = np.clip(humidity, 25, 95)

    heating_effect = np.maximum(16 - temperature, 0) * 0.055
    cooling_effect = np.maximum(temperature - 24, 0) * 0.035
    noise = np.random.normal(0, 0.08, periods)

    power = 1.4 + daily + evening_peak + weekly + heating_effect + cooling_effect + noise
    power = np.clip(power, 0.2, None)

    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "Global_active_power": power,
            "temperature": temperature,
            "humidity": humidity,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
        }
    )
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df.to_csv(csv_path, index=False)
    return df


def load_or_create_dataset(csv_path: str | Path, periods: int = 4380, seed: int = 42) -> pd.DataFrame:
    """Load dataset from CSV or create it if the file is absent."""
    csv_path = Path(csv_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["timestamp"])
    return generate_energy_dataset(csv_path=csv_path, periods=periods, seed=seed)


def make_windows(values: np.ndarray, target_index: int, window_size: int = 48, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Convert a multivariate time series into sliding windows."""
    x, y = [], []
    for start in range(0, len(values) - window_size - horizon + 1):
        end = start + window_size
        x.append(values[start:end])
        y.append(values[end + horizon - 1, target_index])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32).reshape(-1, 1)


def prepare_datasets(
    df: pd.DataFrame,
    window_size: int = 48,
    horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict:
    """Scale features and split windows chronologically."""
    feature_data = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    n_train_raw = int(len(feature_data) * train_ratio)

    scaler = StandardScaler()
    scaler.fit(feature_data[:n_train_raw])
    scaled = scaler.transform(feature_data)

    target_index = FEATURE_COLUMNS.index(TARGET_COLUMN)
    x, y = make_windows(scaled, target_index=target_index, window_size=window_size, horizon=horizon)

    n = len(x)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "x_train": x[:n_train],
        "y_train": y[:n_train],
        "x_val": x[n_train : n_train + n_val],
        "y_val": y[n_train : n_train + n_val],
        "x_test": x[n_train + n_val :],
        "y_test": y[n_train + n_val :],
        "scaler": scaler,
        "target_index": target_index,
        "feature_columns": FEATURE_COLUMNS,
    }


def inverse_target(values: np.ndarray, scaler: StandardScaler, target_index: int) -> np.ndarray:
    """Convert scaled target values back to original units."""
    values = np.asarray(values).reshape(-1)
    return values * scaler.scale_[target_index] + scaler.mean_[target_index]
