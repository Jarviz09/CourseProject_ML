"""Exploratory data analysis charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_eda_charts(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Save main EDA plots to the output folder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.describe().to_csv(output_dir / "eda_summary.csv")

    plt.figure(figsize=(12, 4.5))
    plt.plot(df["timestamp"].iloc[:336], df["Global_active_power"].iloc[:336])
    plt.title("Energy consumption: first 14 days")
    plt.xlabel("Time")
    plt.ylabel("Global active power")
    plt.tight_layout()
    plt.savefig(output_dir / "timeseries_power_14_days.png", dpi=180)
    plt.close()

    hourly = df.groupby("hour")["Global_active_power"].mean()
    plt.figure(figsize=(8, 4.5))
    plt.plot(hourly.index, hourly.values, marker="o")
    plt.title("Average daily consumption profile")
    plt.xlabel("Hour")
    plt.ylabel("Average active power")
    plt.tight_layout()
    plt.savefig(output_dir / "daily_profile.png", dpi=180)
    plt.close()

    daily = df.set_index("timestamp")["Global_active_power"].resample("D").mean()
    plt.figure(figsize=(12, 4.5))
    plt.plot(daily.index, daily.values)
    plt.title("Daily average consumption")
    plt.xlabel("Date")
    plt.ylabel("Average active power")
    plt.tight_layout()
    plt.savefig(output_dir / "daily_average_power.png", dpi=180)
    plt.close()

    corr = df[["Global_active_power", "temperature", "humidity", "hour", "is_weekend"]].corr()
    plt.figure(figsize=(6, 5))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=180)
    plt.close()
