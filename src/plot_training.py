"""Plot a training CSV: uv run src/plot_training.py PATH."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def numeric(value: str | None) -> float | None:
    """
    Parse plain numbers or scalar tensor strings from a metrics CSV.

    Params:
        value (`str | None`): The CSV value to parse.

    Returns:
        float | None: The finite numeric value, or None for missing or non-finite values.

    Raises:
        ValueError: If a non-empty value cannot be parsed as a number.
    """
    if not value or not value.strip():
        return None
    value = value.strip()
    if value.startswith("tensor(") and value.endswith(")"):
        value = value[7:-1].split(",", 1)[0]
    result = float(value)
    return result if math.isfinite(result) else None


def plot_training(csv_path: Path) -> Path:
    """
    Refresh the training loss and evaluation reward plots from a metrics CSV.

    Params:
        csv_path (`Path`): The CSV containing steps and training or evaluation metrics.

    Returns:
        Path: The path to `training_curves.png` saved alongside the CSV.
    """
    with csv_path.open(newline="") as source:
        rows = list(csv.DictReader(source))

    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, key, title, ylabel, color in [
        (axes[0], "epoch_loss", "Training loss", "Mean epoch loss", "#2563eb"),
        (
            axes[1],
            "eval/mean_reward",
            "Evaluation reward",
            "Mean episode maximum reward",
            "#059669",
        ),
    ]:
        points = []
        for row in rows:
            step, value = numeric(row.get("step")), numeric(row.get(key))
            if step is not None and value is not None:
                points.append((step, value))
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.2)
        if points:
            x, y = zip(*points)
            ax.plot(x, y, color=color, linewidth=1.7)
            if len(set(x)) > 1:
                ax.set_xlim(min(x), max(x))
            else:
                ax.plot(x, y, "o", color=color)
        else:
            ax.text(
                0.5,
                0.5,
                "No values recorded yet",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#64748b",
                linespacing=1.7,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(csv_path.parent.name, fontsize=12)
    output = csv_path.parent / "training_curves.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def main() -> None:
    """
    Parse the CSV path from the command line and save the training plots.

    Returns:
        None: Prints the path to the saved plot.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    print(f"Saved {plot_training(args.csv_path)}")


if __name__ == "__main__":
    main()
