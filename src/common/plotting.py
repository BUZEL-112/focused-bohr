"""Headless matplotlib helper shared by the CLI runners (all figures are saved, never shown)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: S101 - headless environments / CI


def get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def save_figure(output_dir, name: str, close: bool = True):
    """Save the current figure as <output_dir>/<name>.png (dpi 150) and optionally close it."""
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if close:
        plt.close()
    return path
