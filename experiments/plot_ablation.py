"""
Visualize ablation results: solution profiles and error curves.

Usage:
    python experiments/plot_ablation.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def main():
    d = np.load(RESULTS / "ablation_metrics.npz")
    tau_star = float(d["tau_star"])
    configs = ["C0", "C2_rapidity", "C4_reciprocal"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Summary bar chart (E_rel_near) ---
    values = [float(d[f"{c}_E_rel_near"]) for c in configs]
    bars = axes[0].bar(configs, values, color=["#c0392b", "#f39c12", "#27ae60"])
    axes[0].axhline(0.1, color="black", linestyle="--",
                    label="target (E_rel_near < 0.1)")
    axes[0].set_ylabel("E_rel_near")
    axes[0].set_title("Relative error near pole")
    axes[0].legend()
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v,
                     f"{v:.2e}", ha="center", va="bottom", fontsize=9)

    # --- Comparison of metrics across configs ---
    labels = ["E_inf_far", "E_inf_near", "E_rel_far", "E_rel_near"]
    x = np.arange(len(labels))
    width = 0.25
    for i, cfg in enumerate(configs):
        vals = [float(d[f"{cfg}_{k}"]) for k in labels]
        axes[1].bar(x + i * width, vals, width, label=cfg)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_yscale("log")
    axes[1].set_title("All metrics (log scale)")
    axes[1].legend()

    plt.tight_layout()
    out = RESULTS / "ablation_summary.png"
    plt.savefig(out, dpi=150)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()