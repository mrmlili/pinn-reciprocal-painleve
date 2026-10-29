"""
Run ablation study comparing three representations of y'' = y^3:
  C0           : direct PINN (baseline)
  C2_rapidity  : theta = log(y)
  C4_reciprocal: w = 1/y   (recommended)

Adaptive weighting is omitted (empirically harmful; see README).

Usage:
    python experiments/run_ablation.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Allow imports from src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reference_dop853 import compute_reference
from pinn_solver import train_pinn, evaluate

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

CONFIGS = [
    ("C0",            "direct"),
    ("C2_rapidity",   "rapidity_log"),
    ("C4_reciprocal", "reciprocal"),
]


def main():
    # --- Reference ---
    ref_path = RESULTS / "reference.npz"
    if ref_path.exists():
        d = np.load(ref_path)
        tau_star = float(d["tau_star"])
        ref_t = torch.tensor(d["t_grid"], dtype=torch.float32)
        ref_y = torch.tensor(d["y_ref"], dtype=torch.float32)
        print(f"[reference] loaded tau_star={tau_star:.6f}")
    else:
        print("[reference] computing...")
        sol, tau_star = compute_reference()
        t_grid = np.linspace(0, 0.99 * tau_star, 2001)
        ref_t = torch.tensor(t_grid, dtype=torch.float32)
        ref_y = torch.tensor(sol.sol(t_grid)[0], dtype=torch.float32)
        np.savez(ref_path, t_grid=t_grid, y_ref=ref_y.numpy(),
                 tau_star=tau_star)
        print(f"[reference] tau_star={tau_star:.6f}")

    # --- Run each configuration ---
    all_results = {}
    for name, mode in CONFIGS:
        print(f"\n{'='*60}\nConfig {name} ({mode})\n{'='*60}")
        try:
            net, hist = train_pinn(mode=mode, tau_star=tau_star,
                                   n_epochs=15000, seed=42)
            metrics = evaluate(net, mode, tau_star, ref_t, ref_y)
            all_results[name] = {"history": hist, **metrics}
            torch.save(net.state_dict(), RESULTS / f"net_{name}.pt")
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            all_results[name] = {"error": str(e)}

    # --- Summary table ---
    print(f"\n{'='*72}")
    print(f"{'Config':<16}{'E_inf_far':>13}{'E_inf_near':>13}"
          f"{'E_rel_far':>13}{'E_rel_near':>13}{'NaN':>6}")
    print("-" * 72)
    for name, _ in CONFIGS:
        r = all_results[name]
        if "error" in r:
            print(f"{name:<16}ERROR: {r['error']}")
        else:
            print(f"{name:<16}{r['E_inf_far']:>13.3e}{r['E_inf_near']:>13.3e}"
                  f"{r['E_rel_far']:>13.3e}{r['E_rel_near']:>13.3e}"
                  f"{r['n_nan']:>6d}")
    print("=" * 72)

    # --- Save metrics for downstream plotting ---
    metrics_to_save = {
        "tau_star": tau_star,
        **{f"{k}_E_inf_far":  v["E_inf_far"]  for k, v in all_results.items() if "error" not in v},
        **{f"{k}_E_inf_near": v["E_inf_near"] for k, v in all_results.items() if "error" not in v},
        **{f"{k}_E_rel_far":  v["E_rel_far"]  for k, v in all_results.items() if "error" not in v},
        **{f"{k}_E_rel_near": v["E_rel_near"] for k, v in all_results.items() if "error" not in v},
    }
    np.savez(RESULTS / "ablation_metrics.npz", **metrics_to_save)
    print(f"[saved] {RESULTS / 'ablation_metrics.npz'}")


if __name__ == "__main__":
    main()