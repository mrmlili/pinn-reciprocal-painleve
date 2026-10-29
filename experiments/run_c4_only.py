"""
Quick test: run ONLY C4_reciprocal to check if w=1/y fixes the pole.

If E_rel_near < 0.1: proceed to full ablation.
If E_rel_near > 0.3: REJECT the PINN direction, pivot to alternatives.

Usage:
    python experiments/run_c4_only.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reference_dop853 import compute_reference
from pinn_solver import train_pinn, evaluate

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


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

    # --- Train C4 only ---
    print(f"\n{'='*60}\nC4_reciprocal (w = 1/y)\n{'='*60}")
    net, hist = train_pinn(mode="reciprocal", tau_star=tau_star,
                           n_epochs=15000, seed=42)
    metrics = evaluate(net, "reciprocal", tau_star, ref_t, ref_y)

    print(f"\n{'='*60}")
    print("C4_reciprocal results:")
    print(f"  E_inf_far   = {metrics['E_inf_far']:.3e}")
    print(f"  E_inf_near  = {metrics['E_inf_near']:.3e}")
    print(f"  E_rel_far   = {metrics['E_rel_far']:.3e}")
    print(f"  E_rel_near  = {metrics['E_rel_near']:.3e}")
    print(f"  NaN count   = {metrics['n_nan']}")
    print(f"  w<=0 count  = {metrics['n_negative_w']}")

    # --- Decision ---
    print(f"\n{'='*60}")
    print("DECISION:")
    if metrics['E_rel_near'] < 0.1:
        print("  PASS: E_rel_near < 0.1 -> run full ablation (C0, C2, C4)")
    elif metrics['E_rel_near'] < 0.3:
        print("  PARTIAL: 0.1 <= E_rel_near < 0.3 -> investigate further "
              "(maybe more epochs, wider net, or better architecture)")
    else:
        print("  FAIL: E_rel_near >= 0.3 -> REJECT PINN direction, "
              "pivot to Neural Operator / classical+ML hybrid")
    print(f"{'='*60}")

    # --- Save ---
    np.savez(RESULTS / "c4_only.npz", **metrics)
    torch.save(net.state_dict(), RESULTS / "net_C4_reciprocal.pt")


if __name__ == "__main__":
    main()