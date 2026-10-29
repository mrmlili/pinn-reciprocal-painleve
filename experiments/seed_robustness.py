"""
Multi-seed robustness test for C4_reciprocal.
Runs C4 with 3 different seeds and reports mean/std of E_rel_near.

Usage:
    python experiments/seed_robustness.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pinn_solver import train_pinn, evaluate

RESULTS = ROOT / "results"


def main():
    d = np.load(RESULTS / "reference.npz")
    tau_star = float(d["tau_star"])
    ref_t = torch.tensor(d["t_grid"], dtype=torch.float32)
    ref_y = torch.tensor(d["y_ref"], dtype=torch.float32)

    seeds = [42, 7, 123]
    results = []
    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        net, _ = train_pinn(mode="reciprocal", tau_star=tau_star,
                            n_epochs=15000, seed=seed)
        m = evaluate(net, "reciprocal", tau_star, ref_t, ref_y)
        results.append({"seed": seed, **m})
        torch.save(net.state_dict(), RESULTS / f"net_C4_seed{seed}.pt")

    E_near = [r["E_rel_near"] for r in results]
    E_far  = [r["E_rel_far"]  for r in results]
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for r in results:
        print(f"seed={r['seed']:4d}  E_rel_far={r['E_rel_far']:.3e}  "
              f"E_rel_near={r['E_rel_near']:.3e}")
    print(f"\nE_rel_near: mean={np.mean(E_near):.3e}, "
          f"std={np.std(E_near):.3e}, min={np.min(E_near):.3e}, "
          f"max={np.max(E_near):.3e}")
    print(f"E_rel_far : mean={np.mean(E_far):.3e}, "
          f"std={np.std(E_far):.3e}")

    np.savez(RESULTS / "seed_robustness.npz",
             seeds=np.array(seeds),
             E_rel_near=np.array(E_near),
             E_rel_far=np.array(E_far))


if __name__ == "__main__":
    main()
