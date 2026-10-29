"""
1-week MODIFY diagnostic: why do seeds 7 and 123 fail R^2?

H1: w-noise is high near pole -> E_rel dominated by noise
H2: E_rel(t) is non-monotonic -> two regimes
H3: seed 7 / 123 fall into different minima (different C)

Prints per-seed verdict. NO retraining.
"""

import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from painleve2 import reference_p2
from pinn_solver import PINN

RESULTS = ROOT / "results"


def main():
    sol, tau_star = reference_p2(alpha=1.0)
    seeds = [42, 7, 123, 2024, 999]

    # Dense grid to see local behavior
    t_grid = np.linspace(0.5 * tau_star, 0.98 * tau_star, 4000)
    tt = torch.tensor(t_grid, dtype=torch.float32).reshape(-1, 1)
    y_ref = sol.sol(t_grid)[0]
    dist = tau_star - t_grid

    print(f"tau_star = {tau_star:.6f}\n")

    for seed in seeds:
        net = PINN(64, 4)
        net.load_state_dict(torch.load(RESULTS / f"net_P2_stable_seed{seed}.pt",
                                       weights_only=True))
        net.eval()
        with torch.no_grad():
            w = net(tt).numpy().flatten()
        y = 1.0 / np.maximum(w, 1e-8)
        E = np.abs(y - y_ref) / np.maximum(np.abs(y_ref), 1e-12)

        # H2: monotonicity — does E_rel decrease monotonically as t increases?
        # near pole (t -> t*), E_rel should grow if gamma < 0
        dE = np.diff(E)
        frac_positive = float((dE > 0).mean())  # fraction of intervals where E grows
        # H1: noise — local variance of E in log-space
        logE = np.log(np.maximum(E, 1e-16))
        # rolling std with window 50
        win = 50
        rolling_std = np.array([logE[i:i+win].std() for i in range(len(logE)-win)])
        # H3: minimum location — compare C across seeds (via log10 mean of E)
        log10_C_eff = float(np.median(np.log10(E) + np.log10(dist)))  # E·dist approx C
        # smoothness of E_rel itself (noise level)
        E_smooth = np.convolve(E, np.ones(win)/win, mode="same")
        noise_ratio = float(np.std(E - E_smooth) / (np.mean(E) + 1e-16))

        print(f"seed={seed}")
        print(f"  H1 noise_ratio           = {noise_ratio:.3f}  "
              f"({'HIGH' if noise_ratio > 0.3 else 'low'})")
        print(f"  H2 frac(E increasing)    = {frac_positive:.3f}  "
              f"({'non-monotonic' if 0.3 < frac_positive < 0.7 else 'monotonic'})")
        print(f"  H3 log10(C_eff)          = {log10_C_eff:.3f}")
        print()


if __name__ == "__main__":
    main()
