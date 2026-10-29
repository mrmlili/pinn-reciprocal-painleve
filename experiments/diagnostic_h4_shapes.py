"""
H4 (final attempt): do seeds 7 and 123 converge to QUALITATIVELY
different error shapes (not just different amplitudes)?

Plots E_rel(t) for all 5 seeds on log-log and log-linear axes.
Computes local slope in 5 windows to detect regime changes.
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

    t_grid = np.linspace(0.5 * tau_star, 0.98 * tau_star, 2000)
    tt = torch.tensor(t_grid, dtype=torch.float32).reshape(-1, 1)
    y_ref = sol.sol(t_grid)[0]
    dist = tau_star - t_grid

    # 5 windows of width 400
    n = len(t_grid)
    windows = [(i, i + n // 5) for i in range(0, n, n // 5)][:5]

    print(f"tau_star = {tau_star:.6f}")
    print(f"Windows: t ranges {[f'[{t_grid[a]:.3f},{t_grid[b-1]:.3f}]' for a,b in windows]}\n")

    print(f"{'Seed':>6} | " + " | ".join(f"slope W{i+1}" for i in range(5)))
    print("-" * 70)

    local_slopes = {}
    for seed in seeds:
        net = PINN(64, 4)
        net.load_state_dict(torch.load(RESULTS / f"net_P2_stable_seed{seed}.pt",
                                       weights_only=True))
        net.eval()
        with torch.no_grad():
            w = net(tt).numpy().flatten()
        y = 1.0 / np.maximum(w, 1e-8)
        E = np.abs(y - y_ref) / np.maximum(np.abs(y_ref), 1e-12)

        slopes = []
        for a, b in windows:
            d = dist[a:b]
            e = E[a:b]
            m = (e > 1e-14) & (d > 0)
            if m.sum() < 20:
                slopes.append(np.nan)
                continue
            ld = np.log(d[m])
            le = np.log(e[m])
            coef = np.polyfit(ld, le, 1)
            slopes.append(coef[0])

        local_slopes[seed] = slopes
        s = " | ".join(f"{v:>7.2f}" if np.isfinite(v) else "   n/a" for v in slopes)
        print(f"{seed:>6} | {s}")

    print("-" * 70)

    # Decision
    print("\nInterpretation:")
    print("  - slopes all similar across windows -> clean power law (good)")
    print("  - slopes change sharply between windows -> regime switching (H4)")
    print("  - slopes vary wildly -> no structure")

    print("\nPer-seed verdict:")
    for seed in seeds:
        s = np.array(local_slopes[seed])
        s = s[np.isfinite(s)]
        if len(s) < 2:
            print(f"  seed={seed}: insufficient data")
            continue
        rng = s.max() - s.min()
        if rng > 1.0:
            print(f"  seed={seed}: REGIME SHIFT (range={rng:.2f})")
        elif rng > 0.5:
            print(f"  seed={seed}: moderate variation (range={rng:.2f})")
        else:
            print(f"  seed={seed}: clean power law (range={rng:.2f})")

    # Global H4 test: seeds 7, 123 vs others
    bad_seeds = [7, 123]
    good_seeds = [42, 2024, 999]
    print("\n" + "=" * 70)
    print("H4 global test:")
    for label, group in [("GOOD (42,2024,999)", good_seeds),
                         ("BAD  (7,123)", bad_seeds)]:
        ranges = []
        means = []
        for seed in group:
            s = np.array(local_slopes[seed])
            s = s[np.isfinite(s)]
            ranges.append(s.max() - s.min())
            means.append(s.mean())
        print(f"  {label}: mean slope={np.mean(means):.2f}, "
              f"range(std)={np.mean(ranges):.2f}")

    # Save
    save = {f"seed{s}_slopes": np.array(local_slopes[s]) for s in seeds}
    np.savez(RESULTS / "h4_local_slopes.npz", **save)
    print(f"\n[saved] {RESULTS / 'h4_local_slopes.npz'}")


if __name__ == "__main__":
    main()
