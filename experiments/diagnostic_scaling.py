"""
Diagnostic: is near-pole error a lawful power-law of (t* - t)?

Fits log(E_rel) = γ · log(t* - t) + log(C) for each seed in the near-pole
region t ∈ [0.5·t*, 0.95·t*].

Decision gate:
  KEEP  (Path A, Q1): γ consistent (std < 0.4) AND mean(R²) > 0.7
  REJECT (Path B, Q2): γ noisy (std > 0.7) OR mean(R²) < 0.5
  MODIFY (investigate): otherwise
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from painleve2 import reference_p2
from pinn_solver import PINN

RESULTS = ROOT / "results"


def main():
    sol, tau_star = reference_p2(alpha=1.0)
    seeds = [42, 7, 123, 2024, 999]

    t_grid = np.linspace(0.5 * tau_star, 0.95 * tau_star, 2000)
    t_tensor = torch.tensor(t_grid, dtype=torch.float32).reshape(-1, 1)
    y_ref = sol.sol(t_grid)[0]
    dist_to_pole = tau_star - t_grid  # positive since t < t*

    print(f"tau_star = {tau_star:.6f}")
    print(f"Region: t in [{0.5*tau_star:.4f}, {0.95*tau_star:.4f}] ({len(t_grid)} pts)")
    print()
    print(f"{'Seed':>6} {'R^2':>8} {'gamma':>8} {'log10(C)':>10} {'E_rel(0.95t*)':>14}")
    print("-" * 55)

    gammas = []
    r2s = []
    all_curves = {}

    for seed in seeds:
        net_path = RESULTS / f"net_P2_stable_seed{seed}.pt"
        if not net_path.exists():
            print(f"{seed:>6}  [net not found]")
            continue

        net = PINN(64, 4)
        net.load_state_dict(torch.load(net_path, weights_only=True))
        net.eval()

        with torch.no_grad():
            w_pred = net(t_tensor).numpy().flatten()

        w_safe = np.maximum(w_pred, 1e-8)
        y_pred = 1.0 / w_safe

        E_rel = np.abs(y_pred - y_ref) / np.maximum(np.abs(y_ref), 1e-12)

        mask = np.isfinite(E_rel) & (E_rel > 1e-12) & (dist_to_pole > 0)
        if mask.sum() < 10:
            print(f"{seed:>6}  [insufficient valid pts]")
            continue

        log_dist = np.log(dist_to_pole[mask])
        log_err = np.log(E_rel[mask])

        result = linregress(log_dist, log_err)
        gamma = float(result.slope)
        log_C = float(result.intercept)
        r2 = float(result.rvalue ** 2)

        gammas.append(gamma)
        r2s.append(r2)
        all_curves[seed] = {"dist": dist_to_pole, "E_rel": E_rel}

        print(f"{seed:>6} {r2:>8.3f} {gamma:>8.3f} {log_C:>10.3f} {E_rel[-1]:>14.3e}")

    print("-" * 55)

    if gammas:
        g_arr = np.array(gammas)
        r2_arr = np.array(r2s)
        print(f"gamma: mean={g_arr.mean():.3f}, std={g_arr.std():.3f}, "
              f"range=[{g_arr.min():.3f}, {g_arr.max():.3f}]")
        print(f"R^2:   mean={r2_arr.mean():.3f}, min={r2_arr.min():.3f}")

    # ---- Correlation with PDE residual (the "gap" itself) ----
    d = np.load(RESULTS / "p2_stable_results.npz")
    best_val = d["best_val"]
    E_near = d["E_rel_near"]
    if len(best_val) == len(E_near) >= 3:
        corr = float(np.corrcoef(best_val, E_near)[0, 1])
        print(f"\nCorrelation(residual best_val, E_rel_near) = {corr:.3f}")
        if abs(corr) < 0.5:
            print("  -> residual does NOT predict error (classic residual-error gap)")

    # ---- Decision ----
    print()
    print("=" * 55)
    if gammas:
        std_g = float(np.std(gammas))
        mean_r2 = float(np.mean(r2s))
        print(f"Decision criterion: std(gamma)={std_g:.3f}, mean(R²)={mean_r2:.3f}")
        if std_g < 0.4 and mean_r2 > 0.7:
            print("DECISION: KEEP -> Path A (Q1)")
            print("  Lawful scaling confirmed across seeds.")
        elif std_g > 0.7 or mean_r2 < 0.5:
            print("DECISION: REJECT -> Path B (Q2)")
            print("  No lawful scaling; error not predictable from |t-t*|.")
        else:
            print("DECISION: MODIFY (investigate further)")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for seed, c in all_curves.items():
            ax.loglog(c["dist"], c["E_rel"], label=f"seed={seed}", alpha=0.75)
        ax.set_xlabel(r"$t^* - t$")
        ax.set_ylabel(r"$E_{rel}(t)$")
        ax.set_title("Near-pole error scaling")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS / "diagnostic_scaling.png", dpi=150)
        print(f"\n[saved] {RESULTS / 'diagnostic_scaling.png'}")
    except Exception as e:
        print(f"[plot skipped] {e}")

    if gammas:
        np.savez(RESULTS / "diagnostic_scaling.npz",
                 gammas=np.array(gammas), r2s=np.array(r2s))
        print(f"[saved] {RESULTS / 'diagnostic_scaling.npz'}")


if __name__ == "__main__":
    main()
