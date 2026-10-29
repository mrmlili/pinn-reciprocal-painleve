"""
Generate all publication figures for the Q2 paper.
Reads saved .npz files; no retraining.

Outputs (all in results/figures/):
  fig1_solutions.pdf      - y(t) for prototype and P-II, three representations
  fig2_error_curves.pdf   - log-scale error vs t for C0/C2/C4 and D0/D1/D2
  fig3_alpha_sweep.pdf    - D0/D1/D2 across alpha=0,1,2
  fig4_seed_stats.pdf     - boxplot of E_rel_near across 5 seeds
  fig5_near_pole.pdf      - log-log E_rel vs (t*-t), gamma fit
  fig6_generalization.pdf - PINN at alpha=0.9/1.0/1.1
  fig7_vs_dop853.pdf      - timing-accuracy Pareto plot
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(exist_ok=True)

# Publication-quality defaults
rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

sys.path.insert(0, str(ROOT / "src"))
from painleve2 import reference_p2, transform_to_y_p2
from pinn_solver import PINN


def safe_load(name):
    p = RESULTS / name
    if not p.exists():
        print(f"[skip] {name} not found")
        return None
    return np.load(p)


# ---------------- Fig 1: Solutions ----------------
def fig1():
    from scipy.integrate import solve_ivp

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: y''=y^3
    def rhs1(t, s): return [s[1], s[0]**3]
    def ev(t, s): return s[0] - 1e6
    ev.terminal = True
    sol1 = solve_ivp(rhs1, [0, 10], [1.0, 0.0], method="DOP853",
                     rtol=1e-12, atol=1e-12, events=ev, dense_output=True)
    t1 = np.linspace(0, 0.95 * sol1.t_events[0][0], 500)
    y1 = sol1.sol(t1)[0]
    axes[0].plot(t1, y1, "k-", lw=2, label="reference")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("y(t)")
    axes[0].set_title(r"$y'' = y^3$, $y(0)=1$, $y'(0)=0$")
    axes[0].legend()

    # Right: P-II
    sol2, tstar = reference_p2(alpha=1.0)
    t2 = np.linspace(0, 0.95 * tstar, 500)
    y2 = sol2.sol(t2)[0]
    axes[1].plot(t2, y2, "k-", lw=2, label="reference")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("y(t)")
    axes[1].set_title(r"P-II: $y''=2y^3+ty+1$, $y(0)=1$, $y'(0)=0$")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGDIR / "fig1_solutions.pdf")
    plt.close()
    print("[saved] fig1_solutions.pdf")


# ---------------- Fig 2: Error curves ----------------
def fig2():
    """Error vs t for each representation. Needs loaded networks."""
    # We'll use saved curves from diagnostic outputs if available,
    # otherwise skip gracefully.
    # For prototype, we don't have t-series saved; load nets if present.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # y''=y^3: net_C0, net_C1, net_C2, net_C3 from old run
    from substitutions import RESIDUALS, transform_to_y
    from scipy.integrate import solve_ivp

    def rhs1(t, s): return [s[1], s[0]**3]
    def ev(t, s): return s[0] - 1e6
    ev.terminal = True
    sol1 = solve_ivp(rhs1, [0, 10], [1.0, 0.0], method="DOP853",
                     rtol=1e-12, atol=1e-12, events=ev, dense_output=True)
    ts1 = sol1.t_events[0][0]
    t1 = np.linspace(0, 0.95 * ts1, 400)
    yr1 = sol1.sol(t1)[0]

    import torch
    configs1 = [("C0", "net_C0.pt", "direct"),
                ("C2_log", "net_C2.pt", "rapidity_log"),
                ("C4_recip", "net_C4_reciprocal.pt", "reciprocal")]
    for label, fname, mode in configs1:
        p = RESULTS / fname
        if not p.exists():
            continue
        net = PINN(64, 4)
        net.load_state_dict(torch.load(p, weights_only=True))
        net.eval()
        with torch.no_grad():
            tt = torch.tensor(t1, dtype=torch.float32).reshape(-1, 1)
            raw = net(tt)
            yp = transform_to_y(raw, mode).numpy().flatten()
        err = np.abs(yp - yr1)
        axes[0].semilogy(t1, np.maximum(err, 1e-16), label=label)
    axes[0].axvline(0.5 * ts1, color="gray", ls=":", label=r"$0.5 t^*$")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$|y - y_{ref}|$")
    axes[0].set_title(r"Prototype $y''=y^3$")
    axes[0].legend()

    # P-II
    sol2, ts2 = reference_p2(alpha=1.0)
    t2 = np.linspace(0, 0.95 * ts2, 400)
    yr2 = sol2.sol(t2)[0]
    for label, fname, mode in [("D0", "net_P2_D0.pt", "direct"),
                                ("D1", "net_P2_D1.pt", "rapidity"),
                                ("D2", "net_P2_stable_seed42.pt", "reciprocal")]:
        p = RESULTS / fname
        if not p.exists():
            continue
        net = PINN(64, 4)
        net.load_state_dict(torch.load(p, weights_only=True))
        net.eval()
        with torch.no_grad():
            tt = torch.tensor(t2, dtype=torch.float32).reshape(-1, 1)
            raw = net(tt)
            yp = transform_to_y_p2(raw, mode).numpy().flatten()
        err = np.abs(yp - yr2)
        axes[1].semilogy(t2, np.maximum(err, 1e-16), label=label)
    axes[1].axvline(0.5 * ts2, color="gray", ls=":", label=r"$0.5 t^*$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$|y - y_{ref}|$")
    axes[1].set_title("P-II, $\\alpha=1$")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGDIR / "fig2_error_curves.pdf")
    plt.close()
    print("[saved] fig2_error_curves.pdf")


# ---------------- Fig 3: Alpha sweep ----------------
def fig3():
    d = safe_load("p2_alpha_sweep.npz")
    if d is None:
        return
    alphas = [0.0, 1.0, 2.0]
    d0 = [float(d[f"a{a}_D0_E_rel_near"]) for a in alphas]
    d1 = [float(d[f"a{a}_D1_E_rel_near"]) for a in alphas]
    d2 = [float(d[f"a{a}_D2_E_rel_near"]) for a in alphas]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(alphas))
    w = 0.25
    ax.bar(x - w, d0, w, label="D0 (direct)", color="#c0392b")
    ax.bar(x, d1, w, label="D1 (rapidity)", color="#f39c12")
    ax.bar(x + w, d2, w, label="D2 (reciprocal)", color="#27ae60")
    ax.set_yscale("log")
    ax.axhline(1e-3, color="black", ls="--", alpha=0.5,
               label=r"$10^{-3}$")
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\alpha={a}$" for a in alphas])
    ax.set_ylabel(r"$E_{\mathrm{rel,near}}$")
    ax.set_title("Painlevé II: error across $\\alpha$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig3_alpha_sweep.pdf")
    plt.close()
    print("[saved] fig3_alpha_sweep.pdf")


# ---------------- Fig 4: Seed statistics ----------------
def fig4():
    d1 = safe_load("p2_stable_results.npz")
    d2 = safe_load("seed_robustness.npz")
    if d1 is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if d2 is not None:
        axes[0].boxplot([d2["E_rel_near"]], tick_labels=["C4 (prototype)"])
        axes[0].set_yscale("log")
        axes[0].set_ylabel(r"$E_{\mathrm{rel,near}}$")
        axes[0].set_title("Prototype ($y''=y^3$), 3 seeds")

    axes[1].boxplot([d1["E_rel_near"]], tick_labels=["D2 (P-II)"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$E_{\mathrm{rel,near}}$")
    axes[1].set_title("P-II ($\\alpha=1$), 5 seeds")

    plt.tight_layout()
    plt.savefig(FIGDIR / "fig4_seed_stats.pdf")
    plt.close()
    print("[saved] fig4_seed_stats.pdf")


# ---------------- Fig 5: Near-pole scaling ----------------
def fig5():
    d = safe_load("diagnostic_scaling.npz")
    if d is None:
        return
    # We have gammas, r2s — plot as bar chart per seed
    gammas = d["gammas"]
    r2s = d["r2s"]
    seeds = [42, 7, 123, 2024, 999]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(seeds))
    ax.bar(x, gammas, color="#2980b9", label=r"fitted $\gamma$")
    ax.axhline(-1.0, color="red", ls="--", lw=2,
               label=r"theory $\gamma=-1$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"s={s}\n$R^2$={r:.2f}" for s, r in zip(seeds, r2s)])
    ax.set_ylabel(r"power-law exponent $\gamma$")
    ax.set_title("Near-pole scaling: fitted exponent vs theory")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig5_near_pole.pdf")
    plt.close()
    print("[saved] fig5_near_pole.pdf")


# ---------------- Fig 6: Generalization ----------------
def fig6():
    """Hardcoded from compare_dop853.log — reproduced in Results section."""
    alphas = [0.9, 1.0, 1.1]
    E_near = [1.77e-1, 2.39e-4, 1.29e-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#c0392b", "#27ae60", "#c0392b"]
    bars = ax.bar([str(a) for a in alphas], E_near, color=colors)
    ax.set_yscale("log")
    ax.axhline(1e-2, color="black", ls="--", label=r"$10^{-2}$ threshold")
    ax.set_xlabel(r"test $\alpha$ (PINN trained at $\alpha=1$)")
    ax.set_ylabel(r"$E_{\mathrm{rel,near}}$")
    ax.set_title("Generalization: PINN trained on $\\alpha=1$ only")
    for bar, v in zip(bars, E_near):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.3,
                f"{v:.1e}", ha="center", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig6_generalization.pdf")
    plt.close()
    print("[saved] fig6_generalization.pdf")


# ---------------- Fig 7: vs DOP853 ----------------
def fig7():
    """Pareto plot: accuracy vs cost."""
    # From compare_dop853.log (hardcoded)
    dop_tols = [1e-6, 1e-8, 1e-10, 1e-12]
    dop_times = [18.15, 26.68, 34.84, 46.17]  # ms
    dop_accs = [1e-6, 1e-8, 1e-10, 1e-12]

    pinn_infer_ms = 1.72
    pinn_acc = 2.39e-4  # seed=42

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(dop_times, dop_accs, "o-", label="DOP853", markersize=8)
    for t, a, tol in zip(dop_times, dop_accs, dop_tols):
        ax.annotate(f"{tol:.0e}", (t, a), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")
    ax.loglog([pinn_infer_ms], [pinn_acc], "s", markersize=12,
              color="#c0392b", label="PINN (inference)")
    ax.set_xlabel("wall time (ms) for one solve")
    ax.set_ylabel("accuracy (absolute error)")
    ax.set_title("Cost-accuracy: DOP853 vs PINN+reciprocal")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig7_vs_dop853.pdf")
    plt.close()
    print("[saved] fig7_vs_dop853.pdf")


if __name__ == "__main__":
    print(f"Output: {FIGDIR}\n")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    print("\nDone.")

