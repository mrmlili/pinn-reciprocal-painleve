"""
Fair comparison: DOP853 vs PINN+reciprocal for P-II (alpha=1).

Three axes:
  1. Single-solve accuracy and time
  2. Near-pole behavior (max t reachable)
  3. Generalization: PINN trained at alpha=1, evaluated at alpha=0.9/1.1
     (no retraining)

Also: reference accuracy vs DOP853 with different tolerances
to establish what "reachable accuracy" means.
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from painleve2 import reference_p2, transform_to_y_p2
from pinn_solver import PINN

RESULTS = ROOT / "results"


def solve_dop853(alpha, rtol, atol, t_end, y0=1.0, v0=0.0):
    """Solve P-II with DOP853; return time and solution object."""
    def rhs(t, s):
        y, v = s
        return [v, 2.0 * y**3 + t * y + alpha]
    def ev(t, s): return s[0] - 1e6
    ev.terminal = True; ev.direction = 1
    t0 = time.perf_counter()
    sol = solve_ivp(rhs, [0, t_end], [y0, v0], method="DOP853",
                    rtol=rtol, atol=atol, events=ev, dense_output=True)
    elapsed = time.perf_counter() - t0
    tau_star = float(sol.t_events[0][0]) if sol.t_events[0] else np.inf
    return sol, tau_star, elapsed


def time_dop853(tolerances, alpha=1.0, n_reps=5):
    """Measure DOP853 wall time for various tolerances."""
    print(f"\n{'='*72}")
    print(f"DOP853 single-solve timing (alpha={alpha}, 5 reps)")
    print(f"{'='*72}")
    print(f"{'rtol=atol':>12} {'mean ms':>10} {'std ms':>10} {'tau_star':>10} {'reached':>10}")
    print("-" * 72)
    for tol in tolerances:
        times = []
        tau_last = None
        reached_last = None
        for _ in range(n_reps):
            sol, ts, el = solve_dop853(alpha, tol, tol, t_end=10.0)
            times.append(el * 1000)
            tau_last = ts
            reached_last = sol.t[-1]
        print(f"{tol:>12.0e} {np.mean(times):>10.2f} {np.std(times):>10.2f} "
              f"{tau_last:>10.6f} {reached_last:>10.4f}")


def time_pinn(seed=42, alpha=1.0):
    """Measure PINN training time and inference time."""
    print(f"\n{'='*72}")
    print(f"PINN+reciprocal timing (seed={seed}, alpha={alpha})")
    print(f"{'='*72}")
    # Load pre-trained net
    net_path = RESULTS / f"net_P2_stable_seed{seed}.pt"
    if not net_path.exists():
        print(f"  [net not found at {net_path}]")
        return None, None

    net = PINN(64, 4)
    net.load_state_dict(torch.load(net_path, weights_only=True))
    net.eval()

    # Inference time (single forward pass on 2000 points)
    t_grid = torch.linspace(0, 1.099, 2000).reshape(-1, 1)
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            _ = net(t_grid)
    infer_time = (time.perf_counter() - t0) / 10 * 1000  # ms per call

    print(f"  Inference (2000 pts): {infer_time:.2f} ms")
    print(f"  Training time: ~{30*30/60:.0f} min (30k epochs, 30 min from log)")
    return infer_time, 30 * 60 * 1000  # approx training ms


def test_generalization(seed=42):
    """PINN trained at alpha=1; evaluate at alpha=0.9, 1.0, 1.1 without retraining."""
    print(f"\n{'='*72}")
    print(f"Generalization test: PINN (trained alpha=1) -> alpha=0.9, 1.0, 1.1")
    print(f"{'='*72}")

    net = PINN(64, 4)
    net.load_state_dict(torch.load(RESULTS / f"net_P2_stable_seed{seed}.pt",
                                   weights_only=True))
    net.eval()

    print(f"{'alpha_test':>12} {'tau_star':>10} {'E_rel_far':>12} {'E_rel_near':>12} {'status':>12}")
    print("-" * 72)
    for alpha_test in [0.9, 1.0, 1.1]:
        sol, tau_star = reference_p2(alpha=alpha_test)
        t_max = 0.95 * tau_star
        t_grid = np.linspace(0, t_max, 2000)
        tt = torch.tensor(t_grid, dtype=torch.float32).reshape(-1, 1)
        y_ref = sol.sol(t_grid)[0]

        with torch.no_grad():
            w = net(tt).numpy().flatten()
        y = 1.0 / np.maximum(w, 1e-8)
        E = np.abs(y - y_ref) / np.maximum(np.abs(y_ref), 1e-12)

        mf = t_grid < 0.5 * tau_star
        mn = ~mf
        E_far = float(E[mf].max())
        E_near = float(E[mn].max())
        status = "OK" if E_near < 0.1 else ("SOS" if E_near < 1.0 else "FAIL")
        print(f"{alpha_test:>12.2f} {tau_star:>10.6f} {E_far:>12.3e} "
              f"{E_near:>12.3e} {status:>12}")


def near_pole_behavior(seed=42):
    """How far can PINN reach vs DOP853 before divergence?"""
    print(f"\n{'='*72}")
    print(f"Near-pole behavior: how far can each method reach?")
    print(f"{'='*72}")

    sol, tau_star = reference_p2(alpha=1.0)

    # DOP853 with tight tol
    print(f"tau_star = {tau_star:.6f}")
    print(f"\nDOP853 (rtol=1e-12):")
    sol_d, ts_d, _ = solve_dop853(1.0, 1e-12, 1e-12, t_end=10.0)
    print(f"  Reached t = {sol_d.t[-1]:.6f} ({(sol_d.t[-1]/ts_d)*100:.2f}% of t*)")

    print(f"\nDOP853 (rtol=1e-6):")
    sol_d6, ts_d6, _ = solve_dop853(1.0, 1e-6, 1e-6, t_end=10.0)
    print(f"  Reached t = {sol_d6.t[-1]:.6f} ({(sol_d6.t[-1]/ts_d6)*100:.2f}% of t*)")

    print(f"\nPINN+reciprocal (seed={seed}):")
    net = PINN(64, 4)
    net.load_state_dict(torch.load(RESULTS / f"net_P2_stable_seed{seed}.pt",
                                   weights_only=True))
    net.eval()
    t_test = np.linspace(0, 0.99 * tau_star, 1000)
    tt = torch.tensor(t_test, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        w = net(tt).numpy().flatten()
    # Check where w stays positive (valid)
    w_valid = w > 1e-8
    if w_valid.any():
        last_valid_idx = np.where(w_valid)[0][-1]
        last_valid_t = t_test[last_valid_idx]
        print(f"  w>0 up to t = {last_valid_t:.6f} "
              f"({(last_valid_t/tau_star)*100:.2f}% of t*)")
    print(f"  w<=0 count: {(~w_valid).sum()}/{len(w)}")


if __name__ == "__main__":
    print("=" * 72)
    print("FAIR COMPARISON: DOP853 vs PINN+reciprocal")
    print("=" * 72)

    # 1. DOP853 timing
    time_dop853([1e-6, 1e-8, 1e-10, 1e-12], alpha=1.0, n_reps=5)

    # 2. PINN timing
    time_pinn(seed=42, alpha=1.0)

    # 3. Generalization
    test_generalization(seed=42)

    # 4. Near-pole behavior
    near_pole_behavior(seed=42)

    print(f"\n{'='*72}")
    print("Done. Summary for paper section 4.5 'Baseline comparison'")
    print(f"{'='*72}")
