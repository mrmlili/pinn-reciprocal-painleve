"""
Alpha sweep for Painleve II: y'' = 2y^3 + t y + alpha

Tests alpha in {0, 1, 2} for each representation (direct, rapidity, reciprocal).

Usage:
    python experiments/run_p2_alpha_sweep.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from painleve2 import (reference_p2, RESIDUALS_P2, ic_loss_p2,
                       transform_to_y_p2)
from pinn_solver import PINN, build_collocation_points

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def train_p2(mode, tau_star, alpha, n_epochs=15000, n_collocation=2000,
             lr=1e-3, width=64, depth=4, T_frac=0.95, ic_weight=10.0,
             seed=42, log_every=5000):
    torch.manual_seed(seed)
    T_max = T_frac * tau_star
    net = PINN(width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2000,
                                                     factor=0.5)
    residual_fn = RESIDUALS_P2[mode]
    t_colloc = build_collocation_points(T_max, n_collocation)

    for epoch in range(n_epochs):
        opt.zero_grad()
        res = residual_fn(net, t_colloc, alpha=alpha)
        loss_pde = (res**2).mean()
        t0 = torch.zeros(1, 1, requires_grad=True)
        loss_ic = ic_loss_p2(mode, t0, net, alpha=alpha)
        loss = loss_pde + ic_weight * loss_ic
        loss.backward()
        opt.step()
        sch.step(loss.detach())
        if epoch % log_every == 0:
            print(f"  [{mode} a={alpha}] ep {epoch:5d} | "
                  f"loss={loss.item():.3e} | pde={loss_pde.item():.3e}")
    return net


def evaluate_p2(net, mode, tau_star, ref_t, ref_y, T_frac=0.95):
    T_max = T_frac * tau_star
    mask = ref_t <= T_max
    t_eval = ref_t[mask].reshape(-1, 1)
    y_ref = ref_y[mask].numpy().flatten()

    with torch.no_grad():
        raw = net(t_eval)
        y_pred = transform_to_y_p2(raw, mode).numpy().flatten()

    t_flat = t_eval.numpy().flatten()
    mask_far = t_flat < 0.5 * tau_star
    mask_near = ~mask_far

    abs_err = np.abs(y_pred - y_ref)
    rel_err = abs_err / np.maximum(np.abs(y_ref), 1e-12)

    return {
        "E_rel_far": float(rel_err[mask_far].max()),
        "E_rel_near": float(rel_err[mask_near].max()),
        "n_nan": int((~np.isfinite(y_pred)).sum()),
    }


if __name__ == "__main__":
    alphas = [0.0, 1.0, 2.0]
    modes = [("D0", "direct"), ("D1", "rapidity"), ("D2", "reciprocal")]
    results = {}

    for alpha in alphas:
        print(f"\n{'#'*60}\nALPHA = {alpha}\n{'#'*60}")
        sol, tau_star = reference_p2(alpha=alpha)
        print(f"[alpha={alpha}] tau_star = {tau_star:.6f}")

        t_grid = np.linspace(0, 0.99 * tau_star, 2001)
        ref_t = torch.tensor(t_grid, dtype=torch.float32)
        ref_y = torch.tensor(sol.sol(t_grid)[0], dtype=torch.float32)

        for name, mode in modes:
            print(f"\n--- {name} (alpha={alpha}) ---")
            net = train_p2(mode, tau_star, alpha, n_epochs=15000, seed=42)
            m = evaluate_p2(net, mode, tau_star, ref_t, ref_y)
            m["tau_star"] = tau_star
            results[(alpha, name)] = m
            torch.save(net.state_dict(),
                       RESULTS / f"net_P2_{name}_alpha{alpha}.pt")
            print(f"  -> E_rel_near = {m['E_rel_near']:.3e}  |  "
                  f"E_rel_far = {m['E_rel_far']:.3e}")

    print(f"\n{'='*75}")
    print(f"{'Alpha':<8}{'D0 E_rel_near':>16}{'D1 E_rel_near':>16}"
          f"{'D2 E_rel_near':>16}{'D2/D0 ratio':>14}")
    print("-" * 75)
    for alpha in alphas:
        d0 = results[(alpha, "D0")]["E_rel_near"]
        d1 = results[(alpha, "D1")]["E_rel_near"]
        d2 = results[(alpha, "D2")]["E_rel_near"]
        ratio = d0 / d2 if d2 > 0 else float('inf')
        print(f"{alpha:<8}{d0:>16.3e}{d1:>16.3e}{d2:>16.3e}{ratio:>14.1f}x")
    print("=" * 75)

    save_dict = {}
    for (alpha, name), m in results.items():
        for k, v in m.items():
            save_dict[f"a{alpha}_{name}_{k}"] = v
    np.savez(RESULTS / "p2_alpha_sweep.npz", **save_dict)
    print(f"[saved] {RESULTS / 'p2_alpha_sweep.npz'}")
