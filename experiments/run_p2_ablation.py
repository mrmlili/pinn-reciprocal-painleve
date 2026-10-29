"""
Ablation for Painleve II: y'' = 2y^3 + t y + alpha, alpha=1.

Configurations:
  D0: direct PINN
  D1: rapidity (theta = log y)
  D2: reciprocal (w = 1/y)

Usage:
    python experiments/run_p2_ablation.py
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


def train_p2(mode: str, tau_star: float, n_epochs=15000,
             n_collocation=2000, lr=1e-3, width=64, depth=4,
             T_frac=0.95, ic_weight=10.0, seed=42, log_every=2000):
    torch.manual_seed(seed)
    T_max = T_frac * tau_star
    net = PINN(width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2000,
                                                     factor=0.5)
    residual_fn = RESIDUALS_P2[mode]
    t_colloc = build_collocation_points(T_max, n_collocation)

    hist = {"loss": [], "pde": [], "ic": []}
    for epoch in range(n_epochs):
        opt.zero_grad()
        res = residual_fn(net, t_colloc)
        loss_pde = (res**2).mean()
        t0 = torch.zeros(1, 1, requires_grad=True)
        loss_ic = ic_loss_p2(mode, t0, net)
        loss = loss_pde + ic_weight * loss_ic
        loss.backward()
        opt.step()
        sch.step(loss.detach())
        hist["loss"].append(loss.item())
        hist["pde"].append(loss_pde.item())
        hist["ic"].append(loss_ic.item())
        if epoch % log_every == 0:
            print(f"[{mode}] epoch {epoch:5d} | loss={loss.item():.3e} | "
                  f"pde={loss_pde.item():.3e} | ic={loss_ic.item():.3e}")
    return net, hist


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
        "E_inf_far": float(abs_err[mask_far].max()),
        "E_inf_near": float(abs_err[mask_near].max()),
        "E_rel_far": float(rel_err[mask_far].max()),
        "E_rel_near": float(rel_err[mask_near].max()),
        "n_nan": int((~np.isfinite(y_pred)).sum()),
    }


if __name__ == "__main__":
    print("[P-II] computing reference...")
    sol, tau_star = reference_p2()
    print(f"[P-II] tau_star = {tau_star:.6f}")

    t_grid = np.linspace(0, 0.99 * tau_star, 2001)
    ref_t = torch.tensor(t_grid, dtype=torch.float32)
    ref_y = torch.tensor(sol.sol(t_grid)[0], dtype=torch.float32)

    configs = [("D0", "direct"), ("D1", "rapidity"), ("D2", "reciprocal")]
    results = {}
    for name, mode in configs:
        print(f"\n{'='*60}\n{name} ({mode})\n{'='*60}")
        net, _ = train_p2(mode, tau_star, n_epochs=15000, seed=42)
        m = evaluate_p2(net, mode, tau_star, ref_t, ref_y)
        results[name] = m
        print(f"  E_rel_near = {m['E_rel_near']:.3e}  |  "
              f"E_rel_far = {m['E_rel_far']:.3e}  |  NaN = {m['n_nan']}")
        torch.save(net.state_dict(), RESULTS / f"net_P2_{name}.pt")

    print(f"\n{'='*60}\nSUMMARY (P-II, alpha=1)\n{'='*60}")
    for name, _ in configs:
        r = results[name]
        print(f"{name}: E_rel_near = {r['E_rel_near']:.3e}  |  "
              f"E_rel_far = {r['E_rel_far']:.3e}")

    np.savez(RESULTS / "p2_ablation.npz", tau_star=tau_star,
             **{f"{k}_{mk}": v[mk]
                for k, v in results.items()
                for mk in ("E_inf_far","E_inf_near","E_rel_far","E_rel_near")})
    print(f"[saved] {RESULTS / 'p2_ablation.npz'}")
