"""
Stable training for P-II D2 (reciprocal) with:
  - best-model checkpointing (validation on held-out PDE points)
  - improved collocation (cubic clustering near pole)
  - multi-seed (42, 7, 123, 2024, 999)

Usage:
    python experiments/run_p2_stable.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from painleve2 import reference_p2, RESIDUALS_P2, ic_loss_p2, transform_to_y_p2
from pinn_solver import PINN

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def build_collocation_v3(T_max, n_points):
    """Three-way split: uniform, quadratic, cubic clustering near T_max."""
    n_u = n_points // 3
    n_q = n_points // 3
    n_c = n_points - n_u - n_q
    t_u = torch.linspace(0, T_max, n_u)
    u = torch.rand(n_q); t_q = T_max * (1 - (1 - u)**2)
    u = torch.rand(n_c); t_c = T_max * (1 - (1 - u)**4)
    return torch.cat([t_u, t_q, t_c]).sort().values.reshape(-1, 1)


def train_stable(mode, tau_star, alpha, seed, n_epochs=30000,
                 n_collocation=2000, n_val=300, lr=1e-3,
                 width=64, depth=4, T_frac=0.95, ic_weight=10.0,
                 val_every=100, log_every=5000):
    """Train with best-model checkpointing on held-out PDE points."""
    torch.manual_seed(seed)
    T_max = T_frac * tau_star

    net = PINN(width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=2000, factor=0.5)

    residual_fn = RESIDUALS_P2[mode]

    # Training points (collocation)
    t_colloc = build_collocation_v3(T_max, n_collocation)
    # Validation points (held out, fixed for the run)
    torch.manual_seed(seed + 1000)
    t_val = build_collocation_v3(T_max, n_val)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    val_history = []

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

        if epoch % val_every == 0 and epoch > 0:
            # NOTE: no torch.no_grad() here — residual_fn needs autograd.
            res_v = residual_fn(net, t_val, alpha=alpha)
            val_loss = (res_v**2).mean().item()
            val_history.append((epoch, val_loss))
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if epoch % log_every == 0:
            print(f"  seed={seed} ep {epoch:5d} | "
                  f"loss={loss.item():.3e} | pde={loss_pde.item():.3e} | "
                  f"best_val={best_val:.3e}")

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_val, best_epoch, val_history


def evaluate_p2(net, mode, tau_star, ref_t, ref_y, T_frac=0.95):
    T_max = T_frac * tau_star
    mask = ref_t <= T_max
    t_eval = ref_t[mask].reshape(-1, 1)
    y_ref = ref_y[mask].numpy().flatten()

    with torch.no_grad():
        raw = net(t_eval)
        y_pred = transform_to_y_p2(raw, mode).numpy().flatten()

    t_flat = t_eval.numpy().flatten()
    mf = t_flat < 0.5 * tau_star
    mn = ~mf
    ae = np.abs(y_pred - y_ref)
    re = ae / np.maximum(np.abs(y_ref), 1e-12)
    return {
        "E_rel_far": float(re[mf].max()),
        "E_rel_near": float(re[mn].max()),
        "n_nan": int((~np.isfinite(y_pred)).sum()),
    }


if __name__ == "__main__":
    sol, tau_star = reference_p2(alpha=1.0)
    print(f"[P-II, alpha=1] tau_star = {tau_star:.6f}")

    tg = np.linspace(0, 0.99 * tau_star, 2001)
    rt = torch.tensor(tg, dtype=torch.float32)
    ry = torch.tensor(sol.sol(tg)[0], dtype=torch.float32)

    seeds = [42, 7, 123, 2024, 999]
    results = []
    for seed in seeds:
        print(f"\n{'='*60}\nSEED {seed} (best-model checkpointing)\n{'='*60}")
        net, bv, be, vh = train_stable(
            "reciprocal", tau_star, 1.0, seed, n_epochs=30000,
            width=64, depth=4)
        m = evaluate_p2(net, "reciprocal", tau_star, rt, ry)
        print(f"  best_val={bv:.3e} @ ep={be}")
        print(f"  E_rel_near={m['E_rel_near']:.3e}, E_rel_far={m['E_rel_far']:.3e}")
        results.append({"seed": seed, **m, "best_val": bv, "best_epoch": be})
        torch.save(net.state_dict(), RESULTS / f"net_P2_stable_seed{seed}.pt")

    # ---- Statistics ----
    print(f"\n{'='*72}")
    print(f"{'Seed':<8}{'best_ep':>10}{'best_val':>14}{'E_rel_far':>14}{'E_rel_near':>14}")
    print("-" * 72)
    for r in results:
        print(f"{r['seed']:<8}{r['best_epoch']:>10}{r['best_val']:>14.3e}"
              f"{r['E_rel_far']:>14.3e}{r['E_rel_near']:>14.3e}")
    E = [r["E_rel_near"] for r in results]
    print(f"\nE_rel_near: mean={np.mean(E):.3e}  median={np.median(E):.3e}")
    print(f"            std={np.std(E):.3e}  CV={np.std(E)/np.mean(E):.3f}")
    print(f"            min={np.min(E):.3e}  max={np.max(E):.3e}")

    np.savez(RESULTS / "p2_stable_results.npz",
             seeds=np.array([r["seed"] for r in results]),
             E_rel_near=np.array(E),
             E_rel_far=np.array([r["E_rel_far"] for r in results]),
             best_val=np.array([r["best_val"] for r in results]),
             best_epoch=np.array([r["best_epoch"] for r in results]))
    print(f"\n[saved] {RESULTS / 'p2_stable_results.npz'}")
