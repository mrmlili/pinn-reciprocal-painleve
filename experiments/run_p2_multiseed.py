"""Multi-seed for P-II D2 (reciprocal), alpha=1."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from painleve2 import reference_p2, RESIDUALS_P2, ic_loss_p2, transform_to_y_p2
from pinn_solver import PINN, build_collocation_points

RESULTS = ROOT / "results"

def train_p2(mode, tau_star, alpha, seed, n_epochs=15000, n_collocation=2000,
             lr=1e-3, width=64, depth=4, T_frac=0.95, ic_weight=10.0):
    torch.manual_seed(seed)
    T_max = T_frac * tau_star
    net = PINN(width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2000, factor=0.5)
    t_colloc = build_collocation_points(T_max, n_collocation)
    residual_fn = RESIDUALS_P2[mode]
    for epoch in range(n_epochs):
        opt.zero_grad()
        res = residual_fn(net, t_colloc, alpha=alpha)
        loss_pde = (res**2).mean()
        t0 = torch.zeros(1, 1, requires_grad=True)
        loss_ic = ic_loss_p2(mode, t0, net, alpha=alpha)
        loss = loss_pde + ic_weight * loss_ic
        loss.backward(); opt.step(); sch.step(loss.detach())
    return net

def eval_p2(net, mode, tau_star, ref_t, ref_y, T_frac=0.95):
    T_max = T_frac * tau_star
    mask = ref_t <= T_max
    t_eval = ref_t[mask].reshape(-1, 1)
    y_ref = ref_y[mask].numpy().flatten()
    with torch.no_grad():
        y_pred = transform_to_y_p2(net(t_eval), mode).numpy().flatten()
    t_flat = t_eval.numpy().flatten()
    mf = t_flat < 0.5 * tau_star; mn = ~mf
    ae = np.abs(y_pred - y_ref)
    re = ae / np.maximum(np.abs(y_ref), 1e-12)
    return {"E_rel_far": float(re[mf].max()), "E_rel_near": float(re[mn].max())}

if __name__ == "__main__":
    sol, tau_star = reference_p2(alpha=1.0)
    tg = np.linspace(0, 0.99 * tau_star, 2001)
    rt = torch.tensor(tg, dtype=torch.float32)
    ry = torch.tensor(sol.sol(tg)[0], dtype=torch.float32)
    for seed in [42, 7, 123]:
        print(f"\n--- seed={seed} ---")
        net = train_p2("reciprocal", tau_star, 1.0, seed)
        m = eval_p2(net, "reciprocal", tau_star, rt, ry)
        print(f"  E_rel_near={m['E_rel_near']:.3e}, E_rel_far={m['E_rel_far']:.3e}")
        torch.save(net.state_dict(), RESULTS / f"net_P2_D2_seed{seed}.pt")
