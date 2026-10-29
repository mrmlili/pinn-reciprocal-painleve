"""Diagnostic: seed=7 with 30000 epochs."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from painleve2 import reference_p2, RESIDUALS_P2, ic_loss_p2, transform_to_y_p2
from pinn_solver import PINN, build_collocation_points

def train_and_eval(seed, n_epochs, log_every=5000):
    torch.manual_seed(seed)
    sol, tau_star = reference_p2(alpha=1.0)
    T_max = 0.95 * tau_star
    tg = np.linspace(0, 0.99 * tau_star, 2001)
    rt = torch.tensor(tg, dtype=torch.float32)
    ry = torch.tensor(sol.sol(tg)[0], dtype=torch.float32)

    net = PINN(64, 4)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2000, factor=0.5)
    t_colloc = build_collocation_points(T_max, 2000)
    residual_fn = RESIDUALS_P2["reciprocal"]

    for epoch in range(n_epochs):
        opt.zero_grad()
        res = residual_fn(net, t_colloc, alpha=1.0)
        loss_pde = (res**2).mean()
        t0 = torch.zeros(1, 1, requires_grad=True)
        loss_ic = ic_loss_p2("reciprocal", t0, net, alpha=1.0)
        loss = loss_pde + 10.0 * loss_ic
        loss.backward(); opt.step(); sch.step(loss.detach())
        if epoch % log_every == 0:
            print(f"  ep {epoch:5d} | loss={loss.item():.3e} | pde={loss_pde.item():.3e}")

    mask = rt <= T_max
    te = rt[mask].reshape(-1, 1)
    yr = ry[mask].numpy().flatten()
    with torch.no_grad():
        yp = transform_to_y_p2(net(te), "reciprocal").numpy().flatten()
    tf = te.numpy().flatten()
    mf = tf < 0.5 * tau_star; mn = ~mf
    ae = np.abs(yp - yr); re = ae / np.maximum(np.abs(yr), 1e-12)
    return float(re[mf].max()), float(re[mn].max())

if __name__ == "__main__":
    print("=== seed=7, 30000 epochs ===")
    far, near = train_and_eval(7, 30000)
    print(f"  E_rel_far={far:.3e}, E_rel_near={near:.3e}")
    print(f"  Improvement vs 15000 ep (3.57e-02): {3.57e-2/near:.2f}x")
