"""P-II multi-seed: 10 seeds × 3 α × 3 representations = 90 runs.
Estimated: ~24 hours on CPU.
Saves to results/p2_multiseed_full.npz.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from painleve2 import reference_p2, RESIDUALS_P2, ic_loss_p2, transform_to_y_p2
from pinn_solver import PINN

RESULTS = ROOT / "results"
SEEDS = [42, 7, 123, 2024, 999, 1, 2, 3, 4, 5]
ALPHAS = [0.0, 1.0, 2.0]
MODES = [("D0", "direct"), ("D1", "rapidity"), ("D2", "reciprocal")]
N_EPOCHS = 30000

def build_colloc(T_max, n):
    n_u = n//3; n_q = n//3; n_c = n - n_u - n_q
    t_u = torch.linspace(0, T_max, n_u)
    u = torch.rand(n_q); t_q = T_max*(1-(1-u)**2)
    u = torch.rand(n_c); t_c = T_max*(1-(1-u)**4)
    return torch.cat([t_u, t_q, t_c]).sort().values.reshape(-1,1)

def train_eval(mode, seed, tau_star, ref_t, ref_y, alpha):
    torch.manual_seed(seed)
    T_max = 0.95 * tau_star
    net = PINN(64, 4)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2000, factor=0.5)
    res_fn = RESIDUALS_P2[mode]
    t_col = build_colloc(T_max, 2000)
    for ep in range(N_EPOCHS):
        opt.zero_grad()
        res = res_fn(net, t_col, alpha=alpha)
        loss_pde = (res**2).mean()
        t0 = torch.zeros(1, 1, requires_grad=True)
        loss_ic = ic_loss_p2(mode, t0, net, alpha=alpha)
        loss = loss_pde + 10.0 * loss_ic
        loss.backward(); opt.step(); sch.step(loss.detach())
    mask = ref_t <= T_max
    te = ref_t[mask].reshape(-1, 1)
    yr = ref_y[mask].numpy().flatten()
    with torch.no_grad():
        yp = transform_to_y_p2(net(te), mode).numpy().flatten()
    tf = te.numpy().flatten()
    mf = tf < 0.5 * tau_star; mn = ~mf
    re = np.abs(yp - yr) / np.maximum(np.abs(yr), 1e-12)
    return float(re[mf].max()), float(re[mn].max())

if __name__ == "__main__":
    save = {}
    for alpha in ALPHAS:
        sol, tau_star = reference_p2(alpha=alpha)
        t = np.linspace(0, 0.95*tau_star, 2001)
        ref_t = torch.tensor(t, dtype=torch.float32)
        ref_y = torch.tensor(sol.sol(t)[0], dtype=torch.float32)
        print(f"\n=== alpha={alpha}, tau_star={tau_star:.6f} ===")
        for cfg, mode in MODES:
            for seed in SEEDS:
                t0 = time.time()
                far, near = train_eval(mode, seed, tau_star, ref_t, ref_y, alpha)
                save[f"a{alpha}_{cfg}_seed{seed}_far"] = far
                save[f"a{alpha}_{cfg}_seed{seed}_near"] = near
                print(f"  a={alpha} {cfg} s={seed}: near={near:.3e} ({time.time()-t0:.0f}s)")
                np.savez(RESULTS / "p2_multiseed_full.npz", **save)
    print(f"\n[saved] {RESULTS / 'p2_multiseed_full.npz'}")
