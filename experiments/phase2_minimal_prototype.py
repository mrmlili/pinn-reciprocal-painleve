"""
Phase 2 (minimal, MCS-ready): 5-seed for Table 1 (prototype only).
Saves results to results/phase2_proto_5seed.npz.

60 total runs → 15 runs (5 seeds × 3 configs).
Estimated time: ~4 hours on CPU.
"""

import sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pinn_solver import PINN
from substitutions import RESIDUALS, initial_conditions, transform_to_y

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 7, 123, 2024, 999]
N_EPOCHS = 30000
N_COLLOC = 2000
WIDTH, DEPTH, LR = 64, 4, 1e-3


def build_collocation_v3(T_max, n_points):
    n_u = n_points // 3
    n_q = n_points // 3
    n_c = n_points - n_u - n_q
    t_u = torch.linspace(0, T_max, n_u)
    u = torch.rand(n_q); t_q = T_max * (1 - (1 - u)**2)
    u = torch.rand(n_c); t_c = T_max * (1 - (1 - u)**4)
    return torch.cat([t_u, t_q, t_c]).sort().values.reshape(-1, 1)


def train_and_eval(mode, seed, tau_star, ref_t, ref_y):
    torch.manual_seed(seed)
    T_max = 0.95 * tau_star
    net = PINN(WIDTH, DEPTH)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2000, factor=0.5)
    residual_fn = RESIDUALS[mode]
    t_colloc = build_collocation_v3(T_max, N_COLLOC)

    for epoch in range(N_EPOCHS):
        opt.zero_grad()
        res = residual_fn(net, t_colloc)
        loss_pde = (res**2).mean()
        t0 = torch.zeros(1, 1, requires_grad=True)
        loss_ic = initial_conditions(mode, t0, net)
        loss = loss_pde + 10.0 * loss_ic
        loss.backward(); opt.step(); sch.step(loss.detach())

    mask = ref_t <= T_max
    te = ref_t[mask].reshape(-1, 1)
    yr = ref_y[mask].numpy().flatten()
    with torch.no_grad():
        yp = transform_to_y(net(te), mode).numpy().flatten()
    tf = te.numpy().flatten()
    mf = tf < 0.5 * tau_star; mn = ~mf
    ae = np.abs(yp - yr); re = ae / np.maximum(np.abs(yr), 1e-12)
    return {"E_rel_far": float(re[mf].max()), "E_rel_near": float(re[mn].max())}


def main():
    # Reference
    def rhs(t, s): return [s[1], s[0]**3]
    def ev(t, s): return s[0] - 1e6
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(rhs, [0, 10], [1.0, 0.0], method="DOP853",
                    rtol=1e-12, atol=1e-12, events=ev, dense_output=True)
    tau_star = float(sol.t_events[0][0])
    t = np.linspace(0, 0.95 * tau_star, 2001)
    ref_t = torch.tensor(t, dtype=torch.float32)
    ref_y = torch.tensor(sol.sol(t)[0], dtype=torch.float32)
    print(f"[prototype] tau_star = {tau_star:.6f}")

    results = {}
    for cfg_name, mode in [("C0","direct"),("C2","rapidity_log"),("C4","reciprocal")]:
        for seed in SEEDS:
            t0 = time.time()
            m = train_and_eval(mode, seed, tau_star, ref_t, ref_y)
            results[f"{cfg_name}_seed{seed}"] = m
            print(f"  {cfg_name} seed={seed}: E_near={m['E_rel_near']:.3e} ({time.time()-t0:.0f}s)")

    save = {"tau_star": tau_star}
    for k, v in results.items():
        save[f"{k}_E_far"] = v["E_rel_far"]
        save[f"{k}_E_near"] = v["E_rel_near"]
    np.savez(RESULTS / "phase2_proto_5seed.npz", **save)
    print(f"[saved] {RESULTS / 'phase2_proto_5seed.npz'}")


if __name__ == "__main__":
    main()
