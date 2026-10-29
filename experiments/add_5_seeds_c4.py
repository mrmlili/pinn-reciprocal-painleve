"""Add 5 more seeds for C4 (prototype). Characterize distribution better."""
import sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pinn_solver import PINN
from substitutions import RESIDUALS, initial_conditions, transform_to_y

def build_colloc_v3(T_max, n_points):
    n_u = n_points // 3; n_q = n_points // 3; n_c = n_points - n_u - n_q
    t_u = torch.linspace(0, T_max, n_u)
    u = torch.rand(n_q); t_q = T_max * (1 - (1 - u)**2)
    u = torch.rand(n_c); t_c = T_max * (1 - (1 - u)**4)
    return torch.cat([t_u, t_q, t_c]).sort().values.reshape(-1, 1)

def train_eval(seed, mode, tau_star, ref_t, ref_y, T_max, n_epochs=30000):
    torch.manual_seed(seed)
    net = PINN(64, 4)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2000, factor=0.5)
    res_fn = RESIDUALS[mode]
    t_col = build_colloc_v3(T_max, 2000)
    for ep in range(n_epochs):
        opt.zero_grad()
        res = res_fn(net, t_col)
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
    re = np.abs(yp - yr) / np.maximum(np.abs(yr), 1e-12)
    return float(re[mn].max())

if __name__ == "__main__":
    def rhs(t, s): return [s[1], s[0]**3]
    def ev(t, s): return s[0] - 1e6
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(rhs, [0, 10], [1.0, 0.0], method="DOP853",
                    rtol=1e-12, atol=1e-12, events=ev, dense_output=True)
    tau_star = float(sol.t_events[0][0])
    T_max = 0.95 * tau_star
    t = np.linspace(0, T_max, 2001)
    ref_t = torch.tensor(t, dtype=torch.float32)
    ref_y = torch.tensor(sol.sol(t)[0], dtype=torch.float32)

    print(f"tau_star = {tau_star:.6f}")
    print("Adding 5 more seeds for C4 (prototype)...")
    new_seeds = [1, 2, 3, 4, 5]
    results = {}
    for seed in new_seeds:
        t0 = time.time()
        E = train_eval(seed, "reciprocal", tau_star, ref_t, ref_y, T_max)
        results[seed] = E
        print(f"  seed={seed}: E_near={E:.3e} ({time.time()-t0:.0f}s)")

    # Save
    existing = np.load("results/phase2_proto_5seed.npz")
    save = {k: existing[k] for k in existing.files}
    for seed, E in results.items():
        save[f"C4_seed{seed}_E_near"] = E
    np.savez("results/phase2_proto_10seed_extended.npz", **save)
    print(f"\n[saved] results/phase2_proto_10seed_extended.npz")

    all_C4 = [float(existing[f"C4_seed{s}_E_near"]) for s in [42,7,123,2024,999]] + list(results.values())
    print(f"\nC4 with {len(all_C4)} seeds:")
    print(f"  median = {np.median(all_C4):.3e}")
    print(f"  IQR = [{np.percentile(all_C4,25):.3e}, {np.percentile(all_C4,75):.3e}]")
    print(f"  min = {min(all_C4):.3e}, max = {max(all_C4):.3e}")
