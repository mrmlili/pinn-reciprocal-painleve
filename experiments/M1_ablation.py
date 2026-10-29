"""
M1: Critical ablation - isolate transform from loss reweighting.
Configs A-E as in reviewer 4's demand.

Prototype y''=y^3, seed=42, 30000 epochs.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pinn_solver import PINN

def build_colloc(T_max, n):
    n_u = n//3; n_q = n//3; n_c = n - n_u - n_q
    t_u = torch.linspace(0, T_max, n_u)
    u = torch.rand(n_q); t_q = T_max*(1-(1-u)**2)
    u = torch.rand(n_c); t_c = T_max*(1-(1-u)**4)
    return torch.cat([t_u, t_q, t_c]).sort().values.reshape(-1,1)

def residual_y(net, t):
    """R_y = y'' - y^3"""
    t = t.clone().requires_grad_(True)
    y = net(t)
    y_t = torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
    y_tt = torch.autograd.grad(y_t, t, torch.ones_like(y_t), create_graph=True)[0]
    return y_tt - y**3, y

def residual_w(net, t):
    """R_w = 2(w')^2 - w w'' - 1, output w"""
    t = t.clone().requires_grad_(True)
    w = net(t)
    w_t = torch.autograd.grad(w, t, torch.ones_like(w), create_graph=True)[0]
    w_tt = torch.autograd.grad(w_t, t, torch.ones_like(w_t), create_graph=True)[0]
    return 2*w_t**2 - w*w_tt - 1, w

def ic_loss_y(net, t0):
    y0 = net(t0)
    y0_t = torch.autograd.grad(y0, t0, torch.ones_like(y0), create_graph=True)[0]
    return (y0 - 1.0)**2 + y0_t**2

def ic_loss_w(net, t0):
    w0 = net(t0)
    w0_t = torch.autograd.grad(w0, t0, torch.ones_like(w0), create_graph=True)[0]
    return (w0 - 1.0)**2 + w0_t**2

def train(config, seed, tau_star, ref_t, ref_y, n_epochs=30000):
    torch.manual_seed(seed)
    T_max = 0.95 * tau_star
    net = PINN(64, 4)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2000, factor=0.5)
    t_col = build_colloc(T_max, 2000)

    for ep in range(n_epochs):
        opt.zero_grad()
        t0 = torch.zeros(1,1, requires_grad=True)
        if config == 'A':
            # y output, R_y loss
            R, y = residual_y(net, t_col)
            loss = (R**2).mean() + 10*ic_loss_y(net, t0)
        elif config == 'B':
            # y output, w^3 * R_y loss
            R, y = residual_y(net, t_col)
            w3 = 1.0 / torch.clamp(y**3, min=1e-6)
            loss = ((w3 * R)**2).mean() + 10*ic_loss_y(net, t0)
        elif config == 'C':
            # w output, R_w loss (current C4)
            R, w = residual_w(net, t_col)
            loss = (R**2).mean() + 10*ic_loss_w(net, t0)
        elif config == 'D':
            # w output, R_w / w^3 loss (transform only, no reweight)
            R, w = residual_w(net, t_col)
            w3 = torch.clamp(w**3, min=1e-6)
            loss = ((R / w3)**2).mean() + 10*ic_loss_w(net, t0)
        elif config == 'E':
            # y output, normalized R_y / (1+|y|^3)
            R, y = residual_y(net, t_col)
            norm = 1.0 + torch.abs(y)**3
            loss = ((R / norm)**2).mean() + 10*ic_loss_y(net, t0)
        loss.backward(); opt.step(); sch.step(loss.detach())

    # Eval in y-space
    mask = ref_t <= T_max
    te = ref_t[mask].reshape(-1,1)
    yr = ref_y[mask].numpy().flatten()
    with torch.no_grad():
        out = net(te)
    if config in ('C','D'):
        yp = 1.0 / torch.clamp(out, min=1e-8)
    else:
        yp = out
    yp = yp.numpy().flatten()
    tf = te.numpy().flatten()
    mf = tf < 0.5 * tau_star; mn = ~mf
    re = np.abs(yp - yr) / np.maximum(np.abs(yr), 1e-12)
    return float(re[mf].max()), float(re[mn].max())

if __name__ == "__main__":
    def rhs(t, s): return [s[1], s[0]**3]
    def ev(t, s): return s[0] - 1e6
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(rhs, [0, 10], [1.0, 0.0], method="DOP853",
                    rtol=1e-12, atol=1e-12, events=ev, dense_output=True)
    tau_star = float(sol.t_events[0][0])
    t = np.linspace(0, 0.95*tau_star, 2001)
    ref_t = torch.tensor(t, dtype=torch.float32)
    ref_y = torch.tensor(sol.sol(t)[0], dtype=torch.float32)
    print(f"tau_star = {tau_star:.6f}")

    results = {}
    for cfg in ['A','B','C','D','E']:
        t0 = time.time()
        far, near = train(cfg, 42, tau_star, ref_t, ref_y, n_epochs=30000)
        results[cfg] = (far, near)
        print(f"  {cfg}: E_far={far:.3e}, E_near={near:.3e} ({time.time()-t0:.0f}s)")

    print("\n=== M1 Verdict ===")
    A, B, C, D, E = [results[k][1] for k in 'ABCDE']
    print(f"  A (y, R_y)        = {A:.3e}")
    print(f"  B (y, w³R_y)      = {B:.3e}")
    print(f"  C (w, R_w)        = {C:.3e}  ← current C4")
    print(f"  D (w, R_w/w³)     = {D:.3e}")
    print(f"  E (y, R_y/(1+|y|³))= {E:.3e}")
    print()
    if B < A * 0.1 and C < A * 0.1:
        print("VERDICT: Reweighting alone (B) matches transform (C).")
        print("  → Article must be REFRAMED as 'residual reweighting'")
    elif D < A * 0.1 and C < A * 0.1:
        print("VERDICT: Transform alone (D) matches combined (C).")
        print("  → Article claim holds, C4 contribution is real")
    elif B > A * 0.5 and D > A * 0.5 and C < A * 0.1:
        print("VERDICT: Only combined B+C works. Transform is necessary.")
        print("  → Article claim holds")
    else:
        print("VERDICT: Mixed. Need to inspect and decide.")
