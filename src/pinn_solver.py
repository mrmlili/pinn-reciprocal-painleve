"""
PINN architecture and training loop for y'' = y^3.

Supports three representations (direct, rapidity_log, reciprocal) via
substitutions.py. Adaptive loss weighting is omitted â€” empirical evidence
(see README) shows it is harmful for this problem.
"""

import numpy as np
import torch
import torch.nn as nn
from substitutions import RESIDUALS, initial_conditions, transform_to_y


class PINN(nn.Module):
    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)


def build_collocation_points(T_max: float, n_points: int, device="cpu"):
    """Half uniform, half clustered near T_max (where the pole is)."""
    n_half = n_points // 2
    t_uniform = torch.linspace(0, T_max, n_half, device=device)
    u = torch.rand(n_half, device=device)
    t_clustered = T_max * (1.0 - (1.0 - u) ** 2)
    return torch.cat([t_uniform, t_clustered]).sort().values.reshape(-1, 1)


def train_pinn(mode: str, tau_star: float, n_epochs: int = 15000,
               n_collocation: int = 2000, lr: float = 1e-3,
               width: int = 64, depth: int = 4,
               T_frac: float = 0.95, ic_weight: float = 10.0,
               seed: int = 42, log_every: int = 2000,
               device: str = "cpu"):
    """Train a PINN under the chosen representation."""
    torch.manual_seed(seed)
    T_max = T_frac * tau_star

    net = PINN(width, depth).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2000, factor=0.5,
    )
    residual_fn = RESIDUALS[mode]
    t_colloc = build_collocation_points(T_max, n_collocation, device)

    history = {"loss": [], "pde": [], "ic": []}
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        res = residual_fn(net, t_colloc)
        loss_pde = (res ** 2).mean()

        # Fresh t0 per epoch (avoids stale computation graph)
        t0 = torch.zeros(1, 1, requires_grad=True, device=device)
        loss_ic = initial_conditions(mode, t0, net)

        loss = loss_pde + ic_weight * loss_ic
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())

        history["loss"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())

        if epoch % log_every == 0:
            print(f"[{mode}] epoch {epoch:5d} | "
                  f"loss={loss.item():.3e} | "
                  f"pde={loss_pde.item():.3e} | ic={loss_ic.item():.3e}")

    return net, history


def evaluate(net, mode: str, tau_star: float, ref_t: torch.Tensor,
             ref_y: torch.Tensor, T_frac: float = 0.95):
    """Compare PINN output to DOP853 reference on a grid."""
    T_max = T_frac * tau_star
    mask = ref_t <= T_max
    t_eval = ref_t[mask].reshape(-1, 1)
    y_ref = ref_y[mask].numpy().flatten()

    with torch.no_grad():
        raw = net(t_eval)
        y_pred = transform_to_y(raw, mode).numpy().flatten()

    # Sanity check: for reciprocal mode, w should stay positive in the domain
    n_negative_w = 0
    if mode == "reciprocal":
        with torch.no_grad():
            w_raw = net(t_eval).numpy().flatten()
        n_negative_w = int((w_raw <= 0).sum())
        if n_negative_w > 0:
            print(f"  [warn] {n_negative_w}/{len(w_raw)} points have w<=0 "
                  f"(reciprocal transform violated positivity)")

    t_flat = t_eval.numpy().flatten()
    mask_far = t_flat < 0.5 * tau_star
    mask_near = ~mask_far

    abs_err = np.abs(y_pred - y_ref)
    rel_err = abs_err / np.maximum(np.abs(y_ref), 1e-12)

    return {
        "t": t_flat,
        "pred": y_pred,
        "ref": y_ref,
        "E_inf_far": float(abs_err[mask_far].max()),
        "E_inf_near": float(abs_err[mask_near].max()),
        "E_rel_far": float(rel_err[mask_far].max()),
        "E_rel_near": float(rel_err[mask_near].max()),
        "n_nan": int((~np.isfinite(y_pred)).sum()),
        "n_negative_w": n_negative_w,
    }
