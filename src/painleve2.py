"""
Painleve II: y'' = 2y^3 + t y + alpha

Three representations:
  D0: direct      -> residual: y'' - 2y^3 - t y - alpha
  D1: rapidity    -> theta = log(y)
  D2: reciprocal  -> w = 1/y
"""

import torch
import numpy as np
from scipy.integrate import solve_ivp


ALPHA = 1.0


def reference_p2(alpha: float = ALPHA, rtol: float = 1e-12,
                 atol: float = 1e-12, t_end: float = 10.0):
    """DOP853 reference for P-II with y(0)=1, y'(0)=0."""
    def rhs(t, state):
        y, v = state
        return [v, 2.0 * y**3 + t * y + alpha]

    def event_blowup(t, state):
        return state[0] - 1e6
    event_blowup.terminal = True
    event_blowup.direction = 1

    sol = solve_ivp(rhs, [0, t_end], [1.0, 0.0], method='DOP853',
                    rtol=rtol, atol=atol, events=event_blowup,
                    dense_output=True)
    tau_star = float(sol.t_events[0][0]) if sol.t_events[0] else np.nan
    return sol, tau_star


def residual_direct_p2(net, t, alpha=ALPHA):
    t = t.clone().requires_grad_(True)
    y = net(t)
    y_t = torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
    y_tt = torch.autograd.grad(y_t, t, torch.ones_like(y_t), create_graph=True)[0]
    return y_tt - 2.0 * y**3 - t * y - alpha


def residual_rapidity_p2(net, t, alpha=ALPHA):
    """theta = log(y)."""
    t = t.clone().requires_grad_(True)
    th = net(t)
    th_t = torch.autograd.grad(th, t, torch.ones_like(th), create_graph=True)[0]
    th_tt = torch.autograd.grad(th_t, t, torch.ones_like(th_t), create_graph=True)[0]
    return (th_tt + th_t**2
            - 2.0 * torch.exp(2.0 * th)
            - t
            - alpha * torch.exp(-th))


def residual_reciprocal_p2(net, t, alpha=ALPHA):
    """w = 1/y."""
    t = t.clone().requires_grad_(True)
    w = net(t)
    w_t = torch.autograd.grad(w, t, torch.ones_like(w), create_graph=True)[0]
    w_tt = torch.autograd.grad(w_t, t, torch.ones_like(w_t), create_graph=True)[0]
    return (2.0 * w_t**2 - w * w_tt - 2.0
            - t * w**2 - alpha * w**3)


RESIDUALS_P2 = {
    "direct":       residual_direct_p2,
    "rapidity":     residual_rapidity_p2,
    "reciprocal":   residual_reciprocal_p2,
}


def ic_loss_p2(mode: str, t0: torch.Tensor, net, alpha=ALPHA):
    """IC: y(0)=1, y'(0)=0."""
    if mode == "direct":
        y0 = net(t0)
        y0_t = torch.autograd.grad(y0, t0, torch.ones_like(y0), create_graph=True)[0]
        return (y0 - 1.0)**2 + y0_t**2
    if mode == "rapidity":
        th0 = net(t0)
        th0_t = torch.autograd.grad(th0, t0, torch.ones_like(th0), create_graph=True)[0]
        return th0**2 + th0_t**2
    if mode == "reciprocal":
        w0 = net(t0)
        w0_t = torch.autograd.grad(w0, t0, torch.ones_like(w0), create_graph=True)[0]
        return (w0 - 1.0)**2 + w0_t**2
    raise ValueError(mode)


def transform_to_y_p2(net_out, mode: str):
    if mode == "direct":
        return net_out
    if mode == "rapidity":
        return torch.exp(net_out)
    if mode == "reciprocal":
        w_safe = torch.where(net_out > 1e-8, net_out,
                             torch.full_like(net_out, 1e-8))
        return 1.0 / w_safe
    raise ValueError(mode)
