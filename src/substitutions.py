import torch


def residual_direct(net, t):
    """Residual of y'' - y^3 = 0."""
    t = t.clone().requires_grad_(True)
    y = net(t)
    y_t = torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
    y_tt = torch.autograd.grad(y_t, t, torch.ones_like(y_t), create_graph=True)[0]
    return y_tt - y ** 3


def residual_rapidity_log(net, t):
    """Residual of theta'' + (theta')^2 - exp(2 theta) = 0, theta = log(y)."""
    t = t.clone().requires_grad_(True)
    th = net(t)
    th_t = torch.autograd.grad(th, t, torch.ones_like(th), create_graph=True)[0]
    th_tt = torch.autograd.grad(th_t, t, torch.ones_like(th_t), create_graph=True)[0]
    return th_tt + th_t ** 2 - torch.exp(2 * th)


def residual_reciprocal(net, t):
    """Residual of 2(w')^2 - w w'' - 1 = 0, w = 1/y."""
    t = t.clone().requires_grad_(True)
    w = net(t)
    w_t = torch.autograd.grad(w, t, torch.ones_like(w), create_graph=True)[0]
    w_tt = torch.autograd.grad(w_t, t, torch.ones_like(w_t), create_graph=True)[0]
    return 2 * w_t ** 2 - w * w_tt - 1.0


RESIDUALS = {
    "direct":       residual_direct,
    "rapidity_log": residual_rapidity_log,
    "reciprocal":   residual_reciprocal,
}


def initial_conditions(mode, t0, net):
    """IC loss for y(0)=1, y'(0)=0 under the chosen representation."""
    if mode == "direct":
        y0 = net(t0)
        y0_t = torch.autograd.grad(y0, t0, torch.ones_like(y0), create_graph=True)[0]
        return (y0 - 1.0) ** 2 + y0_t ** 2
    if mode == "rapidity_log":
        th0 = net(t0)
        th0_t = torch.autograd.grad(th0, t0, torch.ones_like(th0), create_graph=True)[0]
        return th0 ** 2 + th0_t ** 2
    if mode == "reciprocal":
        w0 = net(t0)
        w0_t = torch.autograd.grad(w0, t0, torch.ones_like(w0), create_graph=True)[0]
        return (w0 - 1.0) ** 2 + w0_t ** 2
    raise ValueError(mode)


def transform_to_y(net_output, mode):
    """Convert network output back to physical variable y for evaluation."""
    if mode == "direct":
        return net_output
    if mode == "rapidity_log":
        return torch.exp(net_output)
    if mode == "reciprocal":
        w_safe = torch.where(net_output > 1e-8, net_output,
                             torch.full_like(net_output, 1e-8))
        return 1.0 / w_safe
    raise ValueError(mode)
