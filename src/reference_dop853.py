"""
Reference solution for y'' = y^3 with y(0)=1, y'(0)=0.

Uses scipy DOP853 (adaptive Runge-Kutta, order 8) with tight tolerances
as ground truth. Detects the blow-up time tau_star via event location.

Output: dense solution object + tau_star, cached to results/reference.npz
"""

import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path


def compute_reference(t_end: float = 10.0,
                      y0: float = 1.0,
                      v0: float = 0.0,
                      blowup_threshold: float = 1e6,
                      rtol: float = 1e-12,
                      atol: float = 1e-12):
    """Solve y'' = y^3 and locate blow-up time tau_star."""
    def rhs(t, state):
        y, v = state
        return [v, y ** 3]

    def event_blowup(t, state):
        return state[0] - blowup_threshold
    event_blowup.terminal = True
    event_blowup.direction = 1

    sol = solve_ivp(
        rhs, [0, t_end], [y0, v0], method='DOP853',
        rtol=rtol, atol=atol, events=event_blowup, dense_output=True,
    )
    tau_star = float(sol.t_events[0][0]) if sol.t_events[0] else float('nan')
    return sol, tau_star


def save_reference(sol, tau_star, out_path: Path):
    """Cache reference to disk for reuse across experiments."""
    t_grid = np.linspace(0, 0.99 * tau_star, 2001)
    y_ref = sol.sol(t_grid)[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, t_grid=t_grid, y_ref=y_ref, tau_star=tau_star)


if __name__ == "__main__":
    sol, tau_star = compute_reference()
    print(f"[reference_dop853] tau_star = {tau_star:.6f}")
    out = Path(__file__).resolve().parent.parent / "results" / "reference.npz"
    save_reference(sol, tau_star, out)
    print(f"[reference_dop853] saved to {out}")