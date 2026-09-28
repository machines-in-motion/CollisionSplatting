"""Model Predictive Path Integral (MPPI) control with spline-parameterized controls.

The planner is model-agnostic: you pass a batched ``rollout(x0, U) -> X`` and ``cost(X, U) -> (K,)``. The paper
combines the CollisionSplatting cost (:meth:`CollisionChecker.query` ``.cost``) with task and image-space costs.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from ..paths import spline_basis


class MPPI:
    """Sampling-based MPC (MPPI) over ``n_knots`` spline knots interpolated to a ``horizon``-step control sequence.

    Each :meth:`optimize` call runs ``iterations`` rounds of: perturb the nominal knots with Gaussian noise, roll out
    all samples in parallel, weight them with ``softmax(-cost / temperature)`` and average. :meth:`shift` warm-starts
    the next MPC step by time-shifting the nominal plan.

    Args:
        rollout: ``f(x0, U) -> X``; ``U`` is (K, horizon, nu), ``X`` is (K, horizon, nx) (states after each control).
        cost: ``f(X, U) -> (K,)`` total cost of every sample.
        nu: control dimension.
        horizon: number of control steps.
        n_samples: rollouts per iteration.
        n_knots: spline knots per control dimension.
        noise_sigma: (nu,) standard deviation of the knot perturbations.
        temperature: MPPI temperature ``lambda`` (smaller -> greedier).
        u_min, u_max: (nu,) control bounds.
        iterations: MPPI iterations per :meth:`optimize` call.
        spline: ``"cubic"`` or ``"linear"`` knot interpolation.
    """

    def __init__(self, rollout: Callable, cost: Callable, nu: int, horizon: int, n_samples: int = 512,
                 n_knots: int = 6, noise_sigma: Sequence[float] = (0.2,), temperature: float = 0.05,
                 u_min: Optional[Sequence[float]] = None, u_max: Optional[Sequence[float]] = None,
                 iterations: int = 1, spline: str = "linear", device="cuda", seed: int = 0):
        self.rollout_fn, self.cost_fn = rollout, cost
        self.nu, self.horizon, self.n_samples, self.n_knots = nu, horizon, n_samples, n_knots
        self.device = torch.device(device)
        self.sigma = torch.as_tensor(noise_sigma, dtype=torch.float32, device=self.device).expand(nu)
        self.temperature = float(temperature)
        inf = torch.full((nu,), float("inf"), device=self.device)
        self.u_min = inf.neg() if u_min is None else torch.as_tensor(u_min, dtype=torch.float32, device=self.device)
        self.u_max = inf if u_max is None else torch.as_tensor(u_max, dtype=torch.float32, device=self.device)
        self.iterations = iterations
        self.basis = spline_basis(n_knots, horizon, spline, self.device)          # (H, K)
        self._basis_pinv = torch.linalg.pinv(self.basis)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.knots = torch.zeros(n_knots, nu, device=self.device)

    def reset(self, knots: Optional[torch.Tensor] = None):
        """Reset the nominal plan (zeros by default)."""
        self.knots = torch.zeros(self.n_knots, self.nu, device=self.device) if knots is None else knots.to(self.device)

    @property
    def controls(self) -> torch.Tensor:
        """(horizon, nu) nominal control sequence."""
        return (self.basis @ self.knots).clamp(self.u_min, self.u_max)

    @torch.no_grad()
    def optimize(self, x0: torch.Tensor) -> dict:
        """Improve the nominal plan from state ``x0``; returns the last iteration's rollouts, costs and weights."""
        for _ in range(self.iterations):
            eps = torch.randn(self.n_samples, self.n_knots, self.nu, device=self.device, generator=self.gen) * self.sigma
            eps[0] = 0.0                                                         # keep the nominal plan in the batch
            K = self.knots[None] + eps
            U = (self.basis[None] @ K).clamp(self.u_min, self.u_max)
            X = self.rollout_fn(x0, U)
            J = self.cost_fn(X, U)
            w = torch.softmax(-(J - J.min()) / self.temperature, 0)
            self.knots = (w[:, None, None] * K).sum(0)
        return {"states": X, "controls": U, "costs": J, "weights": w}

    def shift(self, steps: int = 1):
        """Warm start: advance the nominal plan by ``steps`` control steps (holding the last control)."""
        U = self.controls
        U = torch.cat([U[steps:], U[-1:].expand(steps, -1)], 0)
        self.knots = self._basis_pinv @ U


def planar_unicycle_rollout(dt: float) -> Callable:
    """Rollout for a planar base driven by body-frame ``(v_forward, v_lateral, yaw_rate)`` commands.

    State is ``(x, y, yaw)`` in the plane; returns ``f(x0, U) -> X`` with ``X`` of shape (K, H, 3). Map planar
    coordinates to the 3D scene frame yourself (e.g. which axes span the floor and at what body height).
    """

    def rollout(x0: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        K = U.shape[0]
        yaw = x0[2] + torch.cumsum(U[..., 2] * dt, 1)
        yaw_prev = torch.cat([x0[2].expand(K, 1), yaw[:, :-1]], 1)
        c, s = torch.cos(yaw_prev), torch.sin(yaw_prev)
        dx = (c * U[..., 0] - s * U[..., 1]) * dt
        dy = (s * U[..., 0] + c * U[..., 1]) * dt
        return torch.stack([x0[0] + torch.cumsum(dx, 1), x0[1] + torch.cumsum(dy, 1), yaw], -1)

    return rollout
