"""Path utilities: spline knots, shortcutting, collision-checked smoothing and path metrics."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def spline_basis(n_knots: int, n_steps: int, kind: str = "cubic", device="cpu") -> torch.Tensor:
    """(n_steps, n_knots) matrix mapping evenly spaced knots to evenly spaced samples of the interpolant.

    ``samples = basis @ knots``. ``kind="cubic"`` is a natural cubic spline (C2, passes through the knots),
    ``"linear"`` is piecewise-linear interpolation. Used to parameterize MPPI controls and to smooth paths.
    """
    t_k = np.linspace(0.0, 1.0, n_knots)
    t_s = np.linspace(0.0, 1.0, n_steps)
    if kind == "linear" or n_knots < 3:
        B = np.stack([np.interp(t_s, t_k, np.eye(n_knots)[j]) for j in range(n_knots)], 1)
    elif kind == "cubic":
        B = np.stack([_natural_cubic(t_k, np.eye(n_knots)[j], t_s) for j in range(n_knots)], 1)
    else:
        raise ValueError(f"unknown spline kind {kind!r}")
    return torch.tensor(B, dtype=torch.float32, device=device)


def _natural_cubic(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Evaluate the natural cubic spline through ``(x, y)`` at ``xq`` (small dense solve; x is short)."""
    n = len(x)
    h = np.diff(x)
    A = np.zeros((n, n)); r = np.zeros(n)
    A[0, 0] = A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        r[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, r)                       # second derivatives at the knots
    k = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, n - 2)
    t = xq - x[k]
    hk = h[k]
    return (M[k] * (x[k + 1] - xq) ** 3 / (6 * hk) + M[k + 1] * t ** 3 / (6 * hk)
            + (y[k] / hk - M[k] * hk / 6) * (x[k + 1] - xq) + (y[k + 1] / hk - M[k + 1] * hk / 6) * t)


def resample(path: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to ``n`` points evenly spaced in arc length."""
    path = np.asarray(path, dtype=np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    if s[-1] == 0:
        return np.repeat(path[:1], n, 0)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, path[:, k]) for k in range(path.shape[1])], 1)


def path_length(path) -> float:
    """Total length of a polyline."""
    p = np.asarray(path)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def path_smoothness(path, eps: float = 1e-8) -> float:
    """Normalized curvature energy (lower is smoother), as reported in the paper's ablation table."""
    p = np.asarray(path, dtype=np.float64)
    d2 = p[2:] - 2 * p[1:-1] + p[:-2]
    return float((np.linalg.norm(d2, axis=1) ** 2).sum() / (path_length(p) + eps))


def shortcut(path: np.ndarray, segment_free: Callable[[np.ndarray, np.ndarray], bool]) -> np.ndarray:
    """Greedy shortcutting: from each waypoint jump to the farthest later waypoint reachable in a straight line.

    Args:
        path: (K, D) waypoints; consecutive waypoints are assumed collision-free (e.g. an RRT solution).
        segment_free: ``f(a, b) -> bool`` collision test for the straight segment ``a -> b``.
    """
    path = np.asarray(path, dtype=np.float64)
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not segment_free(path[i], path[j]):
            j -= 1
        out.append(path[j]); i = j
    return np.array(out)


def chaikin_smooth(path: np.ndarray, segment_free: Callable[[np.ndarray, np.ndarray], bool],
                   iterations: int = 5) -> np.ndarray:
    """Corner-cutting (Chaikin) smoothing that keeps only iterations whose every segment stays collision-free."""
    path = np.asarray(path, dtype=np.float64)
    for _ in range(iterations):
        q = [path[0]]
        for a, b in zip(path[:-1], path[1:]):
            q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        cand = np.array(q + [path[-1]])
        if not all(segment_free(cand[k], cand[k + 1]) for k in range(len(cand) - 1)):
            break
        path = cand
    return path
