"""Rendering helpers built on `gsplat <https://github.com/nerfstudio-project/gsplat>`_ (optional dependency).

Used for image-conditioned costs (render what the robot camera would see) and for visualizing plans: trees, paths
and robots are drawn as extra Gaussians, so the scene itself handles occlusion.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from .scene import GaussianScene


def look_at(eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """World-to-camera 4x4 matrix (OpenCV camera: x right, y down, z forward) looking from ``eye`` to ``target``.

    ``up`` is the world's up direction (``(0, 0, 1)`` for z-up scenes, ``(0, -1, 0)`` for OpenCV/y-down scenes such
    as Marble exports).
    """
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(z, [0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x); y = np.cross(z, x)
    V = np.eye(4); V[:3, :3] = np.stack([x, y, z]); V[:3, 3] = -V[:3, :3] @ eye
    return V


def intrinsics(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    """Pinhole intrinsics with a horizontal field of view of ``fov_deg``."""
    f = 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)


def render(scene: GaussianScene, viewmat, K, width: int, height: int, background=(0.08, 0.08, 0.08),
           return_depth: bool = False, min_thickness: float = 0.05):
    """Rasterize ``scene`` (colors from the degree-0 SH term) for one or a batch of cameras.

    Args:
        viewmat: (4, 4) or (C, 4, 4) world-to-camera matrices (accepts numpy or torch).
        K: (3, 3) or (C, 3, 3) intrinsics.
        min_thickness: flat 2DGS disks are given a third axis of ``min_thickness x`` their smaller in-plane scale so
            the 3DGS rasterizer can draw them (rendering only; collision checking uses the true scales).
    Returns:
        (C, H, W, 3) float RGB in [0, 1] (and (C, H, W) expected depth if ``return_depth``); C is dropped for a
        single camera.
    """
    from gsplat import rasterization

    dev = scene.device
    V = torch.as_tensor(viewmat, dtype=torch.float32, device=dev)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)
    single = V.ndim == 2
    V, Kt = V.reshape(-1, 4, 4), Kt.reshape(-1, 3, 3).expand(V.reshape(-1, 4, 4).shape[0], 3, 3)
    colors = scene.colors if scene.colors is not None else torch.full_like(scene.means, 0.7)
    s = scene.scales.clone()
    s[:, 2] = torch.maximum(s[:, 2], min_thickness * s[:, :2].min(-1).values)
    img, alpha, _ = rasterization(scene.means, scene.quats, s.clamp_min(1e-7), scene.opacities, colors,
                                  V, Kt, width, height, render_mode="RGB+ED")
    bg = torch.as_tensor(background, dtype=torch.float32, device=dev)
    rgb = (img[..., :3] + (1 - alpha) * bg).clamp(0, 1)
    out = (rgb[0], img[0, ..., 3]) if single else (rgb, img[..., 3])
    return out if return_depth else out[0]


# ---------------------------------------------------------------------- overlay primitives (drawn as Gaussians)
def _splats(points, color, scale: float, opacity: float = 1.0, device="cuda") -> GaussianScene:
    P = torch.as_tensor(np.asarray(points, dtype=np.float32), device=device).reshape(-1, 3)
    n = P.shape[0]
    q = torch.zeros(n, 4, device=device); q[:, 0] = 1.0
    col = torch.as_tensor(color, dtype=torch.float32, device=device).reshape(-1, 3).expand(n, 3).contiguous()
    return GaussianScene(P, q, torch.full((n, 3), scale, device=device), torch.full((n,), opacity, device=device), col)


def polyline_splats(points, color=(1.0, 0.55, 0.1), width: float = 0.01, opacity: float = 1.0, device="cuda") -> GaussianScene:
    """A thick polyline made of small round Gaussians (``width`` is the Gaussian std)."""
    pts = np.asarray(points, dtype=np.float64)
    step = 0.6 * width
    dense = [np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / step))) for a, b in zip(pts[:-1], pts[1:])]
    return _splats(np.concatenate(dense) if dense else pts, color, width, opacity, device)


def tree_splats(nodes, parents, color=(0.2, 0.85, 1.0), width: float = 0.003, opacity: float = 0.85,
                mask: Optional[np.ndarray] = None, device="cuda") -> GaussianScene:
    """All edges ``parents[i] -> i`` of an RRT tree (optionally only nodes where ``mask`` is True)."""
    nodes, parents = np.asarray(nodes), np.asarray(parents)
    idx = np.arange(1, len(nodes)) if mask is None else np.nonzero(mask[1:])[0] + 1
    segs = [np.linspace(nodes[parents[i]], nodes[i], max(2, int(np.linalg.norm(nodes[i] - nodes[parents[i]]) / (1.8 * width))))
            for i in idx]
    pts = np.concatenate(segs) if segs else nodes[:1]
    return _splats(pts, color, width, opacity, device)


def sphere_splats(center, radius: float, color=(1.0, 0.8, 0.15), n: int = 1200,
                  light=(0.3, -0.4, 0.85), device="cuda") -> GaussianScene:
    """A shaded sphere (Fibonacci-sampled surface Gaussians), e.g. to show a spherical robot."""
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    l = np.asarray(light) / np.linalg.norm(light)
    shade = (0.35 + 0.65 * np.clip(dirs @ l, 0, 1))[:, None] * np.asarray(color)[None]
    return _splats(np.asarray(center)[None] + radius * dirs, np.clip(shade, 0, 1), 0.08 * radius, 1.0, device)


def merge(scenes: Iterable[GaussianScene]) -> GaussianScene:
    """Concatenate several scenes (e.g. the environment plus overlay primitives) for one render call."""
    scenes = list(scenes)
    cat = lambda name: torch.cat([getattr(s, name) for s in scenes], 0)
    colors = torch.cat([s.colors if s.colors is not None else torch.full_like(s.means, 0.7) for s in scenes], 0)
    return GaussianScene(cat("means"), cat("quats"), cat("scales"), cat("opacities"), colors)
