"""Batched collision queries against a Gaussian-splat scene (the public entry point of the metric)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch
import warp as wp

from .kernels import ellipsoid_query_kernel
from .scene import GaussianScene

ArrayLike = Union[torch.Tensor, np.ndarray, Sequence]


@dataclass
class QueryResult:
    """Output of :meth:`CollisionChecker.query`; every field has the query batch shape (e.g. ``(B, T, L)``).

    Attributes:
        distance: soft-minimum of the conservative scaling distance ``s~^2`` over nearby splats. ``< 1`` means the
            ellipsoid overlaps the (inflated) scene; very large values mean nothing is nearby.
        min_distance: hard minimum of ``s~^2``.
        count: number of splats with ``s~^2 < 1``.
        cost: the paper's MPPI collision cost, ``mean_i d_max * exp(-decay * max(s~^2_i - s_min, 0))``.
    """

    distance: torch.Tensor
    min_distance: torch.Tensor
    count: torch.Tensor
    cost: torch.Tensor

    @property
    def in_collision(self) -> torch.Tensor:
        """Boolean mask: at least one splat has ``s~^2 < 1`` (the collision test used by the RRT)."""
        return self.count > 0


class CollisionChecker:
    """GPU (or CPU) collision checker for ellipsoidal robot bodies against a :class:`GaussianScene`.

    The checker owns Warp views of the splat parameters and a spatial hash grid. Queries take arbitrarily batched
    ellipsoids -- e.g. ``(num_samples, horizon, num_links)`` -- and return per-ellipsoid distances and costs.

    Example::

        scene = GaussianScene.from_ply("scene.ply").drop_floaters()
        checker = CollisionChecker(scene, n_sigma=1.0)
        res = checker.query_spheres(points, radius=0.05)      # points: (..., 3)
        free = ~res.in_collision

    Args:
        scene: the Gaussian-splat scene. Its tensors define the device used for queries.
        n_sigma: default conservatism factor ``n_sigma >= 0``. ``0`` reduces to a point-cloud (splat-centre) check.
        max_splat_scale: splats whose largest scale exceeds this are ignored as outliers. Defaults to
            ``max_splat_scale_multiple x median`` largest scale.
        max_splat_scale_multiple: see above (the paper uses 30).
        softmin_gamma: sharpness of the soft-minimum aggregation (larger -> closer to a hard min).
        grid_dims: resolution of the Warp hash grid (a hash table size, not a spatial extent).
    """

    def __init__(self, scene: GaussianScene, n_sigma: float = 1.0, max_splat_scale: Optional[float] = None,
                 max_splat_scale_multiple: float = 30.0, softmin_gamma: float = 300.0,
                 grid_dims: tuple = (128, 128, 128)):
        wp.init()
        self.scene = scene
        self.device = scene.device
        self.wp_device = "cpu" if scene.device.type == "cpu" else f"cuda:{scene.device.index or 0}"
        self.n_sigma = float(n_sigma)
        ms = scene.max_scale
        self.max_splat_scale = float(max_splat_scale if max_splat_scale is not None else max_splat_scale_multiple * ms.median())
        self.softmin_gamma = float(softmin_gamma)
        # 99th percentile of splat sizes (below the outlier cutoff): used to size the neighbourhood radius
        kept = ms[ms <= self.max_splat_scale]
        sample = kept[torch.randperm(kept.numel(), device=kept.device)[:200_000]] if kept.numel() > 200_000 else kept
        self.scale_p99 = float(torch.quantile(sample.float(), 0.99)) if kept.numel() else 0.0
        self._means = wp.from_torch(scene.means.contiguous(), dtype=wp.vec3)
        self._quats = wp.from_torch(scene.quats.contiguous(), dtype=wp.vec4)
        self._scales = wp.from_torch(scene.scales.contiguous(), dtype=wp.vec3)
        self._grid = wp.HashGrid(*grid_dims, device=self.wp_device)
        self._grid_cell = None

    # ------------------------------------------------------------------ neighbourhood radius
    def default_radius(self, max_axis: float, n_sigma: Optional[float] = None) -> float:
        """Neighbourhood radius that contains every splat able to reach ``s~^2 < 1`` for a sphere of ``max_axis``.

        For a splat of size ``s`` at distance ``d`` from a sphere of radius ``a``, ``s~^2 < 1`` requires
        ``d < n_sigma*s + sqrt((n_sigma*s)^2 + a^2)``; we use the 99th-percentile splat size plus a 25% margin.
        """
        ns = self.n_sigma if n_sigma is None else n_sigma
        k = ns * self.scale_p99
        return float(k + np.sqrt(k * k + max_axis ** 2) + 0.25 * max_axis)

    def _ensure_grid(self, radius: float):
        if self._grid_cell is None or not (0.5 * self._grid_cell <= radius <= 2.0 * self._grid_cell):
            self._grid.build(self._means, radius)
            self._grid_cell = radius

    # ------------------------------------------------------------------ queries
    def query(self, centers: ArrayLike, axes: ArrayLike, rotations: Optional[ArrayLike] = None,
              n_sigma: Optional[float] = None, search_radius: Optional[float] = None,
              d_max: float = 1.0, decay: float = 1.0, s_min: float = 1.0) -> QueryResult:
        """Evaluate the metric for a batch of ellipsoids.

        Args:
            centers: (..., 3) ellipsoid centres in the scene frame.
            axes: (..., 3) or (3,) semi-axis lengths (broadcast against ``centers``).
            rotations: optional (..., 3, 3) body-to-world rotation matrices (default: axis aligned).
            n_sigma: conservatism factor for this call (default: the checker's).
            search_radius: neighbourhood radius for the hash-grid query (default: :meth:`default_radius` of the
                largest semi-axis).
            d_max, decay, s_min: parameters of the MPPI cost (paper Eq.); ``cost`` is ignored by the RRT.

        Returns:
            :class:`QueryResult` with the batch shape of ``centers[..., 0]``.
        """
        c = torch.as_tensor(centers, dtype=torch.float32, device=self.device)
        batch = c.shape[:-1]
        c = c.reshape(-1, 3).contiguous()
        a = torch.as_tensor(axes, dtype=torch.float32, device=self.device).expand(*batch, 3).reshape(-1, 3).contiguous()
        if rotations is None:
            R = torch.eye(3, device=self.device).expand(c.shape[0], 3, 3).contiguous()
        else:
            R = torch.as_tensor(rotations, dtype=torch.float32, device=self.device).expand(*batch, 3, 3).reshape(-1, 3, 3).contiguous()
        ns = self.n_sigma if n_sigma is None else float(n_sigma)
        r = float(search_radius) if search_radius is not None else self.default_radius(float(a.max()), ns)
        self._ensure_grid(r)
        n = c.shape[0]
        out = [torch.empty(n, device=self.device), torch.empty(n, device=self.device),
               torch.empty(n, dtype=torch.int32, device=self.device), torch.empty(n, device=self.device)]
        if n:
            wp.launch(ellipsoid_query_kernel, dim=n, device=self.wp_device, inputs=[
                self._grid.id, self._means, self._quats, self._scales,
                wp.from_torch(c, dtype=wp.vec3), wp.from_torch(R, dtype=wp.mat33), wp.from_torch(a, dtype=wp.vec3),
                r, ns, self.max_splat_scale, self.softmin_gamma, float(d_max), float(decay), float(s_min),
                wp.from_torch(out[0]), wp.from_torch(out[1]), wp.from_torch(out[2], dtype=wp.int32), wp.from_torch(out[3])])
        return QueryResult(*(t.reshape(batch) for t in out))

    def query_spheres(self, centers: ArrayLike, radius: float, **kw) -> QueryResult:
        """Shortcut for spherical robots / points: ``query(centers, axes=(radius,)*3)``."""
        return self.query(centers, torch.full((3,), float(radius)), **kw)

    def is_free(self, centers: ArrayLike, radius: float, **kw) -> torch.Tensor:
        """Boolean mask, True where a sphere of ``radius`` at each centre is collision-free."""
        return ~self.query_spheres(centers, radius, **kw).in_collision

    def segment_free(self, a: ArrayLike, b: ArrayLike, radius: float, resolution: Optional[float] = None, **kw) -> bool:
        """True if a sphere swept along the straight segment ``a -> b`` stays collision-free.

        Args:
            resolution: spacing of the checked points (default: ``radius / 2``).
        """
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        b = torch.as_tensor(b, dtype=torch.float32, device=self.device)
        step = resolution or 0.5 * radius
        k = max(2, int(torch.ceil((b - a).norm() / step)) + 1)
        t = torch.linspace(0, 1, k, device=self.device)[:, None]
        return bool(self.is_free(a + t * (b - a), radius, **kw).all())
