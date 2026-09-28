"""Gaussian-splat scenes: a small container plus loaders for standard 3DGS / 2DGS files.

CollisionSplatting works on the *standard* representation -- no normalized reconstruction, no mesh. A scene is just
the per-splat parameters every 3DGS exporter writes:

* ``means``      (N, 3)  splat centres
* ``quats``      (N, 4)  unit quaternions, **(w, x, y, z)** order (the INRIA 3DGS / gsplat convention)
* ``scales``     (N, 3)  per-axis standard deviations (already exponentiated). 2DGS disks have ``scales[:, 2] == 0``
* ``opacities``  (N,)    in [0, 1] (already passed through the sigmoid)
* ``colors``     (N, 3)  optional RGB in [0, 1] from the degree-0 SH coefficients (only used for rendering)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

SH_C0 = 0.28209479177387814  # degree-0 spherical-harmonics constant


@dataclass
class GaussianScene:
    """Per-splat parameters of a Gaussian-splat scene (all tensors on the same device).

    Build one with :meth:`from_ply` (INRIA-format ``.ply`` as written by 3DGS, 2DGS, gsplat, nerfstudio, Marble, ...)
    or :meth:`from_gsplat_checkpoint`, or directly from tensors.
    """

    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        n = self.means.shape[0]
        for name, shape in (("means", (n, 3)), ("quats", (n, 4)), ("scales", (n, 3)), ("opacities", (n,))):
            t = getattr(self, name)
            if tuple(t.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(t.shape)}")
        if self.colors is not None and tuple(self.colors.shape) != (n, 3):
            raise ValueError(f"colors must have shape {(n, 3)}, got {tuple(self.colors.shape)}")

    # ------------------------------------------------------------------ basic properties
    def __len__(self) -> int:
        return self.means.shape[0]

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def max_scale(self) -> torch.Tensor:
        """(N,) largest per-axis standard deviation of each splat."""
        return self.scales.max(dim=-1).values

    def bounds(self, quantile: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds of the splat centres. ``quantile > 0`` ignores that fraction of outliers per side."""
        m = self.means.detach().float()
        if quantile > 0:
            lo, hi = torch.quantile(m, quantile, dim=0), torch.quantile(m, 1 - quantile, dim=0)
        else:
            lo, hi = m.min(0).values, m.max(0).values
        return lo.cpu().numpy(), hi.cpu().numpy()

    # ------------------------------------------------------------------ transforms
    def to(self, device: Union[str, torch.device]) -> "GaussianScene":
        """Copy of the scene on ``device``."""
        kw = {f.name: (getattr(self, f.name).to(device) if torch.is_tensor(getattr(self, f.name)) else getattr(self, f.name))
              for f in fields(self)}
        return GaussianScene(**kw)

    def select(self, mask: torch.Tensor) -> "GaussianScene":
        """Sub-scene with the splats where ``mask`` is True (or at the given indices)."""
        kw = {f.name: (getattr(self, f.name)[mask] if torch.is_tensor(getattr(self, f.name)) else getattr(self, f.name))
              for f in fields(self)}
        return GaussianScene(**kw)

    def drop_floaters(self, min_opacity: float = 0.1, size_multiple: float = 5.0) -> "GaussianScene":
        """Remove splats that are **both** nearly transparent and large (paper, Sec. II-C).

        Reconstructions and generative models represent haze, reflections and poorly observed regions with big,
        low-opacity Gaussians. They are not obstacles, so they are discarded before collision checking.

        Args:
            min_opacity: splats with opacity below this ...
            size_multiple: ... *and* a largest scale above ``size_multiple x median`` are removed.
        """
        ms = self.max_scale
        bad = (self.opacities < min_opacity) & (ms > size_multiple * ms.median())
        out = self.select(~bad)
        out.metadata = {**self.metadata, "dropped_floaters": int(bad.sum())}
        return out

    def filter_opacity(self, min_opacity: float = 0.1) -> "GaussianScene":
        """Keep only splats with opacity >= ``min_opacity`` (the filter used for the paper's baseline study)."""
        return self.select(self.opacities >= min_opacity)

    def crop(self, lower, upper) -> "GaussianScene":
        """Keep splats whose centre lies inside the axis-aligned box ``[lower, upper]``."""
        lo = torch.as_tensor(lower, dtype=self.means.dtype, device=self.device)
        hi = torch.as_tensor(upper, dtype=self.means.dtype, device=self.device)
        return self.select(((self.means >= lo) & (self.means <= hi)).all(-1))

    def transformed(self, T: Union[np.ndarray, torch.Tensor], scale: float = 1.0) -> "GaussianScene":
        """Apply a rigid transform ``T`` (4x4, rotation + translation) and an optional uniform ``scale``.

        Useful to bring a scene into a metric, gravity-aligned frame (e.g. Marble worlds are OpenCV y-down).
        """
        T = torch.as_tensor(T, dtype=self.means.dtype, device=self.device)
        R, t = T[:3, :3], T[:3, 3]
        means = scale * (self.means @ R.T) + t
        quats = quat_multiply(matrix_to_quat(R).expand_as(self.quats), self.quats)
        return replace(self, means=means, quats=quats, scales=self.scales * scale)

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_ply(cls, path: Union[str, Path], device: Union[str, torch.device] = "cuda",
                 flatten_2dgs: bool = False) -> "GaussianScene":
        """Load an INRIA-format Gaussian-splat ``.ply`` (3DGS, 2DGS, gsplat, nerfstudio, Marble, ...).

        Reads ``x y z``, ``scale_*`` (log-space), ``rot_*`` (w, x, y, z), ``opacity`` (logit) and ``f_dc_*``; any
        spherical-harmonics degree works (higher-order coefficients are ignored). Files with only two ``scale_*``
        properties (2DGS) get a zero third axis.

        Args:
            path: ``.ply`` file.
            device: torch device for the returned tensors.
            flatten_2dgs: set the third scale to zero, i.e. treat every splat as a flat 2DGS disk.
        """
        from plyfile import PlyData  # imported lazily: only needed for loading

        v = PlyData.read(str(path)).elements[0]
        names = [p.name for p in v.properties]
        get = lambda n: np.asarray(v[n], dtype=np.float32)
        means = np.stack([get("x"), get("y"), get("z")], 1)
        scale_names = sorted([n for n in names if n.startswith("scale_")], key=lambda s: int(s.split("_")[-1]))
        rot_names = sorted([n for n in names if n.startswith("rot")], key=lambda s: int(s.split("_")[-1]))
        if len(rot_names) != 4 or len(scale_names) not in (2, 3):
            raise ValueError(f"{path}: not a Gaussian-splat ply (scale/rot properties missing): {names}")
        scales = np.exp(np.stack([get(n) for n in scale_names], 1))
        if scales.shape[1] == 2:
            scales = np.concatenate([scales, np.zeros_like(scales[:, :1])], 1)
        quats = np.stack([get(n) for n in rot_names], 1)
        opac = 1.0 / (1.0 + np.exp(-get("opacity")))
        colors = None
        if all(f"f_dc_{i}" in names for i in range(3)):
            colors = np.clip(np.stack([get(f"f_dc_{i}") for i in range(3)], 1) * SH_C0 + 0.5, 0.0, 1.0)
        n_rest = sum(n.startswith("f_rest_") for n in names)
        sh_degree = int(round(math.sqrt(n_rest / 3 + 1) - 1)) if n_rest else 0
        return cls._from_numpy(means, quats, scales, opac, colors, device, flatten_2dgs,
                               metadata={"source": str(path), "sh_degree": sh_degree})

    @classmethod
    def from_gsplat_checkpoint(cls, path: Union[str, Path], device: Union[str, torch.device] = "cuda",
                               flatten_2dgs: bool = False) -> "GaussianScene":
        """Load a `gsplat <https://github.com/nerfstudio-project/gsplat>`_ training checkpoint (``ckpt_*.pt``).

        Accepts both ``{"splats": {...}}`` checkpoints and bare splat dicts with keys ``means, quats, scales,
        opacities, sh0`` (log-scales and logit opacities, as gsplat stores them).
        """
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        sp = ck["splats"] if "splats" in ck else ck
        tonp = lambda t: t.detach().float().cpu().numpy()
        colors = None
        if "sh0" in sp:
            colors = np.clip(tonp(sp["sh0"]).reshape(-1, 3) * SH_C0 + 0.5, 0.0, 1.0)
        return cls._from_numpy(tonp(sp["means"]), tonp(sp["quats"]), np.exp(tonp(sp["scales"])),
                               1.0 / (1.0 + np.exp(-tonp(sp["opacities"]).reshape(-1))), colors, device, flatten_2dgs,
                               metadata={"source": str(path)})

    @classmethod
    def _from_numpy(cls, means, quats, scales, opac, colors, device, flatten_2dgs, metadata):
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-12)
        if flatten_2dgs:
            scales = scales.copy(); scales[:, 2] = 0.0
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
        return cls(t(means), t(quats), t(scales), t(opac), None if colors is None else t(colors), metadata)


# ---------------------------------------------------------------------- quaternion helpers (w, x, y, z)
def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (..., 4) quaternions in (w, x, y, z) order."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) (w, x, y, z) quaternions -> (..., 3, 3) rotation matrices."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(3, 3) rotation matrix -> (4,) (w, x, y, z) quaternion."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R.detach().cpu().numpy()).as_quat()
    return torch.tensor([w, x, y, z], dtype=R.dtype, device=R.device)
