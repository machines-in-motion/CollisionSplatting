"""Batched forward kinematics for URDF robots whose collision geometry is a set of ellipsoids.

Collision bodies are read from the links' ``<collision>`` elements. Besides the standard ``<sphere radius=.../>``,
CollisionSplatting URDFs may use an ``<ellipsoid scale="a b c"/>`` geometry (semi-axes in metres)::

    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry><ellipsoid scale="0.08 0.08 0.15"/></geometry>
    </collision>

Kinematics are plain PyTorch, batched over any leading dimensions (e.g. ``(samples, horizon)``) and run on GPU or CPU.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def _floats(s: Optional[str], n: int, default: float = 0.0) -> np.ndarray:
    return np.array([float(v) for v in s.split()], dtype=np.float64) if s else np.full(n, default)


def _origin(el: Optional[ET.Element]) -> np.ndarray:
    """4x4 transform of a URDF ``<origin xyz rpy>`` (rpy = fixed-axis roll-pitch-yaw, R = Rz(y) Ry(p) Rx(r))."""
    T = np.eye(4)
    if el is None:
        return T
    r, p, y = _floats(el.get("rpy"), 3)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    T[:3, :3] = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                          [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                          [-sp, cp * sr, cp * cr]])
    T[:3, 3] = _floats(el.get("xyz"), 3)
    return T


@dataclass
class Joint:
    """A URDF joint (only the fields needed for kinematics)."""

    name: str
    type: str                     # revolute | continuous | prismatic | fixed
    parent: str
    child: str
    origin: np.ndarray            # 4x4 parent-link -> joint frame
    axis: np.ndarray              # (3,) unit axis in the joint frame
    lower: float = -np.inf
    upper: float = np.inf
    velocity: float = np.inf
    index: int = -1               # position in the configuration vector (-1 for fixed joints)


@dataclass
class CollisionEllipsoid:
    """An ellipsoidal collision body rigidly attached to ``link``."""

    link: str
    offset: np.ndarray            # 4x4 link -> ellipsoid frame
    axes: np.ndarray              # (3,) semi-axes


@dataclass
class RobotModel:
    """Kinematic tree + collision ellipsoids parsed from a URDF (see :meth:`from_urdf`)."""

    name: str
    root: str
    joints: list                  # in topological order (parents before children)
    ellipsoids: list
    visuals: dict = field(default_factory=dict)   # link -> list of (4x4 offset, mesh filename)
    urdf_dir: Optional[Path] = None

    @classmethod
    def from_urdf(cls, path: Union[str, Path]) -> "RobotModel":
        """Parse a URDF file (revolute, continuous, prismatic and fixed joints)."""
        path = Path(path)
        root_el = ET.parse(path).getroot()
        joints, children = {}, set()
        for j in root_el.findall("joint"):
            lim = j.find("limit")
            jt = Joint(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                       child=j.find("child").get("link"), origin=_origin(j.find("origin")),
                       axis=_floats(j.find("axis").get("xyz") if j.find("axis") is not None else None, 3, 0.0))
            if jt.type not in ("revolute", "continuous", "prismatic", "fixed"):
                raise NotImplementedError(f"joint type {jt.type!r} ({jt.name})")
            if jt.type != "fixed" and not np.any(jt.axis):
                jt.axis = np.array([1.0, 0.0, 0.0])
            jt.axis = jt.axis / (np.linalg.norm(jt.axis) or 1.0)
            if lim is not None and jt.type != "continuous":
                jt.lower, jt.upper = float(lim.get("lower", -np.inf)), float(lim.get("upper", np.inf))
                jt.velocity = float(lim.get("velocity", np.inf))
            joints[jt.child] = jt                 # keyed by child link: joint names need not be unique in the wild
            children.add(jt.child)
        links = [l.get("name") for l in root_el.findall("link")]
        roots = [l for l in links if l not in children]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root link, found {roots}")
        order, frontier = [], [roots[0]]          # breadth-first topological order
        while frontier:
            nxt = []
            for link in frontier:
                for jt in joints.values():
                    if jt.parent == link:
                        order.append(jt); nxt.append(jt.child)
            frontier = nxt
        k = 0
        for jt in order:
            if jt.type != "fixed":
                jt.index = k; k += 1
        ellipsoids, visuals = [], {}
        for l in root_el.findall("link"):
            for c in l.findall("collision"):
                g = c.find("geometry")
                if g is None:
                    continue
                if g.find("ellipsoid") is not None:
                    axes = _floats(g.find("ellipsoid").get("scale"), 3)
                elif g.find("sphere") is not None:
                    axes = np.full(3, float(g.find("sphere").get("radius")))
                else:
                    continue                      # meshes / boxes are not used by CollisionSplatting
                ellipsoids.append(CollisionEllipsoid(l.get("name"), _origin(c.find("origin")), axes))
            for v in l.findall("visual"):
                m = v.find("geometry/mesh")
                if m is not None:
                    visuals.setdefault(l.get("name"), []).append((_origin(v.find("origin")), m.get("filename")))
        return cls(root_el.get("name", path.stem), roots[0], order, ellipsoids, visuals, path.parent)

    # ------------------------------------------------------------------ properties
    @property
    def dof(self) -> int:
        """Number of actuated joints (length of the configuration vector ``q``)."""
        return sum(j.type != "fixed" for j in self.joints)

    @property
    def joint_limits(self) -> np.ndarray:
        """(dof, 2) position limits in configuration order."""
        return np.array([[j.lower, j.upper] for j in self.joints if j.type != "fixed"])

    @property
    def links(self) -> list:
        return [self.root] + [j.child for j in self.joints]

    # ------------------------------------------------------------------ kinematics
    def forward_kinematics(self, q: torch.Tensor, base_pose: Optional[torch.Tensor] = None) -> dict:
        """World poses of every link.

        Args:
            q: (..., dof) joint positions.
            base_pose: optional (..., 4, 4) pose of the root link (default identity).

        Returns:
            dict ``link name -> (..., 4, 4)`` pose tensor.
        """
        batch, dev, dt = q.shape[:-1], q.device, q.dtype
        eye = torch.eye(4, device=dev, dtype=dt).expand(*batch, 4, 4)
        poses = {self.root: eye if base_pose is None else base_pose.to(dev, dt).expand(*batch, 4, 4)}
        for jt in self.joints:
            T = poses[jt.parent] @ torch.as_tensor(jt.origin, device=dev, dtype=dt)
            if jt.type in ("revolute", "continuous"):
                T = T @ _axis_angle_transform(torch.as_tensor(jt.axis, device=dev, dtype=dt), q[..., jt.index])
            elif jt.type == "prismatic":
                M = torch.eye(4, device=dev, dtype=dt).expand(*batch, 4, 4).clone()
                M[..., :3, 3] = q[..., jt.index, None] * torch.as_tensor(jt.axis, device=dev, dtype=dt)
                T = T @ M
            poses[jt.child] = T
        return poses

    def collision_ellipsoids(self, q: torch.Tensor, base_pose: Optional[torch.Tensor] = None):
        """World-frame collision ellipsoids for configurations ``q``.

        Returns:
            ``(centers (..., E, 3), rotations (..., E, 3, 3), axes (E, 3))`` -- feed directly to
            :meth:`CollisionChecker.query`.
        """
        poses = self.forward_kinematics(q, base_pose)
        dev, dt = q.device, q.dtype
        Ts = torch.stack([poses[e.link] @ torch.as_tensor(e.offset, device=dev, dtype=dt) for e in self.ellipsoids], -3)
        axes = torch.as_tensor(np.stack([e.axes for e in self.ellipsoids]), device=dev, dtype=dt)
        return Ts[..., :3, 3], Ts[..., :3, :3], axes


def _axis_angle_transform(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) homogeneous rotation about a fixed unit ``axis`` by ``angle`` (Rodrigues)."""
    x, y, z = axis
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1 - c
    R = torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s,
                     y * x * C + z * s, c + y * y * C, y * z * C - x * s,
                     z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1).reshape(*angle.shape, 3, 3)
    T = torch.zeros(*angle.shape, 4, 4, device=angle.device, dtype=angle.dtype)
    T[..., :3, :3] = R
    T[..., 3, 3] = 1.0
    return T
