"""Batched GPU RRT for a rigid ellipsoidal (or spherical) body, with CollisionSplatting edge checks."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from ..checker import CollisionChecker
from ..paths import chaikin_smooth, path_length, shortcut


@dataclass
class RRTResult:
    """Output of :meth:`BatchedRRT.plan`.

    Attributes:
        success: whether the goal was connected.
        path: (K, D) waypoints from start to goal (``None`` on failure).
        nodes: (N, D) tree nodes in insertion order; ``nodes[0]`` is the start.
        parents: (N,) parent index of every node (``parents[0] == 0``).
        node_iteration: (N,) RRT iteration at which each node was added (handy for animations).
        iterations: number of iterations run.
        time: wall-clock planning time in seconds.
    """

    success: bool
    path: Optional[np.ndarray]
    nodes: np.ndarray
    parents: np.ndarray
    node_iteration: np.ndarray
    iterations: int
    time: float
    info: dict = field(default_factory=dict)

    @property
    def length(self) -> float:
        return path_length(self.path) if self.path is not None else float("nan")


class BatchedRRT:
    """RRT that grows ``batch_size`` branches per iteration and collision-checks all their edges in one kernel call.

    Every iteration: sample ``batch_size`` states in the box ``[lower, upper]`` (the goal with probability
    ``goal_bias``), find the nearest tree node of each (on the GPU), steer at most ``step_size`` towards it, and check
    ``edge_checks`` points along every new edge with :meth:`CollisionChecker.query`. Collision-free edges are added.

    Planning happens in the scene frame. Set ``lower[k] == upper[k]`` to freeze a coordinate (e.g. a ground robot at a
    fixed body height plans in 2D).

    Args:
        checker: collision checker for the scene.
        lower, upper: (3,) sampling bounds.
        body_axes: semi-axes of the robot body ellipsoid (a float means a sphere of that radius).
        body_rotation: optional (3, 3) fixed body orientation.
        step_size: maximum edge length.
        batch_size: branches added per iteration.
        edge_checks: points checked along every edge (default: enough for a spacing of half the smallest semi-axis).
        goal_bias: probability of sampling the goal.
        goal_tolerance: a node within this distance of the goal connects to it (default ``step_size``).
        max_nodes: capacity of the tree.
        n_sigma: conservatism for this planner (default: the checker's).
        seed: RNG seed for reproducibility.
    """

    def __init__(self, checker: CollisionChecker, lower: Sequence[float], upper: Sequence[float],
                 body_axes=0.05, body_rotation=None, step_size: float = 0.1, batch_size: int = 32,
                 edge_checks: Optional[int] = None, goal_bias: float = 0.1, goal_tolerance: Optional[float] = None,
                 max_nodes: int = 32768, n_sigma: Optional[float] = None, seed: int = 0):
        self.checker = checker
        dev = checker.device
        self.device = dev
        self.lower = torch.as_tensor(lower, dtype=torch.float32, device=dev)
        self.upper = torch.as_tensor(upper, dtype=torch.float32, device=dev)
        ax = np.broadcast_to(np.asarray(body_axes, dtype=np.float32), (3,)).copy()
        self.body_axes = torch.as_tensor(ax, device=dev)
        self.body_rotation = None if body_rotation is None else torch.as_tensor(body_rotation, dtype=torch.float32, device=dev)
        self.step_size = float(step_size)
        self.batch_size = int(batch_size)
        self.edge_checks = int(edge_checks or max(2, int(np.ceil(self.step_size / (0.5 * float(ax.min()))))))
        self.goal_bias = float(goal_bias)
        self.goal_tolerance = float(goal_tolerance or step_size)
        self.max_nodes = int(max_nodes)
        self.n_sigma = n_sigma
        self.gen = torch.Generator(device=dev).manual_seed(seed)

    # ------------------------------------------------------------------ collision helpers
    def _in_collision(self, pts: torch.Tensor) -> torch.Tensor:
        kw = {} if self.body_rotation is None else {"rotations": self.body_rotation}
        return self.checker.query(pts, self.body_axes, n_sigma=self.n_sigma, **kw).in_collision

    def state_free(self, x) -> bool:
        """True if the body placed at ``x`` is collision-free."""
        return not bool(self._in_collision(torch.as_tensor(x, dtype=torch.float32, device=self.device)[None])[0])

    def segment_free(self, a, b) -> bool:
        """True if the body swept along the segment ``a -> b`` (checked at edge resolution) is collision-free."""
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        b = torch.as_tensor(b, dtype=torch.float32, device=self.device)
        k = max(2, int(np.ceil(float((b - a).norm()) / self.step_size * self.edge_checks)) + 1)
        t = torch.linspace(0, 1, k, device=self.device)[:, None]
        return not bool(self._in_collision(a + t * (b - a)).any())

    # ------------------------------------------------------------------ planning
    def plan(self, start, goal, max_iterations: int = 2000) -> RRTResult:
        """Grow the tree from ``start`` until a node connects to ``goal`` or ``max_iterations`` is reached."""
        dev = self.device
        start = torch.as_tensor(start, dtype=torch.float32, device=dev)
        goal = torch.as_tensor(goal, dtype=torch.float32, device=dev)
        if not self.state_free(start):
            raise ValueError("start state is in collision")
        if not self.state_free(goal):
            raise ValueError("goal state is in collision")
        nodes = torch.zeros(self.max_nodes, start.numel(), device=dev); nodes[0] = start
        parents = torch.zeros(self.max_nodes, dtype=torch.long, device=dev)
        node_iter = torch.zeros(self.max_nodes, dtype=torch.long, device=dev)
        n, goal_parent = 1, -1
        t_edge = torch.linspace(1.0 / self.edge_checks, 1.0, self.edge_checks, device=dev)[None, :, None]
        span = self.upper - self.lower
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        it = 0
        for it in range(1, max_iterations + 1):
            u = torch.rand(self.batch_size, start.numel(), device=dev, generator=self.gen)
            samples = self.lower + u * span
            use_goal = torch.rand(self.batch_size, device=dev, generator=self.gen) < self.goal_bias
            samples[use_goal] = goal
            near = torch.cdist(samples, nodes[:n]).argmin(1)
            p = nodes[near]
            d = samples - p
            dist = d.norm(dim=1, keepdim=True).clamp_min(1e-9)
            cand = p + d * (dist.clamp(max=self.step_size) / dist)
            edges = p[:, None] + t_edge * (cand - p)[:, None]                 # (B, E, 3)
            ok = ~self._in_collision(edges).any(1)
            k = int(ok.sum())
            if k == 0:
                continue
            k = min(k, self.max_nodes - n)
            new, par = cand[ok][:k], near[ok][:k]
            nodes[n:n + k] = new; parents[n:n + k] = par; node_iter[n:n + k] = it
            reach = (new - goal).norm(dim=1) <= self.goal_tolerance
            for j in torch.nonzero(reach).flatten().tolist():
                if self.segment_free(new[j], goal):
                    goal_parent = n + j
                    break
            n += k
            if goal_parent >= 0 or n >= self.max_nodes:
                break
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        nodes_np, parents_np, iter_np = nodes[:n].cpu().numpy(), parents[:n].cpu().numpy(), node_iter[:n].cpu().numpy()
        path = None
        if goal_parent >= 0:
            chain, i = [], goal_parent
            while i != 0:
                chain.append(nodes_np[i]); i = parents_np[i]
            path = np.array([nodes_np[0]] + chain[::-1] + [goal.cpu().numpy()], dtype=np.float64)
        return RRTResult(goal_parent >= 0, path, nodes_np, parents_np, iter_np, it, dt)

    def postprocess(self, path: np.ndarray, smooth_iterations: int = 5) -> np.ndarray:
        """Shortcut then corner-smooth an RRT path, keeping every segment collision-free."""
        return chaikin_smooth(shortcut(path, self.segment_free), self.segment_free, smooth_iterations)
