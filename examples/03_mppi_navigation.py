"""Receding-horizon MPPI navigation with the CollisionSplatting cost (simulated ground robot).

A 15 cm (radius) ground robot with body-frame velocity commands (v_forward, v_lateral, yaw rate) crosses a cluttered
Marble library. A batched RRT gives sparse waypoints; MPPI tracks them while the CollisionSplatting cost keeps it
clear of furniture. Every MPC step evaluates 512 rollouts x 25 steps of collision cost in one kernel call.

    python examples/03_mppi_navigation.py
"""
import time

import numpy as np
import torch

from collisionsplatting import MPPI, BatchedRRT, CollisionChecker, GaussianScene, planar_unicycle_rollout, resample
from collisionsplatting.datasets import download_marble_sample

from _common import OUT, save_png

R, N_SIGMA, BODY_HEIGHT = 0.15, 0.5, 0.30
START, GOAL = np.array([-1.02, -3.0]), np.array([-0.99, 1.02])      # (x, z) on the floor plane

scene = GaussianScene.from_ply(download_marble_sample("elegant_library_with_fireplace")).drop_floaters()
checker = CollisionChecker(scene, n_sigma=N_SIGMA)
lo, hi = scene.bounds(quantile=0.02)
# Marble scenes are y-down; the floor is the dense layer of splats with the largest y near the room centre
centre = (scene.means[:, [0, 2]] - torch.tensor(0.5 * (lo + hi)[[0, 2]], device="cuda")).abs().max(1).values < 1.0
floor_y = float(torch.quantile(scene.means[centre, 1], 0.97))
Y = floor_y - BODY_HEIGHT
to3d = lambda xz: torch.stack([xz[..., 0], torch.full_like(xz[..., 0], Y), xz[..., 1]], -1)

# ---- global guidance: planar RRT at body height -> waypoints every ~1 m
rrt = BatchedRRT(checker, lower=[lo[0], Y, lo[2]], upper=[hi[0], Y, hi[2]], body_axes=R, step_size=0.1, seed=0)
res = rrt.plan([START[0], Y, START[1]], [GOAL[0], Y, GOAL[1]], max_iterations=3000)
assert res.success, "RRT failed"
guide = rrt.postprocess(res.path)[:, [0, 2]]
waypoints = resample(guide, max(2, int(np.ceil(np.linalg.norm(np.diff(guide, axis=0), axis=1).sum() / 1.0)) + 1))[1:]
print(f"RRT: {res.time:.2f}s, {len(waypoints)} waypoints")

# ---- MPPI
dev = "cuda"
target = torch.tensor(waypoints[0], dtype=torch.float32, device=dev)
rollout = planar_unicycle_rollout(dt=0.1)


def cost(X, U):
    """Tracking + terminal + CollisionSplatting cost (paper Eq.) + hard penalty for s~^2 < 1 + control effort."""
    d = (X[..., :2] - target).norm(dim=-1)
    q = checker.query_spheres(to3d(X[..., :2]), R, search_radius=R + 0.25, d_max=1.0, decay=1.5)
    return (d.mean(1) + 6.0 * d[:, -1] + 4.0 * q.cost.sum(1) + 200.0 * q.in_collision.float().sum(1)
            + 0.02 * (U ** 2).sum((1, 2)))


mppi = MPPI(rollout, cost, nu=3, horizon=25, n_samples=512, n_knots=6, noise_sigma=(0.25, 0.15, 0.6),
            temperature=0.05, u_min=(-0.15, -0.3, -1.2), u_max=(0.6, 0.3, 1.2), iterations=2, device=dev)
heading = np.arctan2(*(waypoints[0] - START)[::-1])
x = torch.tensor([START[0], START[1], heading], dtype=torch.float32, device=dev)
traj, times, wi = [x.cpu().numpy()], [], 0
for step in range(600):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    mppi.optimize(x)
    torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    x = rollout(x, mppi.controls[:1][None])[0, 0]        # apply the first control for one step
    mppi.shift()
    traj.append(x.cpu().numpy())
    if wi < len(waypoints) - 1 and (x[:2] - target).norm() < 0.4:
        wi += 1; target = torch.tensor(waypoints[wi], dtype=torch.float32, device=dev)
    if wi == len(waypoints) - 1 and (x[:2] - target).norm() < 0.15:
        break
traj = np.array(traj)
dense = torch.tensor(resample(traj[:, :2], 2000), dtype=torch.float32, device=dev)
ok = bool(checker.is_free(to3d(dense), R).all())
print(f"reached goal: {np.linalg.norm(traj[-1, :2] - GOAL) < 0.2} after {len(traj) - 1} steps; "
      f"executed path collision-free: {ok}; MPC step {1e3 * np.median(times):.0f} ms (median)")
np.savez(OUT / "03_mppi_navigation.npz", traj=traj, waypoints=waypoints, y=Y)

try:
    from collisionsplatting.render import intrinsics, look_at, merge, polyline_splats, render, sphere_splats
    view = scene.select(scene.means[:, 1] > floor_y - 1.7)          # drop the ceiling
    path3 = to3d(torch.tensor(traj[:, :2], dtype=torch.float32)).numpy()
    wp3 = to3d(torch.tensor(waypoints, dtype=torch.float32)).numpy()
    overlay = merge([view, polyline_splats(path3, width=0.02), *[sphere_splats(w, 0.05, (1, 1, 1)) for w in wp3],
                     sphere_splats(path3[0], R, (0.2, 1, 0.35)), sphere_splats(path3[-1], R)])
    c = 0.5 * (lo + hi); W, H = 960, 1280
    img = render(overlay, look_at(c + np.array([0, -9.5, 0]), c, up=(0, 0, -1)), intrinsics(W, H, 45), W, H)
    save_png(img.cpu().numpy(), "03_mppi_navigation_topdown.png")
except ImportError:
    print("gsplat not installed: skipping render")
