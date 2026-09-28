"""Batched GPU RRT on a generated 3DGS world, then shortcut + smooth, then render the result.

A 16 cm flying sphere plans from mid-air by the door to a spot by the sink in a Marble (World Labs) kitchen,
directly on the raw splats.

    python examples/02_rrt_marble.py            # add --no-render to skip gsplat
"""
import argparse

import numpy as np
import torch

from collisionsplatting import BatchedRRT, CollisionChecker, GaussianScene, path_length
from collisionsplatting.datasets import download_marble_sample

from _common import OUT, save_png

p = argparse.ArgumentParser()
p.add_argument("--scene", default="rustic_kitchen_with_natural_light")
p.add_argument("--start", type=float, nargs=3, default=[-0.62, -0.04, -0.25])
p.add_argument("--goal", type=float, nargs=3, default=[-0.28, 0.72, 2.38])
p.add_argument("--radius", type=float, default=0.08, help="robot sphere radius (scene units ~ metres)")
p.add_argument("--n-sigma", type=float, default=1.0)
p.add_argument("--no-render", action="store_true")
args = p.parse_args()

scene = GaussianScene.from_ply(download_marble_sample(args.scene)).drop_floaters()
checker = CollisionChecker(scene, n_sigma=args.n_sigma)
lo, hi = scene.bounds(quantile=0.03)

rrt = BatchedRRT(checker, lower=lo, upper=hi, body_axes=args.radius, step_size=0.08, batch_size=32, seed=1)
res = rrt.plan(args.start, args.goal, max_iterations=3000)
print(f"success={res.success} iterations={res.iterations} nodes={len(res.nodes):,} time={res.time:.2f}s")
if not res.success:
    raise SystemExit("no path found")
path = rrt.postprocess(res.path)
print(f"path length: RRT {res.length:.2f} -> smoothed {path_length(path):.2f} "
      f"(straight line {np.linalg.norm(np.subtract(args.goal, args.start)):.2f}); "
      f"collision-free: {all(rrt.segment_free(a, b) for a, b in zip(path[:-1], path[1:]))}")
np.savez(OUT / "02_rrt_result.npz", nodes=res.nodes, parents=res.parents, path=path, raw_path=res.path)

if not args.no_render:
    from collisionsplatting.render import intrinsics, look_at, merge, polyline_splats, render, sphere_splats, tree_splats
    # Marble scenes are y-down: crop the ceiling for a dollhouse view and look from above
    ceiling_cut = scene.select(scene.means[:, 1] > lo[1] + 0.25 * (hi[1] - lo[1]))
    overlay = merge([ceiling_cut, tree_splats(res.nodes, res.parents, width=0.004), polyline_splats(path, width=0.012),
                     sphere_splats(path[len(path) // 2], args.radius), sphere_splats(args.start, 0.035, (0.2, 1, 0.35)),
                     sphere_splats(args.goal, 0.035, (1, 0.2, 0.2))])
    center = 0.5 * (lo + hi)
    W, H = 1280, 960
    top = render(overlay, look_at(center + np.array([0, -7.0, 0.0]), center, up=(0, 0, -1)), intrinsics(W, H, 42), W, H)
    save_png(top.cpu().numpy(), "02_rrt_topdown.png")
