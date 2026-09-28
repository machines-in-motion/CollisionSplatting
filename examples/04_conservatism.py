"""Tunable conservatism: plan the same query at several n_sigma values and compare.

Works on any 3DGS ``.ply``. The defaults reproduce the paper's Stonehenge setting (normalized scene, z-up, planning
at z = 0.08 with a sphere of radius 1.5% of the scene): larger n_sigma inflates uncertain splats, so narrow gaps
close and the planner keeps more clearance.

    python examples/04_conservatism.py --ply path/to/stonehenge_3dgs.ply
"""
import argparse

import numpy as np
import torch

from collisionsplatting import BatchedRRT, CollisionChecker, GaussianScene, path_length

from _common import OUT, save_png

p = argparse.ArgumentParser()
p.add_argument("--ply", required=True)
p.add_argument("--start", type=float, nargs=3, default=[0.08, 0.007, 0.08])
p.add_argument("--goal", type=float, nargs=3, default=[-0.545, 0.401, 0.08])
p.add_argument("--plane-axis", type=int, default=2, help="coordinate held fixed (2 = z for a z-up scene)")
p.add_argument("--radius", type=float, default=0.015)
p.add_argument("--n-sigmas", type=float, nargs="+", default=[0.0, 1.0, 2.0])
p.add_argument("--min-opacity", type=float, default=0.1, help="opacity filter used in the paper's evaluation")
p.add_argument("--no-render", action="store_true")
args = p.parse_args()

scene = GaussianScene.from_ply(args.ply).filter_opacity(args.min_opacity)
lo, hi = scene.bounds(quantile=0.01)
lo[args.plane_axis] = hi[args.plane_axis] = args.start[args.plane_axis]
results = {}
print(f"{'n_sigma':>8} {'length':>8} {'clearance':>10} {'time [s]':>9}")
for ns in args.n_sigmas:
    checker = CollisionChecker(scene, n_sigma=ns)
    rrt = BatchedRRT(checker, lo, hi, body_axes=args.radius, step_size=0.03, seed=0)
    try:
        res = rrt.plan(args.start, args.goal, max_iterations=4000)
    except ValueError as e:                      # start/goal themselves inside the inflated obstacles
        print(f"{ns:8.2f}   {e}"); continue
    if not res.success:
        print(f"{ns:8.2f}   no path"); continue
    path = rrt.postprocess(res.path)
    dense = torch.tensor(np.concatenate([np.linspace(a, b, 20) for a, b in zip(path[:-1], path[1:])]), dtype=torch.float32, device="cuda")
    clearance = min(float((dense[i:i + 256, None] - scene.means[None]).norm(dim=-1).min()) for i in range(0, len(dense), 256))
    results[ns] = (res, path)
    print(f"{ns:8.2f} {path_length(path):8.3f} {clearance:10.4f} {res.time:9.2f}")
np.savez(OUT / "04_conservatism.npz", **{f"path_nsigma_{k}": v[1] for k, v in results.items()})

if not args.no_render and results:
    from collisionsplatting.render import intrinsics, look_at, merge, polyline_splats, render, sphere_splats
    colors = [(0.9, 0.25, 0.2), (1.0, 0.7, 0.1), (0.2, 0.8, 0.3), (0.3, 0.6, 1.0)]
    parts = [scene] + [polyline_splats(path, colors[i % 4], width=0.004) for i, (_, (_, path)) in enumerate(results.items())]
    parts += [sphere_splats(args.start, 0.012, (0.2, 1, 0.35)), sphere_splats(args.goal, 0.012, (1, 0.2, 0.2))]
    c = 0.5 * (np.array(args.start) + np.array(args.goal)); W, H = 1280, 960
    eye = c + np.array([0.9, -0.9, 1.3]) if args.plane_axis == 2 else c + np.array([0.0, -2.0, 0.5])
    up = (0, 0, 1) if args.plane_axis == 2 else (0, -1, 0)
    img = render(merge(parts), look_at(eye, c, up=up), intrinsics(W, H, 50), W, H)
    save_png(img.cpu().numpy(), "04_conservatism.png")
    print("path colours: " + ", ".join(f"n_sigma={k}: {['red', 'amber', 'green', 'blue'][i % 4]}" for i, k in enumerate(results)))
