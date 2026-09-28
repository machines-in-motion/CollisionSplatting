"""Throughput and memory of the CollisionSplatting kernel (cf. paper Table I).

    python examples/06_benchmark.py --ply scene.ply [--radius 0.015] [--device cuda|cpu]

Reports sphere checks per second for batch sizes 4,096 and 32,768 and the GPU memory the checker actually uses
(splat parameters + Warp hash grid + per-batch outputs; the CUDA context itself is excluded).
"""
import argparse
import time

import numpy as np
import torch
import warp as wp

from collisionsplatting import CollisionChecker, GaussianScene
from collisionsplatting.datasets import download_marble_sample

p = argparse.ArgumentParser()
p.add_argument("--ply", default=None, help="3DGS .ply (default: a Marble sample world)")
p.add_argument("--radius", type=float, default=0.015)
p.add_argument("--n-sigma", type=float, default=1.0)
p.add_argument("--device", default="cuda")
p.add_argument("--min-opacity", type=float, default=0.1)
p.add_argument("--search-radius", type=float, default=None, help="fixed neighbourhood radius (paper: 0.1); default adapts to robot and splat size")
args = p.parse_args()

ply = args.ply or download_marble_sample()
scene = GaussianScene.from_ply(ply, device=args.device).filter_opacity(args.min_opacity)
lo, hi = scene.bounds(quantile=0.02)
cuda = args.device.startswith("cuda")
if cuda:
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    torch_before = torch.cuda.memory_allocated()
checker = CollisionChecker(scene, n_sigma=args.n_sigma)
print(f"search radius: {args.search_radius or checker.default_radius(args.radius):.4f}")
print(f"{len(scene):,} splats on {torch.cuda.get_device_name(0) if cuda else 'CPU'}")
for N in (4096, 32768):
    pts = torch.as_tensor(lo + np.random.rand(N, 3) * (hi - lo), dtype=torch.float32, device=args.device)
    checker.query_spheres(pts, args.radius, search_radius=args.search_radius)   # warm-up (kernel compilation, grid build)
    reps = 50 if cuda else 3
    wp.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        checker.query_spheres(pts, args.radius, search_radius=args.search_radius)
    wp.synchronize()
    print(f"batch {N:6d}: {reps * N / (time.perf_counter() - t0):14,.0f} checks/s")
if cuda:
    splats = sum(t.numel() * t.element_size() for t in (scene.means, scene.quats, scene.scales))
    grid = wp.get_mempool_used_mem_current(checker.wp_device) if hasattr(wp, "get_mempool_used_mem_current") else float("nan")
    print(f"memory: splat parameters {splats / 2**20:.1f} MiB, Warp hash grid ~{grid / 2**20:.1f} MiB, "
          f"peak torch allocations during queries {(torch.cuda.max_memory_allocated() - torch_before) / 2**20:.1f} MiB")
