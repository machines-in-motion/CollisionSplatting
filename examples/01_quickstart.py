"""Quickstart: load a raw 3DGS export and query collisions -- no mesh, no retraining.

Downloads a public Marble (World Labs) sample world, checks a slice of points for a 5 cm sphere at several
conservatism levels, and reports throughput.

    python examples/01_quickstart.py
"""
import time

import numpy as np
import torch

from collisionsplatting import CollisionChecker, GaussianScene
from collisionsplatting.datasets import download_marble_sample

from _common import save_png

# 1. load any standard 3DGS .ply (here: a generated kitchen, 500k splats, OpenCV y-down frame)
scene = GaussianScene.from_ply(download_marble_sample("rustic_kitchen_with_natural_light"))
scene = scene.drop_floaters()                       # discard large, nearly transparent splats (paper Sec. II-C)
print(f"{len(scene):,} splats, SH degree {scene.metadata['sh_degree']}, dropped {scene.metadata['dropped_floaters']} floaters")

# 2. a horizontal slice of query points through the room (y is "down" in Marble scenes)
lo, hi = scene.bounds(quantile=0.05)
xs, zs = np.linspace(lo[0], hi[0], 400), np.linspace(lo[2], hi[2], 400)
X, Z = np.meshgrid(xs, zs)
pts = torch.tensor(np.stack([X.ravel(), np.full(X.size, 0.0), Z.ravel()], 1), dtype=torch.float32, device="cuda")

# 3. query a 5 cm sphere at every point for a few conservatism levels
images = []
for n_sigma in (0.0, 1.0, 2.0):
    checker = CollisionChecker(scene, n_sigma=n_sigma)
    res = checker.query_spheres(pts, radius=0.05)
    free = (~res.in_collision).reshape(X.shape).cpu().numpy()
    print(f"n_sigma = {n_sigma}: {100 * free.mean():.1f}% of the slice is free")
    images.append(np.where(free, 0.92, 0.15))

# 4. throughput: 32k sphere queries per call
checker = CollisionChecker(scene, n_sigma=1.0)
batch = pts[torch.randint(0, len(pts), (32768,), device="cuda")]
checker.query_spheres(batch, 0.05); torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    checker.query_spheres(batch, 0.05)
torch.cuda.synchronize()
print(f"throughput: {20 * 32768 / (time.perf_counter() - t0):,.0f} sphere checks / s")

gap = np.ones((X.shape[0], 8))
save_png(np.repeat(np.concatenate([images[0], gap, images[1], gap, images[2]], 1)[..., None], 3, -1), "01_free_space_nsigma_0_1_2.png")
