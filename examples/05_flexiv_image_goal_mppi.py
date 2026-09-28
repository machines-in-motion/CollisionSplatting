"""Image-goal reaching with a 7-DoF arm: DINOv2 image cost + CollisionSplatting cost in one MPPI loop (paper Sec. IV-A).

The robot must move its wrist camera so that what it *sees* matches a goal image -- a book lying inside a crate --
while the CollisionSplatting cost keeps the end effector out of the crate walls. Every MPPI iteration renders the
wrist view of each rollout's final state with the 3DGS rasterizer and scores it with DINOv2 against the goal image.
Run it with and without the collision cost to see the difference.

Requires the lab scene (``--scene``, a gsplat checkpoint of the Flexiv table) plus ``gsplat`` and internet access
for the DINOv2 weights (``torch.hub``).

    python examples/05_flexiv_image_goal_mppi.py --scene flexiv_table_lab.pt
    python examples/05_flexiv_image_goal_mppi.py --scene flexiv_table_lab.pt --no-collision-cost
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from collisionsplatting import CollisionChecker, GaussianScene, RobotModel, spline_basis
from collisionsplatting.render import merge, render

from _common import OUT, save_png

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "flexiv_rizon10s"
p = argparse.ArgumentParser()
p.add_argument("--scene", required=True, help="gsplat checkpoint of the lab table scene (2DGS)")
p.add_argument("--target", type=int, default=1, help="goal index 0..4 (goal images rendered from stored camera poses)")
p.add_argument("--n-sigma", type=float, default=1.0)
p.add_argument("--no-collision-cost", action="store_true")
p.add_argument("--iterations", type=int, default=65)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
dev = "cuda"
torch.manual_seed(args.seed)

# ---------------------------------------------------------------- scene, robot, camera
scene = GaussianScene.from_gsplat_checkpoint(args.scene, flatten_2dgs=True)        # 2DGS surfels
checker = CollisionChecker(scene, n_sigma=args.n_sigma)
robot = RobotModel.from_urdf(ASSETS / "robot_gaussian.urdf")
demo = np.load(ASSETS / "lab_demo.npz")
q0 = torch.tensor(demo["q0"], device=dev)
link7_T_cam = torch.tensor(demo["link7_T_camera"], device=dev)
W = H = 224
fx = 650 / 720 * W
K = torch.tensor([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], device=dev)
lim = torch.tensor(robot.joint_limits, dtype=torch.float32, device=dev)


def camera_poses(q):
    """World pose of the wrist camera for configurations ``q`` (..., 7)."""
    return robot.forward_kinematics(q)["link7"] @ link7_T_cam


def wrist_images(q):
    return render(scene, torch.linalg.inv(camera_poses(q)), K, W, H)


# ---------------------------------------------------------------- DINOv2 image cost
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(dev).eval()
MEAN, STD = torch.tensor([0.485, 0.456, 0.406], device=dev), torch.tensor([0.229, 0.224, 0.225], device=dev)


@torch.no_grad()
def dino_tokens(imgs):                                        # (B, H, W, 3) in [0, 1] -> (B, 1 + patches, D)
    x = ((imgs - MEAN) / STD).permute(0, 3, 1, 2)
    f = dino.forward_features(F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False))
    return torch.cat([f["x_norm_clstoken"][:, None], f["x_norm_patchtokens"]], 1)


goal_img = render(scene, torch.linalg.inv(torch.tensor(demo["target_camera_poses"][args.target], device=dev)), K, W, H)
goal_tok = dino_tokens(goal_img[None])


def image_cost(imgs):
    """Weighted L1 distance of DINOv2 CLS (global) and patch (local) tokens, as in the paper's experiments."""
    t = dino_tokens(imgs)
    glob = (t[:, 0] - goal_tok[:, 0]).abs().mean(-1)
    loc = (t[:, 1:] - goal_tok[:, 1:]).abs().flatten(1).mean(-1)
    return 2.5 * loc + 0.5 * glob


# ---------------------------------------------------------------- joint-velocity MPPI over spline knots
N, T, NK, DT, ITERS = 32, 48, 8, 0.05, args.iterations
B = spline_basis(NK, T, "cubic", dev)                         # (T, NK)
n_idx = torch.arange(50, device=dev)[:, None].float(); h_idx = torch.arange(NK, device=dev)[None].float()
sigma_sched = torch.exp(-n_idx / 50.0 - (NK - h_idx) / (0.8 * NK))   # annealed, larger towards the end of the horizon
w_col = torch.zeros(50, device=dev); w_col[30:] = 0.1                   # collision term enters after 30 iterations


def rollout(knots):                                            # (N, NK, 7) joint velocities -> (N, T, 7) positions
    qd = B[None] @ knots
    return torch.maximum(torch.minimum(q0 + torch.cumsum(qd * DT, 1), lim[:, 1]), lim[:, 0])


def ee_collision(q):                                           # end-effector sphere (8 cm), as in the paper's experiment
    ee = robot.forward_kinematics(q)["endeffector"][..., :3, 3]
    return checker.query_spheres(ee, 0.08, d_max=10.0, decay=2.5)


knots = torch.zeros(NK, 7, device=dev)
torch.cuda.synchronize(); t0 = time.perf_counter()
for it in range(ITERS):
    s = sigma_sched[min(it, 49)].view(1, NK, 1)
    eps = torch.randn(N, NK, 7, device=dev) * 0.6 * s ** 2
    eps[:, 0] = eps[:, -1] = 0.0; eps[:, :, 2] = 0.0; eps[0] = 0.0          # fixed boundary knots, joint 3 held
    cand = knots[None] + eps
    Q = rollout(cand)
    J = image_cost(wrist_images(Q[:, -1]))
    ee = robot.forward_kinematics(Q)["endeffector"][..., :3, 3]
    J = J + (ee[:, 1:] - ee[:, :-1]).norm(dim=-1).sum(1)                  # end-effector path length
    if not args.no_collision_cost:
        J = J + w_col[min(it, 49)] * ee_collision(Q).cost.sum(1)
    w = torch.softmax(-(J - J.min()) / 0.1, 0)
    knots = (w[:, None, None] * cand).sum(0)
torch.cuda.synchronize(); opt_time = time.perf_counter() - t0

# ---------------------------------------------------------------- evaluate the executed plan
q_exec = rollout(knots[None])[0]
ee = robot.forward_kinematics(q_exec)["endeffector"][..., :3, 3]
clearance = min(float((ee[i:i + 8, None] - scene.means[None]).norm(dim=-1).min(-1).values.min()) for i in range(0, T, 8)) - 0.08
Tc, Tg = camera_poses(q_exec[-1]), torch.tensor(demo["target_camera_poses"][args.target], device=dev)
t_err = float((Tc[:3, 3] - Tg[:3, 3]).norm())
r_err = math.degrees(math.acos(float(torch.clamp((torch.trace(Tc[:3, :3].T @ Tg[:3, :3]) - 1) / 2, -1, 1))))
tag = "no_collision_cost" if args.no_collision_cost else "with_collision_cost"
print(f"[{tag}] optimized in {opt_time:.1f}s | closest splat to end effector: {100 * clearance:+.1f} cm "
      f"({'penetrates' if clearance < 0 else 'clear'}) | camera error {100 * t_err:.1f} cm / {r_err:.1f} deg")
final = wrist_images(q_exec[-1:])[0]
save_png(torch.cat([goal_img, torch.ones(H, 6, 3, device=dev), final], 1).cpu().numpy(), f"05_goal_vs_final_{tag}.png")
np.savez(OUT / f"05_flexiv_{tag}.npz", q=q_exec.cpu().numpy(), clearance=clearance, t_err=t_err, r_err=r_err)
