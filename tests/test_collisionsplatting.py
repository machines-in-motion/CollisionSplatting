"""Unit tests (run on CPU: ``pytest``)."""
import math

import numpy as np
import pytest
import torch

from collisionsplatting import BatchedRRT, CollisionChecker, GaussianScene, RobotModel, spline_basis
from collisionsplatting.paths import shortcut
from collisionsplatting.scene import quat_to_matrix

DEV = "cpu"


def scene_from(means, quats=None, scales=None):
    means = torch.as_tensor(means, dtype=torch.float32).reshape(-1, 3)
    n = means.shape[0]
    q = torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4).clone() if quats is None else torch.as_tensor(quats, dtype=torch.float32)
    s = torch.full((n, 3), 1e-4) if scales is None else torch.as_tensor(scales, dtype=torch.float32).reshape(n, 3)
    return GaussianScene(means, q, s, torch.ones(n))


def test_scaling_distance_of_a_point_splat():
    ch = CollisionChecker(scene_from([[0.5, 0.0, 0.0]]), n_sigma=0.0, max_splat_scale=1.0)
    r = ch.query(torch.zeros(1, 3), [0.6, 0.2, 0.2], search_radius=1.0)
    assert r.min_distance.item() == pytest.approx((0.5 / 0.6) ** 2, rel=1e-4)
    assert r.count.item() == 1 and bool(r.in_collision.item())


def test_ellipsoid_rotation_convention():
    ch = CollisionChecker(scene_from([[0.5, 0.0, 0.0]]), n_sigma=0.0, max_splat_scale=1.0)
    Rz90 = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])      # body x-axis -> world y-axis
    r = ch.query(torch.zeros(1, 3), [0.6, 0.2, 0.2], Rz90[None], search_radius=1.0)
    assert r.min_distance.item() == pytest.approx((0.5 / 0.2) ** 2, rel=1e-4)
    assert not bool(r.in_collision.item())


def test_sigma_matches_linearized_covariance_propagation():
    """s~^2 = s^2 - n_sigma * sqrt(g^T Sigma g) with Sigma = R S S^T R^T (3DGS (w,x,y,z) convention)."""
    torch.manual_seed(0)
    a = 0.3
    for _ in range(20):
        mu = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1) * 0.5
        q = torch.nn.functional.normalize(torch.randn(1, 4), dim=-1)
        s = torch.tensor([[0.12, 0.02, 0.005]])
        R = quat_to_matrix(q)[0]
        Sigma = R @ torch.diag(s[0] ** 2) @ R.T
        g = 2 * mu[0] / a ** 2
        expected = max((mu[0] @ mu[0]).item() / a ** 2 - 1.0 * math.sqrt(g @ Sigma @ g), 0.0)
        ch = CollisionChecker(scene_from(mu, q, s), n_sigma=1.0, max_splat_scale=1.0)
        got = ch.query_spheres(torch.zeros(1, 3), a, search_radius=1.0).min_distance.item()
        assert got == pytest.approx(expected, abs=1e-4)


def test_conservatism_is_monotonic():
    torch.manual_seed(1)
    means = torch.rand(400, 3) * 2 - 1
    q = torch.nn.functional.normalize(torch.randn(400, 4), dim=-1)
    sc = scene_from(means, q, torch.rand(400, 3) * 0.05)
    pts = torch.rand(200, 3) * 2 - 1
    d = [CollisionChecker(sc, n_sigma=ns, max_splat_scale=1.0).query_spheres(pts, 0.1, search_radius=0.5).min_distance
         for ns in (0.0, 1.0, 2.0)]
    assert torch.all(d[1] <= d[0] + 1e-6) and torch.all(d[2] <= d[1] + 1e-6)


def test_empty_neighbourhood_is_far():
    ch = CollisionChecker(scene_from([[10.0, 10.0, 10.0]]), max_splat_scale=1.0)
    r = ch.query_spheres(torch.zeros(1, 3), 0.1, search_radius=0.5)
    assert r.distance.item() > 1e9 and r.count.item() == 0 and r.cost.item() == 0.0


def test_ply_roundtrip(tmp_path):
    from plyfile import PlyData, PlyElement
    n = 5
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.zeros(n, dtype=[(p, "f4") for p in props])
    arr["x"] = np.arange(n); arr["scale_0"] = np.log(0.1); arr["scale_1"] = np.log(0.2); arr["scale_2"] = np.log(0.3)
    arr["rot_0"] = 2.0; arr["opacity"] = 0.0
    p = tmp_path / "s.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(p))
    sc = GaussianScene.from_ply(p, device=DEV)
    assert len(sc) == n
    assert torch.allclose(sc.scales[0], torch.tensor([0.1, 0.2, 0.3]), atol=1e-6)
    assert torch.allclose(sc.quats[0], torch.tensor([1.0, 0, 0, 0]))           # normalized
    assert torch.allclose(sc.opacities, torch.full((n,), 0.5))                 # sigmoid(0)
    assert torch.allclose(sc.colors, torch.full((n, 3), 0.5))                  # SH DC 0 -> 0.5 grey


def test_drop_floaters_removes_only_large_transparent_splats():
    sc = scene_from(torch.zeros(4, 3), scales=[[0.01] * 3, [0.01] * 3, [1.0] * 3, [1.0] * 3])
    sc.opacities = torch.tensor([0.9, 0.01, 0.9, 0.01])
    assert len(sc.drop_floaters(min_opacity=0.1, size_multiple=5.0)) == 3


URDF = """<robot name="toy">
  <link name="base"/>
  <link name="l1"><collision><origin xyz="0.5 0 0"/><geometry><ellipsoid scale="0.5 0.05 0.05"/></geometry></collision></link>
  <link name="l2"><collision><geometry><sphere radius="0.1"/></geometry></collision></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
</robot>"""


def test_forward_kinematics_planar_two_link(tmp_path):
    f = tmp_path / "toy.urdf"; f.write_text(URDF)
    robot = RobotModel.from_urdf(f)
    assert robot.dof == 2 and len(robot.ellipsoids) == 2
    q = torch.tensor([[math.pi / 2, 0.0]])
    poses = robot.forward_kinematics(q)
    assert torch.allclose(poses["l2"][0, :3, 3], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    centers, rotations, axes = robot.collision_ellipsoids(q)
    assert torch.allclose(centers[0, 0], torch.tensor([0.0, 0.5, 0.0]), atol=1e-6)   # link-1 ellipsoid mid-link
    assert centers.shape == (1, 2, 3) and rotations.shape == (1, 2, 3, 3) and axes.shape == (2, 3)


def test_spline_basis_interpolates_knots():
    for kind in ("linear", "cubic"):
        B = spline_basis(5, 9, kind)                   # samples at t = 0, .125, ..., 1 include every knot time
        assert torch.allclose(B[::2], torch.eye(5), atol=1e-6)
        assert torch.allclose(B.sum(1), torch.ones(9), atol=1e-6)


def test_rrt_finds_path_through_a_gap():
    ys = torch.linspace(-1, 1, 81)
    wall = torch.stack([torch.zeros_like(ys), ys, torch.zeros_like(ys)], 1)
    wall = wall[(ys.abs() > 0.2)]                                            # gap of width 0.4 at y = 0
    sc = scene_from(wall, scales=torch.full((wall.shape[0], 3), 0.005))
    ch = CollisionChecker(sc, n_sigma=0.0, max_splat_scale=1.0)
    rrt = BatchedRRT(ch, lower=[-1, -1, 0], upper=[1, 1, 0], body_axes=0.05, step_size=0.1, seed=0)
    res = rrt.plan([-0.8, 0.8, 0.0], [0.8, 0.8, 0.0], max_iterations=3000)
    assert res.success
    assert np.abs(res.path[:, 1]).min() < 0.2 + 1e-6                     # must pass through the gap
    short = rrt.postprocess(res.path)
    assert all(rrt.segment_free(a, b) for a, b in zip(short[:-1], short[1:]))


def test_shortcut_straightens_free_space():
    path = np.array([[0, 0, 0], [0.5, 0.5, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    assert len(shortcut(path, lambda a, b: True)) == 2
