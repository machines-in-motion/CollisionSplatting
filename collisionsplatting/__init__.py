"""CollisionSplatting: collision-aware planning directly on standard 3D Gaussian Splatting scenes.

A fast, memory-light GPU collision kernel with tunable conservatism that enables collision-aware planning and
real-time control directly on standard 3DGS scenes -- from SfM/SLAM captures to generative 3D worlds -- with no mesh
conversion required.

Quick start::

    from collisionsplatting import GaussianScene, CollisionChecker
    scene = GaussianScene.from_ply("scene.ply").drop_floaters()
    checker = CollisionChecker(scene, n_sigma=1.0)
    free = checker.is_free(points, radius=0.05)
"""
from .checker import CollisionChecker, QueryResult
from .kinematics import RobotModel
from .paths import chaikin_smooth, path_length, path_smoothness, resample, shortcut, spline_basis
from .planners import MPPI, BatchedRRT, RRTResult, planar_unicycle_rollout
from .scene import GaussianScene

__version__ = "1.0.0"

__all__ = ["GaussianScene", "CollisionChecker", "QueryResult", "BatchedRRT", "RRTResult", "MPPI",
           "planar_unicycle_rollout", "RobotModel", "spline_basis", "resample", "shortcut", "chaikin_smooth",
           "path_length", "path_smoothness", "__version__"]
