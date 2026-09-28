# Assets

* `flexiv_rizon10s/` — Flexiv Rizon 10s model used in the paper's manipulation experiment:
  `robot_gaussian.urdf` (kinematics + ellipsoidal collision bodies via the `<ellipsoid scale="a b c"/>` geometry),
  per-link Gaussian-splat models in `gaussians/` (for rendering the robot inside a 3DGS scene), visual meshes in
  `meshes/`, the wrist-camera calibration `flexiv-ef-cam-jan18.pkl`, and `lab_demo.npz` (initial joint configuration,
  wrist-camera extrinsics and the five goal camera poses used by `examples/05_flexiv_image_goal_mppi.py`).
