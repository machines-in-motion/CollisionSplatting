"""NVIDIA Warp kernels implementing the CollisionSplatting metric (paper, Sec. II).

For a robot link modelled as an ellipsoid ``E = {X : (X - c)^T A (X - c) <= 1}`` with ``A = R diag(1/a^2) R^T`` and a
splat ``G ~ N(mu, Sigma)``, ``Sigma = (R_g S)(R_g S)^T``:

* scaling distance       ``s^2      = (mu - c)^T A (mu - c)``               (< 1 means the splat centre is inside)
* propagated uncertainty ``sigma^2  = grad^T Sigma grad``, ``grad = 2 A (mu - c)``   (first-order / linearized)
* conservative distance  ``s~^2     = max(0, s^2 - n_sigma * sigma)``

Every query aggregates ``s~^2`` over the splats returned by a spatial hash grid with a numerically stable streaming
log-mean-exp soft-minimum, so each thread keeps O(1) state: no per-(query, splat) tensors are ever materialized.
"""
import warp as wp


@wp.func
def splat_axes(q: wp.vec4, s: wp.vec3):
    """``R_g @ diag(s)`` for a splat with (w, x, y, z) quaternion ``q`` and scales ``s``: its scaled principal axes."""
    n = wp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    w = q[0] / n
    x = q[1] / n
    y = q[2] / n
    z = q[3] / n
    R = wp.mat33(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                 2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                 2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))
    return R * wp.diag(s)


@wp.kernel
def ellipsoid_query_kernel(
    grid: wp.uint64,
    means: wp.array(dtype=wp.vec3),
    quats: wp.array(dtype=wp.vec4),
    scales: wp.array(dtype=wp.vec3),
    centers: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.mat33),
    axes: wp.array(dtype=wp.vec3),
    radius: float,
    n_sigma: float,
    max_splat_scale: float,
    softmin_gamma: float,
    d_max: float,
    decay: float,
    s_min: float,
    out_softmin: wp.array(dtype=float),
    out_min: wp.array(dtype=float),
    out_count: wp.array(dtype=wp.int32),
    out_cost: wp.array(dtype=float),
):
    """One thread per query ellipsoid (a robot link at one sample / time step).

    Outputs per query:
      * ``out_softmin`` soft-minimum of ``s~^2`` over neighbouring splats (``1e10`` if there are none)
      * ``out_min``     hard minimum of ``s~^2``
      * ``out_count``   number of splats with ``s~^2 < 1`` (the RRT collision test)
      * ``out_cost``    paper Eq. (MPPI cost): mean over neighbours of ``d_max * exp(-decay * max(s~^2 - s_min, 0))``
    """
    tid = wp.tid()
    c = centers[tid]
    Rl = rotations[tid]
    a = axes[tid]
    A = Rl * wp.diag(wp.vec3(1.0 / (a[0] * a[0]), 1.0 / (a[1] * a[1]), 1.0 / (a[2] * a[2]))) * wp.transpose(Rl)

    query = wp.hash_grid_query(grid, c, radius)
    idx = int(0)
    m = float(1.0e10)        # running minimum (log-mean-exp pivot)
    sum_exp = float(0.0)
    n = float(0.0)
    count = int(0)
    hard_min = float(1.0e10)
    cost_sum = float(0.0)
    while wp.hash_grid_query_next(query, idx):
        d = means[idx] - c
        if wp.length(d) <= radius:
            s = scales[idx]
            if wp.max(s) <= max_splat_scale:       # very large splats are almost always outliers / floaters
                Ad = A * d
                s2 = wp.dot(d, Ad)
                j = wp.transpose(splat_axes(quats[idx], s)) * (2.0 * Ad)   # (R_g S)^T grad  ->  |j| = sigma
                st = wp.max(s2 - n_sigma * wp.length(j), 0.0)
                if st < 1.0:
                    count += 1
                hard_min = wp.min(hard_min, st)
                if st < m:
                    sum_exp = sum_exp * wp.exp(-softmin_gamma * (m - st)) + 1.0
                    m = st
                else:
                    sum_exp = sum_exp + wp.exp(-softmin_gamma * (st - m))
                n = n + 1.0
                cost_sum = cost_sum + d_max * wp.exp(-decay * wp.max(st - s_min, 0.0))
    if n > 0.0:
        out_softmin[tid] = m - wp.log(sum_exp / n) / softmin_gamma
        out_cost[tid] = cost_sum / n
    else:
        out_softmin[tid] = 1.0e10
        out_cost[tid] = 0.0
    out_min[tid] = hard_min
    out_count[tid] = count
