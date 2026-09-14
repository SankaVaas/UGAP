"""
Feature engineering for the evidential uncertainty head. These are cheap,
classical geometric features (density, normal variance, camera distance)
computed on top of the DUSt3R point cloud -- deliberately not learned, so
the uncertainty head has an easy job and the evaluation isolates whether
evidential regression itself helps, not whether a bigger feature extractor
does.
"""

import numpy as np

try:
    import open3d as o3d
except ImportError:  # allow import of this module before open3d is installed
    o3d = None


def compute_local_density(points: np.ndarray, radius: float = 0.02) -> np.ndarray:
    """Number of neighboring points within `radius`, per point (N,)."""
    if o3d is None:
        raise ImportError("open3d is required: pip install open3d")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    density = np.zeros(len(points), dtype=np.float32)
    for i, p in enumerate(points):
        k, idx, _ = kdtree.search_radius_vector_3d(p, radius)
        density[i] = k

    return density


def compute_distance_to_nearest_camera(points: np.ndarray, camera_poses: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest camera center, (N,)."""
    centers = camera_poses[:, :3, 3]  # (V, 3)
    dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)  # (N, V)
    return dists.min(axis=1)


def compute_normal_variance(points: np.ndarray, radius: float = 0.02) -> np.ndarray:
    """Local surface normal variance -- high on thin/transparent structures."""
    if o3d is None:
        raise ImportError("open3d is required: pip install open3d")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    normals = np.asarray(pcd.normals)

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    variance = np.zeros(len(points), dtype=np.float32)
    for i, p in enumerate(points):
        k, idx, _ = kdtree.search_radius_vector_3d(p, radius)
        if k < 3:
            variance[i] = 0.0
            continue
        neighbor_normals = normals[idx]
        variance[i] = neighbor_normals.var(axis=0).sum()

    return variance


def build_evidential_features(
    points: np.ndarray,
    colors: np.ndarray,
    dust3r_confidence: np.ndarray,
    camera_poses: np.ndarray,
    n_views_observed: np.ndarray,
    density_radius: float = 0.02,
) -> np.ndarray:
    """
    Assembles the 6-D feature vector consumed by EvidentialUncertaintyHead:
        [dust3r_confidence, local_density, dist_to_nearest_cam,
         normal_variance, rgb_gradient_proxy, n_views_observed]
    """
    density = compute_local_density(points, radius=density_radius)
    dist_to_cam = compute_distance_to_nearest_camera(points, camera_poses)
    normal_var = compute_normal_variance(points, radius=density_radius)

    # Cheap proxy for local RGB gradient magnitude without needing full
    # image-space neighborhoods: color variance among nearest neighbors.
    rgb_gradient_proxy = np.linalg.norm(np.gradient(colors, axis=0), axis=1)

    features = np.stack(
        [
            dust3r_confidence,
            density,
            dist_to_cam,
            normal_var,
            rgb_gradient_proxy,
            n_views_observed.astype(np.float32),
        ],
        axis=1,
    )
    return features.astype(np.float32)
