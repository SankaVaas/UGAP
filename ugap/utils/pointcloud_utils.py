"""
Feature engineering for the evidential uncertainty head. These are cheap,
classical geometric features (density, normal variance, camera distance)
computed on top of the DUSt3R point cloud -- deliberately not learned, so
the uncertainty head has an easy job and the evaluation isolates whether
evidential regression itself helps, not whether a bigger feature extractor
does.

Uses scipy (cKDTree + manual PCA normal estimation) rather than open3d:
Open3D does not currently ship Python 3.13 wheels (Colab's default
runtime as of writing), and pulling in a full point-cloud library just
for KD-tree queries and normal estimation is unnecessary -- both are a
few lines with scipy/numpy.
"""

import numpy as np
from scipy.spatial import cKDTree


def compute_local_density(points: np.ndarray, radius: float = 0.02) -> np.ndarray:
    """Number of neighboring points within `radius`, per point (N,)."""
    tree = cKDTree(points)
    neighbor_lists = tree.query_ball_point(points, r=radius)
    return np.array([len(n) for n in neighbor_lists], dtype=np.float32)


def compute_distance_to_nearest_camera(points: np.ndarray, camera_poses: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest camera center, (N,)."""
    centers = camera_poses[:, :3, 3]  # (V, 3)
    dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)  # (N, V)
    return dists.min(axis=1)


def compute_normal_variance(points: np.ndarray, radius: float = 0.02, max_nn: int = 30) -> np.ndarray:
    """
    Local surface normal variance -- high on thin/transparent structures.

    For each point: take up to `max_nn` neighbors within `radius`, estimate
    a normal via PCA (eigenvector of the neighborhood covariance matrix
    with the smallest eigenvalue), then measure how much those per-point
    normals disagree with each other within the same neighborhood -- a
    noisy/ambiguous local surface (thin plates, reflective highlights)
    gives inconsistent normals, hence high variance.
    """
    tree = cKDTree(points)
    n_points = len(points)
    variance = np.zeros(n_points, dtype=np.float32)

    # First pass: estimate a normal at every point via local PCA.
    normals = np.zeros((n_points, 3), dtype=np.float32)
    for i, p in enumerate(points):
        idx = tree.query_ball_point(p, r=radius)
        if len(idx) < 3:
            normals[i] = np.array([0.0, 0.0, 1.0])
            continue
        if len(idx) > max_nn:
            idx = np.random.default_rng(0).choice(idx, size=max_nn, replace=False)
        neighborhood = points[idx]
        centered = neighborhood - neighborhood.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # eigenvector for the smallest eigenvalue

    # Second pass: variance of neighboring points' normals around each point.
    for i, p in enumerate(points):
        idx = tree.query_ball_point(p, r=radius)
        if len(idx) < 3:
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
