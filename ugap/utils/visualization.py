"""
Lightweight matplotlib-based visualizations -- deliberately not relying on
open3d's interactive viewer since Colab's headless rendering is finicky.
Good enough for notebook inline plots and figures for a paper draft.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_uncertainty_pointcloud(
    points: np.ndarray,
    uncertainty: np.ndarray,
    title: str = "Per-point epistemic uncertainty",
    max_points: int = 20000,
):
    """Scatter plot of the point cloud colored by uncertainty."""
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points, uncertainty = points[idx], uncertainty[idx]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=uncertainty, cmap="inferno", s=1)
    fig.colorbar(sc, ax=ax, label="epistemic uncertainty")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    return fig


def plot_view_coverage(
    points: np.ndarray,
    camera_positions: np.ndarray,
    proposed_view: np.ndarray = None,
    title: str = "Camera coverage and proposed next-best-view",
    max_points: int = 20000,
):
    """Plot the scene with existing camera positions and (optionally) the
    next proposed viewpoint, so you can sanity-check the NBV planner
    visually before running it on a real robot."""
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="gray", s=1, alpha=0.4, label="scene")
    ax.scatter(
        camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
        c="blue", marker="^", s=80, label="existing views",
    )
    if proposed_view is not None:
        ax.scatter(
            proposed_view[0], proposed_view[1], proposed_view[2],
            c="red", marker="*", s=200, label="proposed next view",
        )
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig
