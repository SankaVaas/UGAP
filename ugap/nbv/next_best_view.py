"""
Deliberately classical (non-learned) next-best-view planner. The research
contribution of UGAP is the uncertainty signal and how it gates the
perception-action loop -- not a novel NBV algorithm -- so this stays simple
and interpretable, which also makes ablations cleaner (swap this module for
"random" or "round robin" to prove the uncertainty signal itself matters).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ViewCandidate:
    position: np.ndarray   # (3,) camera center in world coords
    look_at: np.ndarray    # (3,) point the camera should look at
    score: float           # higher = more useful to visit next


class NextBestViewPlanner:
    """
    Given the current point cloud, per-point uncertainty, and existing
    camera poses, propose the next viewpoint to re-observe the most
    uncertain, currently-underobserved region.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.35,
        max_views: int = 5,
        candidate_view_radius_deg: float = 30.0,
        strategy: str = "entropy_visibility",
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.max_views = max_views
        self.candidate_radius_deg = candidate_view_radius_deg
        self.strategy = strategy

    def has_unreliable_regions(self, uncertainty: np.ndarray) -> bool:
        return bool(np.any(uncertainty > self.uncertainty_threshold))

    def select_target_region(self, points: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
        """Returns the centroid (3,) of the most uncertain point cluster."""
        mask = uncertainty > self.uncertainty_threshold
        if not np.any(mask):
            return points.mean(axis=0)
        # Weight by uncertainty so the noisiest points pull the target most.
        weights = uncertainty[mask]
        return np.average(points[mask], axis=0, weights=weights)

    def propose_view(
        self,
        points: np.ndarray,
        uncertainty: np.ndarray,
        existing_poses: np.ndarray,
        n_candidates: int = 32,
    ) -> Optional[ViewCandidate]:
        """
        Sample candidate camera positions on a sphere around the uncertain
        region and pick the one that (a) is far in angle from all existing
        views (maximizes new information) and (b) is close enough to be
        physically reachable / keep the target in frame.
        """
        if self.strategy == "random":
            return self._random_view(points, uncertainty)

        target = self.select_target_region(points, uncertainty)
        scene_radius = np.linalg.norm(points.std(axis=0)) + 1e-6

        existing_dirs = existing_poses[:, :3, 3] - target
        existing_dirs = existing_dirs / (np.linalg.norm(existing_dirs, axis=1, keepdims=True) + 1e-8)

        rng = np.random.default_rng(0)
        best_candidate, best_score = None, -np.inf

        for _ in range(n_candidates):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction) + 1e-8
            candidate_pos = target + direction * scene_radius * 1.5

            cand_dir = direction
            # Angular novelty: minimum angle to any existing view direction.
            cos_sims = existing_dirs @ cand_dir
            min_angle = np.arccos(np.clip(cos_sims, -1.0, 1.0)).min()
            score = min_angle  # bigger angular gap from existing views = better

            if score > best_score:
                best_score = score
                best_candidate = ViewCandidate(position=candidate_pos, look_at=target, score=score)

        return best_candidate

    def _random_view(self, points: np.ndarray, uncertainty: np.ndarray) -> ViewCandidate:
        target = self.select_target_region(points, uncertainty)
        scene_radius = np.linalg.norm(points.std(axis=0)) + 1e-6
        rng = np.random.default_rng()
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-8
        return ViewCandidate(position=target + direction * scene_radius * 1.5, look_at=target, score=0.0)

    def run_active_loop(
        self,
        reconstruct_fn,
        uncertainty_fn,
        capture_fn,
        image_paths: List[str],
    ):
        """
        Generic driver, decoupled from any specific robot/simulator API.

        Args:
            reconstruct_fn(image_paths) -> ReconstructionOutput
            uncertainty_fn(recon_output) -> (points, uncertainty) arrays
            capture_fn(view_candidate) -> new image path captured from that
                pose (this is the only function that needs to talk to a
                real robot or simulator -- swap it out per experiment)
            image_paths: initial set of images to start from

        Returns:
            final ReconstructionOutput, final uncertainty array, number of
            extra views taken.
        """
        views_taken = 0
        current_paths = list(image_paths)

        while views_taken < self.max_views:
            recon = reconstruct_fn(current_paths)
            points, uncertainty = uncertainty_fn(recon)

            if not self.has_unreliable_regions(uncertainty):
                break

            candidate = self.propose_view(points, uncertainty, recon.per_image_poses)
            if candidate is None:
                break

            new_image_path = capture_fn(candidate)
            current_paths.append(new_image_path)
            views_taken += 1

        recon = reconstruct_fn(current_paths)
        points, uncertainty = uncertainty_fn(recon)
        return recon, uncertainty, views_taken
