"""
Wrapper around GraspNet-baseline (https://github.com/graspnet/graspnet-baseline)
that filters candidate grasps by UGAP's evidential uncertainty before
returning them -- this is the "gate" in Uncertainty-Gated Active Perception.

Expects graspnet-baseline to be cloned and importable, e.g. in Colab:

    !git clone https://github.com/graspnet/graspnet-baseline
    import sys; sys.path.append("graspnet-baseline")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class GraspCandidate:
    translation: np.ndarray   # (3,) grasp point in world coords
    rotation: np.ndarray      # (3, 3) grasp orientation
    width: float              # gripper opening width
    grasp_score: float        # GraspNet's own quality score
    region_uncertainty: float  # UGAP epistemic uncertainty at this point
    accepted: bool             # True if it passed the uncertainty gate


class GraspNetWrapper:
    def __init__(self, checkpoint: str, min_confidence_to_grasp: float = 0.65,
                 collision_check: bool = True, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.min_confidence_to_grasp = min_confidence_to_grasp
        self.collision_check = collision_check
        self.device = device
        self._net = None

    def _load_model(self):
        try:
            from graspnet import GraspNet, pred_decode
        except ImportError as e:
            raise ImportError(
                "Could not import graspnet-baseline. Clone it first:\n"
                "  !git clone https://github.com/graspnet/graspnet-baseline\n"
                "  import sys; sys.path.append('graspnet-baseline')\n"
                "See notebooks/00_setup_colab.ipynb."
            ) from e

        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(
                f"GraspNet checkpoint not found at {self.checkpoint}. "
                "Run scripts/download_models.sh first."
            )

        import torch

        self._net = GraspNet(is_training=False)
        checkpoint = torch.load(self.checkpoint, map_location=self.device)
        self._net.load_state_dict(checkpoint["model_state_dict"])
        self._net.to(self.device)
        self._net.eval()
        self._pred_decode = pred_decode

    def detect_grasps(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        epistemic_uncertainty: np.ndarray,
        top_k: int = 20,
    ) -> List[GraspCandidate]:
        """
        Run GraspNet on the fused point cloud, then reject any grasp whose
        contact region has epistemic uncertainty above the configured
        threshold -- i.e. don't grasp what the model isn't sure it has
        reconstructed correctly.
        """
        if self._net is None:
            self._load_model()

        import torch

        pts_tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
        color_tensor = torch.from_numpy(colors).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            end_points = {"point_clouds": pts_tensor, "cloud_colors": color_tensor}
            end_points = self._net(end_points)
            grasp_preds = self._pred_decode(end_points)[0].detach().cpu().numpy()

        # grasp_preds columns per graspnet-baseline convention:
        # [score, width, height, depth, rotation(9), translation(3), object_id]
        candidates = []
        # Build a KD-tree-free nearest-uncertainty lookup for speed on small N.
        for row in grasp_preds[:top_k]:
            score = float(row[0])
            translation = row[13:16]
            rotation = row[4:13].reshape(3, 3)
            width = float(row[1])

            nearest_idx = np.argmin(np.linalg.norm(points - translation, axis=1))
            region_uncertainty = float(epistemic_uncertainty[nearest_idx])
            accepted = region_uncertainty <= (1.0 - self.min_confidence_to_grasp)

            candidates.append(
                GraspCandidate(
                    translation=translation,
                    rotation=rotation,
                    width=width,
                    grasp_score=score,
                    region_uncertainty=region_uncertainty,
                    accepted=accepted,
                )
            )

        return candidates

    @staticmethod
    def best_accepted_grasp(candidates: List[GraspCandidate]):
        accepted = [c for c in candidates if c.accepted]
        if not accepted:
            return None
        return max(accepted, key=lambda c: c.grasp_score)
