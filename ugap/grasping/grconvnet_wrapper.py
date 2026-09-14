"""
Wrapper around GR-ConvNet (Kumra et al., IROS 2020) via the reference
implementation at https://github.com/skumra/robotic-grasping.

Why this instead of GraspNet-baseline: GR-ConvNet is trained on Cornell
(<1GB) and/or Jacquard (a few GB, distributed in parts) rather than
GraspNet-1Billion (120GB+), and both training sets come with a standard,
widely-reported benchmark protocol (rectangle metric: IoU > 25% and
angle error < 30 deg), so switching datasets doesn't cost benchmark
credibility -- it's what most grasp-detection papers compare against.

GR-ConvNet predicts a 2D grasp rectangle (center pixel, angle, width,
quality) from a single RGB-D image, not a 3D point cloud. We lift the
predicted pixel center to a 3D point using the corresponding depth value
and camera intrinsics, then hand that 3D point + region to the same
evidential-uncertainty gate used elsewhere in UGAP.

Expects the repo to be cloned and importable, e.g. in Colab:

    !git clone https://github.com/skumra/robotic-grasping
    import sys; sys.path.append("robotic-grasping")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class GraspCandidate2D:
    pixel_center: np.ndarray   # (2,) [row, col] in image space
    angle: float               # grasp angle, radians
    width: float                # opening width in pixels
    quality: float               # GR-ConvNet's own quality score, in [0, 1]
    point_3d: Optional[np.ndarray] = None  # (3,) lifted world point, filled in later
    region_uncertainty: Optional[float] = None
    accepted: bool = False


class GRConvNetWrapper:
    """
    Usage:
        wrapper = GRConvNetWrapper(checkpoint="checkpoints/grconvnet/cornell_rgbd_iou_0.96",
                                    device="cuda")
        grasps = wrapper.detect_grasps(rgb, depth, n_grasps=5)
    """

    def __init__(self, checkpoint: str, device: str = "cuda", input_size: int = 224):
        self.checkpoint = checkpoint
        self.device = device
        self.input_size = input_size
        self._net = None

    def _load_model(self):
        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(
                f"GR-ConvNet checkpoint not found at {self.checkpoint}. "
                "Run scripts/download_models.sh, or download a pretrained "
                "checkpoint from https://github.com/skumra/robotic-grasping "
                "(trained on Cornell or Jacquard)."
            )
        import torch

        # The reference repo saves the full model object (torch.save(net)),
        # not just a state_dict -- load accordingly.
        self._net = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self._net.to(self.device)
        self._net.eval()

    def detect_grasps(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        n_grasps: int = 5,
        use_depth: bool = True,
        use_rgb: bool = True,
    ) -> List[GraspCandidate2D]:
        """
        Args:
            rgb: (H, W, 3) uint8 or float image.
            depth: (H, W) depth map, same resolution as rgb.
            n_grasps: number of top grasp candidates to return.
        """
        if self._net is None:
            self._load_model()

        try:
            from utils.data.camera_data import CameraData
            from inference.post_process import post_process_output
            from utils.dataset_processing.grasp import detect_grasps as _detect_grasps_from_maps
        except ImportError as e:
            raise ImportError(
                "Could not import robotic-grasping. Clone it first:\n"
                "  !git clone https://github.com/skumra/robotic-grasping\n"
                "  import sys; sys.path.append('robotic-grasping')\n"
                "See notebooks/00_setup_colab.ipynb."
            ) from e

        import torch

        cam_data = CameraData(
            width=rgb.shape[1], height=rgb.shape[0],
            output_size=self.input_size,
            include_depth=use_depth, include_rgb=use_rgb,
        )
        x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)

        with torch.no_grad():
            xc = x.to(self.device)
            pred = self._net.predict(xc)

        q_img, ang_img, width_img = post_process_output(
            pred["pos"], pred["cos"], pred["sin"], pred["width"]
        )

        grasp_rects = _detect_grasps_from_maps(q_img, ang_img, width_img=width_img, no_grasps=n_grasps)

        candidates = []
        for g in grasp_rects:
            candidates.append(
                GraspCandidate2D(
                    pixel_center=np.array(g.center),
                    angle=float(g.angle),
                    width=float(g.length),
                    quality=float(q_img[g.center[0], g.center[1]]),
                )
            )
        return candidates

    @staticmethod
    def lift_to_3d(
        candidate: GraspCandidate2D,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_pose: np.ndarray,
    ) -> np.ndarray:
        """
        Back-projects the 2D grasp center + its depth value into a 3D point
        in world coordinates, using the pinhole camera model:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = Z
        then transforms camera->world with the given camera_pose (4x4).
        """
        v, u = candidate.pixel_center  # [row, col] = [v, u]
        z = float(depth[int(v), int(u)])

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        point_cam = np.array([x, y, z, 1.0])

        point_world = camera_pose @ point_cam
        candidate.point_3d = point_world[:3]
        return candidate.point_3d

    @staticmethod
    def gate_by_uncertainty(
        candidates: List[GraspCandidate2D],
        recon_points: np.ndarray,
        epistemic_uncertainty: np.ndarray,
        min_confidence_to_grasp: float = 0.65,
    ) -> List[GraspCandidate2D]:
        """
        Same uncertainty gate as GraspNetWrapper, applied to lifted 2D->3D
        candidates: find the nearest reconstructed point to each grasp's
        3D location and reject grasps in high-epistemic-uncertainty regions.
        """
        threshold = 1.0 - min_confidence_to_grasp
        for c in candidates:
            if c.point_3d is None:
                c.accepted = False
                continue
            nearest_idx = np.argmin(np.linalg.norm(recon_points - c.point_3d, axis=1))
            c.region_uncertainty = float(epistemic_uncertainty[nearest_idx])
            c.accepted = c.region_uncertainty <= threshold
        return candidates

    @staticmethod
    def best_accepted_grasp(candidates: List[GraspCandidate2D]) -> Optional[GraspCandidate2D]:
        accepted = [c for c in candidates if c.accepted]
        if not accepted:
            return None
        return max(accepted, key=lambda c: c.quality)
