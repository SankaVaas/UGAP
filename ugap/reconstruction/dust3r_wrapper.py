"""
Thin wrapper around DUSt3R (https://github.com/naver/dust3r) that turns a
handful of pose-free RGB images into a fused point cloud with per-point
DUSt3R confidence. This confidence is NOT the same as our evidential
uncertainty (see ugap/uncertainty) -- it's the model's native output and is
used as one of the input features to the evidential head, not a replacement
for it.

Expects the DUSt3R repo to be cloned and importable, e.g. in Colab:

    !git clone --recursive https://github.com/naver/dust3r
    import sys; sys.path.append("dust3r")

This module is written defensively: it raises a clear error if DUSt3R isn't
on the path yet, rather than failing on a cryptic import error.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class ReconstructionOutput:
    """Container for one reconstruction pass."""

    points: np.ndarray          # (N, 3) fused world-space points
    colors: np.ndarray          # (N, 3) RGB in [0, 1]
    confidence: np.ndarray      # (N,)   DUSt3R native per-point confidence
    per_image_poses: np.ndarray  # (V, 4, 4) estimated camera-to-world poses
    image_paths: List[str]
    depth_maps: Optional[List[np.ndarray]] = None      # per-view (H, W) depth, camera frame
    intrinsics: Optional[List[np.ndarray]] = None       # per-view (3, 3) camera intrinsics
    rgb_images: Optional[List[np.ndarray]] = None       # per-view (H, W, 3) in [0, 1]


class DUSt3RWrapper:
    """
    Minimal wrapper exposing a single method: reconstruct(image_paths).

    Usage:
        wrapper = DUSt3RWrapper(checkpoint="checkpoints/dust3r/DUSt3R_ViT-L_512.pth",
                                 device="cuda")
        out = wrapper.reconstruct(["view1.jpg", "view2.jpg", "view3.jpg"])
    """

    def __init__(self, checkpoint: str, device: str = "cuda", image_size: int = 512):
        self.checkpoint = checkpoint
        self.device = device
        self.image_size = image_size
        self._model = None  # lazy-loaded on first reconstruct() call

    def _load_model(self):
        try:
            from dust3r.model import AsymmetricCroCo3DStereo
        except ImportError as e:
            raise ImportError(
                "Could not import dust3r. Clone it first, e.g. in Colab:\n"
                "  !git clone --recursive https://github.com/naver/dust3r\n"
                "  import sys; sys.path.append('dust3r')\n"
                "See notebooks/00_setup_colab.ipynb."
            ) from e

        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(
                f"DUSt3R checkpoint not found at {self.checkpoint}. "
                "Run scripts/download_models.sh first."
            )

        self._model = AsymmetricCroCo3DStereo.from_pretrained(self.checkpoint).to(
            self.device
        )
        self._model.eval()

    def reconstruct(
        self,
        image_paths: List[str],
        schedule: str = "cosine",
        niter: int = 300,
    ) -> ReconstructionOutput:
        """
        Run DUSt3R on a set of pose-free images and return a fused,
        globally-aligned point cloud with per-point confidence.
        """
        if self._model is None:
            self._load_model()

        from dust3r.inference import inference
        from dust3r.image_pairs import make_pairs
        from dust3r.utils.image import load_images
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

        images = load_images(image_paths, size=self.image_size)
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(pairs, self._model, self.device, batch_size=1)

        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=0.01)

        pts3d = scene.get_pts3d()                  # list of (H, W, 3) per view
        confidence_maps = scene.get_conf()          # list of (H, W) per view
        imgs = scene.imgs                           # list of (H, W, 3) in [0, 1]
        poses = scene.get_im_poses().detach().cpu().numpy()  # (V, 4, 4)
        depthmaps = [d.detach().cpu().numpy() for d in scene.get_depthmaps()]  # list of (H, W)
        intrinsics = [k.detach().cpu().numpy() for k in scene.get_intrinsics()]  # list of (3, 3)

        all_points, all_colors, all_conf = [], [], []
        for pts, conf, img in zip(pts3d, confidence_maps, imgs):
            pts_np = pts.detach().cpu().numpy().reshape(-1, 3)
            conf_np = conf.detach().cpu().numpy().reshape(-1)
            col_np = np.asarray(img).reshape(-1, 3)
            all_points.append(pts_np)
            all_colors.append(col_np)
            all_conf.append(conf_np)

        return ReconstructionOutput(
            points=np.concatenate(all_points, axis=0),
            colors=np.concatenate(all_colors, axis=0),
            confidence=np.concatenate(all_conf, axis=0),
            per_image_poses=poses,
            image_paths=image_paths,
            depth_maps=depthmaps,
            intrinsics=intrinsics,
            rgb_images=[np.asarray(im) for im in imgs],
        )
