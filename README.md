# UGAP — Uncertainty-Gated Active Perception for Robotic Grasping

Feed-forward 3D reconstruction (DUSt3R/MASt3R-style) is fast, but it silently
fails on transparent, reflective, and thin objects — precisely the objects
that are hardest to grasp. UGAP closes the loop:

**Reconstruct → Estimate per-point uncertainty → Actively re-observe uncertain
regions (next-best-view) → Grasp only once confidence clears a threshold.**

This repo is built to run **entirely on Google Colab** (no local GPU
required). Everything heavy (model checkpoints, datasets) is downloaded at
runtime rather than committed to the repo.

## Repo layout

```
UGAP/
├── ugap/                       # Installable Python package
│   ├── reconstruction/         # DUSt3R/MASt3R wrapper -> point maps
│   ├── uncertainty/            # Evidential uncertainty head + training
│   ├── nbv/                    # Next-best-view planner
│   ├── grasping/               # Grasp detector wrappers (GR-ConvNet primary, GraspNet optional)
│   ├── utils/                  # Point cloud + visualization helpers
│   └── data/                   # Dataset loaders (Cornell, Jacquard, ClearGrasp, TOD)
├── notebooks/                  # Colab notebooks — run these, in order
│   ├── 00_setup_colab.ipynb
│   ├── 01_reconstruction_demo.ipynb
│   ├── 02_uncertainty_head_training.ipynb
│   └── 03_active_view_grasping_pipeline.ipynb
├── configs/
│   └── default.yaml
├── scripts/
│   ├── download_models.sh
│   └── download_datasets.sh
├── requirements.txt
└── README.md
```

## Quickstart (Colab)

1. Upload/clone this repo into Colab, or `git clone` it once it's pushed to
   GitHub.
2. Open `notebooks/00_setup_colab.ipynb` first — installs dependencies,
   clones DUSt3R, downloads checkpoints.
3. Run `01_reconstruction_demo.ipynb` to sanity-check reconstruction on a
   couple of sample images.
4. Run `02_uncertainty_head_training.ipynb` to train/fine-tune the evidential
   uncertainty head.
5. Run `03_active_view_grasping_pipeline.ipynb` for the full
   reconstruct → uncertainty → NBV → grasp loop, evaluated against a
   single-shot (no-uncertainty) baseline.

## Milestone 1 scope (current)

- [x] Repo scaffold + interfaces
- [ ] DUSt3R wrapper producing point maps + confidence from real images
- [ ] Evidential (NIG) uncertainty head, trained on ClearGrasp / TOD
- [ ] Entropy/visibility-based NBV planner
- [ ] GR-ConvNet integration (Cornell/Jacquard) + 2D-to-3D lifting via depth
- [ ] Baseline vs. UGAP comparison on transparent/reflective object subset

## Datasets

**Storage note:** full GraspNet-1Billion is 120GB+ and is *not* used by
default. The grasp-detection benchmark instead uses **Cornell** (< 1GB)
and, optionally, **Jacquard** (a few GB) — both are standard, widely-cited
benchmarks with an established evaluation protocol (rectangle metric:
IoU > 25%, angle error < 30°), so this keeps results comparable to the
published literature while fitting comfortably on Colab or a 5GB local
disk. First experiments run on synthetic PyBullet sequences (generated
on-the-fly, no download) for the uncertainty head, plus Cornell for the
grasp benchmark.

| Dataset | Purpose | Size | Required for first experiments? |
|---|---|---|---|
| Self-collected (PyBullet, `ugap/data/sim_generator.py`) | Multi-view sequences + exact ground truth for NBV loop eval | ~0 (generated) | **Yes — primary dataset** |
| [Cornell Grasping Dataset](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp) | Standard grasp-rectangle benchmark (885 RGB-D images) | < 1GB | **Yes — primary grasp benchmark** |
| [Jacquard Dataset](https://jacquard.liris.cnrs.fr/) | Larger grasp-rectangle benchmark; also enables a standard cross-dataset generalization result | ~a few GB (multi-part) | Optional |
| [ClearGrasp](https://sites.google.com/view/cleargrasp) test/val split | Transparent-object uncertainty calibration on real images | ~1-2GB | Yes |
| GraspNet-1Billion, 5-10 scene subset via graspnetAPI | Optional full 6-DoF benchmark, stretch goal | ~3-5GB | Optional / later |
| [TOD (Transparent Object Dataset)](https://github.com/Shreeyak/cleargrasp) | Additional transparent/reflective scenes | varies | Optional |
| ScanNet++ / TUM RGB-D | Sanity-check uncertainty calibration before grasping experiments | varies | Optional |

## Models

| Component | Model |
|---|---|
| Reconstruction | DUSt3R (or MASt3R) — pose-free feed-forward pointmaps |
| Uncertainty | Evidential regression head (Normal-Inverse-Gamma) on reconstruction features |
| Grasp detection (primary) | [GR-ConvNet](https://github.com/skumra/robotic-grasping) — 2D grasp-rectangle prediction, trained on Cornell/Jacquard, lifted to 3D via depth |
| Grasp detection (optional/legacy) | GraspNet-baseline / Contact-GraspNet, inference-only |
| NBV planning | Classical entropy/visibility-based (not learned, by design) |

## Citation targets to build on

- Wang et al., *DUSt3R: Geometric 3D Vision Made Easy*, CVPR 2024
- Leroy et al., *MASt3R*, ECCV 2024
- *Trust3R: Evidential Uncertainty for Feed-Forward 3D Reconstruction*, ICML 2026
- Kumra et al., *Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network (GR-ConvNet)*, IROS 2020
- Depierre et al., *Jacquard: A Large Scale Dataset for Robotic Grasp Detection*, IROS 2018
- Lenz et al., *Deep Learning for Detecting Robotic Grasps* (Cornell Grasping Dataset), IJRR 2015
