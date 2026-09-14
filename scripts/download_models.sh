#!/bin/bash
# Downloads model checkpoints needed by UGAP.
# Intended to be run from a Colab cell: !bash scripts/download_models.sh
set -e

mkdir -p checkpoints/dust3r
mkdir -p checkpoints/grconvnet
mkdir -p checkpoints/graspnet  # optional/legacy

echo "== DUSt3R checkpoint =="
if [ ! -f checkpoints/dust3r/DUSt3R_ViT-L_512.pth ]; then
    wget -q --show-progress \
      https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViT-L-14_384_linear.pth \
      -O checkpoints/dust3r/DUSt3R_ViT-L_512.pth
else
    echo "Already downloaded."
fi

echo "== GraspNet-baseline checkpoint (OPTIONAL / legacy, skip by default) =="
echo "See scripts/download_datasets.sh for why we default to GR-ConvNet +"
echo "Cornell/Jacquard instead. Only fetch this if you want the optional"
echo "stretch-goal comparison against the full 6-DoF GraspNet pipeline."
echo "Manually download 'checkpoint-rs.tar' from:"
echo "  https://github.com/graspnet/graspnet-baseline#pretrained-models"
echo "and place it at checkpoints/graspnet/checkpoint-rs.tar"

echo ""
echo "== GR-ConvNet checkpoint (PRIMARY grasp detector) =="
echo "Pretrained weights for both Cornell and Jacquard are distributed with"
echo "the reference repo: https://github.com/skumra/robotic-grasping"
echo "(see their README's 'trained-models' section / releases page)."
echo "Download the checkpoint you want (e.g. cornell_rgbd_iou_0.96) into:"
echo "  checkpoints/grconvnet/<checkpoint_name>"
echo "Each checkpoint is tens of MB -- no special download tooling needed."

echo "Done. Verify checkpoints/ contents:"
find checkpoints -type f
