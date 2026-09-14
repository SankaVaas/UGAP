#!/bin/bash
# Fetches/points at the datasets UGAP uses.
#
# STORAGE STRATEGY (updated): full GraspNet-1Billion is 120GB+ and is not
# used. The grasp-detection benchmark instead uses Cornell (<1GB) and,
# optionally, Jacquard (a few GB) -- both are standard, widely-cited
# benchmarks with an established rectangle metric (IoU > 25%, angle error
# < 30 deg), so this keeps results comparable to the published literature
# while fitting comfortably on Colab or a 5GB local disk.
set -e

mkdir -p data/cornell data/jacquard data/cleargrasp data/tod data/sim_sequences

echo "== Cornell Grasping Dataset (PRIMARY, < 1GB) =="
echo "Download from Cornell's page or the common Kaggle mirror, e.g.:"
echo "  https://www.kaggle.com/datasets/oneoneliu/cornell-grasp"
echo "extract into data/cornell/ (expects pcd*.txt / pcd*r.png / pcd*cpos.txt"
echo "per the reference repo's utils/data/cornell_data.py)."

echo ""
echo "== Jacquard Dataset (SECONDARY, a few GB, optional) =="
echo "Register and download from https://jacquard.liris.cnrs.fr/"
echo "(distributed as multiple zip parts -- only grab a subset of the parts"
echo "if you want to keep this under ~2-3GB for a pilot; the full set is"
echo "still an order of magnitude smaller than GraspNet-1B)."
echo "Extract into data/jacquard/"

echo ""
echo "== GraspNet-1Billion (OPTIONAL / stretch goal, NOT needed by default) =="
echo "Skip this unless you specifically want the full 6-DoF benchmark later."
echo "If so, download only a handful of scenes via graspnetAPI, never the"
echo "full archive:"
echo "  pip install graspnetAPI"
echo "  python -c \""
echo "    from graspnetAPI.utils.utils import download_scene"
echo "    for i in [100, 101, 102, 103, 104]:  # 5 test scenes, ~3-5GB total"
echo "        download_scene(i, camera='realsense', save_path='data/graspnet1billion_subset')"
echo "  \""

echo ""
echo "== ClearGrasp (TEST/VAL split only) =="
echo "Register and download from https://sites.google.com/view/cleargrasp"
echo "grab only 'cleargrasp-dataset-test-val' (~1-2GB) and extract into"
echo "data/cleargrasp/"

echo ""
echo "== Transparent Object Dataset (TOD) =="
echo "See https://github.com/Shreeyak/cleargrasp for links/instructions"
echo "then extract into data/tod/"

echo ""
echo "== Self-collected sim sequences (uncertainty-head training data) =="
echo "Zero download needed -- generated on-the-fly in Colab with PyBullet."
echo "See ugap/data/sim_generator.py."

echo ""
echo "Reminder: Colab disks are ephemeral. Mount Google Drive and point"
echo "these directories at your Drive copy to avoid re-downloading every"
echo "session, e.g.:"
echo "  from google.colab import drive; drive.mount('/content/drive')"
echo "  !ln -s /content/drive/MyDrive/ugap_data/cornell data/cornell"
