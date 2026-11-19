#!/bin/bash
# One-off CPU-only setup of the UCS-Bench PVC (mounted at /pvc). Idempotent: safe to rerun.
#   /pvc/UCS-Bench                    repo @ UCS_COMMIT + submodules, ckpts/
#   /pvc/UCS-Bench/data/UCS-Bench     Hugging Face dataset (videos + QAs)
#   /pvc/venv                         Python env (uses the image's torch via system site-packages)
#   /pvc/hf                           HF_HOME: Qwen3-VL weights
#   /pvc/ucs-eval                     converted questions + shard lists
set -euo pipefail
step() { echo; echo "================ $* [$(date -u +%H:%M:%S)] ================"; }

step "System packages"
apt-get update -qq
apt-get install -y -qq --no-install-recommends git wget >/dev/null

step "Repo"
REPO=/pvc/UCS-Bench
if [ ! -d "$REPO/.git" ]; then
  git clone -q https://github.com/cocowy1/UCS-Bench.git "$REPO"
fi
cd "$REPO"
git checkout -q "$UCS_COMMIT"
git submodule update --init --recursive
git submodule status
# Upstream configs hard-code the authors' paths.
sed -i "s#/data/ywang/my_projects/VideoUnderstanding/Directme#${REPO}#g" configs/scal3r/*.yaml
# Local fix: vectorised SceneGraph.build_edges (identical output; the upstream all-pairs
# Python loop ran after every chunk and dominated graph-building time). Skipped if applied.
git apply --reverse --check /scripts/build_edges.patch 2>/dev/null || git apply /scripts/build_edges.patch

step "Python env"
if [ ! -f /pvc/venv/.complete ]; then
  rm -rf /pvc/venv
  python -m venv --system-site-packages /pvc/venv
  source /pvc/venv/bin/activate
  cat > /pvc/venv/constraints.txt <<'EOF'
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
xformers==0.0.28.post3
numpy<2
EOF
  PIP="pip install -q -c /pvc/venv/constraints.txt"
  $PIP -e ".[video,perception,viz,api,vlm]"
  $PIP -e third_party/Depth-Anything-3 -e third_party/Scal3R
  SAM2_BUILD_CUDA=0 $PIP --no-build-isolation -e third_party/sam2
  # Undeclared runtime deps (addict: DA3; CLIP: YOLO-World) and the Qwen3-VL eval stack.
  $PIP addict==2.4.0 huggingface_hub \
    "clip @ git+https://github.com/ultralytics/CLIP.git@81ff68ed7ffcac3b40484c914f104f816757308d" \
    transformers==5.9.0 accelerate==1.13.0 qwen-vl-utils
  touch /pvc/venv/.complete
fi
source /pvc/venv/bin/activate
python -c "import torch, transformers, directme, depth_anything_3, sam2, scal3r; from transformers import Qwen3VLForConditionalGeneration; print('torch', torch.__version__, 'transformers', transformers.__version__)"

step "Checkpoints"
mkdir -p ckpts/{sam2,yolo,scal3r,depth_anything_3}
fetch() { [ -s "$2" ] || wget -q -O "$2" "$1"; }
fetch https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt ckpts/sam2/sam2.1_hiera_tiny.pt
fetch https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt ckpts/yolo/yolov8s-worldv2.pt
fetch https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt ckpts/scal3r/dino_salad.ckpt
hf download xbillowy/Scal3R scal3r.pt --local-dir ckpts/scal3r
hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --local-dir ckpts/depth_anything_3/DA3NESTED-GIANT-LARGE-1.1
du -sh ckpts/*

step "Dataset (~126 GB)"
hf download cocowy1/UCS-Bench --repo-type dataset --local-dir data/UCS-Bench --max-workers 16 | tail -1
echo "videos: $(ls data/UCS-Bench/*.mp4 | wc -l)  QA files: $(ls data/UCS-Bench/QAs | wc -l)"

step "Qwen3-VL-8B-Instruct"
hf download Qwen/Qwen3-VL-8B-Instruct | tail -1

step "Questions + shards"
python /scripts/prepare_questions.py data/UCS-Bench /pvc/ucs-eval --shards 8 --subset 10

touch /pvc/.prepare-complete
df -h /pvc
step "Prepare complete"
