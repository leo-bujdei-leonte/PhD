#!/bin/bash
# Build DirectMe scene graphs for one shard of UCS-Bench videos, then (in whichever shard
# finishes last) evaluate DirectMe + Qwen3-VL with examples/evaluate_ucsbench_vlm.py.
#
#   run.sh <run-name> <num-shards>      e.g. run.sh subset 1 | run.sh full 8
#   run.sh <run-name> <num-shards> eval-only    skip graph building, just evaluate
#
# Shard index comes from JOB_COMPLETION_INDEX (Indexed Job). Resumable: videos whose graph
# is complete are skipped, so a retried pod picks up where it stopped.
# Output: /pvc/runs/<run>/{frames,graphs,failed,logs,done,results}
set -euo pipefail
RUN=$1; NSHARDS=$2; MODE=${3:-build}
IDX=${JOB_COMPLETION_INDEX:-0}
R=/pvc/runs/$RUN
REPO=/pvc/UCS-Bench
DATA=$REPO/data/UCS-Bench
MODEL=${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
# Object vocabulary from the README's "Evaluating on UCS-Bench" step.
CLASSES=${CLASSES:-"cup,phone,bottle,chair,table,sink,fridge,door,bag,book"}
log() { echo "[$(date -u +%H:%M:%S)] [$RUN/$IDX] $*"; }

[ -f /pvc/.prepare-complete ] || { echo "PVC not prepared: run PhD/ucs-bench/k8s/prepare.yaml first" >&2; exit 1; }
apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg libgl1 libglib2.0-0 >/dev/null
source /pvc/venv/bin/activate
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$R"/{frames,graphs,failed,logs,done,results}

if [ "$RUN" = subset ]; then
  VIDEOS=/pvc/ucs-eval/videos_subset.txt; QUESTIONS=/pvc/ucs-eval/questions_subset.jsonl
else
  VIDEOS=/pvc/ucs-eval/shards/shard-$IDX.txt; QUESTIONS=/pvc/ucs-eval/questions_full.jsonl
fi

# Called from an `if`, where set -e is off, so every step checks its own status.
build_graph() {
  local vid=$1 F=$R/frames/$1 G=$R/graphs/$1 L=$R/logs/$1.log
  if [ -f "$G/.complete" ]; then return 0; fi
  rm -rf "$G"
  if [ ! -f "$F/.complete" ]; then
    rm -rf "$F"; mkdir -p "$F"
    # 1 fps (frame i <-> t=i s, as ImageFolderFrameSource assumes); long side capped at 1280 px
    # to keep ~600k frames within the PVC. Perception runs at <=640 px anyway.
    ffmpeg -loglevel error -nostdin -i "$DATA/$vid.mp4" \
      -vf "fps=1,scale='if(gt(iw,ih),min(1280,iw),-2)':'if(gt(iw,ih),-2,min(1280,ih))'" \
      -q:v 3 "$F/frame_%06d.jpg" > "$L" 2>&1 || return 1
    touch "$F/.complete"
  fi
  # Graphs reference these frame files as keyframes, so they stay on the PVC.
  python examples/run_real_pipeline.py \
    --frames "$F" \
    --out "$G" \
    --classes "$CLASSES" \
    --storage-backend json \
    --device cuda \
    --depth-model ckpts/depth_anything_3/DA3NESTED-GIANT-LARGE-1.1 \
    --yolo-weights ckpts/yolo/yolov8s-worldv2.pt \
    --sam2-checkpoint ckpts/sam2/sam2.1_hiera_tiny.pt \
    --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
    < /dev/null >> "$L" 2>&1 || return 1
  test -s "$G/scene_graph.json" || return 1
  touch "$G/.complete"
}

if [ "$MODE" = build ]; then
  total=$(grep -c . "$VIDEOS"); n=0; ok=0; failed=0
  log "building graphs for $total videos from $VIDEOS"
  while read -r vid; do
    [ -n "$vid" ] || continue
    n=$((n + 1)); t0=$SECONDS
    if build_graph "$vid"; then
      ok=$((ok + 1))
      log "($n/$total) $vid ok $((SECONDS - t0))s frames=$(ls "$R/frames/$vid" | grep -c jpg)"
    else
      failed=$((failed + 1))
      rm -rf "$R/failed/$vid"; [ -d "$R/graphs/$vid" ] && mv "$R/graphs/$vid" "$R/failed/$vid"
      log "($n/$total) $vid FAILED $((SECONDS - t0))s; last log lines:"
      tail -5 "$R/logs/$vid.log" | sed 's/^/    /'
    fi
  done < "$VIDEOS"
  echo "ok=$ok failed=$failed total=$total" > "$R/done/shard-$IDX"
  log "shard finished: ok=$ok failed=$failed total=$total"
fi

# Evaluate once every shard has finished; mkdir is the lock so only one pod evaluates.
ndone=$(ls "$R/done" | wc -l)
if [ "$MODE" = eval-only ] || [ "$ndone" -ge "$NSHARDS" ]; then
  if [ "$MODE" = eval-only ] || mkdir "$R/eval.lock" 2>/dev/null; then
    log "evaluating DirectMe + $MODEL on $(grep -c . "$QUESTIONS") questions"
    python examples/evaluate_ucsbench_vlm.py \
      --questions "$QUESTIONS" \
      --graphs-dir "$R/graphs" \
      --backend transformers \
      --model "$MODEL" \
      --out "$R/results/directme_qwen3vl.json" 2>&1 | tee "$R/logs/eval.log"
    log "results: $R/results/directme_qwen3vl.json"
  fi
else
  log "$ndone/$NSHARDS shards done; the last shard to finish runs the evaluation"
fi
