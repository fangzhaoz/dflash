#!/usr/bin/env bash
# =============================================================================
# run_drift_2x200.sh — drift experiment: MTP & DFlash, STEPS (default 200) GRPO
# steps, SAMPLED every step, trainer.val_before_train=False (NO greedy step-0).
# Records reward + acceptance per step. Applies both patches, runs both, parses.
#
# GPU keep-alive (the box is terminated if GPUs look idle):
#   - the verl runs use all 8 GPUs, so they collide with a running ghost.py keep-alive;
#   - free_gpus() clears ALL GPU processes the CONTAINER-SAFE way (fuser -k /dev/nvidia*),
#     because nvidia-smi can't list GPU PIDs in this container and ghost.py's spawn-workers
#     don't carry 'ghost.py' in argv;
#   - on exit we revert the patches and (only if the GPUs are usable) relaunch ghost.py, so
#     the GPUs are never idle when the experiment isn't running, and ghosts can't pile up.
#
# Run:  nohup bash rl_spec_decode/run_drift_2x200.sh &> /tmp/drift_2x200.out &
#       tail -f /tmp/drift_2x200.out
# Stop the keep-alive:  touch /tmp/STOP_SPIN     (force: fuser -k /dev/nvidia*)
# =============================================================================
set -eo pipefail

DFLASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERL_DIR=${VERL_DIR:-$HOME/verl}
ENV_NAME=${ENV_NAME:-dflash-verl}
STEPS=${STEPS:-200}
GHOST=${GHOST:-/opt/tiger/lmc_muon/ghost.py}
GHOST_SIZE=${GHOST_SIZE:-65000}
STOP_FILE=${STOP_FILE:-/tmp/STOP_SPIN}
P_MEAS="$DFLASH_DIR/rl_spec_decode/patches/spec_accept_measurement.patch"
P_INJ="$DFLASH_DIR/rl_spec_decode/patches/draft_inject_static.patch"
LOGDIR="$DFLASH_DIR/rl_spec_decode/logs"; mkdir -p "$LOGDIR"
START_GHOST_ON_EXIT=1   # set to 0 if we couldn't free the GPUs (don't pile up more ghosts)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "VERL_DIR=$VERL_DIR  DFLASH_DIR=$DFLASH_DIR  STEPS=$STEPS  keep-alive: python $GHOST --size $GHOST_SIZE"

# container-safe GPU clearer: kill ghost + its spawn-workers + ANY process holding an NVIDIA
# device, then wait until memory is actually released. Returns non-zero if it can't free them.
free_gpus() {
  touch "$STOP_FILE" 2>/dev/null || true; sleep 2
  local i maxused
  for i in $(seq 1 18); do
    pkill -9 -f ghost.py 2>/dev/null || true
    pkill -9 -f 'multiprocessing.spawn' 2>/dev/null || true
    pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null || true
    fuser -k /dev/nvidia* 2>/dev/null || true      # the hammer: kills every NVIDIA-device holder
    sleep 4
    maxused=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ -z "$maxused" ] && return 0
    if [ "$maxused" -lt 2000 ] 2>/dev/null; then echo ">> GPUs free (max used ${maxused} MiB)"; return 0; fi
    echo ">> waiting for GPUs to free (max used ${maxused} MiB) ..."
  done
  echo "!! GPUs STILL not free (max used ${maxused} MiB). NOT starting the runs (would OOM)."
  echo "   inspect:  nvidia-smi ; ps aux | grep -E 'ghost|multiprocessing.spawn' | grep -v grep"
  return 1
}

start_ghost() {
  rm -f "$STOP_FILE"
  echo ">> relaunching GPU keep-alive: python $GHOST --size $GHOST_SIZE --stop-file $STOP_FILE"
  echo "   (stop gracefully: touch $STOP_FILE  |  force: fuser -k /dev/nvidia*)"
  exec python "$GHOST" --size "$GHOST_SIZE" --stop-file "$STOP_FILE"
}
on_exit() {
  set +e
  echo ">> [on_exit] reverting verl patches ..."
  cd "$VERL_DIR"
  git apply -R "$P_INJ"  2>/dev/null
  git apply -R "$P_MEAS" 2>/dev/null
  git diff --quiet && echo ">> verl reverted CLEAN" || echo "!! verl NOT clean after revert — inspect"
  if [ "$START_GHOST_ON_EXIT" = "1" ]; then
    start_ghost   # exec -> this process becomes the keep-alive
  else
    echo ">> NOT relaunching ghost (GPUs were not free / already occupied). Leaving GPUs as-is."
  fi
}

cd "$VERL_DIR"
if ! git diff --quiet; then
  echo "!! verl tree is dirty — clean it first, then relaunch:"
  echo "   cd $VERL_DIR && git checkout -- verl/workers/rollout/vllm_rollout/"
  exit 1   # before trap is armed -> does NOT touch GPUs / ghost
fi
trap on_exit EXIT

echo ">> freeing GPUs for the verl runs ..."
if ! free_gpus; then START_GHOST_ON_EXIT=0; exit 1; fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true

git apply --check "$P_MEAS" && git apply --check "$P_INJ" || { echo "APPLY-CHECK FAILED — aborting"; exit 1; }
git apply "$P_MEAS"; git apply "$P_INJ"
echo ">> patches applied:"; git --no-pager diff --stat

cd "$DFLASH_DIR"
OVR=(data.dataloader_num_workers=2 trainer.val_before_train=False)

echo "===================== RUN MTP ($STEPS steps, sampled, no val) ====================="
STEPS="$STEPS" bash rl_spec_decode/run1_mtp.sh "${OVR[@]}"
echo "----- MTP reward+acceptance trend -----"
python rl_spec_decode/parse_acceptance_trend.py "$LOGDIR/run1_mtp.log" --csv "$LOGDIR/mtp_trend_${STEPS}.csv" | tail -20 || true

echo "===================== RUN DFLASH ($STEPS steps, sampled, no val) ====================="
STEPS="$STEPS" bash rl_spec_decode/run2_dflash.sh "${OVR[@]}"
echo "----- DFlash reward+acceptance trend -----"
python rl_spec_decode/parse_acceptance_trend.py "$LOGDIR/run2_dflash.log" --csv "$LOGDIR/dflash_trend_${STEPS}.csv" | tail -20 || true

echo "===================== DONE — CSVs: $LOGDIR/{mtp,dflash}_trend_${STEPS}.csv ====================="
# fall through -> EXIT trap reverts patches + relaunches the GPU keep-alive
