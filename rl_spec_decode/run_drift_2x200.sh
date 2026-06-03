#!/usr/bin/env bash
# =============================================================================
# run_drift_2x200.sh — drift experiment: MTP & DFlash, STEPS (default 200) GRPO
# steps, SAMPLED every step, trainer.val_before_train=False (NO greedy step-0).
# Records reward + acceptance per step. Applies both patches, runs both, parses.
#
# GPU keep-alive handling (the box is terminated if GPUs look idle):
#   - the verl runs USE all 8 GPUs, so they collide with a running ghost.py keep-alive;
#   - so at start we STOP any ghost.py to free the GPUs for the runs,
#   - and on exit (success OR failure) we revert the patches AND relaunch ghost.py,
#     so the GPUs are never idle when the experiment isn't running.
#
# Run (leave your current ghost.py running — this script will cycle it):
#   nohup bash rl_spec_decode/run_drift_2x200.sh &> /tmp/drift_2x200.out &
#   tail -f /tmp/drift_2x200.out
# Stop EVERYTHING (incl. the keep-alive):  pkill -f run_drift_2x200 ; pkill -f ghost.py
# =============================================================================
set -eo pipefail

DFLASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERL_DIR=${VERL_DIR:-$HOME/verl}
ENV_NAME=${ENV_NAME:-dflash-verl}
STEPS=${STEPS:-200}
GHOST=${GHOST:-/opt/tiger/lmc_muon/ghost.py}   # GPU keep-alive script
GHOST_SIZE=${GHOST_SIZE:-65000}
P_MEAS="$DFLASH_DIR/rl_spec_decode/patches/spec_accept_measurement.patch"
P_INJ="$DFLASH_DIR/rl_spec_decode/patches/draft_inject_static.patch"
LOGDIR="$DFLASH_DIR/rl_spec_decode/logs"; mkdir -p "$LOGDIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "VERL_DIR=$VERL_DIR  DFLASH_DIR=$DFLASH_DIR  STEPS=$STEPS  keep-alive: python $GHOST --size $GHOST_SIZE"

cd "$VERL_DIR"
if ! git diff --quiet; then
  echo "!! verl tree is dirty — clean it first, then relaunch (your ghost.py is left untouched):"
  echo "   cd $VERL_DIR && git checkout -- verl/workers/rollout/vllm_rollout/"
  exit 1   # exits BEFORE the trap is armed -> does NOT stop your running ghost
fi

start_ghost() {
  echo ">> relaunching GPU keep-alive: python $GHOST --size $GHOST_SIZE   (stop: pkill -f ghost.py)"
  exec python "$GHOST" --size "$GHOST_SIZE"
}
on_exit() {
  set +e
  echo ">> [on_exit] reverting verl patches ..."
  cd "$VERL_DIR"
  git apply -R "$P_INJ"  2>/dev/null
  git apply -R "$P_MEAS" 2>/dev/null
  git diff --quiet && echo ">> verl reverted CLEAN" || echo "!! verl NOT clean after revert — inspect manually"
  start_ghost   # exec -> this process becomes the keep-alive; GPUs never sit idle
}
trap on_exit EXIT   # from here on, ANY exit reverts patches + restarts the keep-alive

# free the GPUs for the verl runs
echo ">> stopping any running ghost.py keep-alive to free the GPUs ..."
pkill -f ghost.py 2>/dev/null || true
sleep 10
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true

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
