#!/usr/bin/env bash
# =============================================================================
# run_drift_2x200.sh — the 2-run drift experiment:
#   MTP and DFlash, STEPS (default 200) GRPO steps, SAMPLED every step,
#   trainer.val_before_train=False so there is NO greedy step-0 (consistent decoding).
# Applies both patches (measurement + static injection), runs both, parses the
# reward + acceptance trend for each, and ALWAYS reverts the verl patches on exit
# (incl. on failure, via trap) so the verl tree stays baseline-clean.
#
# Run on the remote:  bash rl_spec_decode/run_drift_2x200.sh
#   (long — ~5 h for 2x200 steps; consider: nohup bash rl_spec_decode/run_drift_2x200.sh &> /tmp/drift.out &)
# =============================================================================
set -eo pipefail

DFLASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERL_DIR=${VERL_DIR:-$HOME/verl}
STEPS=${STEPS:-200}
P_MEAS="$DFLASH_DIR/rl_spec_decode/patches/spec_accept_measurement.patch"
P_INJ="$DFLASH_DIR/rl_spec_decode/patches/draft_inject_static.patch"
LOGDIR="$DFLASH_DIR/rl_spec_decode/logs"; mkdir -p "$LOGDIR"

echo "VERL_DIR=$VERL_DIR  DFLASH_DIR=$DFLASH_DIR  STEPS=$STEPS  (sampled, val_before_train=False)"

cd "$VERL_DIR"
if ! git diff --quiet; then
  echo "!! verl tree is dirty — clean it first:"
  echo "   cd $VERL_DIR && git checkout -- verl/workers/rollout/vllm_rollout/"
  exit 1
fi

revert() {
  echo ">> reverting verl patches ..."
  cd "$VERL_DIR"
  git apply -R "$P_INJ"  2>/dev/null || true
  git apply -R "$P_MEAS" 2>/dev/null || true
  git diff --quiet && echo ">> verl reverted CLEAN" || echo "!! verl NOT clean after revert — inspect manually"
}
trap revert EXIT

git apply --check "$P_MEAS" && git apply --check "$P_INJ" || { echo "APPLY-CHECK FAILED — aborting"; exit 1; }
git apply "$P_MEAS"; git apply "$P_INJ"
echo ">> patches applied:"; git --no-pager diff --stat

cd "$DFLASH_DIR"
OVR=(data.dataloader_num_workers=2 trainer.val_before_train=False)

echo "===================== RUN MTP ($STEPS steps, sampled, no val) ====================="
STEPS="$STEPS" bash rl_spec_decode/run1_mtp.sh "${OVR[@]}"
echo "----- MTP reward+acceptance trend -----"
python rl_spec_decode/parse_acceptance_trend.py "$LOGDIR/run1_mtp.log" --csv "$LOGDIR/mtp_trend_${STEPS}.csv" | tail -20

echo "===================== RUN DFLASH ($STEPS steps, sampled, no val) ====================="
STEPS="$STEPS" bash rl_spec_decode/run2_dflash.sh "${OVR[@]}"
echo "----- DFlash reward+acceptance trend -----"
python rl_spec_decode/parse_acceptance_trend.py "$LOGDIR/run2_dflash.log" --csv "$LOGDIR/dflash_trend_${STEPS}.csv" | tail -20

echo "===================== DONE ====================="
echo "CSVs: $LOGDIR/mtp_trend_${STEPS}.csv  and  $LOGDIR/dflash_trend_${STEPS}.csv"
echo "(verl patches will be reverted on exit by the trap)"
