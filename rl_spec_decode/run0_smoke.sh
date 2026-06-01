#!/usr/bin/env bash
# =============================================================================
# RUN 0 — SMOKE TEST (NO speculative decoding)
#
# Purpose: confirm verl's rollout + FSDP weight-resharding/update path SURVIVES
# the vLLM 0.12 -> 0.20+ nightly jump, on a real GRPO step. A `vllm serve` test
# does NOT exercise weight resharding into the engine — only a training step does.
# This MUST complete cleanly before we touch any spec config (RUN 1 / RUN 2).
#
# Tiny by design: prove integration, not convergence.
# =============================================================================
set -eo pipefail

DFLASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_NAME=${ENV_NAME:-dflash-verl}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
TEST_FILE=${TEST_FILE:-$DATA_DIR/test.parquet}
NGPUS=${NGPUS:-8}
STEPS=${STEPS:-3}
ATTN_IMPL=${ATTN_IMPL:-sdpa}   # HF actor/ref attn; sdpa avoids needing flash-attn. Fallback: eager.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- preflight: model reachable / cached (HF_TOKEN if gated) ----
preflight_hf() {
  python - "$@" <<'PY'
import os, sys
from huggingface_hub import HfApi
tok = os.environ.get("HF_TOKEN")
api, ok = HfApi(), True
for repo in sys.argv[1:]:
    try:
        api.model_info(repo, token=tok)
        print(f"[preflight] reachable: {repo}")
    except Exception as e:
        ok = False
        print(f"[preflight] NOT reachable: {repo} -> {type(e).__name__}: {e}")
        print("            If gated: export HF_TOKEN=hf_xxx  (and `huggingface-cli login`).")
sys.exit(0 if ok else 1)
PY
}
preflight_hf "$MODEL_PATH"

LOGDIR="$DFLASH_DIR/rl_spec_decode/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/run0_smoke.log"

# ---- tiny GRPO core (shared by all three runs) ----
COMMON=(
    algorithm.adv_estimator=grpo
    data.train_files="$TRAIN_FILE"
    data.val_files="$TEST_FILE"
    data.train_batch_size=16
    data.max_prompt_length=512
    data.max_response_length=512
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    +actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPL
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    trainer.critic_warmup=0
    trainer.logger=[console]
    trainer.project_name=verl_specdec_integration
    trainer.experiment_name=run0_smoke
    trainer.n_gpus_per_node="$NGPUS"
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_training_steps="$STEPS"
    trainer.total_epochs=1
)

# RUN 0: NO speculative config.
SPEC=()

set -x
python3 -m verl.trainer.main_ppo "${COMMON[@]}" "${SPEC[@]}" "$@" 2>&1 | tee "$LOG"
set +x

echo
echo "=================================================================="
echo "RUN 0 finished. If it reached the end without a traceback, the"
echo "verl<->vLLM-nightly rollout/weight-sync path is healthy."
echo "Log saved: $LOG"
echo "Paste back the last ~40 lines (the step metrics + any errors)."
echo "Next on success: bash rl_spec_decode/run1_mtp.sh"
echo "=================================================================="
