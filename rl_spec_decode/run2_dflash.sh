#!/usr/bin/env bash
# =============================================================================
# RUN 2 — EXPERIMENT 2: DFlash draft speculative decoding in the vLLM rollout.
#
# Same code path as RUN 1 (generic engine_kwargs passthrough, Path B). The ONLY
# new variables vs RUN 1: method mtp->dflash and the added draft model. Equivalent to:
#   --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-4B-DFlash","num_speculative_tokens":15}'
#
# Requires the vLLM nightly + DFlash env (setup_env.sh). DO NOT run before RUN 0
# and RUN 1 are green — the ladder is the point.
#
# SUCCESS = (a) loop completes, (b) drafting genuinely ACTIVE (see checks at end).
# =============================================================================
set -eo pipefail

DFLASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_NAME=${ENV_NAME:-dflash-verl}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B}
DRAFT_MODEL=${DRAFT_MODEL:-z-lab/Qwen3.5-4B-DFlash}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
TEST_FILE=${TEST_FILE:-$DATA_DIR/test.parquet}
NGPUS=${NGPUS:-8}
STEPS=${STEPS:-3}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-15}
ATTN_IMPL=${ATTN_IMPL:-sdpa}   # HF actor/ref attn; sdpa avoids needing flash-attn. Fallback: eager.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- preflight: model + draft reachable / cached (HF_TOKEN if gated) ----
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
preflight_hf "$MODEL_PATH" "$DRAFT_MODEL"

LOGDIR="$DFLASH_DIR/rl_spec_decode/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/run2_dflash.log"

# ---- tiny GRPO core (shared, verbatim, by all three runs) ----
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
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    trainer.critic_warmup=0
    trainer.logger=[console]
    trainer.project_name=verl_specdec_integration
    trainer.experiment_name=run2_dflash
    trainer.n_gpus_per_node="$NGPUS"
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_training_steps="$STEPS"
    trainer.total_epochs=1
)

# RUN 2: DFlash via the SAME engine_kwargs passthrough as run1 (Path B). Only
# method (mtp->dflash) and the draft model differ. Built key-by-key to avoid
# Hydra inline-dict / slash-in-path quoting pitfalls; '+' adds new keys under the
# pre-existing empty node engine_kwargs.vllm ({} in v0.7.1).
# Fallback if Hydra rejects a '+': try '++' (force) or pass the dict inline:
#   +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config="{method: dflash, model: '$DRAFT_MODEL', num_speculative_tokens: $NUM_SPEC_TOKENS}"
SPEC=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.method=dflash
    +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.model="$DRAFT_MODEL"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.num_speculative_tokens="$NUM_SPEC_TOKENS"
)

set -x
python3 -m verl.trainer.main_ppo "${COMMON[@]}" "${SPEC[@]}" "$@" 2>&1 | tee "$LOG"
set +x

echo
echo "=================================================================="
echo "RUN 2 (DFlash) finished. Two independent success checks:"
echo "------------------------------------------------------------------"
echo "== INTEGRATION: spec-config registered at engine startup =="
grep -iE 'speculative_config|SpeculativeConfig|num_speculative_tokens|dflash' "$LOG" | tail -30 || echo "!! NO speculative_config/dflash at startup -> spec decoding was NOT configured (inspect the 'serve' arg list / engine init above)."
echo "== BEHAVIOR: drafting actually engaged (accepted tokens > 0) =="
grep -iE 'accept|num_accepted|spec_decode|drafted|draft.?acceptance' "$LOG" | tail -30 || echo "(no acceptance metric surfaced in this short run -- not necessarily a fallback; may need more steps / a stats flush. Trust the INTEGRATION check above for config, this for activity.)"
echo "------------------------------------------------------------------"
echo "Paste back BOTH blocks (the 'dflash' startup line is the key integration proof)."
echo "=================================================================="
