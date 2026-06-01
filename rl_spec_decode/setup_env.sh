#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — one conda env for verl + DFlash + vLLM nightly (spec-decode RL)
#
# Strategy (decided): single env. Install ORDER matters and is deliberate:
#   1) verl base deps (its requirements.txt has NO vLLM pin)
#   2) verl package with --no-deps  (so it can't re-pin / downgrade vLLM)
#   3) DFlash with --no-deps
#   4) vLLM NIGHTLY *last* so nothing downgrades it
# Then we capture every resolved version.
#
# You are on miniconda: we create ONE conda env (py3.12) and do all heavy
# installs via pip, as requested.
#
# Run on the remote box:  bash rl_spec_decode/setup_env.sh
# Re-runnable: skips clone if VERL_DIR already exists.
# =============================================================================
set -eo pipefail

# ---- user-adjustable ----
ENV_NAME=${ENV_NAME:-dflash-verl}
PYVER=${PYVER:-3.12}
VERL_REF=${VERL_REF:-v0.7.1}                       # pinned verl release
VERL_DIR=${VERL_DIR:-$HOME/verl}                   # where verl gets cloned
DFLASH_DIR=${DFLASH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}  # this repo

# vLLM nightly install command — installed LAST. Override VLLM_INSTALL_CMD to swap.
#   Default = nightly wheels (what the brief calls for).
#   Stable alternative (DFlash README says v0.20.1+ has core DFlash for std models):
#     VLLM_INSTALL_CMD='pip install -U "vllm>=0.20.1"'
VLLM_INSTALL_CMD=${VLLM_INSTALL_CMD:-'pip install -U --pre vllm --extra-index-url https://wheels.vllm.ai/nightly'}

OUT=${OUT:-$DFLASH_DIR/rl_spec_decode/resolved_versions.txt}
# ---- end user-adjustable ----

echo "============================================================"
echo " env=$ENV_NAME  python=$PYVER  verl=$VERL_REF"
echo " VERL_DIR=$VERL_DIR"
echo " DFLASH_DIR=$DFLASH_DIR"
echo " vLLM install: $VLLM_INSTALL_CMD"
echo "============================================================"

# --- conda env ---
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=$PYVER"
fi
conda activate "$ENV_NAME"
python -m pip install -U pip setuptools wheel

# --- clone verl at the pinned ref ---
if [ ! -d "$VERL_DIR/.git" ]; then
    git clone https://github.com/volcengine/verl.git "$VERL_DIR"
fi
git -C "$VERL_DIR" fetch --tags --quiet
git -C "$VERL_DIR" checkout "$VERL_REF"

# === Install order (do NOT reorder) ====================================
# 1) verl base deps — requirements.txt does NOT pin vLLM (it's commented out),
#    so this is safe; vLLM stays unconstrained.
pip install -r "$VERL_DIR/requirements.txt"

# 2) verl package itself, NO deps — cannot re-pin/downgrade vLLM.
pip install --no-deps -e "$VERL_DIR"

# 3) DFlash, NO deps (its light deps come from verl's set; add the two it adds).
pip install --no-deps -e "$DFLASH_DIR"
pip install rich loguru

# 4) vLLM NIGHTLY — LAST, so nothing downgrades it.
eval "$VLLM_INSTALL_CMD"
# =======================================================================

echo
echo "############ pip check (dependency conflicts, non-fatal) ############"
pip check || true

# --- capture resolved versions ---
{
  echo "# Resolved versions — $(hostname)  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "conda_env: $ENV_NAME"
  echo "verl_ref_requested: $VERL_REF"
  echo "verl_commit: $(git -C "$VERL_DIR" rev-parse HEAD)"
  echo "vllm_install_cmd: $VLLM_INSTALL_CMD"
  echo
  python - <<'PY'
import importlib, importlib.metadata as md
def v(p):
    try: return md.version(p)
    except Exception as e: return f"NOT-INSTALLED ({e.__class__.__name__})"
for p in ["vllm","torch","transformers","tensordict","ray","numpy",
          "flash-attn","flashinfer-python","datasets","huggingface-hub",
          "accelerate","peft","xformers","triton"]:
    print(f"{p}: {v(p)}")
import torch
print("torch.cuda:", torch.version.cuda, "| torch.cuda.is_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
try:
    print("gpu0:", torch.cuda.get_device_name(0))
except Exception as e:
    print("gpu0:", e)
PY
  echo
  echo "# nvidia-smi"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
} | tee "$OUT"

echo
echo "============================================================"
echo " DONE. Resolved versions written to:"
echo "   $OUT"
echo " Please paste that file's contents back."
echo " Next: bash rl_spec_decode/prepare_data.sh"
echo "============================================================"
