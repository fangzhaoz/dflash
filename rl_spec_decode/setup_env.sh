#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — one conda env for verl + DFlash + vLLM nightly (spec-decode RL)
#
# Strategy (decided): single env. Install ORDER matters and is deliberate:
#   1) verl base deps (its requirements.txt has NO vLLM pin)
#   2) verl package with --no-deps  (so it can't re-pin / downgrade vLLM)
#   3) DFlash with --no-deps
#   4) vLLM NIGHTLY *last* so nothing downgrades it — but the CUDA-12 (cu128) build
#      of the nightly, installed via `uv --torch-backend=cu128`, so torch/vLLM match
#      the box's NVIDIA driver. (First attempt's plain-pip nightly pulled a CUDA-13
#      torch that the 535/CUDA-12.4 driver can't run; cu128 runs via CUDA minor-version
#      compatibility. We keep the NIGHTLY because that's the build with the dflash method.)
# Then we capture every resolved version AND hard-gate on two checks:
#   - torch.cuda.is_available() == True   (CUDA build actually usable on this driver)
#   - 'dflash' present in the installed vLLM build  (RUN 2 would die without it)
#
# You are on miniconda: we create ONE conda env (py3.12) and do all heavy
# installs via pip/uv, as requested.
#
# Run on the remote box:  bash rl_spec_decode/setup_env.sh
#   FRESH=1 bash rl_spec_decode/setup_env.sh   # remove + recreate the env first
#     (use FRESH=1 once, to clear the polluted cu13 env from the first attempt)
# Re-runnable: skips clone if VERL_DIR already exists.
# =============================================================================
set -eo pipefail

# ---- user-adjustable ----
ENV_NAME=${ENV_NAME:-dflash-verl}
PYVER=${PYVER:-3.12}
VERL_REF=${VERL_REF:-v0.7.1}                       # pinned verl release
VERL_DIR=${VERL_DIR:-$HOME/verl}                   # where verl gets cloned
DFLASH_DIR=${DFLASH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}  # this repo

# vLLM install (LAST). Install the EXPLICIT CUDA-12 release wheel (deterministic), so the
# compiled vllm._C links libcudart.so.12 and runs on this 535/CUDA-12.4 driver via CUDA
# minor-version compat. Index resolution is NOT reliable here: it pulled vLLM's cu13 default
# wheel, whose _C needs libcudart.so.13 (which a CUDA-12 driver cannot provide).
#   v0.22.0 publishes +cu129 and +cpu wheels (NO +cu128). cu129 = CUDA 12.9, fine on this driver.
VLLM_VERSION=${VLLM_VERSION:-0.22.0}
VLLM_CUDA=${VLLM_CUDA:-cu129}          # vLLM WHEEL's CUDA build (its _C links libcudart.so.12)
VLLM_WHEEL=${VLLM_WHEEL:-https://github.com/vllm-project/vllm/releases/download/v$VLLM_VERSION/vllm-$VLLM_VERSION+$VLLM_CUDA-cp38-abi3-manylinux_2_28_x86_64.whl}
# Torch is pinned INDEPENDENTLY to the validated cu128 build, with its index also pinned to
# cu128, so installing the cu129 vLLM wheel CANNOT drag torch to a different version OR CUDA
# variant. (vLLM 0.22.0 declares torch==2.11.0 with no +cuXXX, so the cu128 build satisfies it;
# vllm._C needs libcudart.so.12, which torch cu128 provides.) The gate asserts torch is unchanged.
TORCH_PIN=${TORCH_PIN:-torch==2.11.0}
TORCH_CUDA=${TORCH_CUDA:-cu128}
EXPECTED_TORCH=${EXPECTED_TORCH:-2.11.0+$TORCH_CUDA}
VLLM_PYTORCH_INDEX=${VLLM_PYTORCH_INDEX:-https://download.pytorch.org/whl/$TORCH_CUDA}
VLLM_INSTALL_CMD=${VLLM_INSTALL_CMD:-}

OUT=${OUT:-$DFLASH_DIR/rl_spec_decode/resolved_versions.txt}
# ---- end user-adjustable ----

echo "============================================================"
echo " env=$ENV_NAME  python=$PYVER  verl=$VERL_REF"
echo " VERL_DIR=$VERL_DIR"
echo " DFLASH_DIR=$DFLASH_DIR"
echo " vLLM install: ${VLLM_INSTALL_CMD:-$VLLM_WHEEL  (+ torch $TORCH_PIN @ $TORCH_CUDA)}"
echo " FRESH=${FRESH:-0}"
echo "============================================================"

# --- conda env ---
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ "${FRESH:-0}" = "1" ] && conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">> FRESH=1: removing existing env '$ENV_NAME' before recreate"
    conda env remove -y -n "$ENV_NAME"
fi
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

# 4) vLLM — LAST. Explicit CUDA-12 release wheel, with torch pinned to the validated cu128
#    build (torch==2.11.0 + cu128 index) so the wheel can't swap torch's version OR CUDA
#    variant. Deterministic: no index ambiguity (which previously yielded a cu13 vllm whose
#    _C needed libcudart.so.13).
pip install -U uv
if [ -n "$VLLM_INSTALL_CMD" ]; then
    eval "$VLLM_INSTALL_CMD"
else
    uv pip install --python "$CONDA_PREFIX/bin/python" --reinstall-package vllm \
        "$VLLM_WHEEL" "$TORCH_PIN" \
        --extra-index-url "$VLLM_PYTORCH_INDEX"
fi
# =======================================================================

echo
echo "############ pip check (dependency conflicts, non-fatal) ############"
pip check || true

echo
echo "############ ENV GATE: CUDA usable + dflash present (FATAL on fail) ############"
# In an `if` so `set -e` doesn't abort before we can print the FAILED banner.
if EXPECTED_TORCH="$EXPECTED_TORCH" python - <<'PY'
import os, sys
ok = True

import torch
expected = os.environ.get("EXPECTED_TORCH", "")
print(f"torch={torch.__version__}  torch.version.cuda={torch.version.cuda}  "
      f"cuda_available={torch.cuda.is_available()}  expected={expected}")
if not torch.cuda.is_available():
    print("FATAL: torch.cuda.is_available() is False -> CUDA build still mismatched to the driver.")
    ok = False
if expected and torch.__version__ != expected:
    print(f"FATAL: torch is {torch.__version__}, expected {expected} -> the vLLM install swapped torch.")
    print("       (We pin torch to the validated cu128 build; a swap can reintroduce a driver mismatch.)")
    ok = False

# Force the COMPILED extension + platform init to load (lazy `import vllm` would NOT catch a
# CUDA-build mismatch like the libcudart.so.13 error; `from vllm import LLM` does).
import vllm
from vllm import LLM  # noqa: F401
print(f"vllm import LLM: OK")
vdir = os.path.dirname(vllm.__file__)
hits = []
for root, _, files in os.walk(vdir):
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            try:
                if "dflash" in open(p, errors="ignore").read().lower():
                    hits.append(os.path.relpath(p, vdir))
            except Exception:
                pass
print(f"vllm={vllm.__version__}  dflash_refs_in_build={len(hits)}")
for h in hits[:8]:
    print("    ", h)
if not hits:
    print("FATAL: no 'dflash' in the installed vLLM build -> RUN 2 (method=dflash) would fail at init.")
    print("       This build lacks DFlash; do NOT proceed to RUN 2 with it.")
    ok = False

sys.exit(0 if ok else 1)
PY
then
    echo "ENV GATE PASSED: CUDA usable and dflash present."
else
    echo "!!!! ENV GATE FAILED — env is NOT good. Fix before running anything. !!!!"
    exit 1
fi

# --- capture resolved versions ---
{
  echo "# Resolved versions — $(hostname)  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "conda_env: $ENV_NAME"
  echo "verl_ref_requested: $VERL_REF"
  echo "verl_commit: $(git -C "$VERL_DIR" rev-parse HEAD)"
  echo "vllm_install_cmd: ${VLLM_INSTALL_CMD:-uv pip install --reinstall-package vllm $VLLM_WHEEL $TORCH_PIN --extra-index-url $VLLM_PYTORCH_INDEX}"
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
