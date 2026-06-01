# Onboarding: spec-decode (MTP / DFlash) in pure vLLM and in verl RL training

A from-scratch guide to reproduce, on an 8×H100 box, that Qwen3.5-4B's **MTP** head and the
**DFlash** drafter accelerate generation — first in **pure vLLM inference**, then **inside a verl
GRPO RL training rollout** at the same acceptance rate.

What you'll see:

| Drafter | mean acceptance length | per-draft-token acceptance |
|---|---|---|
| MTP (n=2)     | **~2.74** / 3 max  | ~86.8% |
| DFlash (n=15) | **~6.88** / 16 max | ~39.2% |

The same numbers appear in pure inference (Step 2) and in the RL rollout (Step 3).

---

## Prerequisites

- **Hardware:** an NVIDIA GPU box. 8×H100 is the tested config; 1 GPU is enough for Step 2.
  Driver ≥ 525.60.13 (CUDA 12.x). The tested box is driver 535 / CUDA 12.4.
- **Software:** `git`, `conda` (miniconda), `bash`. ~80 GB free disk for envs + model weights.
- **Hugging Face access** to `Qwen/Qwen3.5-4B` and `z-lab/Qwen3.5-4B-DFlash`. If either is gated:
  `export HF_TOKEN=hf_xxx` and `huggingface-cli login`. (Weights download on first use.)
- **Repo access:** the owner adds you on GitHub (repo → Settings → Collaborators → Add people),
  then you `git clone` and checkout the work branch:
  ```bash
  git clone git@github.com:fangzhaoz/dflash.git
  cd dflash
  git checkout rl-spec-decode-integration
  ```

All commands below are run **from the repo root** (the `dflash/` you just cloned). The scripts
auto-detect the repo path; verl is cloned to `$HOME/verl` and data to `$HOME/data/gsm8k` by
default (override with `VERL_DIR=` / `DATA_DIR=` if you want).

---

## Step 1 — Build the conda environment

One script does everything: creates the `dflash-verl` conda env (py3.12), clones **verl v0.7.1**,
and installs verl + DFlash + the **vLLM 0.22.0 nightly (cu129)** in a deliberate order so nothing
downgrades vLLM, then hard-gates on `torch.cuda.is_available()` **and** that the DFlash method is
actually compiled into the vLLM build.

```bash
bash rl_spec_decode/setup_env.sh
```

- Re-runnable (skips the verl clone if present). To wipe and recreate: `FRESH=1 bash rl_spec_decode/setup_env.sh`.
- On success it prints `ENV GATE PASSED` and writes `rl_spec_decode/resolved_versions.txt`.
- **Why a special install order / wheel:** plain `pip install vllm` pulls a CUDA-13 build that a
  535/CUDA-12.4 driver can't run (`torch.cuda.is_available()` would be False). The script pins the
  explicit `vllm-0.22.0+cu129` wheel + `torch==2.11.0+cu128`, which run on the 535 driver via CUDA
  minor-version compatibility. If the gate ever fails, fix the env before proceeding — don't run anything.

Then build the tiny gsm8k dataset verl needs (used in Step 3):

```bash
bash rl_spec_decode/prepare_data.sh        # writes $HOME/data/gsm8k/{train,test}.parquet
```

Activate the env for the manual commands below:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate dflash-verl
```

---

## Step 2 — Pure vLLM inference acceptance (the reference numbers)

This runs vLLM **standalone** (no verl) with real weights and measures the spec-decode acceptance
on a handful of gsm8k prompts. ~30–60 s each (first run also downloads weights). Expect the table above.

**MTP** (native head, no separate draft model):

```bash
python rl_spec_decode/measure_spec_accept.py --mode measure \
  --method mtp --num-spec-tokens 2 --draft-model ""
```

**DFlash** (separate drafter repo):

```bash
python rl_spec_decode/measure_spec_accept.py --mode measure \
  --method dflash --num-spec-tokens 15
```

Read the `SPEC-DECODE ACCEPTANCE` block at the end:
- MTP → `mean_acceptance_length ≈ 2.74`, `per_draft_token_acceptance ≈ 0.868`
- DFlash → `mean_acceptance_length ≈ 6.88`, `per_draft_token_acceptance ≈ 0.392`

(Optional, richer: `--mode auto` instead of `--mode measure` prints **greedy and temperature-1.0**
side by side and writes a JSON snapshot — that's the harness used to develop the RL fix.)

---

## Step 3 — Same acceptance, now inside verl RL training

In the verl GRPO rollout the engine is launched with `--load_format dummy` and the colocated
sleep/wake cycle reshards only the **policy** — so out of the box the draft runs on **dummy
weights + sleep-zeroed buffers → ~0% acceptance**. Two committed patches fix this:

- `patches/spec_accept_measurement.patch` — read-only, logs the in-rollout acceptance each step.
- `patches/draft_inject_static.patch` — the fix: on every wake it re-loads the **real** draft
  weights (MTP head from the base checkpoint, or the DFlash repo) and **recomputes** the buffers
  that sleep zeroes (rope cache + attn scales). Works for both `method ∈ {mtp, dflash}`.

The patches modify the **verl** checkout (`$HOME/verl`); verl is installed editable so no reinstall
is needed. Always **revert** them afterward so verl stays baseline-clean.

### 3a. Validate the patch applies cleanly (do this once)

```bash
VERL_DIR=${VERL_DIR:-$HOME/verl}
DFLASH_DIR=$(pwd)
cd "$VERL_DIR"
git diff --quiet && echo "verl clean" || echo "verl dirty — revert/stash first"
git apply --check "$DFLASH_DIR/rl_spec_decode/patches/draft_inject_static.patch" && echo "APPLY-CHECK OK"
cd "$DFLASH_DIR"
```

### 3b. Apply both patches, run, observe, revert

```bash
VERL_DIR=${VERL_DIR:-$HOME/verl}
DFLASH_DIR=$(pwd)

# apply
cd "$VERL_DIR"
git apply "$DFLASH_DIR/rl_spec_decode/patches/spec_accept_measurement.patch"
git apply "$DFLASH_DIR/rl_spec_decode/patches/draft_inject_static.patch"

# RUN 1 = MTP  (or rl_spec_decode/run2_dflash.sh for DFlash)
cd "$DFLASH_DIR"
bash rl_spec_decode/run1_mtp.sh 2>&1 | tee rl_spec_decode/logs/run1.log

# pull the key lines out
grep -iE 'DRAFT-BUF|DRAFT-INJECT|SPEC-ACCEPT' rl_spec_decode/logs/run1.log

# revert (back to baseline verl)
cd "$VERL_DIR"
git apply -R "$DFLASH_DIR/rl_spec_decode/patches/draft_inject_static.patch"
git apply -R "$DFLASH_DIR/rl_spec_decode/patches/spec_accept_measurement.patch"
git diff --quiet && echo "verl reverted clean"
```

What success looks like (steady state, across the 3 steps / 8 replicas):
- `[DRAFT-BUF] rope_recomputed=… rope_fail=0 …` every wake (MTP: `rope_recomputed=1 scales_reset=4`;
  DFlash: `5` / `20`).
- `[DRAFT-INJECT] … changed=… embed=copied` (MTP: `params=13 changed=12`; DFlash: `44/43`).
- `[SPEC-ACCEPT per-step] … per_draft_token_acceptance=… mean_acceptance_length=…`:
  - **MTP → ~88% / ~2.77**
  - **DFlash → ~38.5% / ~6.78**

i.e. the Step-2 inference numbers, now **inside RL training**. (For DFlash use `run2_dflash.sh`,
which sets `method=dflash`, the draft repo, and `num_speculative_tokens=15`.)

> Baseline check (optional): run `run1_mtp.sh` **without** the `draft_inject_static.patch`
> (measurement patch only) and you'll see `per_draft_token_acceptance ≈ 0` — that's the dummy-weight
> baseline the fix corrects.

---

## Notes / troubleshooting

- **`measure_spec_accept.py` modes:** `measure` (one acceptance number, greedy by default),
  `auto` (greedy + temp-1.0, writes JSON), `sleep_wake_recompute` (reproduces verl's exact
  sleep/wake lifecycle standalone — the 10 s/iter harness used to develop the fix),
  `introspect` (dumps the draft model's structure). `--diff A.json B.json` compares two `auto` runs.
- **OOM / memory:** the runs use `gpu_memory_utilization=0.6`; lower it (or `--gpu-mem-util` for
  the standalone tool) on smaller GPUs.
- **HF gated repos:** `export HF_TOKEN=…`; the run scripts preflight-check both repos and tell you
  if one isn't reachable.
- **Patch hygiene:** the patches are applied to `$VERL_DIR` and reverted with `git apply -R`. If a
  run crashes mid-way, `cd $VERL_DIR && git checkout -- verl/workers/rollout/vllm_rollout/` restores
  baseline. Never commit applied patches into the verl tree.
- **Full chronological detail** (every fix, every number, the root-cause investigation) is in
  `rl_spec_decode/NOTES.md`; the design/runbook is in `rl_spec_decode/FINDINGS.md`; the patches are
  documented in `rl_spec_decode/patches/README.md`.
