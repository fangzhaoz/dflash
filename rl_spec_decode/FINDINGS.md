# Findings & Runbook: vLLM speculative decoding in verl's RL rollout

Baseline integration phase. Goal: prove verl's vLLM rollout can run spec decoding
end-to-end in a tiny GRPO loop, two methods (MTP, DFlash). **Not** building the drafter
re-sync / co-training yet — that's the next phase.

Investigation = static source read of verl **v0.7.1** (latest release) + `main`. Nothing
has been executed on the remote box yet; all version claims are to be confirmed against
the actually-installed wheels via `resolved_versions.txt` from `setup_env.sh`.

---

## 1. Environment decision

**One conda env** (py3.12), all heavy installs via pip, in a deliberate order so the
vLLM nightly is never downgraded:

1. verl base deps (`requirements.txt` — has **no** vLLM pin; the pin lives only in the
   `[vllm]` extra, which we don't use)
2. `pip install --no-deps -e verl`  (verl package; can't re-pin/downgrade vLLM)
3. `pip install --no-deps -e dflash`  (+ `rich loguru`)
4. **vLLM nightly LAST**

Why one env / why override the pin: verl v0.7.1's vLLM ceiling **`vllm>=0.8.5,<=0.12.0`
lives ONLY in the `[vllm]` extra** — `VLLM_REQUIRES` (setup.py:52) wired via
`extras_require={"vllm": VLLM_REQUIRES}` (setup.py:68). It is **not** in `install_requires`
(setup.py:26–, no vllm entry) and **not** in `requirements.txt` (vllm is commented at
:19). DFlash's vLLM `method: dflash` needs **vLLM v0.20.1+ (nightly)** (DFlash README) —
above that ceiling. Because our install order uses `requirements.txt` (step 1) and
`verl --no-deps` (step 2, which skips both extras and install_requires), **the `[vllm]`
extra is never resolved, so the ceiling never applies** — no conflict to override; the
`--no-deps` is what keeps it that way. (Verified: install order is consistent with this.)

What requirements.txt DOES pin and we therefore apply: `numpy<2.0.0` (requirements.txt:8)
and `tensordict>=0.8.0,<=0.10.0,!=0.9.0` (requirements.txt:16). These may fight whatever
the nightly pulls — `pip check` in setup surfaces it.

verl's rollout is largely version-decoupled (composes a `vllm serve` CLI list → calls
`AsyncEngineArgs` / `AsyncLLM.from_vllm_config`) and has explicit version branches up to
`>=0.13.0` (`vllm_async_server.py:57–63`, also :333/:692/:747). **Nothing is tested at
0.20+ — treat nightly compatibility as UNVERIFIED; RUN 0 below is the actual gate.**

Script: `setup_env.sh` → writes/pastes back `resolved_versions.txt` (verl commit, vLLM,
torch, transformers, tensordict, ray, CUDA, GPUs, + `pip check` conflicts).

**CUDA build vs driver (resolved, learned from the first attempt).** Plain
`pip install --pre vllm` is driver-unaware and pulled a **CUDA-13 torch** (torch 2.11+cu13),
but the box driver is **535.261.03 (CUDA 12.4)** → `torch.cuda.is_available()==False` (CUDA 13
is a new *major*; needs driver ≥580). Fix: install the **cu128 nightly variant**
(`uv --torch-backend=cu128 --extra-index-url https://wheels.vllm.ai/nightly/cu128`) — still
the nightly (so `dflash` is in the build), but CUDA-12, which runs on the 535 driver via CUDA
*minor-version compatibility* (any 12.x build runs on driver ≥525.60.13). `setup_env.sh`
then **hard-gates** before declaring the env good: asserts `torch.cuda.is_available()` AND
greps the installed vLLM for `dflash` (version-number-independent — don't trust the tag).
Re-run with `FRESH=1` once to clear the polluted cu13 env.

---

## 2. How spec-config reaches the engine (two passthrough paths) — **no verl patch needed**

verl retired the SPMD rollout (PR #4411). As of v0.7.1 the vLLM rollout is
**async-server only**: `vllm_async_server.py :: launch_server()` builds an `args` dict,
converts it to a `vllm serve` CLI list via `build_cli_args_from_config()`
(which **JSON-serializes dict values** — so a dict becomes a valid `--speculative-config
'{...}'`), then calls `AsyncEngineArgs.from_cli_args()`.

### Path B — generic `engine_kwargs` passthrough  ← **what both experiments use**

`launch_server()`:
```python
engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
...
args = { ... , **engine_kwargs }      # merged verbatim into the serve args
```
`engine_kwargs.vllm` is a pre-existing empty node (`vllm: {}` in `rollout.yaml`), so new
children are added with Hydra's `+`. This forwards the spec config exactly like a
standalone `vllm serve`. Used for **both** RUN 1 (mtp) and RUN 2 (dflash) so they share
one code path — RUN 1 de-risks the plumbing, RUN 2 only swaps `method` + adds the draft.

Exact keys (built key-by-key to dodge Hydra inline-dict / slash quoting):
```
+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.method=<mtp|dflash>
+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.model=<draft repo>     # dflash only
+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.num_speculative_tokens=<N>
```
Fallback if Hydra rejects a `+` (key exists): use `++`; or pass the whole dict inline:
`+...vllm.speculative_config="{method: dflash, model: 'z-lab/Qwen3.5-4B-DFlash', num_speculative_tokens: 15}"`.

### Path A — verl-native MTP block (NOT used for baselines; it's the co-training path)

`launch_server()` also has a first-class branch:
```python
if self.config.mtp.enable and self.config.mtp.enable_rollout:
    args["speculative_config"] = {"method": self.config.mtp.method,
                                  "num_speculative_tokens": self.config.mtp.num_speculative_tokens}
```
Configured under **`actor_rollout_ref.model.mtp.*`** (rollout.mtp interpolates from it):
```
actor_rollout_ref.model.mtp.enable=True
actor_rollout_ref.model.mtp.enable_rollout=True
actor_rollout_ref.model.mtp.method=mtp
actor_rollout_ref.model.mtp.num_speculative_tokens=N
```
This path makes **verl manage the MTP/draft weights** in the training model and reshard
them into the engine — i.e. it's the natural hook for the **drafter re-sync / co-training**
phase. We deliberately avoid it for the static-draft baselines: all official examples are
**megatron-only** (`examples/mtp_trainer/*`), so FSDP support is unproven, and it would
exercise a different path than DFlash. Precedence note: if ever both are set, Path A
overwrites `args["speculative_config"]`, so keep `model.mtp.enable=False` for RUN 1/2.

---

## 3. The test ladder (one new variable per rung; required, not optional)

| Run | Script | Spec config | What it proves |
|-----|--------|-------------|----------------|
| **0** | `run0_smoke.sh` | none | verl rollout + FSDP weight-resharding/update **survives the vLLM 0.12→0.20+ nightly jump** on a real GRPO step. A `vllm serve` test does NOT cover this. |
| **1** | `run1_mtp.sh` | `method=mtp, n=2` (Path B) | the generic engine_kwargs → `--speculative-config` plumbing works; native MTP head drafts. |
| **2** | `run2_dflash.sh` | `method=dflash, model=z-lab/Qwen3.5-4B-DFlash, n=15` (Path B) | DFlash draft loads & drafts through the exact same path. |

Order is enforced by us, not the scripts: **do not run 1/2 until the previous rung comes
back green.** If RUN 0 fails on the nightly and the fix isn't quick → STOP, tell the user;
we then stand up a separate verl-blessed-vLLM env for the MTP baseline as a parallel track
(do not build it preemptively).

Tiny by design (prove integration, not convergence): Qwen3.5-4B, gsm8k, train_batch=16,
n=4, prompt/resp 512, `total_training_steps=3`, tp=1, enforce_eager=True,
`disable_log_stats=False` (so vLLM emits spec-decode metrics), no save, no val.
`run0` data via `prepare_data.sh` (verl's own gsm8k preprocessor).

---

## 4. Success criteria (per spec run) — two independent checks

- **(a) Completes:** GRPO loop runs to `total_training_steps` with no traceback.
- **(b) Drafting genuinely ACTIVE** — split into two, so "no acceptance metric in a
  3-step run" is not misread as silent fallback:
  - **INTEGRATION:** the engine startup / `serve` arg list shows `speculative_config`
    with the right `method` (`mtp`/`dflash`) and the draft model. Proves config landed.
  - **BEHAVIOR:** vLLM spec-decode metrics show **accepted tokens > 0 / acceptance rate**
    (`vllm:spec_decode_num_accepted_tokens_total`, `…num_draft_tokens_total`,
    draft acceptance rate). Proves the drafter actually engaged, not plain AR.

  Each run script greps the log for both and prints them at the end.

---

## 5. Known risks / things to confirm at runtime
- **verl × vLLM-nightly compatibility** — UNVERIFIED; RUN 0 is the gate. The `<=0.12.0`
  ceiling is *avoided* (never resolved, because we skip the `[vllm]` extra via `--no-deps`),
  not overridden.
- **Qwen3.5-4B load** under the nightly (target + native MTP head), and that
  **`z-lab/Qwen3.5-4B-DFlash`** is the right draft repo / not gated (preflight checks both).
- **Hydra `+` override** for `engine_kwargs.vllm.speculative_config.*` (fallbacks documented in the scripts).
- **numpy/tensordict pins** applied via requirements.txt (`numpy<2.0.0` :8,
  `tensordict>=0.8.0,<=0.10.0,!=0.9.0` :16) vs what the nightly pulls — `pip check` in
  setup surfaces conflicts; runtime may still be fine.
- Possible spec-decode knobs if drafting errors: `--attention-backend flash_attn`
  (DFlash README uses it for non-Gemma), `max_num_batched_tokens`. Add only if a run
  points to them (keeps "one variable per run").

## 6. Open question this phase does NOT close (the crux of the co-training phase)
Even with all three runs green: **does verl's FSDP weight-sync push the DRAFT / MTP head
weights into the vLLM engine each step, or only the main policy weights?** Irrelevant for
these static-draft baselines (the draft is loaded once by the engine and never needs to
track the policy). But it is exactly the mechanism the drafter re-sync / co-training phase
depends on — to be investigated next, likely via Path A (`model.mtp.*`), which is where
verl already manages draft weights for resharding.
