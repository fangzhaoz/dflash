# Run notes / results (fill in as we go)

See `FINDINGS.md` for the design + runbook. This file records what actually happened.

## Resolved versions
_Paste `resolved_versions.txt` (from `setup_env.sh`) here once installed._

See `resolved_versions.txt` (committed). Headline: vllm **0.22.0** (cu128 nightly),
torch **2.11.0+cu128**, transformers 5.9.0, tensordict 0.10.0, numpy 2.3.5, 8× H100,
driver 535.261.03.

### Env setup log
- **Attempt 1 (cu13, FAILED gate):** plain-pip nightly pulled torch 2.11.0+**cu13**;
  box driver 535.261.03 = CUDA 12.4 → `torch.cuda.is_available()=False` (CUDA-13 needs
  driver ≥580).
- **Attempt 2 (PASSED weak gate, but vLLM was cu13):** `cuda_available=True`, dflash present,
  torch cu128 — but the gate used lazy `import vllm`, which never loads the compiled `_C`. The
  vLLM WHEEL was actually the cu13 default (index resolution ignored `--extra-index-url cu128`).
  RUN 0 then died at rollout init: `import vllm._C -> libcudart.so.13` (cu13 vLLM on a CUDA-12
  driver/torch).
- **Attempt 3 (explicit cu129 wheel):** install `vllm-0.22.0+cu129-...whl` (v0.22.0 has NO
  +cu128; cu129 links libcudart.so.12 → runs on 535 driver). torch pinned 2.11.0+cu128 (cu128
  index) so the wheel can't swap it. **Gate hardened:** `from vllm import LLM` (forces `_C`) +
  assert torch unchanged. ⬜ re-verify pending.

## RUN 0 — smoke (no spec)
- Status: ✅ **PASS** (attempt 4). All 3 GRPO steps completed; `update_weights ~5–6 s/step`
  (FSDP→vLLM-0.22 resharding works each step), throughput ~94–99 tok/s, `critic/score/mean
  0.046875`. **The verl×vLLM-0.22 rollout + weight-sync path is proven.** Env: HF_HUB_OFFLINE=1.
- Watch item (benign): at shutdown, a DataLoader worker was SIGKILLed (CPU RAM ~203 GB across
  16 workers); training had already finished. If a longer run trips the OOM killer mid-train,
  drop `data.dataloader_num_workers` and/or worker count.
- History of fixes to get here:
- **Attempt 1:** model downloaded + instantiated fine (Qwen3.5-4B = hybrid linear/full
  attention VL model w/ native MTP, `mtp_num_hidden_layers=1`), but crashed at ref-model
  build: `ImportError: FlashAttention2 ... not installed`. Cause: verl
  `fsdp_workers.py:394` defaults actor/ref `attn_implementation=flash_attention_2`; we
  never installed flash-attn. **Fix (config-only):** `+actor_rollout_ref.model.override_config.attn_implementation=sdpa`
  (new `ATTN_IMPL` env var, default sdpa; fallback eager). Engine not yet reached, so
  verl×vLLM-0.22 still unproven.
- **Attempt 2 (sdpa):** got past model build, then `import vllm._C: libcudart.so.13`
  (cu13 vLLM wheel — see env log; fixed with cu129 wheel).
- **Attempt 3 (cu129 vLLM):** MAJOR progress — vLLM 0.22 engine fully initialized on the
  hybrid Qwen3.5 model (Mamba/GDN linear attn, FlashInfer GDN prefill JIT, API router up),
  training started, reached the **FSDP→vLLM weight-resharding** path. Failed there on a config
  assertion: embed_tokens (248320×2560 fp32 ≈ 2.54 GB) > default 2048 MB transfer bucket.
  **Fix:** `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096`
  (added to all runs). This PROVES verl×vLLM-0.22 engine init + hybrid-model load work.
- **Attempt 4 (bucket=4096 + qwen-vl-utils + HF_HUB_OFFLINE=1):** ✅ completed end-to-end.

## RUN 1 — MTP (Path B, method=mtp, n=2)
- Status: ✅ **PASS** (Exp 1 criteria met). 3 steps completed; `update_weights ~5.7–6.4 s/step`
  (resharding works with spec on).
- INTEGRATION ✅: serve args contain `--speculative_config '{"method": "mtp",
  "num_speculative_tokens": 2}'`; config echo shows `engine_kwargs.vllm.speculative_config`.
  → the Path B `+...engine_kwargs.vllm.speculative_config.*` Hydra override works, verl forwards it.
- BEHAVIOR ✅ (engaged, not silent AR): spec-decode-only Triton kernels JIT-compiled & run
  during inference — `rejection_greedy_sample_kernel` (verify/accept), `eagle_prepare_inputs_padded_kernel`
  (MTP draft prep) — plus `vllm speculative.py:709` num_spec_tokens warning. Throughput 87 vs
  RUN0 99 tok/s (spec overhead at tiny scale).
- Acceptance NUMBER (standalone, REAL weights, via `measure_spec_accept.py`, greedy):
  **mean acceptance length 2.736 / 3.0, per-draft-token acceptance 86.8%**
  (num_drafts 749, draft_tokens 1498, accepted 1300). → the MTP drafter itself is excellent.
- IN-ROLLOUT ACCEPTANCE (MEASURED via the read-only verl patch, see Patches): **0.000%**
  (`num_drafts` ~88–96k, `num_draft_tokens` ~177–193k, `num_accepted_tokens` = 0; mean acceptance
  length 1.000). CONFIRMED: in the verl rollout the **MTP head runs on dummy weights** — `load_format=dummy`
  inits it random and Path B reshards only the policy weights, never the MTP head. The draft→verify
  machinery genuinely runs (hundreds of k draft tokens proposed, spec kernels execute) but 0% are
  accepted, which is why throughput *dropped* (pure overhead, no payoff).
- CONTRAST: standalone real-weight MTP = **86.8%** (drafter quality) vs in-rollout dummy-weight MTP =
  **0.0%**. So 86.8% is NOT the in-rollout number.
- IMPLICATION: Exp-1 plumbing is proven, but effective in-RL MTP requires syncing the draft head —
  the verl-native `actor_rollout_ref.model.mtp.*` path that reshards it. That is exactly the
  drafter-resync / co-training mechanism (next phase); this 0% is its baseline (a synced head should
  move it toward ~86%). Earlier "we used the MTP head during GRPO" was wrong in the *effective* sense;
  now measured, not inferred.
  Note: vLLM v1's offline `LLM.get_metrics()` needs `disable_log_stats=False`; the verl async-server
  path doesn't emit a spec-decode stat line to stdout, which is why the in-loop number isn't directly
  readable (standalone measurement is the clean way).

## RUN 2 — DFlash (Path B, method=dflash, z-lab/Qwen3.5-4B-DFlash, n=15)
- Status: ✅ **PASS** (Exp 2 criteria met). 3 steps; `update_weights ~5.4 s/step`.
- Fix needed first: `num_speculative_tokens=15` × default `max_num_seqs=1024` made vLLM reserve
  ~14k draft slots > `max_num_batched_tokens=8192` → `VllmConfig` ValidationError
  (`max_num_scheduled_tokens=-6144`) at engine init. Fixed by `max_num_batched_tokens=32768`
  (DFlash README value) + `max_num_seqs=64` (our real concurrency) in COMMON.
- INTEGRATION ✅: serve args show `--speculative_config '{"method":"dflash","model":
  "z-lab/Qwen3.5-4B-DFlash","num_speculative_tokens":15}'`; draft downloaded; dflash kernels
  (`copy_and_expand_dflash_inputs_kernel`) ran. Path B passthrough works for DFlash.
- BEHAVIOR: drafting engaged (drafts ~90k, draft_tokens ~1.45M) but **in-rollout acceptance =
  0.000%** (measured via the patch; accepted=0, mean acceptance length 1.000).
- SMOKING GUN: `WARNING qwen3_dflash.py:363 DFlash buffer initialization was skipped. If dummy
  weights are not in use, this may indicate an error in weight loading.` → vLLM loaded the DFlash
  draft with **dummy weights**. verl's `load_format=dummy` is GLOBAL (applies to the draft too),
  and verl reshards only the policy, never the draft. So my earlier hypothesis (separate draft
  repo dodges dummy) is WRONG — measured.

## Route 1 experiment — `rollout.load_format=auto` (MTP): NEGATIVE
Ran `run1_mtp.sh actor_rollout_ref.rollout.load_format=auto` with the measurement patch.
**In-rollout acceptance stayed 0.000%** (accepted=0, throughput ~81–88 tok/s, no acceleration).
So `load_format=auto` alone does NOT give the draft real, usable weights in the verl rollout —
route 1 is INSUFFICIENT (not written up as working; the measured number did not move).
Diagnostic resolved (sub-case **a**): the override DID reach the engine — config shows
`load_format: auto` and the serve args include `--load_format auto` — yet acceptance stayed 0%
(even at the initial validation, after verl's base weight-sync). So vLLM loaded real weights at
init but the per-step policy reshard interaction leaves the MTP draft head without usable
weights. CONCLUSION: a real draft-weight path is required (load the draft real AND keep the
policy reshard from clobbering it AND resync it) — the co-training engineering, not a config flip.

## WHY CONFIG CAN'T FIX IT — sleep/wake + MiMo-MTP (code-grounded, the key result)
`load_format=auto` for DFlash was also measured **0.000%** (with the patch), even at step 1. Reason
is the colocated sleep/wake cycle, not the init load:
- verl colocates the trainer (FSDP: weights+grads+optimizer+ref) and the rollout (vLLM: weights+KV
  cache) on the SAME GPUs; they can't both fit, so each step they time-share via sleep/wake
  (`free_cache_engine=True`). Per step: wake+`update_weights` (push current policy) → generate →
  `sleep` → train actor on the GPU vLLM just freed.
- Hybrid mode sleeps at **level 2 = DISCARD weights** (not offload), because the policy must be
  re-synced from the FSDP actor every step anyway. Wake (inside `update_weights`) restores **only
  what `_iter_all_models()` yields**.
- `_iter_all_models()` = policy model, PLUS the drafter **iff** `_use_mtp_drafter_weight_sync()`
  (`method=="mtp"` AND `model_runner.drafter` exists). So the DFlash draft (method!=mtp) is
  discarded on the first sleep and **never restored** → dummy forever, regardless of init load_format.

**Why MiMo MTP works (the existence proof for the hook):** with `method=="mtp"` on Megatron +
`model.mtp.enable=True`, the actor's weight stream contains BOTH the policy and the MTP-module
params, and `_iter_all_models()` yields BOTH the policy model and the MTP drafter — so each wake
`model.load_weights(weights)` restores both. The MTP draft IS in verl's per-step restore loop →
survives sleep/wake → stays real (and, with enable_train, synced). DFlash just needs the SAME
"re-push the draft on every wake," but from its OWN weight source (z-lab checkpoint / co-trained
drafter) instead of the policy actor's stream. → the injection write-hook, run per-wake.

**VERSION CORRECTION (verified):** the `_iter_all_models` / `_use_mtp_drafter_weight_sync`
MTP-drafter sync described just above is in verl **`main`**, NOT the remote's **`v0.7.1`**. v0.7.1's
worker extension (`vLLMColocateWorkerExtension._update_weights`, utils.py:208) loads **only
`self.model_runner.model`** (the policy) — it has **no drafter handling whatsoever**. So on the
actual remote, the draft is never synced for ANY method (even MTP), which matches every 0%
measurement. Conclusion unchanged (per-wake draft injection required); the MiMo-MTP "both restored"
description applies to newer verl, not the box we're on. The injection patch is built against the
real v0.7.1 file (`draft_inject_static.patch`).

## WRITE-HOOK (static-real DFlash injection) — IN PROGRESS, partial
`draft_inject_static.patch` (first verl behavior change; built against real v0.7.1 utils.py).
Re-loads real z-lab DFlash weights into the live drafter each wake.
- Attempt 1: hook RAN every wake (`[DRAFT-INJECT] ... buffers_rebuilt=True`); acceptance moved
  **0.000% → ~0.18%** (mean length 1.000 → ~1.027). Non-zero but far below the ~39%/6.9 reference →
  weights landing partially/wrong ("ran but didn't take"). Diagnostics too weak: `load_weights`
  returns None (so loaded-count was -1, uninformative); `qwen3_dflash.py:363` buffer warning is at
  init dummy_run (likely benign). Hypothesis: partial load / name mismatch on the diffusion core.
- Attempt 2 (pending): enhanced hook measures the load by in-place param-NORM delta — reports
  `changed`/`unchanged` param counts + sample unchanged names → pinpoints what didn't load.

## STANDALONE REFERENCE ACCEPTANCE (real weights, greedy, measure_spec_accept.py)
The drafters themselves are strong — these are the targets the in-RL numbers should approach once
real weights are loaded:
| Drafter | mean acceptance length | per-draft-token | throughput |
|---|---|---|---|
| MTP (n=2)     | **2.74** / 3 max  | 86.8% | 245 tok/s |
| DFlash (n=15) | **6.88** / 16 max | 39.2% | 437 tok/s |
DFlash: drafts=302, draft_tokens=4530, accepted=1777 — lower per-token rate than MTP but ~2.5x the
mean acceptance length (drafts a 15-token block; ~6.9 tokens advanced per target forward pass,
~1.8x MTP throughput). Both are 0.0% IN the verl rollout purely due to the dummy-weight loading
gap — the drafters are fine.

## KEY UNIFIED FINDING (both baselines)
In the verl GRPO rollout with Path B, **every** speculative draft (MTP head AND DFlash) runs on
**dummy weights** because verl sets `load_format=dummy` and reshards only the policy → in-rollout
acceptance is **0.0%** for both (vs 86.8% standalone for MTP with real weights). Exp-1/Exp-2
plumbing is fully proven (config passthrough + draft machinery active in-loop), but neither
accelerates RL yet. To make spec decoding *effective* in verl training, the draft must get real
weights — two sub-problems for the co-training phase: (1) **load the draft real** in the rollout
engine (e.g. `rollout.load_format=auto` so vLLM loads the draft from its checkpoint — UNVERIFIED,
to test next), and (2) **resync** the draft as the policy updates (the novel contribution).

## Patches applied to verl/vLLM (if any)
- Baselines (RUN 0/1/2) are **config-only** — no verl/vLLM source changes.
- **`patches/spec_accept_measurement.patch`** (MEASUREMENT ONLY, not part of the baselines):
  first and only verl source modification so far. Read-only hook in `vLLMHttpServer.sleep()`
  that logs in-rollout spec-decode acceptance from the Prometheus registry. Does not alter
  generation/sampling/spec-config/weight-sync. Applied with `git apply`, reverted with `git apply -R`
  (round-trips to byte-identical original — verified). See `patches/README.md`. Used only to obtain
  the in-rollout MTP acceptance number; reverted afterward so the baseline verl is unmodified.
