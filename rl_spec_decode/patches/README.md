# verl patches

These modify the verl checkout (NOT this repo, NOT the baseline run configs). Apply them
in the verl source tree; verl is installed editable (`pip install -e`), so edits take effect
on the next run with no reinstall.

## spec_accept_measurement.patch  (MEASUREMENT ONLY — not part of the baselines)

Adds a **read-only** hook to `vLLMHttpServer` (`verl/workers/rollout/vllm_rollout/vllm_async_server.py`):
a `_log_spec_decode_metrics()` method that reads the vLLM Prometheus registry and logs the
cumulative spec-decode acceptance, called at the top of `sleep()` (runs once per rollout step,
after generation). It does NOT alter generation, sampling, the spec config, or weight sync — it
only reads counters and logs. Requires `rollout.disable_log_stats=False` (all run scripts set this).

Purpose: get the **in-rollout** MTP/DFlash acceptance number directly, instead of inferring it.

## draft_inject_static.patch  (FIRST verl BEHAVIOR change — write-hook, MEASUREMENT/EXPERIMENT)

Adds `_inject_independent_draft_weights()` to `vLLMColocateWorkerExtension`
(`verl/workers/rollout/vllm_rollout/utils.py`), called at the end of `update_weights_from_ipc`
(the per-wake policy-restore path). For `method=="dflash"` only, it re-loads the **real** DFlash
draft weights from the draft's own checkpoint (`load_format=auto`, bypassing the engine's `dummy`)
into the live `model_runner.drafter.model` on every wake, then rebuilds the fused KV buffers
(`_build_fused_kv_buffers`). This counters the sleep(level=2)/wake cycle that discards engine
weights and restores only the policy. STATIC/frozen weights for now (co-training swaps the source).

**Buffer fix (`_recompute_draft_buffers`, the part that actually makes it work).** Re-loading params
was necessary but NOT sufficient: `sleep(level=2)` frees ALL engine GPU memory; `wake` re-maps it
**zeroed**; `load_weights` restores only **parameters**, so the draft's **config-derived buffers**
stay zero — per-layer `rotary_emb.cos_sin_cache` and attn `_k/_q/_v/_prob_scale` (and the DFlash core
`_rope_cos_sin_cache`, which is just an **alias** to layer-0's `cos_sin_cache`, `qwen3_dflash.py:317`).
A zero rope cache makes the draft positionally blind → near-random drafts → ~0% acceptance (mean-len ≈
1.0, the exact symptom). `_recompute_draft_buffers` navigates **directly** to
`layer.self_attn.{rotary_emb, attn}` and **recomputes** the cache from config
(`rotary._compute_cos_sin_cache()`, no-arg, config-only) + resets the zeroed scales to 1.0, then re-runs
`_build_fused_kv_buffers()` (which re-aliases `_rope_cos_sin_cache` to the now-valid cache).

Two design decisions, both forced by what broke earlier (see NOTES):
- **Recompute, not capture.** The hook fires at wake = *after* sleep, so the buffers are already zeroed
  when it first runs — there are no good values to capture. Recompute is timing-independent.
- **Targeted navigation, NEVER a broad attr scan.** An earlier capture version walked *all* tensor
  attrs and hit the vLLM `Attention` op's `kv_cache` (separate KV sleep pool, unmapped at wake) →
  `CUDA illegal memory access` that killed the engine. The recompute touches only the two known buffer
  kinds, all in the weights pool that `load_weights` already proved mapped.

PROVEN standalone in `measure_spec_accept.py --mode sleep_wake_recompute` (mode E — verl-faithful
post-wake timing): `recompute_draft_buffers -> rope_recomputed=5 rope_fail=0 scales_reset=20 err=None`,
recovering DFlash acceptance **0% → greedy 38.5% / 6.78, temp1.0 29.2% / 5.37 (== normal init)**; the
`--diff A vs E` shows every buffer + param matching the auto-init reference. Watch for the `[DRAFT-BUF]
rope_recomputed=5 rope_fail=0 ...` line in RUN 2; `rope_fail>0` or a `[DRAFT-BUF] FAILED` means the
recompute didn't take.

Self-verifying: prints `[DRAFT-INJECT] loaded N draft params ... buffers_rebuilt=...` (N=0 ⇒ silent
no-op / name mismatch) and any exception. **Success criterion = the MEASURED acceptance number**
(via spec_accept_measurement.patch) moving 0% → toward the ~6.88 standalone DFlash reference. "Run
completes" is NOT success. Apply this TOGETHER with spec_accept_measurement.patch.

Apply / run / revert (adjust paths to your boxes — verl at $VERL_DIR, this repo at $DFLASH_DIR):
```bash
VERL_DIR=${VERL_DIR:-$HOME/verl}
DFLASH_DIR=${DFLASH_DIR:-/opt/tiger/lmc_muon/dflash}

# apply
cd "$VERL_DIR" && git apply "$DFLASH_DIR/rl_spec_decode/patches/spec_accept_measurement.patch"
grep -n _log_spec_decode_metrics verl/workers/rollout/vllm_rollout/vllm_async_server.py   # confirm

# measure (real RUN 1 condition + the read-only log)
cd "$DFLASH_DIR" && bash rl_spec_decode/run1_mtp.sh
grep -i 'SPEC-ACCEPT' rl_spec_decode/logs/run1_mtp.log | tail -20

# revert (back to byte-identical baseline verl)
cd "$VERL_DIR" && git apply -R "$DFLASH_DIR/rl_spec_decode/patches/spec_accept_measurement.patch"
```

The hook ALWAYS logs a `[SPEC-ACCEPT per-step] ...` line each step (heartbeat). vLLM resets
the spec-decode counters on each wake_up, so each line is THAT step's acceptance, not cumulative
— read the steady-state steps (2–3), not step 1 (cold-start/JIT warmup). The
`per_draft_token_acceptance` field is the in-rollout acceptance rate. The line also dumps
`info` (incl. `logger_types`), `registry`, and `accum` so we can see exactly which source has
the data. If NO `[SPEC-ACCEPT]` lines appear at all, the patch isn't being picked up (confirm
it's applied; verl is editable so no reinstall is needed).
