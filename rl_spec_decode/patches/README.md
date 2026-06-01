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

The last `[SPEC-ACCEPT ...]` line per replica gives the full-run cumulative numbers; the
`per_draft_token_acceptance` field is the in-rollout acceptance rate. If NO `[SPEC-ACCEPT]`
lines appear, the spec_decode metrics aren't in the front-end Prometheus registry in this build
— tell me and I'll add a `self.engine.get_metrics()` fallback.
