# Findings: speculative decoding passthrough in verl's vLLM rollout

Investigation done by reading verl source on GitHub (tag **v0.7.1**, the latest
release, and `main`). Goal: can verl pass a `speculative-config` to its vLLM rollout
engine, and what are the exact config keys? No remote execution was involved — this is
a static source read, to be confirmed against the actually-installed versions.

## TL;DR

- **No verl patch is needed.** verl exposes speculative decoding to the vLLM rollout
  engine through **two** mechanisms. We can drive both experiments from config alone.
- **Important architecture note:** verl retired the SPMD rollout (PR #4411). As of
  v0.7.1 the vLLM rollout is **async-server only** (`AsyncLLM.from_vllm_config`). The
  engine is launched by composing a `vllm serve ...` CLI arg list. Our spec-config rides
  on that CLI exactly like a standalone `vllm serve`.
- **There is a real version blocker to resolve first** (see "Blocker" below): verl
  v0.7.1 pins `vllm>=0.8.5,<=0.12.0`, but DFlash's vLLM `method: dflash` needs vLLM
  v0.20.1+ (nightly). Those ranges do not overlap.

## Where the spec-config enters the engine

File: `verl/workers/rollout/vllm_rollout/vllm_async_server.py`, in `launch_server()`.
It builds an `args` dict, converts it to a `vllm serve` CLI list via
`build_cli_args_from_config(...)`, then calls `AsyncEngineArgs.from_cli_args(...)`.

`build_cli_args_from_config` (in `.../vllm_rollout/utils.py`) **JSON-serializes any dict
value**, so a dict under `speculative_config` becomes a proper
`--speculative-config '{...}'` on the engine command line.

### Path A — native MTP block (first-class, used for Experiment 1)

`vllm_async_server.py` (v0.7.1, ~L347):

```python
if self.config.mtp.enable and self.config.mtp.enable_rollout:
    speculative_config = {
        "method": self.config.mtp.method,                       # default "mtp"
        "num_speculative_tokens": self.config.mtp.num_speculative_tokens,  # default 1
    }
    args["speculative_config"] = speculative_config
```

`MtpConfig` (`verl/workers/config/model.py`) fields:
`enable`, `enable_train`, `enable_rollout`, `method` (default `"mtp"`),
`num_speculative_tokens` (default 1).

**Exact verl config keys for Experiment 1 (MTP):**
```
actor_rollout_ref.rollout.mtp.enable=True
actor_rollout_ref.rollout.mtp.enable_rollout=True
actor_rollout_ref.rollout.mtp.method=mtp
actor_rollout_ref.rollout.mtp.num_speculative_tokens=2
```
This reproduces `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`.
Note: this block only supports `method` + `num_speculative_tokens` — fine for MTP.

### Path B — generic engine_kwargs passthrough (used for Experiment 2)

`vllm_async_server.py` (v0.7.1, ~L215 and ~L326):

```python
engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
...
args = { ... , **engine_kwargs }   # merged straight into the vLLM serve args
```

So **anything** placed under `rollout.engine_kwargs.vllm.*` is forwarded verbatim to the
engine. Because dict values get JSON-serialized, we can pass the full DFlash spec config:

**Exact verl config keys for Experiment 2 (DFlash):**
```
actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config={"method":"dflash","model":"z-lab/Qwen3.5-4B-DFlash","num_speculative_tokens":15}
```
This reproduces
`--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-4B-DFlash","num_speculative_tokens":15}'`.

Precedence note: the MTP block (Path A) overwrites `args["speculative_config"]` if
`mtp.enable_rollout` is set. For Exp 2 keep `mtp.enable=False` so Path B is the only
writer.

## How we'll prove drafting is genuinely ACTIVE (success criterion b)

verl forwards `disable_log_stats` (rollout config) and has a `prometheus` block. vLLM's
spec-decode metrics (`vllm:spec_decode_num_accepted_tokens_total`,
`...num_draft_tokens_total`, acceptance rate, and per-position acceptance) are emitted
when stats logging is on. Plan: set `rollout.disable_log_stats=False` and scrape the
engine's acceptance counters / log lines. If `num_speculative_tokens>1` but accepted ==
0 or the metric is absent, that means silent AR fallback. Exact wiring TBD once the env
is up and we can see which metrics the installed vLLM emits.

## Blocker to resolve BEFORE running experiments

**verl v0.7.1 declares `vllm>=0.8.5,<=0.12.0`** (setup.py `VLLM_REQUIRES`), but
**DFlash's vLLM `method: dflash` needs vLLM v0.20.1+ / nightly** (DFlash README). The
ranges don't overlap, so a naive `pip install verl[vllm]` + DFlash won't co-resolve.

Consequences:
- **Exp 1 (MTP)** does NOT need DFlash or the nightly. It needs verl + a vLLM that both
  (i) is acceptable to verl's rollout code and (ii) supports Qwen3.5-4B's native MTP.
  Lower risk — matches "do Exp 1 first."
- **Exp 2 (DFlash)** needs vLLM nightly ≥0.20.1, which exceeds verl's declared ceiling.
  verl's rollout is fairly version-decoupled (it just composes CLI args + calls
  `AsyncEngineArgs`/`AsyncLLM`, with `>=0.13.0` feature guards), so it *may* work against
  the nightly, but this is untested and the pip pin must be overridden deliberately.

**Proposed env strategy (to confirm):** one isolated venv built as
`verl (code) + vLLM nightly + DFlash`, installing verl with its vLLM pin overridden
(install verl, then force-upgrade vLLM to the DFlash nightly). Try BOTH experiments in
that single env — MTP works on the nightly too. Fall back to a second venv with a
verl-blessed vLLM for Exp 1 only if verl turns out to be incompatible with the nightly.
All resolved versions to be pinned and recorded once installed.

## Open items needing a command run on the remote box
1. Confirm the verl version we install actually contains the `mtp` block + `engine_kwargs`
   passthrough (greps provided separately).
2. Confirm the installed vLLM nightly supports BOTH `method: mtp` (Qwen3.5-4B) and
   `method: dflash`, and that verl's rollout launches against it without erroring.
3. Capture which spec-decode acceptance metrics that vLLM build actually emits.
