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
- **Attempt 2 (cu128, PASSED gate):** `FRESH=1 VLLM_CUDA=cu128`. `cuda_available=True`
  on the 535 driver (CUDA minor-version compat works); `dflash_refs_in_build=11` incl.
  `v1/spec_decode/dflash.py`. Env is good for both baselines.

## RUN 0 — smoke (no spec)
- Status: ⬜ not run / ⬜ pass / ⬜ fail
- Completed N steps cleanly?
- Errors / fixes needed:
- Last ~40 lines of log:

## RUN 1 — MTP (Path B, method=mtp, n=2)
- Status: ⬜ not run / ⬜ pass / ⬜ fail
- INTEGRATION (startup speculative_config shows method=mtp):
- BEHAVIOR (accepted tokens / acceptance rate):
- Errors / fixes needed:

## RUN 2 — DFlash (Path B, method=dflash, z-lab/Qwen3.5-4B-DFlash, n=15)
- Status: ⬜ not run / ⬜ pass / ⬜ fail
- INTEGRATION (startup speculative_config shows method=dflash + draft):
- BEHAVIOR (accepted tokens / acceptance rate / throughput):
- Errors / fixes needed:

## Patches applied to verl/vLLM (if any)
- None expected (config-only). Record here if that changes.
