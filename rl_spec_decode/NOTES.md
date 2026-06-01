# Run notes / results (fill in as we go)

See `FINDINGS.md` for the design + runbook. This file records what actually happened.

## Resolved versions
_Paste `resolved_versions.txt` (from `setup_env.sh`) here once installed._

```
(pending setup_env.sh — cu128 rebuild)
```

### Env setup log
- **Attempt 1 (cu13, FAILED gate):** plain-pip nightly pulled torch 2.11.0+**cu13**;
  box driver 535.261.03 = CUDA 12.4 → `torch.cuda.is_available()=False` (CUDA-13 needs
  driver ≥580). vLLM 0.22.1rc1.dev24, transformers 5.8.1, numpy 2.3.5 (numpy<2 conflict
  is cosmetic). 8× H100 visible.
- **Attempt 2 (cu128):** `FRESH=1 ... VLLM_CUDA=cu128`. Result: ⬜ pending paste-back of
  `resolved_versions.txt` + ENV GATE (cuda_available + dflash_refs_in_build).

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
