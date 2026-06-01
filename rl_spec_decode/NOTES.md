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
