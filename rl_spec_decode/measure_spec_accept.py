#!/usr/bin/env python3
"""
Standalone speculative-decoding acceptance measurement (vLLM v1).

Why standalone: verl's async-server rollout path doesn't emit a spec-decode stat line
to stdout, and it runs vLLM with load_format=dummy (so the MTP head can be dummy there).
Here we load REAL weights (load_format=auto) and read vLLM v1's spec-decode counters via
LLM.get_metrics(), giving a trustworthy acceptance number for the drafter itself.

Usage (in the dflash-verl env):
  # MTP (Exp 1):
  python rl_spec_decode/measure_spec_accept.py --method mtp --num-spec-tokens 2
  # DFlash (Exp 2):
  python rl_spec_decode/measure_spec_accept.py --method dflash \
      --draft-model z-lab/Qwen3.5-4B-DFlash --num-spec-tokens 15

Prints raw spec-decode counters + mean acceptance length + per-draft-token acceptance rate.
"""
import argparse
import time

# A handful of gsm8k-style prompts (enough to estimate acceptance; mirrors the RL task).
QUESTIONS = [
    "Natalia sold clips to 48 friends in April, then half as many in May. How many clips did she sell altogether?",
    "Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. How much did she earn?",
    "Betty needs $100 for a wallet and has half of it. Her parents give $15 and grandparents twice as much as her parents. How much more does she need?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?",
    "James writes a 3-page letter to 2 friends twice a week. How many pages does he write a year?",
    "Mark has a garden with 28 flowers. Twenty percent are yellow, the rest split evenly between purple and red. How many red flowers?",
    "Ken created a care package. He put in 2 pounds of jelly beans, then added enough brownies to triple the weight, then 2 more pounds of jelly beans, then doubled it. Final weight?",
    "A school has 4 classes of 30 students each. Each student needs 5 notebooks. How many notebooks total?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--method", default="mtp", help="mtp | dflash | ...")
    ap.add_argument("--draft-model", default=None, help="draft repo (required for dflash)")
    ap.add_argument("--num-spec-tokens", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.6)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy; measures the drafter's exact-match acceptance")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    spec = {"method": args.method, "num_speculative_tokens": args.num_spec_tokens}
    if args.draft_model:
        spec["model"] = args.draft_model
    print(f"[measure] model={args.model} speculative_config={spec}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": q}],
                                tokenize=False, add_generation_prompt=True)
        for q in QUESTIONS
    ]

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        speculative_config=spec,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_seqs=len(prompts),
        # load_format defaults to "auto" -> REAL weights (incl. MTP head), unlike the verl run.
    )

    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)

    # ---- pull spec-decode counters from vLLM v1 metrics ----
    def metric_value(metrics, needle):
        for m in metrics:
            if needle in getattr(m, "name", ""):
                # Counter -> .value ; Vector -> .values (sum)
                if hasattr(m, "value") and m.value is not None:
                    return float(m.value), m.name
                vals = getattr(m, "values", None)
                if vals:
                    try:
                        return float(sum(vals)), m.name
                    except TypeError:
                        return None, m.name
        return None, None

    print("\n================ SPEC-DECODE ACCEPTANCE ================", flush=True)
    try:
        metrics = llm.get_metrics()
    except Exception as e:
        print(f"[measure] llm.get_metrics() unavailable ({type(e).__name__}: {e}).")
        print("[measure] Listing all metric names so we can adjust:")
        try:
            metrics = []
            eng = getattr(llm, "llm_engine", None)
            if eng is not None and hasattr(eng, "get_metrics"):
                metrics = eng.get_metrics()
        except Exception as e2:
            print(f"[measure] fallback also failed: {e2}")
            metrics = []

    spec_names = [getattr(m, "name", "?") for m in metrics if "spec_decode" in getattr(m, "name", "")]
    print(f"[measure] spec_decode metric names found: {spec_names}")

    num_drafts, _ = metric_value(metrics, "spec_decode_num_drafts")
    num_draft_tokens, _ = metric_value(metrics, "spec_decode_num_draft_tokens")
    num_accepted, _ = metric_value(metrics, "spec_decode_num_accepted_tokens")

    print(f"  num_drafts            = {num_drafts}")
    print(f"  num_draft_tokens      = {num_draft_tokens}")
    print(f"  num_accepted_tokens   = {num_accepted}")

    if num_drafts and num_accepted is not None:
        mean_accept_len = num_accepted / num_drafts + 1.0  # +1 bonus (always-accepted) token
        print(f"  MEAN ACCEPTANCE LENGTH = {mean_accept_len:.3f} tokens/step  (1.0 == no drafts accepted)")
    if num_draft_tokens and num_accepted is not None:
        rate = num_accepted / num_draft_tokens
        print(f"  PER-DRAFT-TOKEN ACCEPTANCE RATE = {rate:.3%}")
    if not spec_names:
        print("  !! No spec_decode_* metrics found -> spec decoding may not have engaged, "
              "or this vLLM build names them differently (see the full list above).")

    print(f"\n  generated {gen_tokens} tokens in {dt:.1f}s  ({gen_tokens/dt:.1f} tok/s, "
          f"temperature={args.temperature})")
    print("=======================================================", flush=True)


if __name__ == "__main__":
    main()
