#!/usr/bin/env python3
"""
Standalone DFlash spec-decode acceptance + injection REPRO (isolates the verl write-hook
from verl's sleep/wake lifecycle, with ~10s iterations).

Three paths (run as separate processes; A and B each write a JSON snapshot, then --diff):

  (A) --mode auto         : load_format=auto (normal vLLM init). Measure greedy AND temp=1.0.
                            Snapshot the draft object. This is the working reference.
  (B) --mode dummy_inject : load_format=dummy, then reproduce the EXACT verl hook:
                            load_main_real -> inject_draft (get_all_weights -> load_weights ->
                            embed copy -> _build_fused_kv_buffers). Measure greedy AND temp=1.0.
                            Snapshot the draft object. Faithful to verl (real target, hook-loaded draft).
  (C) --diff A.json B.json: object-level diff of the two draft snapshots — param norms,
                            named_buffers, embed sharing (data_ptr), lm_head tie, d2t / index map,
                            fused buffers. Reveals WIRING differences a norm-delta can't see.

Hypothesis under test (instrumented, not assumed): normal init runs _maybe_share_embeddings +
weight-tying + d2t/buffer build that load_weights-from-dummy skips, so B has the right param
VALUES but the wrong draft<->target WIRING. If B's acceptance recovers to A -> injection logic
sound (verl 0% is sleep/wake). If B stays ~0% -> the diff names the missing wiring.

Usage (in the dflash-verl env, from the dflash repo root):
  python rl_spec_decode/measure_spec_accept.py --mode auto         --out /tmp/A.json
  python rl_spec_decode/measure_spec_accept.py --mode dummy_inject --out /tmp/B.json
  python rl_spec_decode/measure_spec_accept.py --diff /tmp/A.json /tmp/B.json

  # legacy single measurement (real weights), e.g. for MTP reference:
  python rl_spec_decode/measure_spec_accept.py --mode measure --method mtp --num-spec-tokens 2
"""
import argparse
import json
import os
import sys
import time

# Make this module importable by the spawned vLLM workers (for worker_extension_cls).
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = _HERE + os.pathsep + os.environ.get("PYTHONPATH", "")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

WORKER_EXT = "measure_spec_accept.DFlashReproExtension"

QUESTIONS = [
    "Natalia sold clips to 48 friends in April, then half as many in May. How many clips did she sell altogether?",
    "Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. How much did she earn?",
    "Betty needs $100 for a wallet and has half of it. Her parents give $15 and grandparents twice as much as her parents. How much more does she need?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?",
    "James writes a 3-page letter to 2 friends twice a week. How many pages does he write a year?",
    "Mark has a garden with 28 flowers. Twenty percent are yellow, the rest split evenly between purple and red. How many red flowers?",
    "Ken put 2 pounds of jelly beans in a box, added brownies to triple the weight, added 2 more pounds of jelly beans, then doubled it. Final weight?",
    "A school has 4 classes of 30 students each. Each student needs 5 notebooks. How many notebooks total?",
]


# ===========================================================================
# Worker extension — runs IN the vLLM worker (has self.model_runner), like verl's.
# ===========================================================================
class DFlashReproExtension:
    def _draft_model(self):
        d = getattr(self.model_runner, "drafter", None)
        return d.model if d is not None and hasattr(d, "model") else None

    def _find_target_embed(self, want_shape):
        for n, p in self.model_runner.model.named_parameters():
            if n.endswith("embed_tokens.weight") and tuple(p.shape) == tuple(want_shape):
                return p
        return None

    def load_main_real(self):
        """Load the MAIN (target/policy) model's REAL weights from its checkpoint, to replicate
        verl's per-step policy reshard (so the embed we copy into the draft is real)."""
        try:
            from vllm.config import LoadConfig
            from vllm.model_executor.model_loader import get_model_loader
            cfg = self.model_runner.vllm_config.model_config
            loader = get_model_loader(LoadConfig(load_format="auto"))
            self.model_runner.model.load_weights(loader.get_all_weights(cfg, self.model_runner.model))
            return "ok"
        except Exception as e:
            import traceback
            return f"FAILED {type(e).__name__}: {e}\n{traceback.format_exc()}"

    def inject_draft(self):
        """The EXACT verl write-hook logic."""
        try:
            spec = self.model_runner.vllm_config.speculative_config
            if spec is None or getattr(spec, "method", None) != "dflash":
                return {"status": "skip (not dflash)"}
            draft_model = self._draft_model()
            draft_cfg = getattr(spec, "draft_model_config", None)
            if draft_model is None or draft_cfg is None:
                return {"status": f"skip draft={draft_model is not None} cfg={draft_cfg is not None}"}
            from vllm.config import LoadConfig
            from vllm.model_executor.model_loader import get_model_loader
            params = dict(draft_model.named_parameters())
            before = {n: float(p.detach().float().norm()) for n, p in params.items()}
            loader = get_model_loader(LoadConfig(load_format="auto"))
            draft_model.load_weights(loader.get_all_weights(draft_cfg, draft_model))
            inner = getattr(draft_model, "model", None)
            embed = "skip"
            dft_embed = getattr(inner, "embed_tokens", None)
            if dft_embed is not None:
                tgt = self._find_target_embed(dft_embed.weight.shape)
                if tgt is not None:
                    dft_embed.weight.data.copy_(tgt.data.to(dft_embed.weight.dtype))
                    embed = "copied"
                else:
                    embed = "no_target_embed"
            changed = [n for n, p in params.items()
                       if abs(float(p.detach().float().norm()) - before[n]) > 1e-6]
            rebuilt = False
            if inner is not None and hasattr(inner, "_build_fused_kv_buffers"):
                inner._build_fused_kv_buffers()
                rebuilt = True
            return {"status": "ok", "changed": len(changed), "unchanged": len(before) - len(changed),
                    "embed": embed, "rebuilt": rebuilt}
        except Exception as e:
            import traceback
            return {"status": f"FAILED {type(e).__name__}: {e}", "tb": traceback.format_exc()}

    def snapshot_draft(self):
        """Object-level snapshot of the draft (WIRING + values). data_ptr only used WITHIN this
        process to compute sharing/tie booleans (ptrs are not comparable across processes)."""
        import torch
        dm = self._draft_model()
        if dm is None:
            return {"error": "no draft model"}
        snap = {"params": {}, "buffers": {}, "extras": {}}
        for n, p in dm.named_parameters():
            snap["params"][n] = {"norm": round(float(p.detach().float().norm()), 5), "shape": list(p.shape)}
        for n, b in dm.named_buffers():
            try:
                nrm = round(float(b.detach().float().norm()), 5) if b.is_floating_point() else None
            except Exception:
                nrm = None
            snap["buffers"][n] = {"norm": nrm, "shape": list(b.shape), "dtype": str(b.dtype)}
        inner = getattr(dm, "model", None)
        dft_embed = getattr(inner, "embed_tokens", None)
        tgt = self._find_target_embed(dft_embed.weight.shape) if dft_embed is not None else None
        e = snap["extras"]
        e["embed_shared_with_target"] = bool(
            dft_embed is not None and tgt is not None and dft_embed.weight.data_ptr() == tgt.data_ptr())
        e["draft_embed_norm"] = round(float(dft_embed.weight.detach().float().norm()), 5) if dft_embed is not None else None
        e["target_embed_norm"] = round(float(tgt.detach().float().norm()), 5) if tgt is not None else None
        lm = getattr(dm, "lm_head", None)
        lm_w = getattr(lm, "weight", None) if lm is not None else None
        e["lm_head_tied_to_draft_embed"] = bool(
            lm_w is not None and dft_embed is not None and lm_w.data_ptr() == dft_embed.weight.data_ptr())
        e["lm_head_norm"] = round(float(lm_w.detach().float().norm()), 5) if isinstance(lm_w, torch.Tensor) else None
        d2t = getattr(dm, "draft_id_to_target_id", None)
        if isinstance(d2t, torch.Tensor):
            e["d2t"] = {"present": True, "shape": list(d2t.shape),
                        "sum": int(d2t.long().sum().item()), "first": d2t.flatten()[:8].tolist()}
        else:
            e["d2t"] = {"present": d2t is not None}
        for attr in ["_num_attn_layers", "_kv_size", "_head_dim", "_num_kv_heads", "_rms_norm_eps"]:
            v = getattr(inner, attr, "MISSING")
            e[attr] = v if isinstance(v, (int, float, str)) else str(v)
        for attr in ["_fused_kv_weight", "_hidden_norm_weight", "_rope_cos_sin_cache"]:
            t = getattr(inner, attr, None)
            e[attr + "_norm"] = round(float(t.detach().float().norm()), 5) if isinstance(t, torch.Tensor) else None
        e["use_aux_hidden_state"] = getattr(inner, "use_aux_hidden_state", "MISSING")
        return snap


# ===========================================================================
# Helpers
# ===========================================================================
def _build_llm(args, load_format, worker_ext):
    from vllm import LLM
    spec = {"method": args.method, "num_speculative_tokens": args.num_spec_tokens}
    if args.draft_model:
        spec["model"] = args.draft_model
    print(f"[build] load_format={load_format} spec={spec}", flush=True)
    kw = dict(
        model=args.model, trust_remote_code=True, speculative_config=spec,
        enforce_eager=True, gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len, max_num_seqs=len(QUESTIONS),
        disable_log_stats=False, load_format=load_format,
    )
    if worker_ext:
        kw["worker_extension_cls"] = WORKER_EXT
    return LLM(**kw)


def _spec_totals(llm):
    out = {"num_drafts": 0.0, "num_draft_tokens": 0.0, "num_accepted_tokens": 0.0}
    try:
        for m in llm.get_metrics():
            name = getattr(m, "name", "")
            for k in out:
                if name.endswith("spec_decode_" + k):
                    v = getattr(m, "value", None)
                    if v is None:
                        vs = getattr(m, "values", None)
                        v = sum(vs) if vs else 0.0
                    out[k] = float(v)
    except Exception as e:
        print(f"[metrics] get_metrics failed: {e}", flush=True)
    return out


def _acc(d):
    dr, dt, ac = d["num_drafts"], d["num_draft_tokens"], d["num_accepted_tokens"]
    return {
        "num_drafts": dr, "num_draft_tokens": dt, "num_accepted_tokens": ac,
        "per_draft_token_acceptance": (ac / dt) if dt else None,
        "mean_acceptance_length": (ac / dr + 1.0) if dr else None,
    }


def _measure(llm, prompts, temperature, max_tokens, prev_totals):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    tot = _spec_totals(llm)
    delta = {k: tot[k] - prev_totals.get(k, 0.0) for k in tot}
    res = _acc(delta)
    res["gen_tokens"] = toks
    res["throughput_tok_s"] = round(toks / dt, 1) if dt else None
    res["temperature"] = temperature
    return res, tot


def _run_path(args, load_format, do_inject):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True) for q in QUESTIONS]
    llm = _build_llm(args, load_format, worker_ext=True)

    inject_status = None
    if do_inject:
        print("[inject] loading MAIN real (replicate verl reshard)...", flush=True)
        print("  load_main_real ->", llm.collective_rpc("load_main_real"), flush=True)
        print("[inject] running verl hook (inject_draft)...", flush=True)
        inject_status = llm.collective_rpc("inject_draft")
        print("  inject_draft ->", inject_status, flush=True)

    totals = {"num_drafts": 0.0, "num_draft_tokens": 0.0, "num_accepted_tokens": 0.0}
    greedy, totals = _measure(llm, prompts, 0.0, args.max_tokens, totals)
    temp1, totals = _measure(llm, prompts, 1.0, args.max_tokens, totals)
    snap = llm.collective_rpc("snapshot_draft")
    snap = snap[0] if isinstance(snap, list) else snap

    result = {"load_format": load_format, "inject": inject_status,
              "greedy": greedy, "temp1.0": temp1, "snapshot": snap}
    print("\n================ RESULT (" + ("dummy_inject" if do_inject else "auto") + ") ================", flush=True)
    for lab in ("greedy", "temp1.0"):
        r = result[lab]
        print(f"  {lab}: per_draft_token_acceptance={r['per_draft_token_acceptance']}  "
              f"mean_acceptance_length={r['mean_acceptance_length']}  "
              f"(drafts={r['num_drafts']} draft_tokens={r['num_draft_tokens']} accepted={r['num_accepted_tokens']}  "
              f"{r['throughput_tok_s']} tok/s)", flush=True)
    if isinstance(snap, dict) and "extras" in snap:
        e = snap["extras"]
        print(f"  WIRING: embed_shared={e.get('embed_shared_with_target')}  "
              f"lm_head_tied={e.get('lm_head_tied_to_draft_embed')}  "
              f"d2t={e.get('d2t')}  _num_attn_layers={e.get('_num_attn_layers')}", flush=True)
    print("====================================================\n", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[out] wrote {args.out}", flush=True)


def _diff(a_path, b_path):
    A = json.load(open(a_path))
    B = json.load(open(b_path))
    print("================ DIFF (A=auto  vs  B=dummy_inject) ================")
    for lab in ("greedy", "temp1.0"):
        ra, rb = A.get(lab, {}), B.get(lab, {})
        print(f"  ACCEPTANCE [{lab}]: A per_token={ra.get('per_draft_token_acceptance')} "
              f"mean_len={ra.get('mean_acceptance_length')}  ||  "
              f"B per_token={rb.get('per_draft_token_acceptance')} mean_len={rb.get('mean_acceptance_length')}")
    sa, sb = A.get("snapshot", {}), B.get("snapshot", {})
    ea, eb = sa.get("extras", {}), sb.get("extras", {})
    print("\n  --- WIRING (the payload) ---")
    keys = sorted(set(ea) | set(eb))
    for k in keys:
        va, vb = ea.get(k, "MISSING"), eb.get(k, "MISSING")
        flag = "  <<< DIFF" if va != vb else ""
        print(f"    {k}: A={va}  B={vb}{flag}")
    # param norm diffs
    pa, pb = sa.get("params", {}), sb.get("params", {})
    names = sorted(set(pa) | set(pb))
    print("\n  --- PARAM NORMS (A vs B; only diffs/missing) ---")
    ndiff = 0
    for n in names:
        a = pa.get(n, {}).get("norm")
        b = pb.get(n, {}).get("norm")
        if a is None or b is None or abs((a or 0) - (b or 0)) > 1e-3:
            print(f"    {n}: A={a}  B={b}")
            ndiff += 1
    if ndiff == 0:
        print("    (all param norms match within 1e-3)")
    # buffer diffs
    ba, bb = sa.get("buffers", {}), sb.get("buffers", {})
    bnames = sorted(set(ba) | set(bb))
    print("\n  --- NAMED_BUFFERS (A vs B; only diffs/missing) ---")
    bdiff = 0
    for n in bnames:
        a, b = ba.get(n), bb.get(n)
        if a != b:
            print(f"    {n}: A={a}  B={b}")
            bdiff += 1
    if bdiff == 0:
        print("    (all named_buffers match)")
    print("==================================================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["measure", "auto", "dummy_inject"], default="measure")
    ap.add_argument("--diff", nargs=2, metavar=("A.json", "B.json"), default=None)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--method", default="dflash")
    ap.add_argument("--draft-model", default="z-lab/Qwen3.5-4B-DFlash")
    ap.add_argument("--num-spec-tokens", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.6)
    ap.add_argument("--temperature", type=float, default=0.0, help="only for --mode measure")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.diff:
        _diff(args.diff[0], args.diff[1])
        return
    if args.mode == "auto":
        _run_path(args, load_format="auto", do_inject=False)
    elif args.mode == "dummy_inject":
        _run_path(args, load_format="dummy", do_inject=True)
    else:  # legacy single measurement (real weights, given temperature)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prompts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                           tokenize=False, add_generation_prompt=True) for q in QUESTIONS]
        llm = _build_llm(args, load_format="auto", worker_ext=False)
        res, _ = _measure(llm, prompts, args.temperature, args.max_tokens,
                          {"num_drafts": 0.0, "num_draft_tokens": 0.0, "num_accepted_tokens": 0.0})
        print("\n================ SPEC-DECODE ACCEPTANCE ================", flush=True)
        print(f"  method={args.method} temperature={args.temperature}", flush=True)
        print(f"  per_draft_token_acceptance={res['per_draft_token_acceptance']}", flush=True)
        print(f"  mean_acceptance_length={res['mean_acceptance_length']}", flush=True)
        print(f"  (drafts={res['num_drafts']} draft_tokens={res['num_draft_tokens']} "
              f"accepted={res['num_accepted_tokens']}  {res['throughput_tok_s']} tok/s)", flush=True)
        print("=======================================================", flush=True)


if __name__ == "__main__":
    main()
