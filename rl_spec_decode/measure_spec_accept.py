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

  (D) --mode sleep_wake_inject : load_format=dummy + enable_sleep_mode, then replicate verl's
                            per-step lifecycle exactly: init(dummy) -> llm.sleep(level=2) [DISCARDS
                            engine weights] -> llm.wake_up() [re-allocs] -> load_main_real + inject
                            -> measure. This is the ONE variable B omits. B already == A (injection
                            logic is sound, proven); if D drops to ~0% the sleep/wake discard/realloc
                            is the verl culprit (reproduced standalone @ ~10s/iter). --diff A.json D.json
                            then names exactly what wake breaks that the post-wake inject can't restore.
                            --cycles N repeats sleep/wake to mimic multi-step. RESULT (proven): wake
                            zeroes config-derived BUFFERS (rope cos_sin_cache, attn scales) that
                            load_weights doesn't restore; capture-before-sleep + restore-after-wake
                            recovers 0% -> 38.5%/6.78 == A.

  (E) --mode sleep_wake_recompute : the VERL-FAITHFUL variant + the actual verl fix. Same as D but
                            the DRAFT gets NO pre-sleep capture (verl's hook fires post-sleep, so there
                            are no good values to capture); instead the draft's wake-zeroed buffers are
                            RECOMPUTED from config post-wake (rotary._compute_cos_sin_cache() + scales
                            reset to 1.0), navigating DIRECTLY to layer.self_attn.{rotary_emb,attn} --
                            never a broad attr scan (that reaches the Attention op's kv_cache in the
                            separate KV sleep pool, unmapped at wake -> CUDA illegal access; that scan
                            crashed verl). Target kept coherent via main-only capture/restore. This is
                            the exact logic ported into draft_inject_static.patch; expect == A.

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

    # --- buffer capture/restore (the sleep/wake fix) -----------------------------------------
    # sleep(level=2) frees ALL engine memory; wake re-maps it ZEROED; load_weights only restores
    # PARAMETERS. Config-derived buffers (rope cos_sin_cache, attn _k/_q/_v/_prob_scale) and
    # DFlash plain-attr caches (_rope_cos_sin_cache) are computed at init (correct even under dummy)
    # and NEVER repopulated by load_weights -> they stay zero -> draft is positionally blind -> ~0%.
    # Fix: clone the good (nonzero) buffers BEFORE the first sleep, copy them back AFTER each wake.
    # Needs no model internals; these tensors are constants across steps, so save/restore is exact.
    def _state_tensors(self, root):
        """(key, owner_module, attr_name, is_registered_buffer) for every registered buffer AND
        plain tensor attribute in the tree. Plain attrs catch DFlash caches named_buffers() misses."""
        import torch
        out = []
        for mod_name, mod in root.named_modules():
            for bname, b in list(mod._buffers.items()):
                if isinstance(b, torch.Tensor):
                    out.append((mod_name + ".#buf#." + bname, mod, bname, True))
            for aname, a in list(vars(mod).items()):
                if isinstance(a, torch.Tensor) and aname not in mod._buffers:
                    out.append((mod_name + ".#attr#." + aname, mod, aname, False))
        return out

    @staticmethod
    def _is_zero(t):
        try:
            return float(t.detach().float().norm()) == 0.0
        except Exception:
            return False

    def _get_t(self, owner, attr, is_buf):
        return owner._buffers[attr] if is_buf else getattr(owner, attr, None)

    # Only the small config-derived buffers get zeroed-and-not-restored (rope cos_sin_cache <=64MiB,
    # attn scales are scalars). Cap well above those but below big weight-like plain-attr tensors so
    # we (a) don't OOM the GPU and (b) don't clone things load_weights already restores.
    _BUF_CAP_BYTES = 256 * 1024 * 1024

    def capture_buffers(self, which="both"):
        """Snapshot nonzero, small buffers/caches to CPU, BEFORE any sleep. which: main|draft|both."""
        import torch
        self._buf_cache = {}
        out = {}
        for tag, mod in (("main", self.model_runner.model), ("draft", self._draft_model())):
            if which != "both" and tag != which:
                continue
            if mod is None:
                out[tag] = None
                continue
            d, skipped = {}, 0
            for key, owner, attr, is_buf in self._state_tensors(mod):
                t = self._get_t(owner, attr, is_buf)
                if not (isinstance(t, torch.Tensor) and not self._is_zero(t)):
                    continue
                if t.numel() * t.element_size() > self._BUF_CAP_BYTES:
                    skipped += 1
                    continue
                d[key] = t.detach().to("cpu", copy=True)  # CPU to avoid GPU OOM (host RAM is ample)
            self._buf_cache[tag] = d
            out[tag] = {"captured": len(d), "skipped_oversize": skipped}
        return out

    def restore_buffers(self, which="both"):
        """Copy captured buffers back into any tensor that wake left zeroed. which: main|draft|both."""
        import torch
        if not getattr(self, "_buf_cache", None):
            return {"status": "no cache (capture_buffers never ran)"}
        counts = {}
        for tag, mod in (("main", self.model_runner.model), ("draft", self._draft_model())):
            if which != "both" and tag != which:
                continue
            if mod is None:
                continue
            saved = self._buf_cache.get(tag, {})
            n = 0
            for key, owner, attr, is_buf in self._state_tensors(mod):
                s = saved.get(key)
                if s is None:
                    continue
                t = self._get_t(owner, attr, is_buf)
                if isinstance(t, torch.Tensor) and tuple(t.shape) == tuple(s.shape) and self._is_zero(t):
                    t.data.copy_(s.to(t.device, t.dtype))
                    n += 1
            counts[tag] = n
        return counts

    # --- THE verl FIX: recompute (don't capture) the draft's wake-zeroed config buffers ----------
    # verl's hook fires at wake = AFTER sleep, so the buffers are already zeroed -> capture can't see
    # good values. Instead RECOMPUTE from config (timing-independent). And navigate DIRECTLY to the
    # two orphaned buffer kinds -- per-layer rotary_emb.cos_sin_cache + attn _k/_q/_v/_prob_scale --
    # NEVER scanning all attrs (that reaches the Attention op's kv_cache in the separate KV sleep pool,
    # unmapped at wake -> CUDA illegal access; that scan crashed verl). _rope_cos_sin_cache auto-fixes
    # because _build_fused_kv_buffers aliases it to layer-0's cos_sin_cache (verified in qwen3_dflash.py).
    def recompute_draft_buffers(self):
        import torch
        dm = self._draft_model()
        if dm is None:
            return {"status": "no draft"}
        inner = getattr(dm, "model", dm)
        layers = getattr(inner, "layers", None)
        if layers is None:
            return {"status": "no layers"}
        rope_done = rope_fail = scale_done = 0
        last_err = None
        for layer in layers:
            sa = getattr(layer, "self_attn", None)
            if sa is None:
                continue
            rotary = getattr(sa, "rotary_emb", None)
            cache = getattr(rotary, "cos_sin_cache", None) if rotary is not None else None
            if isinstance(cache, torch.Tensor):
                try:
                    # _compute_cos_sin_cache(): no-arg, config-only (base/rotary_dim/max_pos), builds
                    # fp32 -> we cast to the buffer's device+dtype. Verified in rotary_embedding/base.py.
                    new = rotary._compute_cos_sin_cache().to(cache.device, cache.dtype)
                    cache.data.copy_(new)
                    rope_done += 1
                except Exception as e:
                    rope_fail += 1
                    last_err = f"{type(e).__name__}: {e}"
            attn = getattr(sa, "attn", None)
            if attn is not None:
                for sname in ("_k_scale", "_q_scale", "_v_scale", "_prob_scale"):
                    s = getattr(attn, sname, None)
                    if isinstance(s, torch.Tensor) and self._is_zero(s):
                        s.data.fill_(1.0)
                        scale_done += 1
        # re-establish _rope_cos_sin_cache (alias) + fused KV from the now-valid params/cache
        if hasattr(inner, "_build_fused_kv_buffers"):
            try:
                inner._build_fused_kv_buffers()
            except Exception as e:
                last_err = last_err or f"rebuild: {type(e).__name__}: {e}"
        return {"status": "ok", "rope_recomputed": rope_done, "rope_fail": rope_fail,
                "scales_reset": scale_done, "err": last_err}

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
def _build_llm(args, load_format, worker_ext, enable_sleep=False):
    from vllm import LLM
    spec = {"method": args.method, "num_speculative_tokens": args.num_spec_tokens}
    if args.draft_model:
        spec["model"] = args.draft_model
    print(f"[build] load_format={load_format} spec={spec} enable_sleep={enable_sleep}", flush=True)
    kw = dict(
        model=args.model, trust_remote_code=True, speculative_config=spec,
        enforce_eager=True, gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len, max_num_seqs=len(QUESTIONS),
        disable_log_stats=False, load_format=load_format,
    )
    if enable_sleep:
        kw["enable_sleep_mode"] = True
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


def _run_path(args, load_format, do_inject, sleep_wake=False, recompute=False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True) for q in QUESTIONS]
    llm = _build_llm(args, load_format, worker_ext=True, enable_sleep=sleep_wake)

    if sleep_wake:
        # Replicate verl's per-step lifecycle: init(dummy) -> sleep(level=2) DISCARDS engine weights
        # -> wake re-allocs ZEROED -> (inject restores params) -> fix buffers.
        # recompute mode (verl-faithful): the DRAFT gets NO pre-sleep capture (verl's hook is post-
        # sleep); only the MAIN snapshot is kept, so the verification target is coherent. The draft is
        # fixed solely by the post-wake recompute under test.
        cap_which = "main" if recompute else "both"
        print(f"[capture] snapshot config-derived buffers ({cap_which}) BEFORE sleep ...", flush=True)
        print("  capture_buffers ->", llm.collective_rpc("capture_buffers", args=(cap_which,)), flush=True)
        print(f"[sleep_wake] llm.sleep(level=2) x{args.cycles} cycle(s) ...", flush=True)
        for i in range(args.cycles):
            llm.sleep(level=2)
            llm.wake_up()
            print(f"  cycle {i+1}/{args.cycles}: slept(2)+woke", flush=True)

    inject_status = None
    if do_inject:
        print("[inject] loading MAIN real (replicate verl reshard)...", flush=True)
        print("  load_main_real ->", llm.collective_rpc("load_main_real"), flush=True)
        if sleep_wake and recompute:
            # restore MAIN buffers so the target is coherent (the draft is left to recompute alone).
            print("  restore_buffers(main) ->", llm.collective_rpc("restore_buffers", args=("main",)), flush=True)
        print("[inject] running verl hook (inject_draft)...", flush=True)
        inject_status = llm.collective_rpc("inject_draft")
        print("  inject_draft ->", inject_status, flush=True)
        if recompute:
            print("[recompute] recompute_draft_buffers (THE verl FIX, post-wake) ...", flush=True)
            print("  recompute_draft_buffers ->", llm.collective_rpc("recompute_draft_buffers"), flush=True)

    if sleep_wake and not recompute:
        # mode D: copy the captured (main+draft) buffers back into the wake-zeroed tensors.
        print("[restore] restoring wake-zeroed buffers (main+draft) ...", flush=True)
        print("  restore_buffers ->", llm.collective_rpc("restore_buffers"), flush=True)

    totals = {"num_drafts": 0.0, "num_draft_tokens": 0.0, "num_accepted_tokens": 0.0}
    greedy, totals = _measure(llm, prompts, 0.0, args.max_tokens, totals)
    temp1, totals = _measure(llm, prompts, 1.0, args.max_tokens, totals)
    snap = llm.collective_rpc("snapshot_draft")
    snap = snap[0] if isinstance(snap, list) else snap

    result = {"load_format": load_format, "inject": inject_status, "sleep_wake": sleep_wake,
              "recompute": recompute, "greedy": greedy, "temp1.0": temp1, "snapshot": snap}
    _label = ("sleep_wake_recompute" if recompute else "sleep_wake_inject") if sleep_wake \
        else ("dummy_inject" if do_inject else "auto")
    print("\n================ RESULT (" + _label + ") ================", flush=True)
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
    ap.add_argument("--mode", choices=["measure", "auto", "dummy_inject", "sleep_wake_inject",
                                        "sleep_wake_recompute"], default="measure")
    ap.add_argument("--cycles", type=int, default=1, help="sleep/wake cycles before inject")
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
    elif args.mode == "sleep_wake_inject":
        _run_path(args, load_format="dummy", do_inject=True, sleep_wake=True)
    elif args.mode == "sleep_wake_recompute":
        _run_path(args, load_format="dummy", do_inject=True, sleep_wake=True, recompute=True)
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
