#!/usr/bin/env python3
"""
Turn a verl run log (with the spec_accept_measurement.patch applied) into a per-step
in-rollout spec-decode acceptance TREND, to watch how a STATIC drafter's acceptance changes
as the GRPO policy drifts.

Primary source = the `>>> SPEC-TREND ...` lines (per-step DELTAS; the patch emits one per
replica per rollout step). We aggregate across replicas per step (sum the deltas, then ratio),
because Prometheus counters are cumulative and a lifetime average would mask the trend.

Usage:
  python rl_spec_decode/parse_acceptance_trend.py rl_spec_decode/logs/run1.log
  python rl_spec_decode/parse_acceptance_trend.py rl_spec_decode/logs/run2.log --csv /tmp/run2_trend.csv
"""
import argparse
import re
import sys

TREND = re.compile(
    r"SPEC-TREND replica=(?P<rep>\d+) step=(?P<step>\d+) "
    r"step_acc=(?P<acc>[\d.]+) step_mean_len=(?P<ml>[\d.]+) "
    r"step_drafts=(?P<d>\d+) step_draft_tokens=(?P<t>\d+) step_accepted=(?P<a>\d+) "
    r"cum_acc=(?P<cum>[\d.]+)"
)
# Fallback if a log predates SPEC-TREND: raw cumulative registry values per replica.
ACCEPT = re.compile(
    r"SPEC-ACCEPT per-step info=\{'replica': (?P<rep>\d+).*?"
    r"num_drafts': (?P<d>[\d.]+).*?num_draft_tokens': (?P<t>[\d.]+).*?"
    r"num_accepted_tokens': (?P<a>[\d.]+)"
)
# GRPO reward per training step (one long metric line per step).
REWARD = re.compile(
    r"training/global_step:(?P<step>\d+).*?critic/score/mean:(?:np\.float64\()?(?P<r>[-\d.eE+]+)"
)


def from_reward(lines):
    """global_step -> mean GRPO reward (critic/score/mean)."""
    out = {}
    for ln in lines:
        m = REWARD.search(ln)
        if m:
            try:
                out[int(m["step"])] = float(m["r"])
            except ValueError:
                pass
    return out


def _linfit(xs, ys):
    """Least-squares slope + Pearson r (pure python). Returns (slope, intercept, r) or None."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return None
    slope = sxy / sxx
    return slope, my - slope * mx, sxy / ((sxx * syy) ** 0.5)


def _spark(vals):
    if not vals:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return blocks[3] * len(vals)
    return "".join(blocks[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def from_trend(lines):
    """step -> summed deltas across replicas."""
    steps = {}
    for ln in lines:
        m = TREND.search(ln)
        if not m:
            continue
        s = int(m["step"])
        e = steps.setdefault(s, {"d": 0, "t": 0, "a": 0, "reps": set()})
        e["d"] += int(m["d"]); e["t"] += int(m["t"]); e["a"] += int(m["a"]); e["reps"].add(int(m["rep"]))
    return steps


def from_accept_fallback(lines):
    """Reconstruct per-step deltas from cumulative SPEC-ACCEPT registry values, per replica."""
    prev = {}            # replica -> (d,t,a)
    rep_step = {}        # replica -> next step index
    steps = {}
    for ln in lines:
        m = ACCEPT.search(ln)
        if not m:
            continue
        rep = int(m["rep"]); cd, ct, ca = float(m["d"]), float(m["t"]), float(m["a"])
        if ct == 0:
            continue
        pd, pt, pa = prev.get(rep, (0.0, 0.0, 0.0))
        dd = cd - pd if cd >= pd else cd
        dt = ct - pt if ct >= pt else ct
        da = ca - pa if ca >= pa else ca
        prev[rep] = (cd, ct, ca)
        if dt <= 0:
            continue
        s = rep_step.get(rep, 0); rep_step[rep] = s + 1
        e = steps.setdefault(s, {"d": 0, "t": 0, "a": 0, "reps": set()})
        e["d"] += int(dd); e["t"] += int(dt); e["a"] += int(da); e["reps"].add(rep)
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    lines = open(args.log, errors="ignore").read().splitlines()
    reward = from_reward(lines)  # global_step -> GRPO reward (critic/score/mean)

    def rwd(s):
        for k in (s, s + 1, s - 1):  # align acceptance step to global_step (tolerate ±1 offset)
            if k in reward:
                return reward[k]
        return None

    steps = from_trend(lines)
    source = "SPEC-TREND (per-step deltas)"
    if not steps:
        steps = from_accept_fallback(lines)
        source = "SPEC-ACCEPT fallback (deltas reconstructed from cumulative registry)"
    if not steps:
        print("No spec-decode metrics found. Is spec_accept_measurement.patch applied and "
              "rollout.disable_log_stats=False? (Looked for SPEC-TREND / SPEC-ACCEPT lines.)")
        sys.exit(1)

    rows = []
    for s in sorted(steps):
        e = steps[s]
        if e["t"] <= 0:
            continue
        acc = e["a"] / e["t"]
        ml = (e["a"] / e["d"] + 1.0) if e["d"] > 0 else 0.0
        rows.append((s, len(e["reps"]), e["d"], e["t"], e["a"], acc, ml))

    print(f"\n=== in-rollout REWARD + spec-decode acceptance TREND  ({source}) ===")
    print(f"source log: {args.log}   steps: {len(rows)}   reward points: {len(reward)}")
    print(f"{'step':>4}  {'reward':>7}  {'per_tok_acc':>11}  {'mean_len':>8}  {'reps':>4}  "
          f"{'drafts':>9}  {'accepted':>9}")
    for (s, nr, d, t, a, acc, ml) in rows:
        rv = rwd(s)
        rs = f"{rv:.4f}" if rv is not None else "   -   "
        print(f"{s:>4}  {rs:>7}  {acc:>11.4f}  {ml:>8.3f}  {nr:>4}  {d:>9}  {a:>9}")

    accs = [r[5] for r in rows]
    mls = [r[6] for r in rows]
    if accs:
        print(f"\nper_tok_acceptance: first={accs[0]:.4f}  last={accs[-1]:.4f}  "
              f"min={min(accs):.4f}  max={max(accs):.4f}  delta(last-first)={accs[-1]-accs[0]:+.4f}")
        print(f"  acc  {_spark(accs)}")
        print(f"  len  {_spark(mls)}   (mean_len first={mls[0]:.3f} last={mls[-1]:.3f})")

        # TREND on the sampled TRAINING steps only: drop the initial greedy-validation outlier
        # (its drafts are >>3x the median because it's the whole test set), then linear-fit
        # per_tok_acc vs step. This separates real drift from the greedy->sampled jump.
        med = sorted(r[2] for r in rows)[len(rows) // 2]
        train = [r for r in rows if r[2] <= 3 * med]
        dropped = [r[0] for r in rows if r[2] > 3 * med]
        if dropped:
            print(f"\n(excluding step(s) {dropped} as the greedy validation rollout: drafts >> median)")
        if len(train) >= 3:
            xs = [r[0] for r in train]; ys = [r[5] for r in train]; mly = [r[6] for r in train]
            fa = _linfit(xs, ys); fm = _linfit(xs, mly)
            span = xs[-1] - xs[0]
            if fa:
                sl, _, r = fa
                print(f"TREND per_tok_acc (steps {xs[0]}..{xs[-1]}): slope={sl:+.5f}/step  "
                      f"total Δ over run={sl*span:+.4f}  Pearson_r={r:+.3f}  "
                      f"mean={sum(ys)/len(ys):.4f}")
            if fm:
                sl, _, r = fm
                print(f"TREND mean_len      (steps {xs[0]}..{xs[-1]}): slope={sl:+.5f}/step  "
                      f"total Δ over run={sl*span:+.4f}  Pearson_r={r:+.3f}  "
                      f"mean={sum(mly)/len(mly):.3f}")
            print("verdict: |Pearson_r|<~0.3 and |total Δ| within step-to-step noise => essentially "
                  "FLAT (no drift); a clearly negative slope with |r|>~0.5 => the frozen drafter is "
                  "going stale as the policy drifts.")

    # REWARD trend (independent of acceptance alignment — keyed by global_step directly).
    if reward:
        rs_xs = sorted(reward); rs_ys = [reward[s] for s in rs_xs]
        print(f"\nGRPO reward: first={rs_ys[0]:.4f}  last={rs_ys[-1]:.4f}  "
              f"min={min(rs_ys):.4f}  max={max(rs_ys):.4f}  delta={rs_ys[-1]-rs_ys[0]:+.4f}")
        print(f"  rwd  {_spark(rs_ys)}")
        fr = _linfit(rs_xs, rs_ys)
        if fr:
            sl, _, r = fr
            print(f"TREND reward        (steps {rs_xs[0]}..{rs_xs[-1]}): slope={sl:+.5f}/step  "
                  f"total Δ over run={sl*(rs_xs[-1]-rs_xs[0]):+.4f}  Pearson_r={r:+.3f}")
        print("=> reads as: does the policy MOVE (reward rises) and, if so, does the FROZEN drafter's "
              "acceptance fall with it (drift) or hold (robust)?")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("step,reward,per_draft_token_acceptance,mean_acceptance_length,"
                    "replicas,drafts,draft_tokens,accepted\n")
            for (s, nr, d, t, a, acc, ml) in rows:
                rv = rwd(s)
                f.write(f"{s},{'' if rv is None else f'{rv:.6f}'},{acc:.6f},{ml:.6f},{nr},{d},{t},{a}\n")
        print(f"\n[csv] wrote {args.csv}")


if __name__ == "__main__":
    main()
