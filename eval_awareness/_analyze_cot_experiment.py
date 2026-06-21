#!/usr/bin/env python3
"""Analyze the 4-arm reasoning-language (CoT) experiment, qwen3.7-plus, F1+F3+F8.

2x2: prompt language {ZH, EN} x forced chain-of-thought language {EN, ZH}.
  A = ZH prompt -> force EN CoT      (condition 1)
  B = ZH prompt -> force ZH CoT      (control for A)
  C = EN prompt -> force ZH CoT      (condition 2)
  D = EN prompt -> force EN CoT      (control for C)

Awareness verdicts: judge_outputs_full/ (LLM-judged, 200 traces).
CoT-language compliance: judge_inputs_full/_compliance.json (CJK heuristic).
Natural no-system-prompt baselines: judge_outputs/{en,zh}_batch_*.json.

The scientific contrast holds the PROMPT language fixed and varies the forced CoT
language, on tasks where BOTH arms actually complied with the intended CoT
language (paired McNemar).
"""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
JOF = ROOT / "judge_outputs_full"
JO = ROOT / "judge_outputs"


def load_full():
    out = {}
    for f in JOF.glob("?_batch_*.json"):
        for r in json.loads(f.read_text(encoding="utf-8")):
            out[r["uid"]] = r
    return out


def load_baseline(prefix):
    out = {}
    for k in range(10):
        f = JO / f"{prefix}_batch_{k:02d}.json"
        if f.exists():
            for r in json.loads(f.read_text(encoding="utf-8")):
                out[r["uid"].split(":", 1)[1]] = r
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n))


def main():
    V = load_full()
    comp = json.loads((ROOT / "judge_inputs_full" / "_compliance.json").read_text(encoding="utf-8"))
    arms = {"A": "ZH->EN", "B": "ZH->ZH", "C": "EN->ZH", "D": "EN->EN"}

    def recs(arm):
        return {u.split(":", 1)[1]: V[u] for u in V if u.startswith(arm + ":")}

    R = {a: recs(a) for a in arms}
    comp_ok = lambda arm, t: comp[f"{arm}:{t}"]["compliant"]

    print("=" * 74)
    print("REASONING-LANGUAGE (CoT) EXPERIMENT — qwen3.7-plus, F1+F3+F8, n=50/arm")
    print("=" * 74)
    print(f"{'arm':3} {'prompt->CoT':11} {'aware(all50)':14} {'compliant':10} {'aware(compliant-only)':22}")
    for a, lab in arms.items():
        rs = R[a]
        aw_all = sum(int(bool(rs[t]["aware"])) for t in rs)
        comp_ts = [t for t in rs if comp_ok(a, t)]
        aw_c = sum(int(bool(rs[t]["aware"])) for t in comp_ts)
        nc = len(comp_ts)
        lo, hi = wilson(aw_c, nc)
        print(f"{a:3} {lab:11} {aw_all:2}/50 = {aw_all/50:4.0%}   {nc:2}/50      "
              f"{aw_c:2}/{nc:<2} = {aw_c/nc if nc else 0:4.0%}  CI[{lo:.0%},{hi:.0%}]")

    # natural baselines (no system prompt)
    en0, zh0 = load_baseline("en"), load_baseline("zh")
    en0_aw = sum(int(bool(r["aware"])) for r in en0.values())
    zh0_aw = sum(int(bool(r["aware"])) for r in zh0.values())
    print(f"\nNatural baselines (no system prompt): "
          f"EN prompt {en0_aw}/{len(en0)}={en0_aw/len(en0):.0%} | "
          f"ZH prompt {zh0_aw}/{len(zh0)}={zh0_aw/len(zh0):.0%}")

    print("\n" + "-" * 74)
    print("KEY CONTRASTS — paired McNemar on tasks where BOTH arms complied")
    print("-" * 74)
    for x, y, desc in (("A", "B", "ZH prompt: force-EN vs force-ZH CoT"),
                       ("D", "C", "EN prompt: force-EN vs force-ZH CoT")):
        ts = [t for t in R[x] if comp_ok(x, t) and comp_ok(y, t)]
        ax = sum(int(bool(R[x][t]["aware"])) for t in ts)
        ay = sum(int(bool(R[y][t]["aware"])) for t in ts)
        b = sum(1 for t in ts if R[x][t]["aware"] and not R[y][t]["aware"])
        c = sum(1 for t in ts if not R[x][t]["aware"] and R[y][t]["aware"])
        p = mcnemar_exact(b, c)
        print(f"\n{desc}")
        print(f"  both-compliant tasks: {len(ts)}")
        print(f"  {x}({arms[x]}) aware {ax}/{len(ts)}={ax/len(ts) if ts else 0:.0%}  vs  "
              f"{y}({arms[y]}) aware {ay}/{len(ts)}={ay/len(ts) if ts else 0:.0%}")
        print(f"  discordant: {x}-only={b}, {y}-only={c}  ->  McNemar exact p={p:.4f}  "
              f"{'SIG' if p < 0.05 else 'ns'}")

    # judge-detected reasoning language cross-check
    print("\n" + "-" * 74)
    print("Judge-detected reasoning_lang per arm (cross-check of compliance):")
    for a, lab in arms.items():
        rl = Counter(R[a][t].get("reasoning_lang") for t in R[a])
        print(f"  {a} ({lab}): {dict(rl)}")


if __name__ == "__main__":
    main()
