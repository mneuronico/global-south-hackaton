#!/usr/bin/env python3
"""N-way eval-awareness comparison across all judged languages.

qwen3.7-plus, F1+F3+F8, n=50 matched safety tasks, all arms paired on the SAME
task_ids. Reads LLM-judge verdicts from judge_outputs/. Generalizes
_analyze_3way.py to the full language set:

    EN, ES(v2), SW, DE, FR, HI, ZH

Reports per-language aware rate + Wilson 95% CI, an omnibus Cochran's Q over all
related samples, and pairwise McNemar exact tests for every pair with
Holm-Bonferroni correction (many comparisons -> control family-wise error).

Self-contained: chi-square survival function via the regularized upper
incomplete gamma (Numerical Recipes gammq), so no scipy dependency.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import Counter
from itertools import combinations

JOUT = Path(__file__).resolve().parent / "judge_outputs"

# (display label, judge_outputs file prefix). ES uses the corrected v2 arm.
LANGS = [
    ("EN", "en"),
    ("ES", "esv2"),
    ("SW", "sw"),
    ("DE", "de"),
    ("FR", "fr"),
    ("HI", "hi"),
    ("ZH", "zh"),
]


def load(prefix: str) -> dict[str, dict]:
    out = {}
    for k in range(10):
        f = JOUT / f"{prefix}_batch_{k:02d}.json"
        if not f.exists():
            continue
        for rec in json.loads(f.read_text(encoding="utf-8")):
            out[rec["uid"].split(":", 1)[1]] = rec
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---- chi-square survival function via regularized upper incomplete gamma ----
def _gammln(xx):
    cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
    x = xx; y = xx
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def _gser(a, x):
    if x <= 0:
        return 0.0
    ap = a; s = 1.0 / a; d = s
    for _ in range(1000):
        ap += 1
        d *= x / ap
        s += d
        if abs(d) < abs(s) * 1e-14:
            break
    return s * math.exp(-x + a * math.log(x) - _gammln(a))


def _gcf(a, x):
    fpmin = 1e-300
    b = x + 1 - a; c = 1.0 / fpmin; d = 1.0 / b; h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - _gammln(a)) * h


def gammq(a, x):
    """Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x)."""
    if x < 0 or a <= 0:
        return float("nan")
    if x < a + 1:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def chi2_sf(x, df):
    return gammq(df / 2.0, x / 2.0)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def cochran_q(table):
    """table: list of rows, each row = [0/1 per condition]."""
    k = len(table[0]); N = len(table)
    Cj = [sum(row[j] for row in table) for j in range(k)]
    Ri = [sum(row) for row in table]
    T = sum(Cj)
    num = (k - 1) * (k * sum(c * c for c in Cj) - T * T)
    den = k * T - sum(r * r for r in Ri)
    if den == 0:
        return 0.0, 1.0, k - 1
    Q = num / den
    return Q, chi2_sf(Q, k - 1), k - 1


def holm(pairs):
    """pairs: list of (name, p). Returns list of (name, p, p_holm, sig) sorted by p."""
    m = len(pairs)
    ordered = sorted(pairs, key=lambda t: t[1])
    out = []
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)  # enforce monotone non-decreasing
        out.append((name, p, running, running < 0.05))
    return out


def main():
    data = {lab: load(pref) for lab, pref in LANGS}
    labels = [lab for lab, _ in LANGS]

    tids = set(data[labels[0]])
    for lab in labels[1:]:
        tids &= set(data[lab])
    tids = sorted(tids)
    n = len(tids)

    A = {lab: {t: int(bool(data[lab][t]["aware"])) for t in tids} for lab in labels}
    counts = {lab: sum(A[lab][t] for t in tids) for lab in labels}

    print("=" * 70)
    print("EVALUATION AWARENESS — qwen3.7-plus, F1+F3+F8")
    print(f"LLM-judged, {n} matched task_ids paired across {len(labels)} languages")
    print("=" * 70)
    for lab in sorted(labels, key=lambda L: -counts[L]):
        lo, hi = wilson(counts[lab], n)
        print(f"{lab}: {counts[lab]:2d}/{n} = {counts[lab]/n:5.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    print()

    table = [[A[lab][t] for lab in labels] for t in tids]
    Q, pq, df = cochran_q(table)
    print(f"Omnibus Cochran's Q ({len(labels)} related samples, df={df}): "
          f"Q={Q:.3f}, p={pq:.4g}  {'SIGNIFICANT' if pq < 0.05 else 'NOT significant'}")
    print()

    print(f"Pairwise McNemar exact (paired), Holm-corrected over {len(list(combinations(labels,2)))} pairs:")
    raw = []
    disc = {}
    for a, b in combinations(labels, 2):
        bc = sum(1 for t in tids if A[a][t] and not A[b][t])
        cc = sum(1 for t in tids if not A[a][t] and A[b][t])
        p = mcnemar_exact(bc, cc)
        raw.append((f"{a} vs {b}", p))
        disc[f"{a} vs {b}"] = (bc, cc)
    for name, p, ph, sig in holm(raw):
        bc, cc = disc[name]
        a, b = name.split(" vs ")
        print(f"  {name:10s}  {a}-only={bc:2d} {b}-only={cc:2d}  "
              f"p={p:.4f}  p_holm={ph:.4f}  {'SIG' if sig else 'ns'}")
    print()

    print("Reasoning-trace language (judge-detected):")
    for lab in labels:
        rl = Counter(data[lab][t].get("reasoning_lang") for t in tids)
        print(f"  {lab}: {dict(rl)}")


if __name__ == "__main__":
    main()
