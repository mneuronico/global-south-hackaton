#!/usr/bin/env python3
"""p-values for the qwen-judge awareness vs Wikipedia-size correlations (n=7).

For each awareness threshold (certain / partial+certain) and each set (all 7
languages, and excluding the ZH outlier), report Pearson r and Spearman rho with:
  - exact permutation p (gold standard for tiny n; all 7!/6! relabelings), and
  - parametric t-test p (betai), for reference.
x = log10(Wikipedia articles of the prompt language); y = qwen awareness rate.
"""
from __future__ import annotations
import json, math
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOR = ROOT / "judge_openrouter_outputs"
LANGS = [("EN", "en", 7_197_853), ("ES", "esv2", 2_120_474), ("SW", "sw", 120_601),
         ("DE", "de", 3_130_681), ("FR", "fr", 2_765_349), ("HI", "hi", 170_050),
         ("ZH", "zh", 1_540_727)]


def rates(prefix):
    cats = [json.loads(l)["category"] for l in (JOR / f"qwen3.7-plus_{prefix}.jsonl").open(encoding="utf-8")]
    cats = [c for c in cats if c]
    n = len(cats)
    return sum(c == "certain" for c in cats) / n * 100, sum(c in ("partial", "certain") for c in cats) / n * 100


def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / (sx*sy)


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
    while i < len(v):
        j = i
        while j+1 < len(v) and v[o[j+1]] == v[o[i]]:
            j += 1
        for k in range(i, j+1):
            r[o[k]] = (i+j)/2 + 1
        i = j+1
    return r


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


# --- parametric t-test p via regularized incomplete beta ------------------
def _gammln(xx):
    cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
    x = xx; y = xx; tmp = x + 5.5; tmp -= (x + 0.5) * math.log(tmp); ser = 1.000000000190015
    for c in cof:
        y += 1; ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def _betacf(a, b, x):
    fpmin = 1e-300; qab = a + b; qap = a + 1; qam = a - 1
    c = 1.0; d = 1 - qab * x / qap
    if abs(d) < fpmin: d = fpmin
    d = 1 / d; h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = fpmin if abs(d) < fpmin else d
        c = 1 + aa / c; c = fpmin if abs(c) < fpmin else c
        d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = fpmin if abs(d) < fpmin else d
        c = 1 + aa / c; c = fpmin if abs(c) < fpmin else c
        d = 1 / d; delta = d * c; h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return h


def betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(_gammln(a + b) - _gammln(a) - _gammln(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b


def t_pvalue(corr, n):
    if abs(corr) >= 1: return 0.0
    df = n - 2
    t = corr * math.sqrt(df / (1 - corr * corr))
    return betai(df / 2, 0.5, df / (df + t * t))


def perm_p(x, y, fn, observed):
    # exact two-sided permutation p: relabel y over all permutations
    total = cnt = 0
    for perm in permutations(range(len(y))):
        yp = [y[i] for i in perm]
        if abs(fn(x, yp)) >= abs(observed) - 1e-12:
            cnt += 1
        total += 1
    return cnt / total


def report(label, xs, ys):
    n = len(xs)
    for fname, fn in (("Pearson r ", pearson), ("Spearman rho", spearman)):
        c = fn(xs, ys)
        pp = perm_p(xs, ys, fn, c)
        tp = t_pvalue(c, n)
        sig = "SIG" if pp < 0.05 else "ns"
        print(f"   {fname}= {c:+.3f}   perm p={pp:.4f}   t-test p={tp:.4f}   [{sig} @perm]")


def main():
    data = {lab: rates(pref) for lab, pref, _ in LANGS}
    wiki = {lab: w for lab, _, w in LANGS}
    order = [lab for lab, _, _ in LANGS]
    lx = [math.log10(wiki[l]) for l in order]
    for mi, mlab in ((0, "certain"), (1, "partial+certain")):
        ys = [data[l][mi] for l in order]
        print(f"\n=== awareness = '{mlab}'  vs  log10(Wikipedia articles) ===")
        print(f" all {len(order)} languages:")
        report(mlab, lx, ys)
        keep = [i for i, l in enumerate(order) if l != "ZH"]
        print(" excluding ZH (n=6):")
        report(mlab, [lx[i] for i in keep], [ys[i] for i in keep])
    print("\n(perm p = exact permutation test, the reliable one at this n)")


if __name__ == "__main__":
    main()
