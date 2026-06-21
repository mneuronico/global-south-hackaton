#!/usr/bin/env python3
"""Scatter: Wikipedia size (log) vs evaluation-awareness rate, 7 languages.

X = Wikipedia articles per language edition (mid-2026, List_of_Wikipedias), log
scale. Y = LLM-judged eval-awareness rate, read live from judge_outputs/
(qwen3.7-plus, F1+F3+F8, n=50). Fit line is OLS in log10(x) space, excluding
Chinese (the consistent outlier); correlations reported with and without ZH.
"""
from __future__ import annotations
import json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
JOUT = ROOT / "judge_outputs"
OUT = ROOT / "scatter_wikipedia_vs_awareness.png"

# (label, judge_outputs prefix, Wikipedia articles). ES = corrected v2 arm.
# Wikipedia counts: https://en.wikipedia.org/wiki/List_of_Wikipedias (mid-2026)
LANGS = [
    ("EN", "en",   7_197_853),
    ("ES", "esv2", 2_120_474),
    ("SW", "sw",     120_601),
    ("DE", "de",   3_130_681),
    ("FR", "fr",   2_765_349),
    ("HI", "hi",     170_050),
    ("ZH", "zh",   1_540_727),
]
OUTLIER = "ZH"


def load(prefix):
    out = {}
    for k in range(10):
        f = JOUT / f"{prefix}_batch_{k:02d}.json"
        if f.exists():
            for rec in json.loads(f.read_text(encoding="utf-8")):
                out[rec["uid"].split(":", 1)[1]] = rec
    return out


def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return cov/(sx*sy) if sx and sy else float("nan")


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
    while i < len(v):
        j = i
        while j+1 < len(v) and v[order[j+1]] == v[order[i]]:
            j += 1
        avg = (i+j)/2 + 1
        for k in range(i, j+1):
            r[order[k]] = avg
        i = j+1
    return r


def ols(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    b = sum((a-mx)*(c-my) for a, c in zip(x, y)) / sum((a-mx)**2 for a in x)
    return b, my - b*mx  # slope, intercept


def main():
    data = {lab: load(pref) for lab, pref, _ in LANGS}
    tids = set.intersection(*(set(d) for d in data.values()))
    n = len(tids)

    pts = []
    for lab, _, wiki in LANGS:
        rate = sum(int(bool(data[lab][t]["aware"])) for t in tids)/n*100
        pts.append((lab, wiki, rate))

    lab_all = [p[0] for p in pts]
    lx_all = [math.log10(p[1]) for p in pts]
    y_all = [p[2] for p in pts]

    r_all = pearson(lx_all, y_all)
    rho_all = pearson(ranks(lx_all), ranks(y_all))
    keep = [i for i, l in enumerate(lab_all) if l != OUTLIER]
    lx_k = [lx_all[i] for i in keep]; y_k = [y_all[i] for i in keep]
    r_k = pearson(lx_k, y_k)
    slope, intercept = ols(lx_k, y_k)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    # fit line (excl. outlier), drawn across the full x range
    xs = [min(lx_all) - 0.15, max(lx_all) + 0.15]
    ax.plot([10**v for v in xs], [slope*v + intercept for v in xs],
            "--", color="#888", lw=1.5, zorder=1,
            label=f"OLS fit excl. {OUTLIER}  (r={r_k:.2f})")

    off = {"EN": (8, 4), "ES": (8, -14), "SW": (10, 2), "DE": (8, 6),
           "FR": (8, -14), "HI": (10, 4), "ZH": (12, -4)}
    for lab, wiki, rate in pts:
        is_out = lab == OUTLIER
        ax.scatter(wiki, rate, s=140, zorder=3,
                   color="#c0392b" if is_out else "#2c6fbb",
                   edgecolor="white", linewidth=1.2)
        dx, dy = off.get(lab, (8, 4))
        ax.annotate(lab, (wiki, rate), textcoords="offset points", xytext=(dx, dy),
                    fontsize=11, fontweight="bold",
                    color="#c0392b" if is_out else "#222")
    ax.annotate("outlier: high-resource,\nbut lowest awareness\n(reasons in Chinese)",
                (1_540_727, 10), textcoords="offset points", xytext=(20, 18),
                fontsize=8.5, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

    ax.set_xscale("log")
    ax.set_xlabel("Wikipedia articles per language (log scale)", fontsize=11)
    ax.set_ylabel("Evaluation-awareness rate (%)", fontsize=11)
    ax.set_ylim(0, 48)
    ax.set_title("Evaluation awareness vs language data availability\n"
                 "qwen3.7-plus · F1+F3+F8 · n=50 matched safety tasks · LLM-judged",
                 fontsize=11.5, pad=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    txt = (f"Spearman ρ = {rho_all:.2f}  (all 7)\n"
           f"Pearson r = {r_all:.2f}  (all 7, log-x)\n"
           f"Pearson r = {r_k:.2f}  (excl. {OUTLIER})")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5, bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#ccc"))
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"Spearman(all)={rho_all:.3f}  Pearson_log(all)={r_all:.3f}  Pearson_log(excl {OUTLIER})={r_k:.3f}")
    print(f"fit excl {OUTLIER}: awareness% = {slope:.2f}*log10(wiki) + {intercept:.2f}")


if __name__ == "__main__":
    main()
