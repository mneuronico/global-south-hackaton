#!/usr/bin/env python3
"""Scatter: Wikipedia size (prompt language) vs eval-awareness AS JUDGED BY QWEN.

Independent auto-judge (qwen/qwen3.7-plus via OpenRouter, sees only the CoT,
3-way none/partial/certain). Plots TWO series vs Wikipedia articles of the prompt
language: 'certain' only, and 'partial+certain'. Each series gets its own OLS fit
(excl. ZH) and correlations. A faint connector shows the 'partial' gap per language.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
JOR = ROOT / "judge_openrouter_outputs"
OUT = ROOT / "scatter_wikipedia_vs_awareness_qwenjudge_both.png"

LANGS = [("EN", "en", 7_197_853), ("ES", "esv2", 2_120_474), ("SW", "sw", 120_601),
         ("DE", "de", 3_130_681), ("FR", "fr", 2_765_349), ("HI", "hi", 170_050),
         ("ZH", "zh", 1_540_727)]
CERT_C, PC_C = "#2c6fbb", "#e08214"   # blue = certain, orange = partial+certain


def rates(prefix):
    cats = [json.loads(l)["category"] for l in (JOR / f"qwen3.7-plus_{prefix}.jsonl").open(encoding="utf-8")]
    cats = [c for c in cats if c]
    n = len(cats)
    return (sum(c == "certain" for c in cats) / n * 100,
            sum(c in ("partial", "certain") for c in cats) / n * 100, n)


def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    return (sum((a-mx)*(b-my) for a, b in zip(x, y)) /
            (math.sqrt(sum((a-mx)**2 for a in x)) * math.sqrt(sum((b-my)**2 for b in y))))


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


def ols(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    b = sum((a-mx)*(c-my) for a, c in zip(x, y)) / sum((a-mx)**2 for a in x)
    return b, my - b*mx


def fit_line(ax, lx, ys, color, label):
    keep = [i for i in range(len(lx)) if LANGS[i][0] != "ZH"]
    slope, inter = ols([lx[i] for i in keep], [ys[i] for i in keep])
    r_k = pearson([lx[i] for i in keep], [ys[i] for i in keep])
    xs = [min(lx) - 0.2, max(lx) + 0.2]
    ax.plot([10**v for v in xs], [slope*v + inter for v in xs], "--", color=color, lw=1.4, zorder=1)
    return r_k, pearson(lx, ys), pearson(ranks(lx), ranks(ys))


def main():
    data = [(lab, w, *rates(pref)) for lab, pref, w in LANGS]  # lab,w,cert,pc,n
    lx = [math.log10(w) for _, w, *_ in data]
    cert = [d[2] for d in data]
    pc = [d[3] for d in data]

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    # connector showing the 'partial' gap per language
    for lab, w, c, p, n in data:
        ax.plot([w, w], [c, p], color="#bbb", lw=1.0, zorder=1)
    rkC, raC, rhoC = fit_line(ax, lx, cert, CERT_C, "certain")
    rkP, raP, rhoP = fit_line(ax, lx, pc, PC_C, "partial+certain")
    ax.scatter([w for _, w, *_ in data], pc, s=130, marker="s", color=PC_C,
               edgecolor="white", linewidth=1.0, zorder=3, label="partial + certain")
    ax.scatter([w for _, w, *_ in data], cert, s=140, marker="o", color=CERT_C,
               edgecolor="white", linewidth=1.0, zorder=4, label="certain only")
    for lab, w, c, p, n in data:
        ax.annotate(lab, (w, p), textcoords="offset points", xytext=(7, 5),
                    fontsize=10, fontweight="bold", color="#333")

    ax.set_xscale("log")
    ax.set_xlabel("Wikipedia articles of the prompt language (log scale)", fontsize=11)
    ax.set_ylabel("Eval-awareness rate per qwen judge (%)", fontsize=11)
    ax.set_ylim(0, max(pc) + 14)
    ax.set_title("Data availability vs eval-awareness — independent qwen judge (CoT-only)\n"
                 "qwen3.7-plus judge · F1+F3+F8 · n=50/lang · two awareness thresholds",
                 fontsize=11, pad=10)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    box = (f"certain:          ρ={rhoC:.2f}  r={raC:.2f}  r(excl ZH)={rkC:.2f}\n"
           f"partial+certain:  ρ={rhoP:.2f}  r={raP:.2f}  r(excl ZH)={rkP:.2f}")
    ax.text(0.03, 0.97, box, transform=ax.transAxes, va="top", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#ccc"))
    ax.legend(loc="lower right", fontsize=9.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)
    for lab, w, c, p, n in data:
        print(f"  {lab}: certain {c:.0f}%  partial+certain {p:.0f}%  (n={n})")


if __name__ == "__main__":
    main()
