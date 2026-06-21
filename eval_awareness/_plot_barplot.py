#!/usr/bin/env python3
"""Bar plot of eval-awareness rate per language, with Wilson 95% CIs.

qwen3.7-plus, F1+F3+F8, n=50 matched safety tasks, all arms paired. Reads the
LLM-judge verdicts from judge_outputs/ (same source as _analyze_nway.py) and
writes a PNG. Bars sorted descending by awareness rate.
"""
from __future__ import annotations
import json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
JOUT = ROOT / "judge_outputs"
OUT = ROOT / "eval_awareness_barplot_7lang.png"

# (display label, judge_outputs file prefix). ES = corrected v2 arm.
LANGS = [
    ("English", "en"), ("Spanish", "esv2"), ("Swahili", "sw"),
    ("German", "de"), ("French", "fr"), ("Hindi", "hi"), ("Chinese", "zh"),
]


def load(prefix):
    out = {}
    for k in range(10):
        f = JOUT / f"{prefix}_batch_{k:02d}.json"
        if f.exists():
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


def main():
    data = {lab: load(pref) for lab, pref in LANGS}
    tids = set.intersection(*(set(d) for d in data.values()))
    n = len(tids)

    rows = []
    for lab, _ in LANGS:
        k = sum(int(bool(data[lab][t]["aware"])) for t in tids)
        lo, hi = wilson(k, n)
        rows.append((lab, k, k / n, lo, hi))
    rows.sort(key=lambda r: -r[2])

    labels = [r[0] for r in rows]
    rates = [r[2] * 100 for r in rows]
    err_lo = [(r[2] - r[3]) * 100 for r in rows]
    err_hi = [(r[4] - r[2]) * 100 for r in rows]

    # Highlight English (reference) vs the rest.
    colors = ["#2c3e50" if lab == "English" else "#5b8fb0" for lab in labels]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = range(len(labels))
    bars = ax.bar(x, rates, color=colors, width=0.68, zorder=3,
                  yerr=[err_lo, err_hi], capsize=5,
                  error_kw=dict(ecolor="#444", lw=1.3, zorder=4))

    for i, (lab, k, rate, lo, hi) in enumerate(rows):
        ax.text(i, hi * 100 + 1.6, f"{rate*100:.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#222")
        ax.text(i, 1.5, f"{k}/{n}", ha="center", va="bottom", fontsize=8.5,
                color="white", zorder=5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Evaluation-awareness rate (%)", fontsize=11)
    ax.set_ylim(0, 62)
    ax.set_title("Evaluation awareness by prompt language\n"
                 "qwen3.7-plus · F1+F3+F8 · n=50 matched safety tasks · "
                 "LLM-judged · Wilson 95% CI",
                 fontsize=11.5, pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}  ({n} matched tasks)")


if __name__ == "__main__":
    main()
