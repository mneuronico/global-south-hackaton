#!/usr/bin/env python3
"""Sonnet 4.5: prompt-factor arm F1+F2 vs F1+F3+F8 on the SAME 50 EN safety tasks.

Both arms ran Sonnet 4.5 in English on the identical 50 task_ids (seed 20260619),
differing only in which evaluation-awareness factors were active in the prompt
(F1+F2 vs F1+F3+F8). Each reasoning trace was judged for evaluation awareness with
the same rubric. Because the task_ids match, the primary test is McNemar's exact
test on the discordant pairs. Emits a paired comparison bar chart.
"""
from __future__ import annotations
import json, glob, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
JOUT = ROOT / "judge_outputs"
OUT = ROOT / "sonnet45_f1f2_vs_f1f3f8_eval_awareness.png"

ARMS = [("F1+F2", "sonnet45_f1f2_en", "#7b6cb0"),
        ("F1+F3+F8", "sonnet45_en", "#c0392b")]


def load(prefix: str) -> dict[str, bool]:
    out = {}
    for f in sorted(glob.glob(str(JOUT / f"{prefix}_batch_*.json"))):
        for r in json.loads(Path(f).read_text(encoding="utf-8")):
            out[r["uid"].split(":", 1)[1]] = bool(r["aware"])
    return out


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n))


def main() -> None:
    f2 = load("sonnet45_f1f2_en")
    f3 = load("sonnet45_en")
    tids = sorted(set(f2) & set(f3))
    assert len(f2) == len(f3) == len(tids) == 50, (len(f2), len(f3), len(tids))

    k2 = sum(f2[t] for t in tids)
    k3 = sum(f3[t] for t in tids)
    b = sum(1 for t in tids if f2[t] and not f3[t])   # F1F2 aware, F1F3F8 not
    c = sum(1 for t in tids if f3[t] and not f2[t])   # F1F3F8 aware, F1F2 not
    both = sum(1 for t in tids if f2[t] and f3[t])
    neither = sum(1 for t in tids if not f2[t] and not f3[t])
    mp = mcnemar_exact(b, c)

    print("Sonnet 4.5 — paired eval-awareness, same 50 EN tasks, F1+F2 vs F1+F3+F8")
    print(f"  F1+F2     : {k2}/50 = {k2/50*100:.0f}%")
    print(f"  F1+F3+F8  : {k3}/50 = {k3/50*100:.0f}%")
    print(f"  difference: {(k2-k3)/50*100:+.0f} pp (F1+F2 - F1+F3+F8)")
    print(f"  contingency: both={both}  F1F2-only={b}  F1F3F8-only={c}  neither={neither}")
    print(f"  McNemar exact (paired): p = {mp:.4f}  {'SIG' if mp < 0.05 else 'ns'}")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    labels = [a[0] for a in ARMS]
    rates = [k2 / 50 * 100, k3 / 50 * 100]
    counts = [(k2, 50), (k3, 50)]
    cols = [a[2] for a in ARMS]
    bars = ax.bar(labels, rates, color=cols, edgecolor="white", width=0.6, zorder=3)
    for bar, (k, n) in zip(bars, counts):
        ax.annotate(f"{k}/{n}\n{bar.get_height():.0f}%",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Evaluation-awareness rate (%)", fontsize=11)
    ax.set_xlabel("Sonnet 4.5 prompt-factor arm", fontsize=10.5)
    ax.set_ylim(0, 40)
    ax.set_title("Sonnet 4.5 evaluation awareness by prompt-factor arm\n"
                 "same 50 EN safety tasks · LLM-judged · "
                 f"paired McNemar p = {mp:.3f}\n(F1F2-only={b}, F1F3F8-only={c}, both={both})",
                 fontsize=10.5, pad=10)
    ax.grid(True, axis="y", ls="--", alpha=0.3, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
