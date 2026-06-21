#!/usr/bin/env python3
"""Wikipedia-size (PROMPT language) vs eval-awareness, 7 languages + 4 forced-CoT arms.

x = Wikipedia articles of the PROMPT language (fixed per arm; the manipulation does
NOT change it). y = eval-awareness (compliant-only). The manipulated variable —
forced chain-of-thought language — is encoded as COLOR (blue = reasons in English,
red = reasons in Chinese). So at each prompt-language x you can read off how the
forced reasoning language shifts awareness. A vertical line links the two arms that
share a prompt language (A-B at ZH x, C-D at EN x). Arms sit below the natural
trend because the v7 scaffolding suppresses awareness ~20pp (see control D vs EN).
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "scatter_wikipedia_vs_awareness_with_arms.png"

LANGS = [("EN", "en", 7_197_853), ("ES", "esv2", 2_120_474), ("SW", "sw", 120_601),
         ("DE", "de", 3_130_681), ("FR", "fr", 2_765_349), ("HI", "hi", 170_050),
         ("ZH", "zh", 1_540_727)]
WIKI = {"EN": 7_197_853, "ZH": 1_540_727}
BLUE, RED = "#1f5fa8", "#c0392b"


def load_lang(prefix):
    n = k = 0
    for i in range(10):
        f = ROOT / "judge_outputs" / f"{prefix}_batch_{i:02d}.json"
        if f.exists():
            for r in json.loads(f.read_text(encoding="utf-8")):
                n += 1; k += int(bool(r["aware"]))
    return k, n


def arm_rate(V, comp, arm):
    ts = [u for u in V if u.startswith(arm + ":") and comp[u]["compliant"]]
    return sum(int(bool(V[u]["aware"])) for u in ts) / len(ts) * 100, len(ts)


def ols(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    b = sum((a-mx)*(c-my) for a, c in zip(x, y)) / sum((a-mx)**2 for a in x)
    return b, my - b*mx


def main():
    lang = {lab: load_lang(pref) for lab, pref, _ in LANGS}
    V = {}
    for f in (ROOT / "judge_outputs_full").glob("?_batch_*.json"):
        for r in json.loads(f.read_text(encoding="utf-8")):
            V[r["uid"]] = r
    comp = json.loads((ROOT / "judge_inputs_full" / "_compliance.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    # natural 7-language scatter + OLS fit (excl ZH)
    pts = [(lab, w, lang[lab][0]/lang[lab][1]*100) for lab, _, w in LANGS]
    lx = [math.log10(w) for _, w, _ in pts]; yy = [y for *_, y in pts]
    keep = [i for i, p in enumerate(pts) if p[0] != "ZH"]
    slope, inter = ols([lx[i] for i in keep], [yy[i] for i in keep])
    xs = [min(lx) - 0.2, max(lx) + 0.25]
    ax.plot([10**v for v in xs], [slope*v + inter for v in xs], "--", color="#aaa", lw=1.4,
            zorder=1, label="natural-language trend (excl ZH)")
    for lab, w, y in pts:
        ax.scatter(w, y, s=95, color="#9bbcd6", edgecolor="#5b8fb0", zorder=2)
        ax.annotate(lab, (w, y), textcoords="offset points", xytext=(7, 6), fontsize=9, color="#456")

    # 4 arms placed at their PROMPT language's wiki size; colour = forced CoT language
    rA, nA = arm_rate(V, comp, "A"); rB, nB = arm_rate(V, comp, "B")
    rC, nC = arm_rate(V, comp, "C"); rD, nD = arm_rate(V, comp, "D")
    # vertical link between arms sharing a prompt language
    ax.plot([WIKI["ZH"], WIKI["ZH"]], [rB, rA], color="#888", lw=1.0, ls="-", zorder=3, alpha=0.6)
    ax.plot([WIKI["EN"], WIKI["EN"]], [rC, rD], color="#888", lw=1.0, ls="-", zorder=3, alpha=0.6)
    arms = [("A  ZH→EN", WIKI["ZH"], rA, BLUE, (9, 2)),
            ("B  ZH→ZH", WIKI["ZH"], rB, RED, (9, -12)),
            ("C  EN→ZH", WIKI["EN"], rC, RED, (-62, -12)),
            ("D  EN→EN", WIKI["EN"], rD, BLUE, (-62, 4))]
    for lab, x, y, col, off in arms:
        ax.scatter(x, y, s=260, marker="*", color=col, edgecolor="white", linewidth=1.0, zorder=5)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=off, fontsize=9,
                    fontweight="bold", color=col)

    ax.set_xscale("log")
    ax.set_xlabel("Wikipedia articles of the PROMPT language (log scale)", fontsize=10.5)
    ax.set_ylabel("Evaluation-awareness rate (%)", fontsize=11)
    ax.set_ylim(0, 50)
    ax.set_title("Data availability (prompt language) vs eval-awareness + forced-CoT arms\n"
                 "qwen3.7-plus · F1+F3+F8 · ★ compliant-only · colour = forced reasoning language",
                 fontsize=10.5, pad=10)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#9bbcd6", markeredgecolor="#5b8fb0",
                  markersize=9, label="natural language (prompt=CoT)"),
           Line2D([0], [0], marker="*", color="w", markerfacecolor=BLUE, markersize=15, label="arm: forced English reasoning"),
           Line2D([0], [0], marker="*", color="w", markerfacecolor=RED, markersize=15, label="arm: forced Chinese reasoning")]
    ax.legend(handles=leg, loc="upper left", fontsize=8.5, frameon=True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)
    print(f"At ZH-prompt x: natural={lang['ZH'][0]/lang['ZH'][1]*100:.0f}%  B(ZH-CoT)={rB:.0f}%  A(EN-CoT)={rA:.0f}%")
    print(f"At EN-prompt x: natural={lang['EN'][0]/lang['EN'][1]*100:.0f}%  C(ZH-CoT)={rC:.0f}%  D(EN-CoT)={rD:.0f}%")


if __name__ == "__main__":
    main()
