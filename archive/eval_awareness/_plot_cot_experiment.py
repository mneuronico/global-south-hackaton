#!/usr/bin/env python3
"""Interaction plot: eval-awareness vs forced CoT language, by prompt language.

Two lines (ZH prompt, EN prompt); x = forced chain-of-thought language (EN -> ZH).
Compliant-only rates with Wilson 95% CIs, from judge_outputs_full/ + the CJK
compliance map. Dashed refs = natural no-system-prompt baselines (EN 42%, ZH 10%).
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cot_experiment_interaction.png"


def load_full():
    out = {}
    for f in (ROOT / "judge_outputs_full").glob("?_batch_*.json"):
        for r in json.loads(f.read_text(encoding="utf-8")):
            out[r["uid"]] = r
    return out


def baseline(prefix):
    n = k = 0
    for i in range(10):
        f = ROOT / "judge_outputs" / f"{prefix}_batch_{i:02d}.json"
        if f.exists():
            for r in json.loads(f.read_text(encoding="utf-8")):
                n += 1; k += int(bool(r["aware"]))
    return k / n * 100


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p * 100, (p - (c - h)) * 100, ((c + h) - p) * 100


def rate(V, comp, arm):
    ks = [u.split(":", 1)[1] for u in V if u.startswith(arm + ":") and comp[u]["compliant"]]
    k = sum(int(bool(V[f"{arm}:{t}"]["aware"])) for t in ks)
    return wilson(k, len(ks)), len(ks), k


def main():
    V = load_full()
    comp = json.loads((ROOT / "judge_inputs_full" / "_compliance.json").read_text(encoding="utf-8"))
    # arm -> (x index by CoT lang). x=0 force EN, x=1 force ZH
    pts = {a: rate(V, comp, a) for a in "ABCD"}
    en_nat, zh_nat = baseline("en"), baseline("zh")

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    x = [0, 1]
    # ZH prompt: A(forceEN,x0) , B(forceZH,x1)
    (yA, lA, hA), nA, _ = pts["A"]; (yB, lB, hB), nB, _ = pts["B"]
    (yC, lC, hC), nC, _ = pts["C"]; (yD, lD, hD), nD, _ = pts["D"]

    ax.errorbar(x, [yA, yB], yerr=[[lA, lB], [hA, hB]], marker="o", ms=9, lw=2,
                capsize=5, color="#c0392b", label=f"ZH prompt (n={nA},{nB})")
    ax.errorbar(x, [yD, yC], yerr=[[lD, lC], [hD, hC]], marker="s", ms=9, lw=2,
                capsize=5, color="#2c6fbb", label=f"EN prompt (n={nD},{nC})")

    ax.axhline(en_nat, ls=":", color="#2c6fbb", alpha=0.7, lw=1.3)
    ax.text(1.02, en_nat, f"EN natural {en_nat:.0f}%", color="#2c6fbb", fontsize=8.5, va="center")
    ax.axhline(zh_nat, ls=":", color="#c0392b", alpha=0.7, lw=1.3)
    ax.text(1.02, zh_nat, f"ZH natural {zh_nat:.0f}%", color="#c0392b", fontsize=8.5, va="center")

    for xi, yi, lab in [(0, yA, "A"), (1, yB, "B"), (0, yD, "D"), (1, yC, "C")]:
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(8, 8),
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(["force EN CoT", "force ZH CoT"], fontsize=11)
    ax.set_xlim(-0.25, 1.45); ax.set_ylim(0, 50)
    ax.set_ylabel("Evaluation-awareness rate (%)", fontsize=11)
    ax.set_title("Forced chain-of-thought language vs eval-awareness\n"
                 "qwen3.7-plus · F1+F3+F8 · compliant traces only · Wilson 95% CI",
                 fontsize=11, pad=10)
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
