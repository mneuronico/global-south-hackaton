#!/usr/bin/env python3
"""Scatter: eval-awareness vs refusal rate, per language (7).

x = refusal rate (share of trials the refusal judge marks 'refuse'; qwen judge,
prompt+response only). y = eval-awareness rate. Default awareness = our gold
binary labels (judge_outputs/); pass --awareness qwen for the qwen 'certain' rate.
Reports Pearson r, Spearman rho and an exact-permutation p-value (n=7).
"""
from __future__ import annotations
import argparse, json, math
from itertools import permutations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
LANGS = [("EN", "en"), ("ES", "esv2"), ("SW", "sw"), ("DE", "de"),
         ("FR", "fr"), ("HI", "hi"), ("ZH", "zh")]


def gold_aware(prefix):
    n = k = 0
    for i in range(10):
        f = ROOT / "judge_outputs" / f"{prefix}_batch_{i:02d}.json"
        if f.exists():
            for r in json.loads(f.read_text(encoding="utf-8")):
                n += 1; k += int(bool(r["aware"]))
    return k / n * 100


def qwen_aware(prefix, aware_set):
    cats = [json.loads(l)["category"] for l in (ROOT / "judge_openrouter_outputs" / f"qwen3.7-plus_{prefix}.jsonl").open(encoding="utf-8")]
    cats = [c for c in cats if c]
    return sum(c in aware_set for c in cats) / len(cats) * 100


def refusal_rate(prefix, metric):
    behs = [json.loads(l)["behavior"] for l in (ROOT / "judge_refusal_outputs" / f"qwen3.7-plus_{prefix}.jsonl").open(encoding="utf-8")]
    behs = [b for b in behs if b]
    aware_set = {"refuse"} if metric == "refuse" else {"refuse", "partial"}
    return sum(b in aware_set for b in behs) / len(behs) * 100, len(behs)


def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / (sx*sy) if sx and sy else 0.0


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


def perm_p(x, y, fn):
    obs = abs(fn(x, y)); cnt = tot = 0
    for pm in permutations(range(len(y))):
        if abs(fn(x, [y[i] for i in pm])) >= obs - 1e-12:
            cnt += 1
        tot += 1
    return cnt / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--awareness", choices=["gold", "qwen-cert", "qwen-pc"], default="gold")
    ap.add_argument("--refusal", choices=["refuse", "noncomply"], default="refuse")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="language labels (e.g. ZH) to drop from the trend line and correlations; still plotted.")
    args = ap.parse_args()
    if args.awareness == "gold":
        awf, alabel = gold_aware, "gold (our labels)"
    elif args.awareness == "qwen-cert":
        awf, alabel = (lambda p: qwen_aware(p, {"certain"})), "qwen 'certain'"
    else:
        awf, alabel = (lambda p: qwen_aware(p, {"partial", "certain"})), "qwen 'partial+certain'"

    pts = []
    for lab, pref in LANGS:
        ref, n = refusal_rate(pref, args.refusal)
        pts.append((lab, ref, awf(pref), n))
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    keep = [i for i, p in enumerate(pts) if p[0] not in args.exclude]
    xf = [xs[i] for i in keep]; yf = [ys[i] for i in keep]
    r = pearson(xf, yf); rho = pearson(ranks(xf), ranks(yf))
    pr = perm_p(xf, yf, pearson); prho = perm_p(xf, yf, lambda a, b: pearson(ranks(a), ranks(b)))

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for lab, x, y, n in pts:
        ax.scatter(x, y, s=150, color="#c0392b" if lab == "ZH" else "#2c6fbb",
                   edgecolor="white", linewidth=1.1, zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(8, 5),
                    fontsize=11, fontweight="bold", color="#c0392b" if lab == "ZH" else "#222")
    # OLS line (fit on the kept subset only; drawn across the full x-range)
    mx = sum(xf)/len(xf); my = sum(yf)/len(yf)
    b = sum((a-mx)*(c-my) for a, c in zip(xf, yf)) / sum((a-mx)**2 for a in xf)
    xr = [min(xs)-3, max(xs)+3]
    fitlab = f"OLS fit excl. {','.join(args.exclude)}" if args.exclude else "OLS fit"
    ax.plot(xr, [b*(v-mx)+my for v in xr], "--", color="#888", lw=1.4, zorder=1, label=fitlab)

    rlabel = "refuse" if args.refusal == "refuse" else "refuse+partial"
    ax.set_xlabel(f"Refusal rate ('{rlabel}', qwen judge on prompt+response, %)", fontsize=10.5)
    ax.set_ylabel(f"Eval-awareness rate — {alabel} (%)", fontsize=10.5)
    ax.set_title("Eval-awareness vs refusal rate, by language\n"
                 "qwen3.7-plus · F1+F3+F8 · n=50/lang", fontsize=11, pad=10)
    ax.grid(True, ls="--", alpha=0.35)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    suffix = f"  (excl {','.join(args.exclude)}, n={len(keep)})" if args.exclude else ""
    box = f"Pearson r = {r:+.2f}  (perm p={pr:.3f}){suffix}\nSpearman ρ = {rho:+.2f}  (perm p={prho:.3f})"
    ax.text(0.03, 0.97, box, transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#ccc"))
    if args.exclude:
        ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    excl = ("_excl" + "".join(args.exclude)) if args.exclude else ""
    out = ROOT / f"scatter_awareness_{args.awareness}_vs_refusal_{args.refusal}{excl}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    for lab, x, y, n in pts:
        print(f"  {lab}: refusal({rlabel})={x:.0f}%  awareness({alabel})={y:.0f}%  (n={n})")
    print(f"Pearson r={r:.3f} (perm p={pr:.3f}) | Spearman rho={rho:.3f} (perm p={prho:.3f})")


if __name__ == "__main__":
    main()
