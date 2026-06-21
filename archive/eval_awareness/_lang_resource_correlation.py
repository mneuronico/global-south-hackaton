#!/usr/bin/env python3
"""Correlate eval-awareness rate with language-resource proxies (n=7 langs).

Answers: "how do I quantitatively differentiate languages by data availability /
speakers, and does it track evaluation awareness?"

Resource figures are external, sourced (see comments). Awareness rates are read
live from judge_outputs/ (same source as _analyze_nway.py). With only n=7
languages this is descriptive, not inferential — Spearman critical value for
alpha=.05 (two-tailed, n=7) is ~0.786, so treat anything below that as suggestive.
"""
from __future__ import annotations
import json, math
from pathlib import Path

JOUT = Path(__file__).resolve().parent / "judge_outputs"
LANGS = [("EN", "en"), ("ES", "esv2"), ("SW", "sw"),
         ("DE", "de"), ("FR", "fr"), ("HI", "hi"), ("ZH", "zh")]

# --- Resource proxies per language (higher = more resourced) ---------------
# joshi  : Joshi et al. 2020 (ACL) resource class 0-5. EN/ES/DE/FR/ZH=5 "Winners",
#          HI=4 "Underdogs", SW=2 "Hopefuls".  https://aclanthology.org/2020.acl-main.560/
# w3      : % of web content, W3Techs 20-Jun-2026. HI/SW reported only as "<0.1%".
#          https://w3techs.com/technologies/overview/content_language
# oscar_w : deduplicated word count, OSCAR 23.01 (Common Crawl Nov/Dec-2022).
#          NB: sw is anomalously tiny in this release (crawl/lang-ID artifact).
#          https://oscar-project.github.io/documentation/versions/oscar-2301/
# wiki    : Wikipedia articles per language edition, mid-2026.
#          https://en.wikipedia.org/wiki/List_of_Wikipedias
# spk_m   : total speakers L1+L2 in millions, ~Ethnologue 2024 (wide uncertainty
#          for SW: 100-200M).  https://en.wikipedia.org/wiki/List_of_languages_by_total_number_of_speakers
RES = {
    "EN": dict(joshi=5, w3=49.7, oscar_w=523_869_288_690, wiki=7_197_853, spk_m=1456),
    "ES": dict(joshi=5, w3=6.1,  oscar_w=63_388_237_965,  wiki=2_120_474, spk_m=560),
    "SW": dict(joshi=2, w3=0.05, oscar_w=164_459,         wiki=120_601,   spk_m=150),
    "DE": dict(joshi=5, w3=6.0,  oscar_w=73_848_586_648,  wiki=3_130_681, spk_m=135),
    "FR": dict(joshi=5, w3=4.6,  oscar_w=62_127_088_294,  wiki=2_765_349, spk_m=310),
    "HI": dict(joshi=4, w3=0.06, oscar_w=2_475_605_444,   wiki=170_050,   spk_m=609),
    "ZH": dict(joshi=5, w3=1.2,  oscar_w=44_378_380_161,  wiki=1_540_727, spk_m=1140),
}


def load(prefix):
    out = {}
    for k in range(10):
        f = JOUT / f"{prefix}_batch_{k:02d}.json"
        if f.exists():
            for rec in json.loads(f.read_text(encoding="utf-8")):
                out[rec["uid"].split(":", 1)[1]] = rec
    return out


def ranks(vals):
    """Average (fractional) ranks, 1=smallest."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy) if sx and sy else float("nan")


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def main():
    data = {lab: load(pref) for lab, pref in LANGS}
    tids = set.intersection(*(set(d) for d in data.values()))
    n = len(tids)
    aware = {lab: sum(int(bool(data[lab][t]["aware"])) for t in tids) / n for lab, _ in LANGS}

    labs = [lab for lab, _ in LANGS]
    aw = [aware[L] * 100 for L in labs]

    print(f"Awareness rate vs language-resource proxies ({n} tasks, {len(labs)} langs)")
    print(f"{'':4}" + "".join(f"{L:>9}" for L in labs))
    print("aware%" + "".join(f"{aware[L]*100:9.0f}" for L in labs))
    print()

    proxies = [
        ("Joshi class (0-5)", [RES[L]["joshi"] for L in labs], False),
        ("W3Techs web %",     [RES[L]["w3"] for L in labs], True),
        ("OSCAR-23.01 words", [RES[L]["oscar_w"] for L in labs], True),
        ("Wikipedia articles",[RES[L]["wiki"] for L in labs], True),
        ("Speakers (M)",      [RES[L]["spk_m"] for L in labs], True),
    ]
    print(f"{'proxy':22}{'Spearman rho':>14}   biggest positive-awareness outlier")
    for name, vals, logscale in proxies:
        rho = spearman(aw, vals)
        # residual outlier on rank scale
        ra, rv = ranks(aw), ranks(vals)
        resid = {labs[i]: ra[i] - rv[i] for i in range(len(labs))}
        hi = max(resid, key=lambda L: resid[L])   # more aware than resources predict
        lo = min(resid, key=lambda L: resid[L])   # less aware than resources predict
        print(f"{name:22}{rho:>14.3f}   over:{hi}(+{resid[hi]:.0f})  under:{lo}({resid[lo]:.0f})")
    print()
    print("Reminder: n=7 -> Spearman |rho|>=0.786 needed for p<.05 (two-tailed).")


if __name__ == "__main__":
    main()
