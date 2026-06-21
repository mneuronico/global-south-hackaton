"""Frequentist mixed/clustered logistic GLM for the AI-narrator effect (statsmodels).

Two frequentist estimators of the paired AI-vs-human effect on refusal:
  * ConditionalLogit (matched pairs): pair intercepts conditioned out -> exact
    within-pair MLE. Overall + per stratum/subgroup. (the "mixed" paired model)
  * GEE logistic, groups = pair, cluster-robust SE, covariate-adjusted
    (mode+scale+lang+domain+context) + narrator x (scale,lang,mode) interactions
    -> adjusted main effect p and effect-modification p.
Prints OR / 95% CI / p. Appendable to RESULTS_aiagent.md.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit

import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ALL = json.load(open("aiagent/data/experiment_ai_results.json"))
DOMS = {"Rank", "Wealth", "Legal", "Epistemic", "Status", "Attentional"}
PAIRS_LANG = [("en", "en_ai"), ("es", "es_ai"), ("zh", "zh_ai")]
graded = lambda r: r.get("behavior") in ("comply", "partial", "refuse") and (r.get("response") or "").strip()
cell = lambda r: (r["domain"], r["context"], r["mode"], r["scale"])

hb, ab = {}, {}
for base, ail in PAIRS_LANG:
    for r in ALL:
        if not graded(r) or r["domain"] not in DOMS:
            continue
        (hb if r["lang"] == base else ab if r["lang"] == ail else {}).__setitem__((base,) + cell(r), r) \
            if r["lang"] in (base, ail) else None
KEYS = sorted(set(hb) & set(ab))
ref = lambda r: 1 if r["behavior"] == "refuse" else 0

rec = []
for pid, k in enumerate(KEYS):
    lang, dom, ctx, mode, scale = k
    for nar, src in ((0, hb[k]), (1, ab[k])):
        rec.append(dict(pair=pid, ai=nar, refuse=ref(src), lang=lang, domain=dom,
                        context=ctx, mode=mode, scale=scale,
                        grab=0 if mode == "positive" else 1))
df = pd.DataFrame(rec)
print(f"rows={len(df)} pairs={df.pair.nunique()}")


def OR(beta, se=None):
    return np.exp(beta)


# ---- ConditionalLogit (matched pairs) overall + subgroups ----
def clogit(sub):
    sub = sub[sub.groupby("pair")["refuse"].transform("nunique") == 2]  # keep discordant-info pairs
    if sub.ai.nunique() < 2 or len(sub) < 4:
        return None
    try:
        r = ConditionalLogit(sub["refuse"], sub[["ai"]], groups=sub["pair"]).fit(disp=0)
        b, se, p = r.params["ai"], r.bse["ai"], r.pvalues["ai"]
        ci = r.conf_int().loc["ai"]
        return dict(OR=np.exp(b), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=p, n=sub.pair.nunique())
    except Exception as e:  # noqa: BLE001
        return {"err": str(e)[:60]}


print("\n== ConditionalLogit (matched pairs) ==")
blocks = [("overall", df), ("control", df[df.grab == 0]), ("grab", df[df.grab == 1])]
for s in ["individual", "group", "society"]:
    blocks.append((f"scale:{s}", df[df.scale == s]))
for l in ["en", "es", "zh"]:
    blocks.append((f"lang:{l}", df[df.lang == l]))
for m in ["positive", "positive+negative", "negative"]:
    blocks.append((f"mode:{m}", df[df["mode"] == m]))
CL = {}
for name, sub in blocks:
    res = clogit(sub)
    CL[name] = res
    if res and "err" not in res:
        print(f"  {name:22s} OR={res['OR']:.2f} [{res['lo']:.2f},{res['hi']:.2f}] p={res['p']:.4f} (npairs~{res['n']})")
    else:
        print(f"  {name:22s} {res}")


# ---- GEE logistic, cluster-robust by pair, covariate-adjusted ----
fam = sm.families.Binomial()
ind = sm.cov_struct.Independence()
print("\n== GEE logistic (cluster=pair, robust SE) ==")
m_main = smf.gee("refuse ~ ai + C(mode) + C(scale) + C(lang) + C(domain) + C(context)",
                 "pair", df, family=fam, cov_struct=ind).fit()
b = m_main.params["ai"]; ci = m_main.conf_int().loc["ai"]; p = m_main.pvalues["ai"]
GEE_MAIN = dict(OR=np.exp(b), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=p)
print(f"  adjusted main: ai OR={GEE_MAIN['OR']:.2f} [{GEE_MAIN['lo']:.2f},{GEE_MAIN['hi']:.2f}] p={p:.4f}")

# effect modification: narrator x scale, x lang, x mode
m_int = smf.gee("refuse ~ ai*C(scale) + ai*C(lang) + ai*C(mode) + C(domain) + C(context)",
                "pair", df, family=fam, cov_struct=ind).fit()
print("\n  interaction terms (Wald p):")
INT = {}
for term in m_int.params.index:
    if term.startswith("ai:"):
        INT[term] = dict(coef=m_int.params[term], OR=np.exp(m_int.params[term]), p=m_int.pvalues[term])
        print(f"    {term:30s} OR={INT[term]['OR']:.2f} p={INT[term]['p']:.4f}")

# joint Wald test that ALL scale & lang interactions are zero
def joint(prefixes):
    terms = [t for t in m_int.params.index if any(t.startswith(pre) for pre in prefixes)]
    if not terms:
        return None
    R = np.zeros((len(terms), len(m_int.params)))
    for i, t in enumerate(terms):
        R[i, list(m_int.params.index).index(t)] = 1
    wt = m_int.wald_test(R, scalar=True)
    return float(wt.statistic), float(wt.pvalue), len(terms)

for label, pre in [("scale", ["ai:C(scale)"]), ("lang", ["ai:C(lang)"]), ("mode", ["ai:C(mode)"])]:
    j = joint(pre)
    if j:
        print(f"  joint narrator×{label}: chi2={j[0]:.2f} df={j[2]} p={j[1]:.4f}")

# save a compact json for the writeup
out = {"clogit": {k: v for k, v in CL.items()},
       "gee_main": GEE_MAIN,
       "gee_interactions": INT,
       "joint": {lab: joint(pre) for lab, pre in
                 [("scale", ["ai:C(scale)"]), ("lang", ["ai:C(lang)"]), ("mode", ["ai:C(mode)"])]}}
json.dump(out, open("aiagent/reports/glmm_freq_results.json", "w"), indent=2, default=float)
print("\nwrote glmm_freq_results.json")
