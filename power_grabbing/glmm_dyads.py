"""Mixed logistic GLM: does the user's / affected party's country (US, China, Other) move
refusal, controlling for everything else?

DESIGN NOTE — why "prompt" is the grouping that controls for the other variables:
  Each prompt id `i` is ONE tensor cell (fixed domain, context, mode, scale, power). Every prompt
  is run across all 8 directed dyads, so WITHIN a prompt the ONLY things that vary are the user's
  and the affected party's nationality. Therefore grouping by prompt (random intercept, or
  conditioning it out) controls EXACTLY for domain/context/mode/scale/power — they are constant
  within the group. The country effects are identified purely from within-prompt variation
  (the same scenario with the nationalities swapped) — the paired structure of the design.

Predictors of interest (reference = "Other"):
  user_cat   in {US, China, Other}   (American -> US, Chinese -> China, else Other)
  aff_cat    in {US, China, Other}   (affected party's nationality)

Two estimators (matching the repo's prior glmm_freq.py house style):
  1) ConditionalLogit stratified by prompt  -> exact within-prompt MLE, prompt intercepts
     conditioned out (= controls for all prompt-constant covariates). Frequentist OR / 95% CI / p.
     Reported overall and split by positive (over-refusal) vs grab.
  2) BinomialBayesMixedGLM  -> Bayesian mixed logistic, random intercept by prompt, with the
     other covariates ALSO entered as fixed effects (regularized, so the ~100% refusal on grabs
     does not break estimation). Corroborates (1) in a single full model.

No user_cat x aff_cat interaction: the design never includes (US user, US affected) or
(China, China), so the in-group superpower cells are absent and the interaction is not estimable.
We estimate MAIN effects only.

Run:  python glmm_dyads.py
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
warnings.filterwarnings("ignore")

GRADED = ("comply", "partial", "refuse")


def cat(nat):
    return "US" if nat == "American" else "China" if nat == "Chinese" else "Other"


TARGET_LABEL = {"minimax/minimax-m3": "MiniMax-M3", "google/gemini-2.5-flash-lite": "Gemini-2.5-FL"}
rows = [r for r in json.load(open("experiment_dyads_results.json", encoding="utf-8"))
        if r["behavior"] in GRADED]
df_all = pd.DataFrame([dict(
    refuse=1 if r["behavior"] == "refuse" else 0,
    user_cat=cat(r["user_nationality"]),
    aff_cat=cat(r["affected_nationality"]),
    mode=r["mode"], power=r["power"], domain=r["domain"], context=r["context"],
    prompt=r["i"], grab=0 if r["mode"] == "positive" else 1,
    target=r["target"],
) for r in rows])

TERMS = ["user_US", "user_China", "aff_US", "aff_China"]


def design(d):
    return pd.DataFrame({
        "user_US": (d.user_cat == "US").astype(float),
        "user_China": (d.user_cat == "China").astype(float),
        "aff_US": (d.aff_cat == "US").astype(float),
        "aff_China": (d.aff_cat == "China").astype(float),
    }, index=d.index)


# ====================================================================== 1) ConditionalLogit
def clogit(d, label):
    # keep only strata (prompts) whose refusal outcome varies -> informative for within-prompt MLE
    info = d[d.groupby("prompt")["refuse"].transform("nunique") == 2].copy()
    X = design(info)
    X = X.loc[:, X.nunique() > 1]  # drop terms with no remaining variation
    n_str = info.prompt.nunique()
    if X.empty or n_str < 2:
        print(f"  [{label}] no informative strata"); return
    res = ConditionalLogit(info["refuse"], X, groups=info["prompt"]).fit(disp=0, method="bfgs")
    ci = res.conf_int()
    print(f"  [{label}]  informative prompts={n_str}  (obs={len(info)})")
    for t in TERMS:
        if t not in res.params:
            print(f"      {t:<11} (dropped — no within-stratum variation)"); continue
        b, p = res.params[t], res.pvalues[t]
        lo, hi = np.exp(ci.loc[t, 0]), np.exp(ci.loc[t, 1])
        star = " *" if p < 0.05 else ""
        print(f"      {t:<11} OR={np.exp(b):5.2f}  95%CI=[{lo:4.2f},{hi:5.2f}]  p={p:.3f}{star}")


def bayes_glmm(grabs):
    fml = ("refuse ~ C(user_cat, Treatment('Other')) + C(aff_cat, Treatment('Other')) "
           "+ C(mode) + C(power) + C(domain) + C(context)")
    model = BinomialBayesMixedGLM.from_formula(fml, {"prompt": "0 + C(prompt)"}, grabs)
    res = model.fit_vb(verbose=False)
    names, fe_mean, fe_sd = model.exog_names, res.fe_mean, res.fe_sd
    label_map = {
        "C(user_cat, Treatment('Other'))[T.US]": "user = US",
        "C(user_cat, Treatment('Other'))[T.China]": "user = China",
        "C(aff_cat, Treatment('Other'))[T.US]": "affected = US",
        "C(aff_cat, Treatment('Other'))[T.China]": "affected = China",
    }
    for nm, lab in label_map.items():
        if nm not in names:
            continue
        j = names.index(nm)
        m, s = fe_mean[j], fe_sd[j]
        lo, hi = np.exp(m - 1.96 * s), np.exp(m + 1.96 * s)
        sig = " *" if (lo > 1 or hi < 1) else ""
        print(f"   {lab:<16} OR={np.exp(m):5.2f}  95%CI=[{lo:4.2f},{hi:5.2f}]  (logit b={m:+.2f}, sd={s:.2f}){sig}")


def run_for_target(df, tname):
    print("\n" + "#" * 78)
    print(f"# TARGET: {tname}   rows={len(df)}  prompts={df.prompt.nunique()}  refuse={df.refuse.mean():.1%}")
    print("#" * 78)
    print("1) CONDITIONAL LOGIT (stratified by prompt = controls for all prompt-constant vars)")
    print("   OR > 1 => that category REFUSES MORE than 'Other'. Reference = Other.")
    print("   PRIMARY — POWER-GRABS only (positives excluded by design: affected party is neutral).")
    clogit(df[df.grab == 1], "GRABS only")
    print("   (context only — not the target of inference:)")
    clogit(df[df.grab == 0], "POSITIVE only")
    clogit(df, "ALL modes pooled")
    print("\n2) BAYESIAN MIXED GLM on POWER-GRABS only (random intercept by prompt + controls)")
    print("   (variational Bayes; 95% CI = mean +/- 1.96*sd, on the OR scale)")
    bayes_glmm(df[df.grab == 1].copy())


for tgt in [t for t in TARGET_LABEL if (df_all.target == t).any()] + \
           sorted(set(df_all.target) - set(TARGET_LABEL)):
    run_for_target(df_all[df_all.target == tgt].copy(), TARGET_LABEL.get(tgt, tgt))
print("\n* = 95% interval excludes OR=1.  (note: design omits user==affected superpower cells -> main effects only)")
