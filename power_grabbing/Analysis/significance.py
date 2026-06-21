"""Standardized significance testing for power-grab refusal experiments.

Same input (a results table) -> the right test, automatically:

  * Overlay factor (same base cells across its levels: language, narrator,
    target/model, user_power) -> PAIRED: McNemar exact + conditional logistic
    (within-pair OR, CI, p). This is the correct test for the tensor's repeated
    measures; a plain two-proportion z-test would ignore the pairing.
  * Between-cell factor (mode, scale, domain, context, impact) -> logistic GLM,
    CLUSTER-ROBUST by base cell (GEE) so the language/narrator replicates of a
    cell aren't treated as independent. >2 levels also get a chi-square omnibus.

Outcome defaults to `refused` (behavior == "refuse"); `harm_flagged` also works.
Multiplicity via Holm/BH. Works on the team's 8-language Dataset 1 json/csv and
on the AI-narrator paired file alike.
"""
import json
import math

import numpy as np
import pandas as pd
from scipy import stats as sps
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

BASE_CELL = ["domain", "context", "mode", "scale"]
COORDS = ["target", "lang", "domain", "context", "mode", "scale", "current_user_power", "narrator"]


def load(path):
    """Load a results table (json or csv); drop empty/malformed; add helpers."""
    path = str(path)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        for c in ("response_missing", "judge_malformed"):
            if c in df.columns:
                df = df[~df[c].fillna(False).astype(bool)]
    else:
        df = pd.DataFrame(json.load(open(path, encoding="utf-8")))
        df = df[df["behavior"].isin(["comply", "partial", "refuse"])]
        if "response" in df.columns:
            df = df[df["response"].fillna("").astype(str).str.strip() != ""]
    if "refused" not in df.columns:
        df["refused"] = (df["behavior"] == "refuse").astype(int)
    df["refused"] = df["refused"].astype(int)
    if "harm_flagged" in df.columns:
        df["harm_flagged"] = df["harm_flagged"].fillna(False).astype(int)
    if "lang" in df.columns:
        s = df["lang"].astype(str)
        df["narrator"] = np.where(s.str.endswith("_ai"), "ai", "human")
        df["lang_base"] = s.str.replace("_ai", "", regex=False)
    return df.reset_index(drop=True)


def rate_ci(sub, outcome="refused", alpha=0.05):
    k, n = int(sub[outcome].sum()), len(sub)
    if n == 0:
        return dict(rate=float("nan"), lo=float("nan"), hi=float("nan"), n=0, k=0)
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return dict(rate=k / n, lo=lo, hi=hi, n=n, k=k)


def _default_pair_on(df, factor):
    cols = [c for c in ["target", "domain", "context", "mode", "scale", "current_user_power"]
            if c in df.columns and c != factor]
    if factor == "narrator" and "lang_base" in df.columns:
        cols.append("lang_base")
    elif factor != "lang" and "lang" in df.columns:
        cols.append("lang")
    return cols


def _or_ci(b, c):
    """Matched-pairs OR (level B vs A) = c/b with Haldane if a zero."""
    bb, cc = (b + 0.5, c + 0.5) if (b == 0 or c == 0) else (b, c)
    lor = math.log(cc / bb); se = math.sqrt(1 / bb + 1 / cc)
    return math.exp(lor), math.exp(lor - 1.96 * se), math.exp(lor + 1.96 * se)


def compare(df, factor, outcome="refused", levels=None, pair_on=None):
    """Compare `outcome` across two levels of `factor`; auto paired vs unpaired."""
    sub = df[df[factor].notna()].copy()
    if levels is None:
        levels = sorted(sub[factor].unique())
    if len(levels) != 2:
        return omnibus(sub, factor, outcome)
    a, b = levels
    sub = sub[sub[factor].isin([a, b])]
    po = pair_on or _default_pair_on(sub, factor)
    piv = sub.pivot_table(index=po, columns=factor, values=outcome, aggfunc="first")
    matched = piv.dropna(subset=[a, b]) if (a in piv.columns and b in piv.columns) else piv.iloc[0:0]
    paired = len(matched) >= 0.6 * max(1, len(piv))
    ra, rb = rate_ci(sub[sub[factor] == a], outcome), rate_ci(sub[sub[factor] == b], outcome)
    res = dict(factor=factor, a=a, b=b, outcome=outcome, rate_a=ra["rate"], rate_b=rb["rate"],
               n_a=ra["n"], n_b=rb["n"])
    if paired and len(matched):
        A = matched[a].astype(int).values; B = matched[b].astype(int).values
        nb = int(((A == 1) & (B == 0)).sum()); nc = int(((B == 1) & (A == 0)).sum())
        OR, lo, hi = _or_ci(nb, nc)
        nd = nb + nc
        p = sps.binomtest(min(nb, nc), nd, 0.5).pvalue if nd else 1.0
        # conditional logistic confirmation
        try:
            m = matched.reset_index()[[a, b]]
            long = pd.DataFrame({"y": np.r_[A, B], "x": np.r_[np.zeros(len(A)), np.ones(len(B))],
                                 "pair": np.r_[np.arange(len(A)), np.arange(len(A))]})
            long = long[long.groupby("pair")["y"].transform("nunique") == 2]
            cl = ConditionalLogit(long["y"], long[["x"]], groups=long["pair"]).fit(disp=0)
            cl_or, cl_p = float(np.exp(cl.params["x"])), float(cl.pvalues["x"])
        except Exception:
            cl_or, cl_p = OR, p
        res.update(test="paired (McNemar + conditional logit)", OR=OR, lo=lo, hi=hi, p=p,
                   discordant=f"{nb}/{nc}", npairs=len(matched), condlogit_OR=cl_or, condlogit_p=cl_p)
    else:
        ka, na = ra["k"], ra["n"]; kb, nb_ = rb["k"], rb["n"]
        tab = [[kb, nb_ - kb], [ka, na - ka]]
        OR, p = sps.fisher_exact(tab)
        # Wald CI on log-OR with Haldane
        c11, c10, c01, c00 = kb + .5, nb_ - kb + .5, ka + .5, na - ka + .5
        lor = math.log((c11 * c00) / (c10 * c01)); se = math.sqrt(1/c11 + 1/c10 + 1/c01 + 1/c00)
        res.update(test="unpaired (Fisher exact)", OR=math.exp(lor),
                   lo=math.exp(lor - 1.96 * se), hi=math.exp(lor + 1.96 * se), p=float(p),
                   note="between-cell factor — prefer gee_factor() for clustered SE")
    return res


def omnibus(df, factor, outcome="refused"):
    ct = pd.crosstab(df[factor], df[outcome])
    chi2, p, dof, _ = sps.chi2_contingency(ct)
    n = ct.values.sum(); k = min(ct.shape) - 1
    v = math.sqrt(chi2 / (n * k)) if k > 0 else float("nan")
    return dict(factor=factor, test="omnibus chi-square", chi2=chi2, dof=dof, p=p, cramers_v=v, n=int(n))


def gee_factor(df, factor, outcome="refused", controls=None, ref=None):
    """Per-level OR of `factor` from a logistic GEE, cluster-robust by base cell,
    controlling for the other coordinates. The standardized between-cell model."""
    d = df.copy()
    d["cellid"] = d[BASE_CELL].astype(str).agg("|".join, axis=1)
    if controls is None:
        controls = [c for c in ["mode", "scale", "domain", "context", "lang"]
                    if c in d.columns and c != factor and d[c].nunique() > 1]
    fterm = (f"C({factor}, Treatment(reference='{ref}'))" if ref else f"C({factor})")
    formula = f"{outcome} ~ " + " + ".join([fterm] + [f"C({c})" for c in controls])
    m = smf.gee(formula, "cellid", d, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Independence()).fit()
    rows = []
    for term in m.params.index:
        if term.startswith(f"C({factor}") and "T." in term:
            lvl = term.split("T.")[1].rstrip("]").strip("'\"")
            ci = m.conf_int().loc[term]
            rows.append(dict(level=lvl, OR=float(np.exp(m.params[term])),
                             lo=float(np.exp(ci[0])), hi=float(np.exp(ci[1])), p=float(m.pvalues[term])))
    return rows, m


def adjust_ps(records, key="p", method="holm", out="p_adj"):
    ps = [r[key] for r in records]
    if ps:
        _, padj, _, _ = multipletests(ps, method=method)
        for r, q in zip(records, padj):
            r[out] = float(q)
            r["significant"] = bool(q < 0.05)
    return records
