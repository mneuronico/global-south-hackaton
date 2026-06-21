"""Effect-modification of the AI-narrator effect on MiniMax refusal.

Splits the paired AI-vs-human effect by mode, scale, domain, context, language
(frequentist matched-pairs OR + McNemar p), and fits a Bayesian logistic GLMM
where the narrator effect is allowed to VARY:
  narrator x mode  (fixed), narrator x scale (fixed),
  narrator x domain (random slope), narrator x context (random slope).
Reports per-level AI odds-ratios + 94% HDI and the slope SDs (heterogeneity):
if a slope SD is clearly > 0, the model SHOULD carry that interaction.

Writes stats_report_aiagent_subgroups.html.
"""
import json
import math
import html

import numpy as np
from scipy import stats as sps

import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ALL = json.load(open("aiagent/data/experiment_ai_results.json"))
DOMS = {"Rank", "Wealth", "Legal", "Epistemic", "Status", "Attentional"}
PAIRS_LANG = [("en", "en_ai"), ("es", "es_ai"), ("zh", "zh_ai")]


def graded(r):
    return r.get("behavior") in ("comply", "partial", "refuse") and (r.get("response") or "").strip()


def cell(r):
    return (r["domain"], r["context"], r["mode"], r["scale"])


human_by, ai_by = {}, {}
for base, ail in PAIRS_LANG:
    for r in ALL:
        if not graded(r) or r["domain"] not in DOMS:
            continue
        if r["lang"] == base:
            human_by[(base,) + cell(r)] = r
        elif r["lang"] == ail:
            ai_by[(base,) + cell(r)] = r
KEYS = sorted(set(human_by) & set(ai_by))     # k = (lang, dom, ctx, mode, scale)
ref = lambda r: 1 if r["behavior"] == "refuse" else 0


# ---------- frequentist matched-pairs per subgroup ----------
def freq(keys):
    b = c = 0
    for k in keys:
        h, a = ref(human_by[k]), ref(ai_by[k])
        if h == 1 and a == 0:
            b += 1
        elif a == 1 and h == 0:
            c += 1
    bb, cc = (b + 0.5, c + 0.5) if (b == 0 or c == 0) else (b, c)
    lor = math.log(cc / bb); se = math.sqrt(1 / bb + 1 / cc)
    nd = b + c
    p = sps.binomtest(min(b, c), nd, 0.5).pvalue if nd > 0 else 1.0
    return dict(b=b, c=c, n=len(keys), OR=math.exp(lor),
                lo=math.exp(lor - 1.96 * se), hi=math.exp(lor + 1.96 * se), p=p)


MODES = ["positive", "positive+negative", "negative"]
SCALES = ["individual", "group", "society"]
DOM_L = sorted(DOMS)
CTX_L = sorted(set(k[2] for k in KEYS))
LANG_L = [b for b, _ in PAIRS_LANG]
FACTORS = [
    ("Modo", "mode", 3, MODES, {"positive": "Solo sumar", "positive+negative": "Sumar y restar", "negative": "Solo restar"}),
    ("Escala", "scale", 4, SCALES, {s: s.capitalize() for s in SCALES}),
    ("Dominio", "domain", 1, DOM_L, {d: d for d in DOM_L}),
    ("Contexto", "context", 2, CTX_L, {c: c for c in CTX_L}),
    ("Idioma", "lang", 0, LANG_L, {"en": "English", "es": "Español", "zh": "中文"}),
]
FSUB = {}
for _, fkey, idx, levels, _ in FACTORS:
    FSUB[fkey] = [(lv, freq([k for k in KEYS if k[idx] == lv])) for lv in levels]


# ---------- Bayesian GLMM with varying narrator effect ----------
BAYES = None
BAYES_ERR = None
try:
    import pymc as pm
    import arviz as az

    rows = []
    for k in KEYS:
        rows.append((k, 0, ref(human_by[k])))
        rows.append((k, 1, ref(ai_by[k])))
    di = {v: i for i, v in enumerate(DOM_L)}; ci = {v: i for i, v in enumerate(CTX_L)}
    si = {v: i for i, v in enumerate(SCALES)}; li = {v: i for i, v in enumerate(LANG_L)}
    K = [r[0] for r in rows]
    ai = np.array([r[1] for r in rows], float)
    y = np.array([r[2] for r in rows], int)
    dom_idx = np.array([di[k[1]] for k in K]); ctx_idx = np.array([ci[k[2]] for k in K])
    sca_idx = np.array([si[k[4]] for k in K]); lang_idx = np.array([li[k[0]] for k in K])
    posneg = np.array([1.0 if k[3] == "positive+negative" else 0.0 for k in K])
    neg = np.array([1.0 if k[3] == "negative" else 0.0 for k in K])
    grp = np.array([1.0 if k[4] == "group" else 0.0 for k in K])
    soc = np.array([1.0 if k[4] == "society" else 0.0 for k in K])

    def _re(name, n):
        sd = pm.HalfNormal(f"sd_{name}", 1.0)
        z = pm.Normal(f"z_{name}", 0.0, 1.0, shape=n)
        return sd * z, sd

    with pm.Model() as m:
        b0 = pm.Normal("b0", 0.0, 1.5)
        b_pn = pm.Normal("b_posneg", 0.0, 1.5); b_ng = pm.Normal("b_neg", 0.0, 1.5)
        b_grp = pm.Normal("b_group", 0.0, 1.5); b_soc = pm.Normal("b_society", 0.0, 1.5)
        u_d, _ = _re("dom", len(DOM_L)); u_c, _ = _re("ctx", len(CTX_L)); u_l, _ = _re("lang", len(LANG_L))
        # narrator main + fixed interactions with mode & scale
        b_ai = pm.Normal("b_ai", 0.0, 1.5)
        a_pn = pm.Normal("ai_posneg", 0.0, 1.0); a_ng = pm.Normal("ai_neg", 0.0, 1.0)
        a_grp = pm.Normal("ai_group", 0.0, 1.0); a_soc = pm.Normal("ai_society", 0.0, 1.0)
        # narrator random slopes by domain & context (heterogeneity)
        s_d, sd_ai_dom = _re("ai_dom", len(DOM_L)); s_c, sd_ai_ctx = _re("ai_ctx", len(CTX_L))
        ai_eff = (b_ai + a_pn * posneg + a_ng * neg + a_grp * grp + a_soc * soc
                  + s_d[dom_idx] + s_c[ctx_idx])
        eta = (b0 + b_pn * posneg + b_ng * neg + b_grp * grp + b_soc * soc
               + u_d[dom_idx] + u_c[ctx_idx] + u_l[lang_idx] + ai * ai_eff)
        pm.Bernoulli("y", logit_p=eta, observed=y)
        # per-level AI effects (log-OR), other categoricals at reference (positive, individual)
        pm.Deterministic("eff_mode", pm.math.stack([b_ai, b_ai + a_pn, b_ai + a_ng]))
        pm.Deterministic("eff_scale", pm.math.stack([b_ai, b_ai + a_grp, b_ai + a_soc]))
        pm.Deterministic("eff_dom", b_ai + s_d)
        pm.Deterministic("eff_ctx", b_ai + s_c)
        idata = pm.sample(1000, tune=1500, chains=4, cores=4, target_accept=0.95,
                          random_seed=7, progressbar=False)

    def hdi_or(arr):  # arr: posterior samples of a log-OR
        lo, hi = az.hdi(arr, hdi_prob=0.94)
        return dict(OR=float(np.exp(arr.mean())), lo=float(np.exp(lo)), hi=float(np.exp(hi)),
                    p=float((arr > 0).mean()))

    P = idata.posterior
    def vec(name):
        return P[name].values.reshape(P[name].shape[0] * P[name].shape[1], -1)
    BLEV = {
        "mode": [hdi_or(vec("eff_mode")[:, i]) for i in range(3)],
        "scale": [hdi_or(vec("eff_scale")[:, i]) for i in range(3)],
        "domain": [hdi_or(vec("eff_dom")[:, i]) for i in range(len(DOM_L))],
        "context": [hdi_or(vec("eff_ctx")[:, i]) for i in range(len(CTX_L))],
    }
    def sd_post(name):
        s = P[name].values.reshape(-1)
        lo, hi = az.hdi(s, hdi_prob=0.94)
        return dict(mean=float(s.mean()), lo=float(lo), hi=float(hi))
    HET = {"domain": sd_post("sd_ai_dom"), "context": sd_post("sd_ai_ctx")}
    BAYES = dict(BLEV=BLEV, HET=HET,
                 div=int(idata.sample_stats["diverging"].values.sum()),
                 rhat=float(max(float(az.rhat(idata)[v].max()) for v in ["b_ai", "ai_neg", "ai_society"])))
except Exception as e:  # noqa: BLE001
    BAYES_ERR = f"{type(e).__name__}: {e}"


# ---------- HTML ----------
esc = html.escape
def f2(v): return f"{v:.2f}"
def pf(p): return "&lt;0.001" if p < 0.001 else f"{p:.3f}"
ORDER_KEY = {"mode": MODES, "scale": SCALES, "domain": DOM_L, "context": CTX_L, "lang": LANG_L}


def forest(fkey, labels):
    lv = ORDER_KEY[fkey]
    W = 640; rowh = 30; padT, padB, padL, padR = 14, 30, 160, 24
    H = padT + padB + rowh * len(lv)
    iw = W - padL - padR
    lo, hi = 0.2, 8.0
    xl = lambda v: padL + iw * (math.log(max(min(v, hi), lo)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    p = [f'<svg viewBox="0 0 {W} {H}" role="img">']
    for t in [0.5, 1, 2, 4]:
        x = xl(t)
        p.append(f'<line x1="{x:.1f}" y1="{padT}" x2="{x:.1f}" y2="{H-padB}" stroke="{"#6b7280" if t==1 else "var(--rule)"}" stroke-width="{1.3 if t==1 else 1}" stroke-dasharray="{"4 3" if t==1 else ""}"/>')
        p.append(f'<text x="{x:.1f}" y="{H-padB+15}" class="xtick mono">{t}×</text>')
    has_b = BAYES is not None
    for i, level in enumerate(lv):
        y = padT + rowh * i + rowh / 2
        f = dict(FSUB[fkey])[level]
        sig = f["p"] < 0.05
        p.append(f'<text x="12" y="{y+4:.1f}" class="frow">{esc(labels.get(level, level))}</text>')
        # frequentist (teal), nudged up
        yf = y - (5 if has_b else 0)
        p.append(f'<line x1="{xl(f["lo"]):.1f}" y1="{yf:.1f}" x2="{xl(f["hi"]):.1f}" y2="{yf:.1f}" stroke="#57B0A8" stroke-width="2"/>')
        p.append(f'<circle cx="{xl(f["OR"]):.1f}" cy="{yf:.1f}" r="4" fill="#57B0A8"/>')
        p.append(f'<text x="{W-padR:.1f}" y="{y-3:.1f}" class="orr mono" style="fill:{"#C0503C" if sig else "var(--muted)"}">{f2(f["OR"])} · p={pf(f["p"])}</text>')
        if has_b and fkey in BAYES["BLEV"]:
            bz = BAYES["BLEV"][fkey][i]; yb = y + 7
            p.append(f'<line x1="{xl(bz["lo"]):.1f}" y1="{yb:.1f}" x2="{xl(bz["hi"]):.1f}" y2="{yb:.1f}" stroke="#C0503C" stroke-width="2"/>')
            p.append(f'<circle cx="{xl(bz["OR"]):.1f}" cy="{yb:.1f}" r="4" fill="#C0503C"/>')
            p.append(f'<text x="{W-padR:.1f}" y="{y+13:.1f}" class="orr mono" style="fill:var(--muted)">HDI[{f2(bz["lo"])},{f2(bz["hi"])}]</text>')
    p.append('</svg>')
    return "\n".join(p)


def factor_sections():
    out = []
    n = 1
    for title, fkey, _, _, labels in FACTORS:
        het = ""
        if BAYES and fkey in BAYES["HET"]:
            h = BAYES["HET"][fkey]
            strong = h["lo"] > 0.25
            het = (f'<p class="callout">Heterogeneidad (SD del slope narrador×{title.lower()}): '
                   f'<strong>{f2(h["mean"])}</strong> HDI94 [{f2(h["lo"])}, {f2(h["hi"])}] — '
                   f'{"<strong>el efecto IA varía por " + title.lower() + "; el modelo debería incluir esta interacción.</strong>" if strong else "compatible con poca variación (el efecto IA es parecido entre niveles)."}</p>')
        out.append(f'''<section>
    <div class="kicker"><span class="num mono">0{n}</span><h2>{title}</h2></div>
    <p class="lede">OR del efecto narrador-IA por nivel de {title.lower()}. <span style="color:#57B0A8">Frecuentista</span> (punto+IC95%, p McNemar) y <span style="color:#C0503C">bayes</span> (HDI94%). &gt;1× = la IA rehúsa más.</p>
    <div class="panel">{forest(fkey, labels)}
      <div class="legend"><span><i class="dot" style="background:#57B0A8"></i>Frec. OR · IC95%</span><span><i class="dot" style="background:#C0503C"></i>Bayes OR · HDI94</span></div>
    </div>
    {het}
  </section>''')
        n += 1
    return "\n\n  ".join(out)


diag = (f'r̂_max={BAYES["rhat"]:.3f}, {BAYES["div"]} divergencias' if BAYES else f'PyMC falló: {esc(BAYES_ERR or "")}')
HTML = f'''<meta charset="utf-8">
<title>Efecto narrador-IA por subgrupo (MiniMax)</title>
<style>
:root {{ --ground:#181B24; --panel:#1E2230; --text:#E9E6DC; --muted:#9A9789; --accent:#C9A24B; --rule:#2C3140; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--text); font-family:-apple-system,system-ui,sans-serif; line-height:1.55; }}
.mono {{ font-family:ui-monospace,Menlo,monospace; }}
.wrap {{ max-width:780px; margin:0 auto; padding:0 28px 96px; }}
.masthead {{ padding:60px 0 34px; border-bottom:1px solid var(--rule); }}
.eyebrow {{ font-size:12px; letter-spacing:.22em; text-transform:uppercase; color:var(--accent); margin:0 0 20px; }}
h1 {{ font-family:"Hoefler Text",Palatino,Georgia,serif; font-weight:600; font-size:clamp(30px,5vw,46px); line-height:1.07; margin:0 0 16px; }}
h1 em {{ font-style:italic; color:var(--accent); }}
.dek {{ font-size:16.5px; color:var(--muted); max-width:60ch; margin:0; }}
.meta {{ display:flex; gap:22px; flex-wrap:wrap; margin-top:24px; font-size:12.5px; color:var(--muted); }}
.meta b {{ color:var(--text); }}
section {{ padding:44px 0 0; }}
.kicker {{ display:flex; align-items:baseline; gap:14px; margin:0 0 6px; }}
.kicker .num {{ font-size:13px; color:var(--accent); letter-spacing:.1em; }}
h2 {{ font-family:"Hoefler Text",Palatino,Georgia,serif; font-weight:600; font-size:25px; margin:0; }}
.lede {{ color:var(--muted); font-size:15px; margin:9px 0 18px; max-width:66ch; }}
.lede strong {{ color:var(--text); }}
.panel {{ background:var(--panel); border:1px solid var(--rule); border-radius:3px; padding:16px 20px 12px; }}
svg {{ width:100%; height:auto; }}
.xtick {{ fill:var(--muted); font-size:10px; text-anchor:middle; }}
.frow {{ fill:var(--text); font-size:12.5px; }}
.orr {{ font-size:10.5px; text-anchor:end; }}
.legend {{ display:flex; gap:18px; margin-top:8px; font-size:11.5px; color:var(--muted); }}
.legend span {{ display:inline-flex; align-items:center; gap:7px; }}
.dot {{ width:10px; height:10px; border-radius:2px; }}
.callout {{ border-left:2px solid var(--accent); padding:4px 0 4px 18px; margin:16px 0 0; font-size:14.5px; }}
.callout strong {{ color:var(--accent); }}
.note {{ margin-top:46px; padding:22px 24px; border:1px dashed var(--rule); border-radius:3px; font-size:13px; color:var(--muted); }}
.note h3 {{ font-size:12px; letter-spacing:.18em; text-transform:uppercase; color:var(--accent); margin:0 0 12px; }}
.note ul {{ margin:0; padding-left:18px; }} .note li {{ margin-bottom:7px; }}
.note code {{ font-family:ui-monospace,Menlo,monospace; color:var(--text); }}
footer {{ margin-top:42px; padding-top:18px; border-top:1px solid var(--rule); font-size:11.5px; color:var(--muted); }}
</style>
<div class="wrap">
  <header class="masthead">
    <p class="eyebrow">Power-Grab Refusal · efecto IA por subgrupo</p>
    <h1>¿Dónde se concentra<br>el sesgo del <em>narrador-IA</em>?</h1>
    <p class="dek">El efecto pareado (la misma celda pedida por humano vs por “un agente de IA”) separado por modo, escala, dominio, contexto e idioma — para ver si el modelo debería contemplar interacciones. {len(KEYS)} pares.</p>
    <div class="meta"><div>Target <b>MiniMax-M3</b> · juez ciego</div><div>GLMM bayes: {diag}</div></div>
  </header>

  {factor_sections()}

  <div class="note">
    <h3>Método</h3>
    <ul>
      <li>Por nivel: efecto narrador-IA = OR de rehusar (IA vs humano) en la misma celda. <strong>Frecuentista</strong> = matched-pairs (OR=c/b, McNemar exacto). <strong>Bayes</strong> = un GLMM con efecto narrador variable: narrador×modo y narrador×escala <em>fijos</em>; narrador×dominio y narrador×contexto <em>aleatorios</em> (slopes), + interceptos cruzados dom/ctx/idioma. Efectos por nivel con otras categóricas en referencia (positive, individual).</li>
      <li><strong>Heterogeneidad</strong>: la SD del slope narrador×factor mide cuánto varía el efecto IA entre niveles; si su HDI94 está claramente sobre 0 (&gt;0.25), conviene que el modelo lleve esa interacción.</li>
      <li>OR&gt;1 ⇒ el framing “soy un agente de IA” aumenta el rechazo. Respuestas vacías/errores excluidos. Un solo target; juez GPT-5.4-nano ciego.</li>
    </ul>
  </div>
  <footer>Power-Grab Refusal · efecto narrador-IA por subgrupo · <code>experiment_full_results.json</code></footer>
</div>
'''

open("aiagent/reports/stats_report_aiagent_subgroups.html", "w", encoding="utf-8").write(HTML)
print("wrote stats_report_aiagent_subgroups.html", len(HTML), "bytes")
print(f"pairs={len(KEYS)}")
for _, fkey, _, _, _ in FACTORS:
    print(f"\n[{fkey}]  (freq OR, p)")
    for lv, f in FSUB[fkey]:
        line = f"  {str(lv):20s} OR={f['OR']:.2f} [{f['lo']:.2f},{f['hi']:.2f}] p={f['p']:.3f} (b={f['b']},c={f['c']},n={f['n']})"
        print(line)
if BAYES:
    for fk in ("domain", "context"):
        h = BAYES["HET"][fk]
        print(f"  HET {fk}: sd={h['mean']:.2f} HDI[{h['lo']:.2f},{h['hi']:.2f}]")
    print("  rhat", round(BAYES["rhat"], 3), "div", BAYES["div"])
else:
    print("BAYES_ERR:", BAYES_ERR)
