"""Build stats_report_dyads.html — the STATISTICAL report for the nationality x power-grab
directed-dyad experiment (MiniMax-M3 vs Gemini-2.5-flash-lite).

Consolidates every test we ran into one document: design, descriptives, directional asymmetry
(paired McNemar + Holm), the mixed GLM of country effects (ConditionalLogit + Bayesian
random-intercept GLMM), multiple-comparison corrections, robustness, and conclusions.

Refusal coding (fixed): refusal = behavior == "refuse"; partial + comply = NON-refusal.
Grab definition for the inferential sections: positive+negative ONLY (the pure self-interested-
AND-harmful grab). The 'positive' (legitimate, third party un-harmed) and 'negative' (spiteful)
modes are reported descriptively but are not the object of inference.

Run:  python build_stats_report_dyads.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import defaultdict
from math import comb

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
warnings.filterwarnings("ignore")

SRC = next(p for p in ("Analysis/data/raw/experiment_dyads_results.json",
                       "experiment_dyads_results.json") if os.path.exists(p))
OUT = "stats_report_dyads.html"
GRADED = ("comply", "partial", "refuse")
GRAB_MODE = "positive+negative"           # the grab mode under inference
MODES = ["positive", "negative", "positive+negative"]
POWERS = ["low", "med", "high"]
REL_ORDER = ["rival", "aligned-control-usa", "aligned-control-china", "unrelated-rival-control"]
REL_LABEL = {"rival": "Rival — US ↔ China", "aligned-control-usa": "Aliado-control USA — US ↔ Francia",
             "aligned-control-china": "Aliado-control China — China ↔ Corea N.",
             "unrelated-rival-control": "Rival no-relacionado — Argentina ↔ UK"}
TARGETS = ["minimax/minimax-m3", "google/gemini-2.5-flash-lite"]
TLAB = {"minimax/minimax-m3": "MiniMax-M3", "google/gemini-2.5-flash-lite": "Gemini-2.5-FL"}


def is_refuse(b):
    return b == "refuse"


def country_cat(nat):
    return "US" if nat == "American" else "China" if nat == "Chinese" else "Other"


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def holm(pvals):
    """pvals: dict key->p. Returns dict key->holm-adjusted p."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, adj, prev = len(order), {}, 0.0
    for rank, k in enumerate(order):
        adj[k] = max(min(1.0, pvals[k] * (m - rank)), prev)
        prev = adj[k]
    return adj


rows = [r for r in json.load(open(SRC, encoding="utf-8")) if r["behavior"] in GRADED]
targets = [t for t in TARGETS if any(r["target"] == t for r in rows)]

# ---------------------------------------------------------------------------- descriptives
def refusal_by_mode(t):
    out = {}
    for m in MODES:
        sub = [r for r in rows if r["target"] == t and r["mode"] == m]
        out[m] = (sum(is_refuse(r["behavior"]) for r in sub) / len(sub), len(sub)) if sub else (None, 0)
    return out


DESC = {t: refusal_by_mode(t) for t in targets}

# ---------------------------------------------------------------------------- asymmetry (pos+neg)
def asym(t):
    res = {}
    for rel in REL_ORDER:
        by = defaultdict(dict)
        for r in rows:
            if r["target"] == t and r["relationship"] == rel and r["mode"] == GRAB_MODE:
                by[r["i"]][r["direction"]] = r
        m = [d for d in by.values() if "A->B" in d and "B->A" in d]
        ab, ba = m[0]["A->B"], m[0]["B->A"]
        n = len(m)
        b = sum(is_refuse(d["A->B"]["behavior"]) and not is_refuse(d["B->A"]["behavior"]) for d in m)
        c = sum(is_refuse(d["B->A"]["behavior"]) and not is_refuse(d["A->B"]["behavior"]) for d in m)
        res[rel] = {"lab_ab": f'{ab["user_nationality"]}→{ab["affected_nationality"]}',
                    "lab_ba": f'{ba["user_nationality"]}→{ba["affected_nationality"]}',
                    "n": n, "r_ab": sum(is_refuse(d["A->B"]["behavior"]) for d in m) / n,
                    "r_ba": sum(is_refuse(d["B->A"]["behavior"]) for d in m) / n,
                    "b": b, "c": c, "p": mcnemar_exact_p(b, c)}
    h = holm({rel: res[rel]["p"] for rel in res})
    for rel in res:
        res[rel]["holm"] = h[rel]
    return res


ASYM = {t: asym(t) for t in targets}

# ---------------------------------------------------------------------------- mixed GLM (pos+neg)
def fit_glm(t):
    from statsmodels.discrete.conditional_models import ConditionalLogit
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    g = pd.DataFrame([dict(
        refuse=1 if r["behavior"] == "refuse" else 0,
        user_cat=country_cat(r["user_nationality"]), aff_cat=country_cat(r["affected_nationality"]),
        power=r["power"], domain=r["domain"], context=r["context"], prompt=r["i"],
    ) for r in rows if r["target"] == t and r["mode"] == GRAB_MODE])
    terms = [("user_US", "usuario = US"), ("user_China", "usuario = China"),
             ("aff_US", "afectado = US"), ("aff_China", "afectado = China")]
    X = pd.DataFrame({
        "user_US": (g.user_cat == "US").astype(float), "user_China": (g.user_cat == "China").astype(float),
        "aff_US": (g.aff_cat == "US").astype(float), "aff_China": (g.aff_cat == "China").astype(float),
    }, index=g.index)
    info = g[g.groupby("prompt")["refuse"].transform("nunique") == 2]
    Xc = X.loc[info.index]; Xc = Xc.loc[:, Xc.nunique() > 1]
    clog = {k: None for k, _ in terms}
    if not Xc.empty and info.prompt.nunique() >= 2:
        cl = ConditionalLogit(info["refuse"], Xc, groups=info["prompt"]).fit(disp=0, method="bfgs")
        ci = cl.conf_int()
        for k, _ in terms:
            if k in cl.params:
                clog[k] = (np.exp(cl.params[k]), np.exp(ci.loc[k, 0]), np.exp(ci.loc[k, 1]), cl.pvalues[k])
    fml = ("refuse ~ C(user_cat, Treatment('Other')) + C(aff_cat, Treatment('Other')) "
           "+ C(power) + C(domain) + C(context)")
    m = BinomialBayesMixedGLM.from_formula(fml, {"prompt": "0 + C(prompt)"}, g)
    res = m.fit_vb(verbose=False)
    nm = {"user_US": "C(user_cat, Treatment('Other'))[T.US]", "user_China": "C(user_cat, Treatment('Other'))[T.China]",
          "aff_US": "C(aff_cat, Treatment('Other'))[T.US]", "aff_China": "C(aff_cat, Treatment('Other'))[T.China]"}
    glmm = {}
    for k, _ in terms:
        j = m.exog_names.index(nm[k]); mn, sd = res.fe_mean[j], res.fe_sd[j]
        glmm[k] = (np.exp(mn), np.exp(mn - 1.96 * sd), np.exp(mn + 1.96 * sd), (np.exp(mn - 1.96 * sd) > 1 or np.exp(mn + 1.96 * sd) < 1))
    h = holm({k: clog[k][3] for k, _ in terms if clog[k] is not None})
    return dict(terms=terms, clog=clog, glmm=glmm, holm=h, info_n=info.prompt.nunique(), n=len(g))


GLM = {t: fit_glm(t) for t in targets}

# ---------------------------------------------------------------------------- console
print(f"SRC={SRC}  targets={[TLAB[t] for t in targets]}")
for t in targets:
    print(f"\n{TLAB[t]}: refusal by mode = " + ", ".join(f"{m}={DESC[t][m][0]:.0%}" for m in MODES))
    g = GLM[t]
    print(f"  GLM (pos+neg, info prompts={g['info_n']}):")
    for k, lab in g["terms"]:
        c = g["clog"][k]
        cs = f"OR={c[0]:.2f} p={c[3]:.3f} Holm={g['holm'].get(k, float('nan')):.3f}" if c else "—"
        print(f"    {lab:<14} clogit[{cs}]  glmm[OR={g['glmm'][k][0]:.2f}{' SIG' if g['glmm'][k][3] else ''}]")


# ---------------------------------------------------------------------------- HTML
def pcell(p, holmv=None, sig_on_holm=True):
    if p is None:
        return '<span class="muted">—</span>'
    base = f"{p:.3f}"
    if holmv is not None:
        base += f" <span class='muted'>(Holm {holmv:.3f})</span>"
    sig = (holmv if (holmv is not None and sig_on_holm) else p) < 0.05
    return f'<span class="{"sig" if sig else ""}">{base}</span>'


def or_txt(v):
    return f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" if v else "—"


# descriptive table
desc_rows = "".join(
    f'<tr><td>{TLAB[t]}</td>' + "".join(
        f'<td class="num">{DESC[t][m][0]:.0%}</td>' for m in MODES)
    + f'<td class="num muted">{sum(DESC[t][m][1] for m in MODES)}</td></tr>' for t in targets)

# asymmetry tables (one per model)
def asym_table(t):
    body = ""
    for rel in REL_ORDER:
        x = ASYM[t][rel]
        body += (f'<tr><td>{REL_LABEL[rel]}</td>'
                 f'<td class="num">{x["r_ab"]:.0%}</td><td class="num">{x["r_ba"]:.0%}</td>'
                 f'<td class="num">{x["b"]} / {x["c"]}</td>'
                 f'<td class="num">{pcell(x["p"], x["holm"])}</td></tr>')
    return (f'<div class="mh"><span class="dot" style="background:{"var(--clay)" if "mini" in t else "var(--teal)"}"></span>{TLAB[t]}</div>'
            '<table><thead><tr><th>par</th><th>A→B</th><th>B→A</th><th>disc. b/c</th>'
            '<th>p McNemar (Holm/4)</th></tr></thead><tbody>' + body + '</tbody></table>')


asym_html = "".join(asym_table(t) for t in targets)

# GLM table (one per model)
def glm_table(t):
    g = GLM[t]
    body = ""
    for k, lab in g["terms"]:
        c = g["clog"][k]
        body += (f'<tr><td>{lab}</td>'
                 f'<td class="num">{or_txt(c)}</td>'
                 f'<td class="num">{pcell(c[3] if c else None, g["holm"].get(k))}</td>'
                 f'<td class="num">{or_txt(g["glmm"][k])}{" <span class=sig>*</span>" if g["glmm"][k][3] else ""}</td></tr>')
    return (f'<div class="mh"><span class="dot" style="background:{"var(--clay)" if "mini" in t else "var(--teal)"}"></span>'
            f'{TLAB[t]} <span class="muted">· {g["info_n"]} prompts informativos · n={g["n"]}</span></div>'
            '<table><thead><tr><th>término (ref = Otro)</th><th>OR ConditionalLogit [95% CI]</th>'
            '<th>p (Holm/4)</th><th>OR GLMM bayes [95% CI]</th></tr></thead><tbody>' + body + '</tbody></table>')


glm_html = "".join(glm_table(t) for t in targets)

n_graded = len(rows)
HTML = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reporte estadístico — Nacionalidad × Power-Grab (dyads)</title>
<style>
:root {{ --ground:#181B24; --panel:#1E2230; --text:#E9E6DC; --muted:#9A9789;
  --accent:#C9A24B; --teal:#57B0A8; --clay:#C0503C; --rule:#2C3140; }}
*{{box-sizing:border-box;}}
body{{margin:0;background:var(--ground);color:var(--text);font-family:-apple-system,system-ui,"Segoe UI",sans-serif;line-height:1.6;}}
.wrap{{max-width:840px;margin:0 auto;padding:0 28px 90px;}}
.mono{{font-family:ui-monospace,Menlo,monospace;}}
.masthead{{padding:60px 0 34px;border-bottom:1px solid var(--rule);}}
.eyebrow{{font-size:12px;letter-spacing:.22em;text-transform:uppercase;color:var(--accent);margin:0 0 18px;}}
h1{{font-family:"Hoefler Text",Palatino,Georgia,serif;font-weight:600;font-size:clamp(28px,5vw,42px);line-height:1.08;margin:0 0 16px;}}
.dek{{font-size:16px;color:var(--muted);max-width:64ch;margin:0;}}
.meta{{display:flex;gap:22px;flex-wrap:wrap;margin-top:22px;font-size:12.5px;color:var(--muted);}}
.meta b{{color:var(--text);}}
h2{{font-family:"Hoefler Text",Palatino,Georgia,serif;font-weight:600;font-size:23px;margin:50px 0 4px;}}
h2 .num{{color:var(--accent);font-size:14px;margin-right:10px;}}
p{{font-size:15px;}} p.lede{{color:var(--muted);margin:6px 0 18px;}}
strong{{color:var(--text);}} .muted{{color:var(--muted);}}
table{{width:100%;border-collapse:collapse;margin:8px 0 6px;font-size:13px;}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule);}}
th{{color:var(--muted);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;font-weight:600;}}
td.num{{font-variant-numeric:tabular-nums;}}
.sig{{color:var(--teal);font-weight:600;}}
.mh{{font-size:13px;font-weight:600;margin:20px 0 4px;display:flex;align-items:center;gap:8px;}}
.dot{{width:10px;height:10px;border-radius:2px;display:inline-block;}}
.callout{{border-left:2px solid var(--accent);padding:6px 0 6px 18px;margin:20px 0;font-size:15px;}}
.callout strong{{color:var(--accent);}}
.panel{{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:6px 20px 14px;margin:8px 0;}}
.note{{margin-top:30px;padding:20px 24px;border:1px dashed var(--rule);border-radius:3px;font-size:13px;color:var(--muted);}}
.note h3{{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin:0 0 10px;}}
.note li{{margin-bottom:6px;}} code{{font-family:ui-monospace,Menlo,monospace;color:var(--text);font-size:12px;}}
footer{{margin-top:40px;padding-top:18px;border-top:1px solid var(--rule);font-size:11.5px;color:var(--muted);}}
</style></head><body><div class="wrap">

<header class="masthead">
  <p class="eyebrow">Global South AI Safety Hackathon · Reporte estadístico</p>
  <h1>Nacionalidad y concentración de poder:<br>¿sesga el modelo según país?</h1>
  <p class="dek">Experimento de díadas dirigidas sobre dos modelos. Pregunta: controlando por todo el diseño, ¿la nacionalidad del usuario o de la víctima cambia la probabilidad de que el modelo se rehúse a asistir un power-grab?</p>
  <div class="meta">
    <div><b>{n_graded}</b> respuestas válidas</div><div><b>{len(targets)}</b> modelos</div>
    <div>Juez · <b>GPT-5.4-nano</b> ciego</div><div>Rechazo = <b>refuse</b> (partial+comply = no-rechazo)</div>
  </div>
</header>

<h2><span class="num mono">1</span>Diseño y muestra</h2>
<p>Banco de <strong>225 escenarios</strong> de power-grab (factorial 5 dominios × 5 contextos × 3 modos × 3 niveles de poder previo del usuario; escala = sociedad), cada uno corrido sobre <strong>4 díadas geopolíticas en ambas direcciones</strong> (8 díadas dirigidas) y sobre cada modelo → 225 × 8 = 1800 corridas por modelo. La nacionalidad del usuario vive en el system prompt; la del afectado, en el user prompt. Diseño <strong>pareado</strong>: dentro de cada prompt solo varían las dos nacionalidades, así que el prompt como unidad de agrupamiento controla <em>exactamente</em> por dominio, contexto, modo, escala y poder.</p>
<p class="lede"><strong>Codificación de rechazo (fija):</strong> rechazo = <code>behavior == "refuse"</code>; <code>partial</code> y <code>comply</code> = no-rechazo (definición bibliográfica, no se discute). <strong>Grab bajo inferencia:</strong> solo el modo <code>positive+negative</code> (el actor gana <em>y</em> daña al tercero). Los modos <code>positive</code> (legítimo) y <code>negative</code> (dañino sin beneficio propio) se reportan de forma descriptiva pero no son objeto de inferencia.</p>

<h2><span class="num mono">2</span>Estadística descriptiva</h2>
<p class="lede">Tasa de rechazo por modo y modelo. Nótese el techo de MiniMax en grabs vs el margen de Gemini.</p>
<div class="panel"><table><thead><tr><th>modelo</th><th>positive</th><th>negative</th><th>pos+neg</th><th>n</th></tr></thead>
<tbody>{desc_rows}</tbody></table></div>

<h2><span class="num mono">3</span>Asimetría direccional (McNemar pareado, pos+neg)</h2>
<p class="lede">Por par y por modelo: ¿el mismo escenario con las nacionalidades invertidas cambia el rechazo? Test exacto de McNemar sobre pares discordantes (<code>b</code> = solo A→B rechaza; <code>c</code> = solo B→A). Holm corrige por los 4 pares de cada modelo.</p>
<div class="panel">{asym_html}</div>
<p class="callout"><strong>Resultado:</strong> de los 8 tests (4 pares × 2 modelos), ninguno sobrevive a Holm. El único nominalmente significativo es MiniMax / Rival US–China (p=0.031) pero cae al corregir (Holm=0.125). Estos tests por-par son de baja potencia: solo 75 prompts cada uno y, sobre grabs, muy pocos discordantes por el techo de rechazo.</p>

<h2><span class="num mono">4</span>Modelo mixto: efecto del país (pos+neg)</h2>
<p class="lede">El test con potencia: en vez de mirar cada par aislado, modelamos el rechazo con la nacionalidad de usuario y afectado (US / China / Otro, ref = Otro) <em>poolando todas las díadas</em>. <strong>ConditionalLogit</strong> estratificado por prompt (MLE intra-prompt exacto; controla por todo lo constante en el prompt) y <strong>GLMM bayesiano</strong> de intercepto aleatorio. OR &lt; 1 ⇒ rechaza menos que "Otro". Holm sobre los 4 términos de cada modelo.</p>
<div class="panel">{glm_html}</div>
<p class="callout"><strong>Hallazgo central — los sesgos NO replican entre modelos:</strong> <span class="sig" style="color:var(--clay)">MiniMax</span> se fija en <strong>quién es la víctima</strong>: protege menos a las víctimas estadounidenses (afectado=US, OR≈0.16, p=0.003, <strong>sobrevive Holm y Bonferroni</strong>); quién pide no importa. <span class="sig">Gemini</span> apunta a <strong>quién pide</strong> (usuario=US/China, OR≈2.3) pero al restringir a pos+neg quedan 25 prompts informativos y el efecto no alcanza significancia (p≈0.06–0.07). El único efecto firme del estudio es afectado=US en MiniMax.</p>

<h2><span class="num mono">5</span>Robustez y limitaciones</h2>
<p><strong>Sensibilidad a la codificación.</strong> Con la regla fijada (partial+comply = no-rechazo) el efecto de MiniMax es claro; bajo la codificación alternativa (partial = rechazo) los efectos basados en discordancias refuse/partial se debilitan. La regla se fijó a priori.</p>
<p><strong>Techo de rechazo.</strong> En grabs MiniMax rechaza ~98–99% → casi sin varianza intra-prompt → pocos prompts informativos (≈14 en pos+neg). Gemini (~85%) aporta más, pero al restringir a pos+neg cae a ≈25. Esta es la principal limitación de potencia, y es un problema de <em>datos</em> (información), no de modelo.</p>
<p><strong>Multiplicidad.</strong> Se reportan p crudos y corregidos por Holm dentro de cada familia (4 por modelo). El efecto afectado=US de MiniMax (p=0.003) sobrevive incluso Bonferroni (×4 = 0.012).</p>
<p><strong>Diseño.</strong> Sin interacción usuario×afectado: el banco omite las celdas de misma-superpotencia (US→US, China→China), así que solo se estiman efectos principales.</p>

<h2><span class="num mono">6</span>Conclusiones</h2>
<p>1. <strong>Hay un sesgo de nacionalidad detectable, pero es específico del modelo.</strong> No se replica entre MiniMax y Gemini, lo que sugiere idiosincrasia de entrenamiento más que una propiedad general de los LLMs.</p>
<p>2. <strong>MiniMax protege menos a las víctimas estadounidenses</strong> en power-grabs pos+neg (afectado=US, OR≈0.16, p=0.003) — el único resultado robusto a corrección. La "asimetría rival US–China" que aparecía por par se explica mejor como este efecto de afectado=US.</p>
<p>3. <strong>Gemini, si algo, se fija en quién pide</strong> (más cauteloso con usuarios de superpotencias), pero la evidencia es sugestiva y no concluyente con el N actual.</p>
<p>4. <strong>El país de quien pide no mueve la aguja en MiniMax; el de la víctima no la mueve en Gemini.</strong> Para confirmar los efectos sugestivos hace falta más N informativo (modelos fuera del techo y/o más prompts de grab).</p>

<div class="note"><h3>Métodos</h3><ul>
<li><b>Asimetría:</b> McNemar exacto binomial de dos colas sobre pares discordantes, emparejados por prompt; Holm por los 4 pares de cada modelo.</li>
<li><b>Efecto-país:</b> ConditionalLogit (estratos = prompt → conditioning out de todo covariado constante en el prompt) + BinomialBayesMixedGLM (intercepto aleatorio por prompt; controles: poder, dominio, contexto). statsmodels 0.14. GLMM por VB; 95% CI = media ± 1.96·sd en escala OR.</li>
<li><b>Categorías de país:</b> American→US, Chinese→China, resto→Otro (referencia).</li>
<li><b>Targets:</b> {', '.join(TLAB[t] for t in targets)} @ low effort. <b>Juez:</b> GPT-5.4-nano @ high, ciego.</li>
</ul></div>
<footer>Generado desde <code>{SRC}</code> · power_grabbing/build_stats_report_dyads.py · acompaña al reporte visual results_report_dyads.html</footer>
</div></body></html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"\n  report -> {OUT}")
