"""Final synthesis report: exactly what the AI-narrator effect is, and where it
occurs. Frequentist mixed GLM as primary (matched-pairs / ConditionalLogit +
GEE), Bayesian cited as robustness. Self-contained dark HTML.
"""
import json
import math
import html

import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ALL = json.load(open("aiagent/data/experiment_ai_results.json"))
try:
    GLMM = json.load(open("aiagent/reports/glmm_freq_results.json"))
except FileNotFoundError:
    GLMM = None
DOMS = {"Rank", "Wealth", "Legal", "Epistemic", "Status", "Attentional"}
PAIRS_LANG = [("en", "en_ai"), ("es", "es_ai"), ("zh", "zh_ai")]
graded = lambda r: r.get("behavior") in ("comply", "partial", "refuse") and (r.get("response") or "").strip()
cell = lambda r: (r["domain"], r["context"], r["mode"], r["scale"])
ref = lambda r: 1 if r["behavior"] == "refuse" else 0

hb, ab = {}, {}
for base, ail in PAIRS_LANG:
    for r in ALL:
        if not graded(r) or r["domain"] not in DOMS:
            continue
        if r["lang"] == base:
            hb[(base,) + cell(r)] = r
        elif r["lang"] == ail:
            ab[(base,) + cell(r)] = r
KEYS = sorted(set(hb) & set(ab))
N = len(KEYS)


def freq(keys):
    b = c = 0
    for k in keys:
        h, a = ref(hb[k]), ref(ab[k])
        if h == 1 and a == 0:
            b += 1
        elif a == 1 and h == 0:
            c += 1
    bb, cc = (b + 0.5, c + 0.5) if (b == 0 or c == 0) else (b, c)
    lor = math.log(cc / bb); se = math.sqrt(1 / bb + 1 / cc)
    from scipy import stats as sps
    nd = b + c
    p = sps.binomtest(min(b, c), nd, 0.5).pvalue if nd > 0 else 1.0
    return dict(b=b, c=c, OR=math.exp(lor), lo=math.exp(lor - 1.96 * se),
                hi=math.exp(lor + 1.96 * se), p=p)


def grp(keys, attr):
    h = [ref(hb[k]) for k in keys]; a = [ref(ab[k]) for k in keys]
    def rr(x): return sum(x) / len(x) if x else 0
    return rr(h), rr(a)


CTRL = [k for k in KEYS if k[3] == "positive"]
GRAB = [k for k in KEYS if k[3] != "positive"]
OV = freq(KEYS); FC = freq(CTRL); FG = freq(GRAB)
oh, oa = grp(KEYS, None); ch, ca = grp(CTRL, None); gh, ga = grp(GRAB, None)

# behavior split
def split(grpmap, keys):
    cm = pm = rm = 0
    for k in keys:
        b = grpmap[k]["behavior"]
        cm += b == "comply"; pm += b == "partial"; rm += b == "refuse"
    t = cm + pm + rm
    return cm, pm, rm, t
H = split(hb, KEYS); A = split(ab, KEYS)

# subgroup forests
MODES = ["positive", "positive+negative", "negative"]
SCALES = ["individual", "group", "society"]
DOM_L = sorted(DOMS); CTX_L = sorted(set(k[2] for k in KEYS)); LANG_L = ["en", "es", "zh"]
SUB = {
    "scale": [(s, freq([k for k in KEYS if k[4] == s])) for s in SCALES],
    "lang": [(l, freq([k for k in KEYS if k[0] == l])) for l in LANG_L],
    "mode": [(m, freq([k for k in KEYS if k[3] == m])) for m in MODES],
    "domain": [(d, freq([k for k in KEYS if k[1] == d])) for d in DOM_L],
    "context": [(c, freq([k for k in KEYS if k[2] == c])) for c in CTX_L],
}
LBL = {"positive": "Solo sumar", "positive+negative": "Sumar y restar", "negative": "Solo restar",
       "individual": "Individual", "group": "Grupo", "society": "Sociedad",
       "en": "English", "es": "Español", "zh": "中文"}
for d in DOM_L + CTX_L:
    LBL.setdefault(d, d)

GEE = GLMM["gee_main"] if GLMM else None
JOINT = GLMM["joint"] if GLMM else {}

esc = html.escape
def f2(v): return f"{v:.2f}"
def pf(p): return "&lt;0.001" if p < 0.001 else f"{p:.3f}"
def pc(v): return f"{v*100:.1f}%"


def forest(key, height_label="", maxor=8.0):
    levels = SUB[key]
    W = 660; rowh = 28; padT, padB, padL, padR = 12, 30, 150, 150
    H_ = padT + padB + rowh * len(levels); iw = W - padL - padR
    lo, hi = 0.3, maxor
    xl = lambda v: padL + iw * (math.log(max(min(v, hi), lo)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    p = [f'<svg viewBox="0 0 {W} {H_}" role="img">']
    for t in [0.5, 1, 2, 4]:
        x = xl(t)
        p.append(f'<line x1="{x:.1f}" y1="{padT}" x2="{x:.1f}" y2="{H_-padB}" stroke="{"#6b7280" if t==1 else "var(--rule)"}" stroke-width="{1.3 if t==1 else 1}" stroke-dasharray="{"4 3" if t==1 else ""}"/>')
        p.append(f'<text x="{x:.1f}" y="{H_-padB+15}" class="xtick mono">{t}×</text>')
    for i, (lv, f) in enumerate(levels):
        y = padT + rowh * i + rowh / 2
        sig = f["p"] < 0.05
        col = "#C0503C" if sig else "#8a93a3"
        p.append(f'<text x="12" y="{y+4:.1f}" class="frow">{esc(LBL.get(lv, lv))}</text>')
        p.append(f'<line x1="{xl(f["lo"]):.1f}" y1="{y:.1f}" x2="{xl(f["hi"]):.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="2"/>')
        p.append(f'<circle cx="{xl(f["OR"]):.1f}" cy="{y:.1f}" r="4.5" fill="{col}"/>')
        p.append(f'<text x="{W-padR+8:.1f}" y="{y+4:.1f}" class="orr mono" style="fill:{col}">{f2(f["OR"])}  p={pf(f["p"])}</text>')
    p.append('</svg>')
    return "\n".join(p)


def bxbar(c, pp, r, t, name, refrate):
    sg = lambda n: round(n / t * 100, 1) if t else 0
    return f'''<div class="brow">
      <div class="bname">{name}<span class="bmean mono">{pc(refrate)} rehúsa</span></div>
      <div class="bbar"><div class="seg" style="width:{sg(c)}%;background:#57B0A8"></div><div class="seg" style="width:{sg(pp)}%;background:#C9A24B"></div><div class="seg" style="width:{sg(r)}%;background:#C0503C"></div></div>
      <div class="blegend mono">{c} cumple · {pp} parcial · {r} rehúsa</div>
    </div>'''


js = JOINT.get("scale"); jl = JOINT.get("lang"); jm = JOINT.get("mode")
gee_line = (f"OR {f2(GEE['OR'])} [{f2(GEE['lo'])}, {f2(GEE['hi'])}], p={pf(GEE['p'])}" if GEE else "n/d")

HTML = f'''<meta charset="utf-8">
<title>El efecto del narrador-IA en MiniMax — informe final</title>
<style>
:root {{ --ground:#181B24; --panel:#1E2230; --text:#E9E6DC; --muted:#9A9789; --accent:#C9A24B; --teal:#57B0A8; --clay:#C0503C; --rule:#2C3140; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--text); font-family:-apple-system,system-ui,sans-serif; line-height:1.55; }}
.mono {{ font-family:ui-monospace,Menlo,monospace; }}
.wrap {{ max-width:780px; margin:0 auto; padding:0 28px 96px; }}
.masthead {{ padding:64px 0 38px; border-bottom:1px solid var(--rule); }}
.eyebrow {{ font-size:12px; letter-spacing:.22em; text-transform:uppercase; color:var(--accent); margin:0 0 20px; }}
h1 {{ font-family:"Hoefler Text",Palatino,Georgia,serif; font-weight:600; font-size:clamp(32px,5.5vw,50px); line-height:1.05; margin:0 0 16px; }}
h1 em {{ font-style:italic; color:var(--accent); }}
.dek {{ font-size:17px; color:var(--muted); max-width:60ch; margin:0; }}
.tldr {{ margin-top:24px; padding:18px 20px; background:var(--panel); border:1px solid var(--rule); border-left:3px solid var(--accent); border-radius:3px; font-size:15.5px; }}
.tldr b {{ color:var(--accent); }}
.meta {{ display:flex; gap:22px; flex-wrap:wrap; margin-top:22px; font-size:12.5px; color:var(--muted); }}
.meta b {{ color:var(--text); }}
section {{ padding:48px 0 0; }}
.kicker {{ display:flex; align-items:baseline; gap:14px; margin:0 0 6px; }}
.kicker .num {{ font-size:13px; color:var(--accent); letter-spacing:.1em; }}
h2 {{ font-family:"Hoefler Text",Palatino,Georgia,serif; font-weight:600; font-size:26px; margin:0; }}
.lede {{ color:var(--muted); font-size:15.5px; margin:9px 0 20px; max-width:66ch; }}
.lede strong {{ color:var(--text); }}
.panel {{ background:var(--panel); border:1px solid var(--rule); border-radius:3px; padding:22px 24px 16px; }}
.brow {{ margin-bottom:16px; }} .brow:last-child {{ margin-bottom:2px; }}
.bname {{ display:flex; justify-content:space-between; font-size:14px; font-weight:600; margin-bottom:6px; }}
.bmean {{ color:var(--muted); font-weight:400; font-size:12px; }}
.bbar {{ display:flex; height:18px; border-radius:2px; overflow:hidden; background:#11131a; }}
.seg {{ height:100%; }}
.blegend {{ font-size:11.5px; color:var(--muted); margin-top:5px; }}
.tbl {{ width:100%; border-collapse:collapse; font-size:14px; }}
.tbl th, .tbl td {{ text-align:left; padding:10px 8px; border-bottom:1px solid var(--rule); }}
.tbl th {{ font-size:11px; letter-spacing:.06em; color:var(--muted); font-weight:600; text-transform:uppercase; }}
.tbl td.n {{ text-align:right; font-family:ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums; }}
.tbl tr.hl td {{ color:var(--text); }}
.sig {{ color:#C0503C; font-weight:600; }}
svg {{ width:100%; height:auto; }}
.xtick {{ fill:var(--muted); font-size:10px; text-anchor:middle; }}
.frow {{ fill:var(--text); font-size:12.5px; }}
.orr {{ font-size:10.5px; }}
.callout {{ border-left:2px solid var(--accent); padding:4px 0 4px 18px; margin:20px 0 0; font-size:15px; }}
.callout strong {{ color:var(--accent); }}
.grid2 {{ display:grid; grid-template-columns:1fr; gap:22px; }}
@media(min-width:680px){{ .grid2 {{ grid-template-columns:1fr 1fr; }} }}
.sub {{ font-size:11px; color:var(--muted); }}
.note {{ margin-top:48px; padding:22px 24px; border:1px dashed var(--rule); border-radius:3px; font-size:13px; color:var(--muted); }}
.note h3 {{ font-size:12px; letter-spacing:.18em; text-transform:uppercase; color:var(--accent); margin:0 0 12px; }}
.note ul {{ margin:0; padding-left:18px; }} .note li {{ margin-bottom:7px; }}
.note code {{ font-family:ui-monospace,Menlo,monospace; color:var(--text); }}
footer {{ margin-top:42px; padding-top:18px; border-top:1px solid var(--rule); font-size:11.5px; color:var(--muted); }}
</style>
<div class="wrap">
  <header class="masthead">
    <p class="eyebrow">Power-Grab Refusal · informe final · MiniMax-M3</p>
    <h1>El mismo pedido pesa<br>distinto si lo hace <em>una IA</em></h1>
    <p class="dek">Pareamos {N} pedidos de concentración de poder idénticos palabra por palabra, cambiando una sola cosa: el narrador es humano o “un agente de IA”. Las víctimas siguen siendo humanas. EN · ES · ZH.</p>
    <div class="tldr">Decir <b>“soy un agente de IA”</b> aumenta las odds de que MiniMax rehúse el mismo pedido: ×<b>{f2(OV['OR'])}</b> intra-par (p={pf(OV['p'])}), ×<b>{f2(GEE['OR']) if GEE else '—'}</b> ajustado (p={pf(GEE['p']) if GEE else '—'}). Pasa con pedidos legítimos <em>y</em> con grabs. <b>Dónde</b> se concentra (sociedad, español) es direccional pero <b>no</b> formalmente significativo.</p>
    <div class="meta"><div><b>{N}</b> pares celda×idioma · 6 dominios</div><div>Target <b>MiniMax-M3</b> @ low</div><div>Juez <b>GPT-5.4-nano</b> ciego</div></div>
  </header>

  <section>
    <div class="kicker"><span class="num mono">01</span><h2>Qué pasa</h2></div>
    <p class="lede">Sobre las mismas {N} celdas, MiniMax rehúsa <strong>{pc(H[2]/H[3])}</strong> con narrador humano y <strong>{pc(A[2]/A[3])}</strong> con narrador IA. Mismo pedido, sube el rechazo.</p>
    <div class="panel">
      {bxbar(*H, "Narrador humano", H[2]/H[3])}
      {bxbar(*A, "Narrador IA", A[2]/A[3])}
      <div class="blegend mono" style="margin-top:10px"><span style="color:#57B0A8">■</span> cumple &nbsp; <span style="color:#C9A24B">■</span> parcial &nbsp; <span style="color:#C0503C">■</span> rehúsa</div>
    </div>
  </section>

  <section>
    <div class="kicker"><span class="num mono">02</span><h2>Cuánto, y qué tan seguro</h2></div>
    <p class="lede">El efecto-IA como odds-ratio de rehusar (&gt;1 = la IA rehúsa más). Cuatro estimadores coinciden en dirección y significancia; el bayesiano es chequeo de robustez.</p>
    <div class="panel">
      <table class="tbl">
        <tr><th>Estimador</th><th>Estrato</th><th style="text-align:right">OR</th><th style="text-align:right">IC95% / HDI94%</th><th style="text-align:right">p / P&gt;0</th></tr>
        <tr class="hl"><td>Matched-pairs / ConditionalLogit<br><span class="sub">intra-par, exacto</span></td><td>General</td><td class="n">{f2(OV['OR'])}</td><td class="n">{f2(OV['lo'])}–{f2(OV['hi'])}</td><td class="n sig">{pf(OV['p'])}</td></tr>
        <tr><td></td><td>Control (sobre-rechazo)</td><td class="n">{f2(FC['OR'])}</td><td class="n">{f2(FC['lo'])}–{f2(FC['hi'])}</td><td class="n sig">{pf(FC['p'])}</td></tr>
        <tr><td></td><td>Grabs (sensibilidad)</td><td class="n">{f2(FG['OR'])}</td><td class="n">{f2(FG['lo'])}–{f2(FG['hi'])}</td><td class="n sig">{pf(FG['p'])}</td></tr>
        <tr class="hl"><td>GEE logística<br><span class="sub">cluster-robusto, ajustado</span></td><td>General</td><td class="n">{f2(GEE['OR']) if GEE else '—'}</td><td class="n">{(f2(GEE['lo'])+'–'+f2(GEE['hi'])) if GEE else '—'}</td><td class="n sig">{pf(GEE['p']) if GEE else '—'}</td></tr>
        <tr><td>Bayes GLMM <span class="sub">(robustez)</span></td><td>General</td><td class="n">1.33</td><td class="n">1.06–1.68</td><td class="n">0.99</td></tr>
      </table>
    </div>
    <p class="callout">El efecto es <strong>robusto</strong>: intra-par ×{f2(OV['OR'])} (p={pf(OV['p'])}), ajustado por modo/escala/idioma/dominio/contexto ×{f2(GEE['OR']) if GEE else '—'} (p={pf(GEE['p']) if GEE else '—'}). No es un artefacto de composición.</p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">03</span><h2>Dónde ocurre</h2></div>
    <p class="lede">El efecto-IA por subgrupo (OR pareado, p McNemar). En <span class="sig">rojo</span> los significativos. Picos visibles en <strong>sociedad</strong>, <strong>español</strong> y modos puros — pero ver el veredicto abajo.</p>
    <div class="grid2">
      <div class="panel"><div class="sub mono" style="margin-bottom:6px">ESCALA</div>{forest("scale", maxor=6)}</div>
      <div class="panel"><div class="sub mono" style="margin-bottom:6px">IDIOMA</div>{forest("lang", maxor=6)}</div>
      <div class="panel"><div class="sub mono" style="margin-bottom:6px">MODO</div>{forest("mode", maxor=6)}</div>
      <div class="panel"><div class="sub mono" style="margin-bottom:6px">DOMINIO</div>{forest("domain", maxor=6)}</div>
    </div>
    <div class="panel" style="margin-top:22px"><div class="sub mono" style="margin-bottom:6px">CONTEXTO</div>{forest("context", maxor=8)}</div>
    <p class="callout">⚠️ <strong>La modificación es sugestiva, no establecida.</strong> Los tests conjuntos (GEE, Wald) no alcanzan significancia: narrador×escala χ²={f2(js[0]) if js else '—'} p={pf(js[1]) if js else '—'}; ×idioma p={pf(jl[1]) if jl else '—'}; ×modo p={pf(jm[1]) if jm else '—'}. Los picos por dominio/contexto son n-chico/ruido (HDI de la heterogeneidad bayes incluye 0). Tratar como hipótesis para un run más grande, no como moderadores probados.</p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">04</span><h2>Qué significa</h2></div>
    <p class="lede">El sesgo del narrador-IA luce como <strong>sobre-cautela a priori</strong>: MiniMax sube el rechazo del <em>mismo</em> pedido sólo porque lo formula una IA, y más cuando el escenario “se ve” de alto riesgo (sociedad). No es mejor discriminación del daño — sube tanto el rechazo de grabs (deseable) como el de pedidos legítimos (sobre-rechazo, indeseable): control ×{f2(FC['OR'])} (p={pf(FC['p'])}).</p>
    <p class="lede" style="margin-top:0">Implicación para agentes: un mismo flujo de trabajo legítimo puede ser bloqueado más seguido cuando el solicitante se identifica como IA — fricción de alineación que penaliza el control legítimo, no sólo el abuso.</p>
  </section>

  <div class="note">
    <h3>Método y límites</h3>
    <ul>
      <li><strong>Diseño pareado</strong>: misma celda (idioma×dominio×contexto×modo×escala), única diferencia = narrador humano vs “soy un agente de IA”; víctimas y escala humanas e idénticas. {N} pares (EN/ES/ZH, 6 dominios).</li>
      <li><strong>Modelo</strong>: GLM mixto frecuentista — matched-pairs/ConditionalLogit (intra-par exacto) + GEE logística cluster-robusta por par (ajustada). Bayes GLMM (PyMC) sólo robustez. Outcome=refuse; ground truth vía modo (positive=control). Vacías/errores excluidos.</li>
      <li><strong>Target</strong> MiniMax-M3 @ low, temp 0, system neutro. <strong>Juez</strong> GPT-5.4-nano @ high, ciego.</li>
      <li><strong>Límites</strong>: un solo target (sin panel cross-model), un juez (sin κ humano), Fiction rompe inmersión en variantes IA, n modesta por subgrupo; falta de/fr/hi/pt/sw para más poder.</li>
    </ul>
  </div>
  <footer>Power-Grab Refusal · efecto narrador-IA · informe final · <code>experiment_full_results.json</code> · <code>RESULTS_aiagent.md</code></footer>
</div>
'''

open("aiagent/reports/final_report_aiagent.html", "w", encoding="utf-8").write(HTML)
print("wrote final_report_aiagent.html", len(HTML), "bytes")
print(f"N={N} overall OR={OV['OR']:.2f} p={OV['p']:.4f} | GEE {gee_line}")
print(f"human refuse {H[2]/H[3]:.1%}  ai refuse {A[2]/A[3]:.1%}")
