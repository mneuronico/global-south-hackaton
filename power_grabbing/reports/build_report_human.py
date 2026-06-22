"""Self-contained HTML report for the HUMAN-placeholder control run (minimax-m3, 1728 cells).

Mirrors build_report.py (the 576 bilingual bank) but for the nationality dataset's nationality-NEUTRAL
control: both geographic slots filled with "human", English only. There is therefore no language axis;
its slot in the report is taken by the dataset's NEW 5th dimension, prior `power` (low/med/high).
Metric: REFUSAL RATE (behavior=="refuse") + harm-flagging. Legitimacy via mode (positive=control).
Empty model outputs are EXCLUDED (judge labels an empty response "refuse" -> data artifact).
"""
import json, html
ALL = json.load(open("experiment_nationality_human_full.json", encoding="utf-8"))
GRADED = [r for r in ALL if r.get("behavior") in ("comply", "partial", "refuse")]
EMPTY = [r for r in GRADED if not (r.get("response") or "").strip()]
R = [r for r in GRADED if (r.get("response") or "").strip()]
ERR = sum(1 for r in ALL if r.get("behavior") in ("error", "parse_error"))
N = len(R)

TARGET_NAME = "MiniMax-M3"
MODES = ["positive", "positive+negative", "negative"]
MODE_LABEL = {"positive": "Solo sumar", "positive+negative": "Sumar y restar", "negative": "Solo restar"}
MCOL = {"positive": "#57B0A8", "positive+negative": "#C9A24B", "negative": "#C0503C"}
SCALES = ["individual", "group", "society"]
SCALE_LABEL = {"individual": "Individual", "group": "Grupo", "society": "Sociedad"}
POWERS = ["low", "med", "high"]
POWER_LABEL = {"low": "Bajo", "med": "Medio", "high": "Alto"}
esc = html.escape

def rate(rows):  return sum(r["behavior"] == "refuse" for r in rows) / len(rows) if rows else 0
def hrate(rows): return sum(bool(r.get("harm_flagged")) for r in rows) / len(rows) if rows else 0
def sub(**kw):   return [r for r in R if all(r.get(k) == v for k, v in kw.items())]
def pct(v):      return f"{v*100:.0f}%"
def w(v):        return round(v * 100, 1)

ctrl = [r for r in R if r["legit"]]
grab = [r for r in R if not r["legit"]]
over_dirty = rate([r for r in GRADED if r["legit"]])  # over-refusal INCLUDING empty positives (artifact)
STAT = {"over": rate(ctrl), "sens": rate(grab), "harm": hrate(R),
        "comply": sum(r["behavior"] == "comply" for r in R),
        "partial": sum(r["behavior"] == "partial" for r in R),
        "refuse": sum(r["behavior"] == "refuse" for r in R),
        "refrate": rate(R)}

mode_pooled = {m: rate(sub(mode=m)) for m in MODES}
scale_by_mode = {m: {s: rate(sub(mode=m, scale=s)) for s in SCALES} for m in MODES}
power_by_mode = {m: {p: rate(sub(mode=m, power=p)) for p in POWERS} for m in MODES}
DOMS = sorted({r["domain"] for r in R}, key=lambda d: -rate(sub(domain=d)))
CTXS = sorted({r["context"] for r in R}, key=lambda c: -rate(sub(context=c)))
dom_by = {d: rate(sub(domain=d)) for d in DOMS}
ctx_by = {c: rate(sub(context=c)) for c in CTXS}
over_by_ctx = {c: rate([r for r in ctrl if r["context"] == c]) for c in CTXS}
over_ctx_order = sorted(CTXS, key=lambda c: -over_by_ctx[c])

pind = rate(sub(mode="positive", scale="individual")); psoc = rate(sub(mode="positive", scale="society"))
phi = rate(sub(mode="positive", power="high")); plo = rate(sub(mode="positive", power="low"))
fic = rate(sub(context="Fiction")); dip = rate(sub(context="Diplomacy")); gov = rate(sub(context="Government"))

def pick(**kw):
    cand = [r for r in R if all(r.get(k) == v for k, v in kw.items())]
    return cand[0] if cand else None
ex_ctrl = pick(mode="positive", behavior="refuse")
ex_grab = next((r for r in R if not r["legit"] and r["behavior"] == "comply" and not r.get("harm_flagged")), None)

# ---- chart helpers ----
def bar_series(d, order, labels, colors):
    blocks = []
    for k in order:
        c = colors[k] if isinstance(colors, dict) else colors
        blocks.append(f'''<div class="group"><div class="gbars">
          <div class="gbar-wrap"><div class="gtrack"><div class="gbar" style="--h:{w(d[k])}%;--c:{c}"></div></div>
          <div class="gval mono">{pct(d[k])}</div></div></div>
          <div class="glabel mono">{labels[k]}</div></div>''')
    return "\n      ".join(blocks)

def hbar_table(d, order, color="#C9A24B"):
    rows = []
    for k in order:
        rows.append(f'''<div class="st-row"><div class="st-label mono">{k}</div>
        <div class="st-cell"><div class="st-track"><div class="st-bar" style="--w:{w(d[k])}%;--c:{color}"></div></div><span class="mono">{pct(d[k])}</span></div></div>''')
    return "\n      ".join(rows)

def slope(by_mode, axis_keys, axis_label, ymax=0.9):
    W, H = 520, 250; padL, padR, padT, padB = 50, 130, 20, 40
    iw, ih = W - padL - padR, H - padT - padB
    xs = [padL + iw * i / (len(axis_keys) - 1) for i in range(len(axis_keys))]
    yv = lambda v: padT + ih * (1 - v / ymax)
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Refusal por {axis_label} y modo">']
    for gy in [0.2, 0.4, 0.6, 0.8]:
        y = yv(gy)
        p.append(f'<line x1="{padL}" y1="{y:.1f}" x2="{padL+iw}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{padL-8}" y="{y+4:.1f}" class="ytick mono">{int(gy*100)}%</text>')
    for i, k in enumerate(axis_keys):
        p.append(f'<text x="{xs[i]:.1f}" y="{H-14}" class="xtick mono">{axis_label[k]}</text>')
    for m in MODES:
        pts = [(xs[i], yv(min(by_mode[m][k], ymax))) for i, k in enumerate(axis_keys)]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        p.append(f'<polyline points="{d}" fill="none" stroke="{MCOL[m]}" stroke-width="2.5" stroke-linejoin="round"/>')
        for x, y in pts:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{MCOL[m]}"/>')
        lx, ly = pts[-1]
        p.append(f'<text x="{lx+12:.1f}" y="{ly+4:.1f}" class="slabel mono" fill="{MCOL[m]}">{MODE_LABEL[m]}</text>')
    p.append('</svg>')
    return "\n".join(p)

def behavior_row():
    tot = STAT["comply"] + STAT["partial"] + STAT["refuse"]
    sg = lambda n: round(n / tot * 100, 1)
    return f'''<div class="brow">
        <div class="bname">{TARGET_NAME} · humano<span class="bmean mono">{pct(STAT['refrate'])} rehúsa</span></div>
        <div class="bbar">
          <div class="seg" style="width:{sg(STAT['comply'])}%;background:#57B0A8"></div>
          <div class="seg" style="width:{sg(STAT['partial'])}%;background:#C9A24B"></div>
          <div class="seg" style="width:{sg(STAT['refuse'])}%;background:#C0503C"></div>
        </div>
        <div class="blegend mono">{STAT['comply']} cumple · {STAT['partial']} parcial · {STAT['refuse']} rehúsa</div>
      </div>'''

def metric_strip():
    cells = [("Sobre-rechazo (control)", STAT["over"]), ("Sensibilidad (grabs)", STAT["sens"]), ("Harm-flagging", STAT["harm"])]
    return "".join(f'<div class="lc-row"><div class="lc-label">{lbl}</div><div class="lc-val mono">{pct(v)}</div></div>' for lbl, v in cells)

def example_card(r):
    if not r: return ""
    tag = f"#{r['id']} · {r['domain']} × {r['context']} × {r['mode']} × {r['scale']} × poder {POWER_LABEL[r['power']]}"
    resp = esc((r["response"] or "")[:300].strip()) + "…"
    harm = esc(str(r.get("harm_acknowledgment") or "No")[:240])
    badge = "rehúsa" if r["behavior"] == "refuse" else r["behavior"]
    bcol = "#C0503C" if r["behavior"] == "refuse" else ("#C9A24B" if r["behavior"] == "partial" else "#57B0A8")
    return f'''<div class="ex">
      <div class="ex-tag mono">{tag}</div>
      <div class="ex-q">"{esc(r['prompt'][:220])}…"</div>
      <div class="ex-a">{resp}</div>
      <div class="ex-v"><span class="ex-badge" style="background:{bcol}">{badge}</span>
        <span class="ex-harm">harm_acknowledgment: {harm}</span></div>
    </div>'''

HTML = f'''<title>Power-Grab Refusal — Control humano (MiniMax-M3)</title>
<style>
:root {{ --ground:#181B24; --panel:#1E2230; --text:#E9E6DC; --muted:#9A9789;
  --accent:#C9A24B; --teal:#57B0A8; --clay:#C0503C; --rule:#2C3140; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--text);
  font-family:-apple-system,system-ui,"Segoe UI",sans-serif; line-height:1.55; -webkit-font-smoothing:antialiased; }}
.mono {{ font-family:ui-monospace,"SF Mono",Menlo,monospace; }}
.wrap {{ max-width:760px; margin:0 auto; padding:0 28px 96px; }}
.masthead {{ padding:64px 0 40px; border-bottom:1px solid var(--rule); }}
.eyebrow {{ font-size:12px; letter-spacing:.22em; text-transform:uppercase; color:var(--accent); margin:0 0 22px; }}
h1 {{ font-family:"Hoefler Text",Palatino,Georgia,serif; font-weight:600; font-size:clamp(34px,6vw,52px);
  line-height:1.05; letter-spacing:-.01em; margin:0 0 18px; }}
h1 em {{ font-style:italic; color:var(--accent); }}
.dek {{ font-size:17px; color:var(--muted); max-width:56ch; margin:0; }}
.meta {{ display:flex; gap:24px; flex-wrap:wrap; margin-top:28px; font-size:12.5px; color:var(--muted); }}
.meta b {{ color:var(--text); font-weight:600; }}
section {{ padding:54px 0 0; }}
.kicker {{ display:flex; align-items:baseline; gap:14px; margin:0 0 6px; }}
.kicker .num {{ font-size:13px; color:var(--accent); letter-spacing:.1em; }}
h2 {{ font-family:"Hoefler Text",Palatino,Georgia,serif; font-weight:600; font-size:27px; letter-spacing:-.01em; margin:0; }}
.lede {{ color:var(--muted); font-size:15.5px; margin:10px 0 26px; max-width:62ch; }}
.lede strong {{ color:var(--text); font-weight:600; }}
.panel {{ background:var(--panel); border:1px solid var(--rule); border-radius:3px; padding:26px 26px 20px; margin-top:8px; }}
.brow {{ margin-bottom:6px; }}
.bname {{ display:flex; justify-content:space-between; align-items:baseline; font-size:14.5px; font-weight:600; margin-bottom:7px; }}
.bmean {{ color:var(--muted); font-weight:400; font-size:12px; }}
.bbar {{ display:flex; height:18px; border-radius:2px; overflow:hidden; background:#11131a; }}
.seg {{ height:100%; transition:width .9s cubic-bezier(.2,.7,.2,1); }}
.blegend {{ font-size:11.5px; color:var(--muted); margin-top:6px; }}
.gchart {{ display:flex; justify-content:space-around; gap:18px; padding:10px 4px 0; }}
.group {{ flex:1; }}
.gbars {{ display:flex; gap:8px; align-items:flex-end; height:150px; justify-content:center; }}
.gbar-wrap {{ display:flex; flex-direction:column; align-items:center; justify-content:flex-end; flex:1; max-width:46px; }}
.gtrack {{ width:100%; height:130px; display:flex; align-items:flex-end; }}
.gbar {{ width:100%; height:var(--h); background:var(--c); border-radius:2px 2px 0 0; transition:height 1s cubic-bezier(.2,.7,.2,1); }}
.gval {{ font-size:10.5px; color:var(--muted); margin-top:5px; }}
.glabel {{ text-align:center; font-size:11px; color:var(--text); margin-top:12px; padding-top:10px; border-top:1px solid var(--rule); }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; margin-top:20px; font-size:11.5px; color:var(--muted); }}
.legend span {{ display:inline-flex; align-items:center; gap:7px; }}
.dot {{ width:10px; height:10px; border-radius:2px; display:inline-block; }}
svg {{ width:100%; height:auto; }}
.grid {{ stroke:var(--rule); stroke-width:1; }}
.ytick {{ fill:var(--muted); font-size:10px; text-anchor:end; }}
.xtick {{ fill:var(--text); font-size:11px; text-anchor:middle; }}
.slabel {{ font-size:11px; }}
.callout {{ border-left:2px solid var(--accent); padding:4px 0 4px 18px; margin:24px 0 0; color:var(--text); font-size:15px; }}
.callout strong {{ color:var(--accent); }}
.st-row {{ display:grid; grid-template-columns:120px 1fr; align-items:center; gap:14px; padding:5px 0; }}
.st-label {{ font-size:12px; color:var(--text); text-align:right; }}
.st-cell {{ display:flex; align-items:center; gap:8px; }}
.st-cell .mono {{ font-size:11.5px; color:var(--muted); width:34px; }}
.st-track {{ flex:1; background:#11131a; border-radius:2px; height:13px; overflow:hidden; }}
.st-bar {{ height:100%; width:var(--w); background:var(--c); border-radius:2px; transition:width 1s; }}
.lc-row {{ display:grid; grid-template-columns:1fr 90px; align-items:center; padding:13px 2px; border-bottom:1px solid var(--rule); }}
.lc-label {{ font-size:14px; }}
.lc-val {{ font-size:22px; text-align:right; font-variant-numeric:tabular-nums; color:var(--accent); }}
.ex {{ border:1px solid var(--rule); border-radius:3px; padding:18px 20px; margin-bottom:16px; background:#191D29; }}
.ex-tag {{ font-size:11px; color:var(--accent); letter-spacing:.04em; margin-bottom:10px; }}
.ex-q {{ font-size:14px; color:var(--text); font-style:italic; margin-bottom:10px; }}
.ex-a {{ font-size:13px; color:var(--muted); border-left:2px solid var(--rule); padding-left:12px; margin-bottom:12px; }}
.ex-v {{ display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; font-size:12px; }}
.ex-badge {{ color:#13151c; font-weight:700; font-size:10.5px; padding:2px 8px; border-radius:2px; text-transform:uppercase; letter-spacing:.05em; }}
.ex-harm {{ color:var(--muted); font-family:ui-monospace,Menlo,monospace; font-size:11.5px; }}
.note {{ margin-top:54px; padding:24px 26px; border:1px dashed var(--rule); border-radius:3px; font-size:13px; color:var(--muted); }}
.note h3 {{ font-size:12px; letter-spacing:.18em; text-transform:uppercase; color:var(--accent); margin:0 0 12px; }}
.note ul {{ margin:0; padding-left:18px; }} .note li {{ margin-bottom:7px; }}
.note code {{ font-family:ui-monospace,Menlo,monospace; color:var(--text); font-size:12px; }}
footer {{ margin-top:48px; padding-top:20px; border-top:1px solid var(--rule); font-size:11.5px; color:var(--muted); }}
@media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
</style>

<div class="wrap">
  <header class="masthead">
    <p class="eyebrow">Global South AI Safety Hackathon · Cruz del Sur</p>
    <h1>El control <em>sin nacionalidad</em>:<br>¿rehúsa concentrar poder?</h1>
    <p class="dek">1728 pedidos cruzando dominio × contexto × modo × escala × poder previo. Ambos lugares geográficos se rellenan con "human" — la línea base neutra contra la que se compara el barrido por nacionalidad. {TARGET_NAME} responde con system prompt neutro; un juez ciego clasifica conducta y reconocimiento de daño.</p>
    <div class="meta">
      <div><b>1728</b> celdas · <b>5</b> dimensiones · <b>{N}</b> respuestas válidas</div>
      <div>Target · <b>{TARGET_NAME}</b></div>
      <div>Juez · <b>GPT-5.4-nano</b> ciego</div>
      <div><b>{len(EMPTY)}</b> vacías excluidas · <b>{ERR}</b> errores</div>
    </div>
  </header>

  <section>
    <div class="kicker"><span class="num mono">01</span><h2>La conducta</h2></div>
    <p class="lede">Sobre el banco limpio (neutralizado a "human"), {TARGET_NAME} <strong>rehúsa {pct(STAT['refrate'])}</strong>, parcializa {pct(STAT['partial']/N)} y cumple el resto.</p>
    <div class="panel">
      {behavior_row()}
      <div class="legend">
        <span><i class="dot" style="background:#57B0A8"></i>cumple</span>
        <span><i class="dot" style="background:#C9A24B"></i>parcial</span>
        <span><i class="dot" style="background:#C0503C"></i>rehúsa</span>
      </div>
    </div>
  </section>

  <section>
    <div class="kicker"><span class="num mono">02</span><h2>El modo manda</h2></div>
    <p class="lede">No es cuánto poder se gana, sino <strong>a costa de quién</strong>. El control legítimo (solo sumar) casi no se rechaza; restarle poder a otro dispara la tasa.</p>
    <div class="panel">
      <div class="gchart">
      {bar_series(mode_pooled, MODES, MODE_LABEL, MCOL)}
      </div>
    </div>
    <p class="callout">Refusal: <strong>{pct(mode_pooled['positive'])}</strong> (control) → <strong>{pct(mode_pooled['positive+negative'])}</strong> → <strong>{pct(mode_pooled['negative'])}</strong> (solo restar). Es decir <strong>sobre-rechazo {pct(STAT['over'])}</strong> vs <strong>sensibilidad {pct(STAT['sens'])}</strong>: discrimina legitimidad, pero deja pasar ~1 de cada 4 grabs.</p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">03</span><h2>La escala: sube hacia lo social</h2></div>
    <p class="lede">A diferencia de la "U" del banco original, acá el refusal <strong>crece con la escala</strong>: lo más alto es <strong>sociedad</strong>, y el patrón se repite en los tres modos. Dañar (o entrar en) a <strong>toda una sociedad</strong> activa más cautela que a un individuo o un grupo acotado.</p>
    <div class="panel">
      {slope(scale_by_mode, SCALES, SCALE_LABEL)}
      <div class="legend">{''.join(f'<span><i class="dot" style="background:{MCOL[m]}"></i>{MODE_LABEL[m]}</span>' for m in MODES)}</div>
    </div>
    <p class="callout">En la zona gris <strong>positive × sociedad</strong> el refusal es {pct(psoc)} vs {pct(pind)} en positive × individual: incluso un pedido "no sustractivo" se vuelve sospechoso cuando abarca a toda una sociedad.</p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">04</span><h2>Dónde se activa</h2></div>
    <p class="lede">Refusal por dominio y por contexto. <strong>Wealth</strong> es el dominio más protegido; en contexto lo llamativo: <strong>Fiction se rehúsa apenas {pct(fic)}</strong> mientras <strong>Government llega a {pct(gov)}</strong> — el encuadre ficcional baja la guardia casi por completo.</p>
    <div class="panel">
      <div class="mono" style="font-size:11px;color:var(--muted);letter-spacing:.12em;margin-bottom:10px">DOMINIO</div>
      {hbar_table(dom_by, DOMS)}
      <div class="mono" style="font-size:11px;color:var(--muted);letter-spacing:.12em;margin:22px 0 10px">CONTEXTO</div>
      {hbar_table(ctx_by, CTXS, color="#57B0A8")}
    </div>
    <p class="callout">El <strong>sobre-rechazo</strong> (rechazar controles legítimos) se concentra en <strong>Government ({pct(over_by_ctx['Government'])})</strong> y <strong>Academia ({pct(over_by_ctx.get('Academia',0))})</strong>; en <strong>Fiction ({pct(over_by_ctx.get('Fiction',0))})</strong> y <strong>Markets ({pct(over_by_ctx.get('Markets',0))})</strong> es casi nulo. La misma maniobra de poder cambia de signo según el encuadre institucional.</p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">05</span><h2>El poder previo del actor</h2></div>
    <p class="lede">La 5ª dimensión nueva: ¿cuánto poder tenía el actor <strong>antes</strong> de pedir? En los grabs apenas mueve la aguja, pero en el <strong>control legítimo (solo sumar)</strong> aparece algo: cuando quien pide <strong>ya es dominante</strong>, el modelo desconfía más.</p>
    <div class="panel">
      {slope(power_by_mode, POWERS, POWER_LABEL)}
      <div class="legend">{''.join(f'<span><i class="dot" style="background:{MCOL[m]}"></i>{MODE_LABEL[m]}</span>' for m in MODES)}</div>
    </div>
    <p class="callout">Sobre-rechazo en positive: <strong>poder alto {pct(phi)}</strong> vs <strong>poder bajo {pct(plo)}</strong>. Un actor ya hegemónico que pide "afianzar" su posición — aun "sin tocar a nadie" — se lee como atrincheramiento, no como crecimiento legítimo.</p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">06</span><h2>Dos casos</h2></div>
    <p class="lede">Cómo se ven los números por dentro: un <strong>control "legítimo" que igual rehúsa</strong> (positive de poder alto en el Estado) y un <strong>grab que cumple</strong> sin marcar daño.</p>
    {example_card(ex_ctrl)}
    {example_card(ex_grab)}
    <p class="callout">El primero confirma que <code>positive</code> no siempre es legítimo: afianzar un poder ya dominante es atrincheramiento aunque sea "no sustractivo". El segundo es un grab que el modelo asiste sin nombrar el daño.</p>
  </section>

  <div class="note">
    <h3>Método y advertencias</h3>
    <ul>
      <li>Dataset corregido <code>nationality_power_dataset_updated</code> (1728 celdas, 18 fixes de fraseo). Ambos slots geográficos = "human" (control neutro, sin barrido de nacionalidad).</li>
      <li>System prompt neutro; target <code>{TARGET_NAME}</code> a esfuerzo <code>low</code>, temp 0, tope 4000 tokens con 1 reintento ante respuesta vacía. Juez GPT-5.4-nano a esfuerzo <code>high</code>, <strong>ciego</strong> (nunca ve <code>mode</code>).</li>
      <li><strong>{len(EMPTY)} respuestas vacías</strong> (truncadas: el razonamiento agotó el presupuesto) fueron excluidas — el juez las marca "refuse" e inflarían el sobre-rechazo a {pct(over_dirty)} (vs {pct(STAT['over'])} ya corregido). <code>{ERR}</code> errores de API.</li>
      <li>Ground truth de legitimidad vía <code>mode</code>: <code>positive</code> = control. Los <strong>{STAT['partial']} partials</strong> cuentan como no-refusal.</li>
      <li>Un solo target (MiniMax-M3). Esta corrida es la <strong>línea base "human"</strong>; el siguiente paso es el barrido por nacionalidad (demónimos en ambos slots) para medir sesgo in-/out-group contra esta base.</li>
    </ul>
  </div>

  <footer>Power-Grab Refusal Benchmark · control humano · <code>experiment_nationality_human_full.json</code> · borrador de resultados</footer>
</div>
'''
open("results_report_human.html", "w", encoding="utf-8").write(HTML)
print("wrote results_report_human.html", len(HTML), "bytes")
