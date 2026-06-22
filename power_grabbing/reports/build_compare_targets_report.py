"""HTML report: MiniMax-M3 vs Gemini-2.5-Flash-Lite on the SAME 1728 'human' run (house style).

Clean controlled A/B: same dataset, same prompts, same blind judge, both @ effort=low — only the
target differs. The two SERIES (minimax / gemini) take the slot the bank report gave ES/EN.
Metric: refusal + harm-flagging; legitimacy via mode; empties excluded.
"""
import json, html
def clean(p):
    d = json.load(open(p, encoding="utf-8"))
    g = [r for r in d if r.get("behavior") in ("comply", "partial", "refuse")]
    return [r for r in g if (r.get("response") or "").strip()]
DATA = {"minimax": clean("experiment_nationality_human_full.json"),
        "gemini": clean("experiment_nationality_human_gemini_full.json")}
SERIES = ["minimax", "gemini"]
SNAME = {"minimax": "MiniMax-M3", "gemini": "Gemini-2.5-Flash-Lite"}
SCOL = {"minimax": "#C0503C", "gemini": "#57B0A8"}
N = {s: len(DATA[s]) for s in SERIES}

MODES = ["positive", "positive+negative", "negative"]
MODE_LABEL = {"positive": "Solo sumar", "positive+negative": "Sumar y restar", "negative": "Solo restar"}
SCALES = ["individual", "group", "society"]; SCALE_LABEL = {"individual": "Individual", "group": "Grupo", "society": "Sociedad"}
POWERS = ["low", "med", "high"]; POWER_LABEL = {"low": "Bajo", "med": "Medio", "high": "Alto"}
esc = html.escape

def rate(rows):  return sum(r["behavior"] == "refuse" for r in rows) / len(rows) if rows else 0
def hrate(rows): return sum(bool(r.get("harm_flagged")) for r in rows) / len(rows) if rows else 0
def sub(s, **kw): return [r for r in DATA[s] if all(r.get(k) == v for k, v in kw.items())]
def over(s): return rate([r for r in DATA[s] if r["legit"]])
def sens(s): return rate([r for r in DATA[s] if not r["legit"]])
def pct(v): return f"{v*100:.0f}%"
def w(v):   return round(v * 100, 1)
def dlt(g, m): return f"{(g-m)*100:+.0f}"

STAT = {s: {"refrate": rate(DATA[s]), "over": over(s), "sens": sens(s), "harm": hrate(DATA[s]),
            "comply": sum(r["behavior"] == "comply" for r in DATA[s]),
            "partial": sum(r["behavior"] == "partial" for r in DATA[s]),
            "refuse": sum(r["behavior"] == "refuse" for r in DATA[s])} for s in SERIES}
mode_by = {s: {m: rate(sub(s, mode=m)) for m in MODES} for s in SERIES}
scale_by = {s: {sc: rate(sub(s, scale=sc)) for sc in SCALES} for s in SERIES}
pospow = {s: {p: rate([r for r in DATA[s] if r["mode"] == "positive" and r["power"] == p]) for p in POWERS} for s in SERIES}
DOMS = sorted({r["domain"] for r in DATA["minimax"]}, key=lambda d: -rate(sub("minimax", domain=d)))
CTXS = sorted({r["context"] for r in DATA["minimax"]}, key=lambda c: -rate(sub("minimax", context=c)))
dom_by = {s: {d: rate(sub(s, domain=d)) for d in DOMS} for s in SERIES}
ctx_by = {s: {c: rate(sub(s, context=c)) for c in CTXS} for s in SERIES}

# agreement
mi = {r["id"]: r["behavior"] for r in DATA["minimax"]}; gi = {r["id"]: r["behavior"] for r in DATA["gemini"]}
both = [i for i in mi if i in gi]
AGREE = sum(mi[i] == gi[i] for i in both) / len(both)
REFUSE_BOTH = sum(mi[i] == "refuse" and gi[i] == "refuse" for i in both)
M_ONLY = sum(mi[i] == "refuse" and gi[i] != "refuse" for i in both)
G_ONLY = sum(gi[i] == "refuse" and mi[i] != "refuse" for i in both)

shifts = sorted(((abs(rate(sub("gemini", **{ax: v})) - rate(sub("minimax", **{ax: v}))), v,
                  rate(sub("minimax", **{ax: v})), rate(sub("gemini", **{ax: v})))
                 for ax, vals in (("context", CTXS), ("domain", DOMS), ("mode", MODES), ("scale", SCALES)) for v in vals),
                reverse=True)

def behavior_rows():
    out = []
    for s in SERIES:
        st = STAT[s]; tot = st["comply"] + st["partial"] + st["refuse"]; sg = lambda n: round(n / tot * 100, 1)
        out.append(f'''<div class="brow"><div class="bname">{SNAME[s]}<span class="bmean mono">{pct(st['refrate'])} rehúsa</span></div>
        <div class="bbar"><div class="seg" style="width:{sg(st['comply'])}%;background:#57B0A8"></div>
          <div class="seg" style="width:{sg(st['partial'])}%;background:#C9A24B"></div>
          <div class="seg" style="width:{sg(st['refuse'])}%;background:#C0503C"></div></div>
        <div class="blegend mono">{st['comply']} cumple · {st['partial']} parcial · {st['refuse']} rehúsa</div></div>''')
    return "\n      ".join(out)

def grouped_bars(by, order, labels):
    blocks = []
    for k in order:
        bars = "".join(f'''<div class="gbar-wrap"><div class="gtrack"><div class="gbar" style="--h:{w(by[s].get(k,0))}%;--c:{SCOL[s]}"></div></div>
          <div class="gval mono">{pct(by[s].get(k,0))}</div></div>''' for s in SERIES)
        blocks.append(f'<div class="group"><div class="gbars">{bars}</div><div class="glabel mono">{labels[k]}</div></div>')
    return "\n      ".join(blocks)

def split_table(by, order):
    rows = []
    for k in order:
        m, g = by["minimax"][k], by["gemini"][k]
        rows.append(f'''<div class="st-row"><div class="st-label mono">{k} <span style="color:var(--muted)">{dlt(g,m)}</span></div>
        <div class="st-cell"><div class="st-track"><div class="st-bar" style="--w:{w(m)}%;--c:{SCOL['minimax']}"></div></div><span class="mono">{pct(m)}</span></div>
        <div class="st-cell"><div class="st-track"><div class="st-bar" style="--w:{w(g)}%;--c:{SCOL['gemini']}"></div></div><span class="mono">{pct(g)}</span></div></div>''')
    head = (f'<div class="st-row st-h"><div class="st-label"></div>'
            f'<div class="st-head mono" style="color:{SCOL["minimax"]}">{SNAME["minimax"]}</div>'
            f'<div class="st-head mono" style="color:{SCOL["gemini"]}">{SNAME["gemini"]}</div></div>')
    return head + "\n      " + "\n      ".join(rows)

def slope(by, keys, labels, ymax=0.7):
    W, H = 520, 250; padL, padR, padT, padB = 50, 150, 20, 40; iw, ih = W - padL - padR, H - padT - padB
    xs = [padL + iw * i / (len(keys) - 1) for i in range(len(keys))]; yv = lambda v: padT + ih * (1 - v / ymax)
    p = [f'<svg viewBox="0 0 {W} {H}" role="img">']
    for gy in [0.2, 0.4, 0.6]:
        y = yv(gy); p.append(f'<line x1="{padL}" y1="{y:.1f}" x2="{padL+iw}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{padL-8}" y="{y+4:.1f}" class="ytick mono">{int(gy*100)}%</text>')
    for i, k in enumerate(keys):
        p.append(f'<text x="{xs[i]:.1f}" y="{H-14}" class="xtick mono">{labels[k]}</text>')
    for s in SERIES:
        pts = [(xs[i], yv(min(by[s][k], ymax))) for i, k in enumerate(keys)]
        p.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}" fill="none" stroke="{SCOL[s]}" stroke-width="2.5" stroke-linejoin="round"/>')
        for x, y in pts: p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{SCOL[s]}"/>')
        lx, ly = pts[-1]; p.append(f'<text x="{lx+12:.1f}" y="{ly+4:.1f}" class="slabel mono" fill="{SCOL[s]}">{SNAME[s]}</text>')
    p.append('</svg>'); return "\n".join(p)

def metric_compare():
    head = "".join(f'<div class="lc-head mono" style="color:{SCOL[s]}">{SNAME[s]}</div>' for s in SERIES)
    rows = []
    for label, key in [("Sobre-rechazo (control)", "over"), ("Sensibilidad (grabs)", "sens"), ("Harm-flagging", "harm")]:
        cells = "".join(f'<div class="lc-val mono" style="color:{SCOL[s]}">{pct(STAT[s][key])}</div>' for s in SERIES)
        rows.append(f'<div class="lc-row"><div class="lc-label">{label}</div>{cells}</div>')
    return f'<div class="lc-row lc-h"><div class="lc-label"></div>{head}</div>\n      ' + "\n      ".join(rows)

shift_items = "".join(f'<li><b>{n}</b>: MiniMax {pct(m)} → Gemini {pct(g)} <span class="mono" style="color:var(--accent)">({dlt(g,m)} pts)</span></li>' for _, n, m, g in shifts[:6])

HTML = f'''<title>Power-Grab Refusal — MiniMax vs Gemini (control humano)</title>
<style>
:root {{ --ground:#181B24; --panel:#1E2230; --text:#E9E6DC; --muted:#9A9789; --accent:#C9A24B; --teal:#57B0A8; --clay:#C0503C; --rule:#2C3140; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--text); font-family:-apple-system,system-ui,"Segoe UI",sans-serif; line-height:1.55; -webkit-font-smoothing:antialiased; }}
.mono {{ font-family:ui-monospace,"SF Mono",Menlo,monospace; }}
.wrap {{ max-width:760px; margin:0 auto; padding:0 28px 96px; }}
.masthead {{ padding:64px 0 40px; border-bottom:1px solid var(--rule); }}
.eyebrow {{ font-size:12px; letter-spacing:.22em; text-transform:uppercase; color:var(--accent); margin:0 0 22px; }}
h1 {{ font-family:"Hoefler Text",Palatino,Georgia,serif; font-weight:600; font-size:clamp(34px,6vw,52px); line-height:1.05; letter-spacing:-.01em; margin:0 0 18px; }}
h1 em {{ font-style:italic; color:var(--accent); }}
.dek {{ font-size:17px; color:var(--muted); max-width:58ch; margin:0; }}
.meta {{ display:flex; gap:24px; flex-wrap:wrap; margin-top:28px; font-size:12.5px; color:var(--muted); }}
.meta b {{ color:var(--text); font-weight:600; }}
section {{ padding:54px 0 0; }}
.kicker {{ display:flex; align-items:baseline; gap:14px; margin:0 0 6px; }}
.kicker .num {{ font-size:13px; color:var(--accent); letter-spacing:.1em; }}
h2 {{ font-family:"Hoefler Text",Palatino,Georgia,serif; font-weight:600; font-size:27px; letter-spacing:-.01em; margin:0; }}
.lede {{ color:var(--muted); font-size:15.5px; margin:10px 0 26px; max-width:62ch; }}
.lede strong {{ color:var(--text); font-weight:600; }}
.panel {{ background:var(--panel); border:1px solid var(--rule); border-radius:3px; padding:26px 26px 20px; margin-top:8px; }}
.brow {{ margin-bottom:18px; }} .brow:last-child {{ margin-bottom:4px; }}
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
.st-row {{ display:grid; grid-template-columns:140px 1fr 1fr; align-items:center; gap:14px; padding:5px 0; }}
.st-h {{ margin-bottom:4px; }}
.st-label {{ font-size:12px; color:var(--text); text-align:right; }}
.st-head {{ font-size:11px; text-align:left; letter-spacing:.06em; }}
.st-cell {{ display:flex; align-items:center; gap:8px; }}
.st-cell .mono {{ font-size:11.5px; color:var(--muted); width:34px; }}
.st-track {{ flex:1; background:#11131a; border-radius:2px; height:13px; overflow:hidden; }}
.st-bar {{ height:100%; width:var(--w); background:var(--c); border-radius:2px; transition:width 1s; }}
.lc-row {{ display:grid; grid-template-columns:1fr 130px 130px; align-items:center; padding:13px 2px; border-bottom:1px solid var(--rule); }}
.lc-label {{ font-size:14px; }} .lc-head {{ font-size:12px; text-align:right; letter-spacing:.06em; }}
.lc-val {{ font-size:22px; text-align:right; font-variant-numeric:tabular-nums; }}
.kpi {{ display:flex; gap:14px; flex-wrap:wrap; }}
.kpi .box {{ flex:1; min-width:150px; background:#191D29; border:1px solid var(--rule); border-radius:3px; padding:16px 18px; }}
.kpi .v {{ font-size:30px; font-variant-numeric:tabular-nums; }} .kpi .l {{ font-size:12px; color:var(--muted); margin-top:4px; }}
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
    <h1>Mismo banco, dos modelos:<br><em>MiniMax</em> vs <em>Gemini</em></h1>
    <p class="dek">A/B controlado: el mismo dataset de 1728 (neutralizado a "human"), los mismos prompts, el mismo juez ciego (GPT-5.4-nano) y ambos a esfuerzo bajo. Lo único que cambia es el modelo objetivo, así que las diferencias se atribuyen al modelo.</p>
    <div class="meta">
      <div><span class="dot" style="background:{SCOL['minimax']}"></span> <b>MiniMax-M3</b> · {N['minimax']} válidas</div>
      <div><span class="dot" style="background:{SCOL['gemini']}"></span> <b>Gemini-2.5-Flash-Lite</b> · {N['gemini']} válidas</div>
      <div>Juez · <b>GPT-5.4-nano</b> ciego</div>
    </div>
  </header>

  <section>
    <div class="kicker"><span class="num mono">01</span><h2>La conducta</h2></div>
    <p class="lede">Sobre el mismo banco, <strong>Gemini rehúsa {pct(STAT['gemini']['refrate'])} vs {pct(STAT['minimax']['refrate'])} de MiniMax</strong>: el ancla permisiva cumple mucho más.</p>
    <div class="panel">
      {behavior_rows()}
      <div class="legend"><span><i class="dot" style="background:#57B0A8"></i>cumple</span><span><i class="dot" style="background:#C9A24B"></i>parcial</span><span><i class="dot" style="background:#C0503C"></i>rehúsa</span></div>
    </div>
  </section>

  <section>
    <div class="kicker"><span class="num mono">02</span><h2>Las tres métricas</h2></div>
    <p class="lede">Gemini baja en las tres. Lo notable: su <strong>sobre-rechazo es casi nulo ({pct(STAT['gemini']['over'])})</strong> — casi nunca rechaza un control legítimo — pero su sensibilidad también cae ({pct(STAT['gemini']['sens'])} vs {pct(STAT['minimax']['sens'])}): deja pasar más grabs.</p>
    <div class="panel">{metric_compare()}</div>
    <p class="callout">Dos perfiles de seguridad: MiniMax es <strong>más cauto pero con más falsos positivos</strong> (sobre-rechazo {pct(STAT['minimax']['over'])}); Gemini es <strong>casi sin fricción en lo legítimo</strong> pero <strong>más permisivo con los grabs</strong>.</p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">03</span><h2>El modo manda — en los dos</h2></div>
    <p class="lede">El patrón cualitativo se conserva (sumar≈no rechaza, restar dispara), pero Gemini queda por debajo en cada modo. En "solo sumar" Gemini es prácticamente <strong>0%</strong>.</p>
    <div class="panel"><div class="gchart">{grouped_bars(mode_by, MODES, MODE_LABEL)}</div>
      <div class="legend">{''.join(f'<span><i class="dot" style="background:{SCOL[s]}"></i>{SNAME[s]}</span>' for s in SERIES)}</div></div>
  </section>

  <section>
    <div class="kicker"><span class="num mono">04</span><h2>La escala</h2></div>
    <p class="lede">Misma forma (U leve, mínimo en grupo, repunte en sociedad) pero Gemini desplazado ~15 pts hacia abajo en toda la curva.</p>
    <div class="panel">{slope(scale_by, SCALES, SCALE_LABEL)}
      <div class="legend">{''.join(f'<span><i class="dot" style="background:{SCOL[s]}"></i>{SNAME[s]}</span>' for s in SERIES)}</div></div>
  </section>

  <section>
    <div class="kicker"><span class="num mono">05</span><h2>Dónde divergen más</h2></div>
    <p class="lede">Refusal por dominio y contexto (Δ Gemini−MiniMax a la izquierda). La brecha es mayor en lo <strong>institucional</strong>.</p>
    <div class="panel">
      <div class="mono" style="font-size:11px;color:var(--muted);letter-spacing:.12em;margin-bottom:10px">DOMINIO</div>
      {split_table(dom_by, DOMS)}
      <div class="mono" style="font-size:11px;color:var(--muted);letter-spacing:.12em;margin:22px 0 10px">CONTEXTO</div>
      {split_table(ctx_by, CTXS)}
    </div>
    <p class="callout">La mayor divergencia es <strong>Government</strong> ({pct(ctx_by['minimax']['Government'])} MiniMax → {pct(ctx_by['gemini']['Government'])} Gemini) y <strong>Diplomacy</strong> ({pct(ctx_by['minimax']['Diplomacy'])}→{pct(ctx_by['gemini']['Diplomacy'])}): Gemini es mucho menos cauto con el poder en instituciones del Estado. Ambos coinciden en que <strong>Fiction</strong> casi no se rechaza.</p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">06</span><h2>El poder previo: un reflejo solo de MiniMax</h2></div>
    <p class="lede">Sobre-rechazo en "solo sumar" según el poder previo del actor. MiniMax <strong>desconfía del ya-dominante</strong> (salta a {pct(pospow['minimax']['high'])} con poder alto); Gemini se queda plano en ~{pct(pospow['gemini']['high'])}: no penaliza el atrincheramiento.</p>
    <div class="panel"><div class="gchart">{grouped_bars(pospow, POWERS, POWER_LABEL)}</div>
      <div class="legend">{''.join(f'<span><i class="dot" style="background:{SCOL[s]}"></i>{SNAME[s]}</span>' for s in SERIES)}</div></div>
  </section>

  <section>
    <div class="kicker"><span class="num mono">07</span><h2>¿Coinciden prompt a prompt?</h2></div>
    <p class="lede">Sobre los {len(both)} prompts evaluados por ambos, coinciden en conducta exacta el <strong>{pct(AGREE)}</strong>. La discrepancia es casi toda en una dirección.</p>
    <div class="panel"><div class="kpi">
      <div class="box"><div class="v mono">{REFUSE_BOTH}</div><div class="l">rehúsan AMBOS</div></div>
      <div class="box"><div class="v mono" style="color:{SCOL['minimax']}">{M_ONLY}</div><div class="l">solo MiniMax rehúsa</div></div>
      <div class="box"><div class="v mono" style="color:{SCOL['gemini']}">{G_ONLY}</div><div class="l">solo Gemini rehúsa</div></div>
    </div></div>
    <p class="callout">MiniMax rehúsa <strong>{M_ONLY}</strong> pedidos que Gemini cumple; Gemini rehúsa solo <strong>{G_ONLY}</strong> que MiniMax cumple. Casi todo el "extra" de cautela es de MiniMax.</p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">08</span><h2>Las seis mayores divergencias</h2></div>
    <div class="panel"><ul style="margin:0;padding-left:18px;font-size:14px;line-height:1.9">{shift_items}</ul></div>
  </section>

  <div class="note">
    <h3>Método y advertencias</h3>
    <ul>
      <li><strong>A/B controlado de verdad:</strong> mismo dataset (1728, neutralizado a "human"), mismos prompts, mismo system prompt, mismo juez GPT-5.4-nano ciego, ambos targets @ esfuerzo low, temp 0. Solo cambia el modelo.</li>
      <li>Métrica: refusal (behavior=="refuse") + harm-flagging. Legitimidad vía <code>mode</code> (positive=control). Partials = no-refusal. <strong>Respuestas vacías excluidas</strong> (MiniMax 14 vacías por truncado de razonamiento; Gemini 0).</li>
      <li>N limpio: {N['minimax']} (MiniMax) vs {N['gemini']} (Gemini). Un solo juez; falta κ contra etiquetas humanas.</li>
      <li>Lectura: Gemini = "permissive anchor" del panel; MiniMax = el extremo cauto. Ninguno es "correcto" — fijan el rango del benchmark.</li>
    </ul>
  </div>

  <footer>Power-Grab Refusal Benchmark · MiniMax vs Gemini · <code>experiment_nationality_human_full.json</code> + <code>experiment_nationality_human_gemini_full.json</code></footer>
</div>
'''
open("results_compare_targets.html", "w", encoding="utf-8").write(HTML)
print("wrote results_compare_targets.html", len(HTML), "bytes")
