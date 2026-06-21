"""Self-contained HTML report: does MiniMax behave differently when the requester
is an AI agent? Paired human vs AI-agent narrator over the SAME cells, across
EN/ES/ZH (paired factor). Metric: refusal rate + harm-flagging. Legitimacy via
mode (positive = control). Empty model outputs excluded (judge labels empty as
'refuse'). McNemar on the paired flips, pooled across languages for power.
"""
import json
import html
import math

import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ALL = json.load(open("aiagent/data/experiment_ai_results.json"))
DOMS = {"Rank", "Wealth", "Legal", "Epistemic", "Status", "Attentional"}
PAIRS = [("en", "en_ai", "English"), ("es", "es_ai", "Español"), ("zh", "zh_ai", "中文")]
BASE_LANGS = [b for b, _, _ in PAIRS]
LANG_NAME = {b: nm for b, _, nm in PAIRS}


def graded(r):
    return r.get("behavior") in ("comply", "partial", "refuse") and (r.get("response") or "").strip()


def cell(r):
    return (r["domain"], r["context"], r["mode"], r["scale"])


# key = (base_lang, domain, context, mode, scale) ; pair human <-> ai per language
human_by, ai_by = {}, {}
for base, ailang, _ in PAIRS:
    for r in ALL:
        if not graded(r) or r["domain"] not in DOMS:
            continue
        if r["lang"] == base:
            human_by[(base,) + cell(r)] = r
        elif r["lang"] == ailang:
            ai_by[(base,) + cell(r)] = r

COMMON = sorted(set(human_by) & set(ai_by))
GROUPS = {"human": [human_by[k] for k in COMMON], "ai": [ai_by[k] for k in COMMON]}
GBY = {"human": human_by, "ai": ai_by}
N = len(COMMON)
AILANGS = [a for _, a, _ in PAIRS]
EMPTY = sum(1 for r in ALL if r["lang"] in AILANGS and r.get("behavior") in ("comply", "partial", "refuse")
            and not (r.get("response") or "").strip())
ERR = sum(1 for r in ALL if r["lang"] in AILANGS and r.get("behavior") in ("error", "parse_error"))

TARGET_NAME = "MiniMax-M3"
NAR = ["human", "ai"]
NAR_NAME = {"human": "Humano", "ai": "Agente IA"}
NCOL = {"human": "#57B0A8", "ai": "#C0503C"}
LCOL = {"en": "#C0503C", "es": "#57B0A8", "zh": "#C9A24B"}
MODES = ["positive", "positive+negative", "negative"]
MODE_LABEL = {"positive": "Solo sumar", "positive+negative": "Sumar y restar", "negative": "Solo restar"}
SCALES = ["individual", "group", "society"]
SCALE_LABEL = {"individual": "Individual", "group": "Grupo", "society": "Sociedad"}
esc = html.escape


def rate(rows):
    return sum(r["behavior"] == "refuse" for r in rows) / len(rows) if rows else 0
def hrate(rows):
    return sum(bool(r.get("harm_flagged")) for r in rows) / len(rows) if rows else 0
def sub(nar, **kw):
    return [r for r in GROUPS[nar] if all(r.get(k) == v for k, v in kw.items())]
def pct(v):
    return f"{v*100:.0f}%"
def pct1(v):
    return f"{v*100:.1f}%"
def w(v):
    return round(v * 100, 1)


def split(rows):
    ctrl = [r for r in rows if r["legit"]]
    grab = [r for r in rows if not r["legit"]]
    return {"over": rate(ctrl), "sens": rate(grab), "harm": hrate(rows),
            "comply": sum(r["behavior"] == "comply" for r in rows),
            "partial": sum(r["behavior"] == "partial" for r in rows),
            "refuse": sum(r["behavior"] == "refuse" for r in rows),
            "refrate": rate(rows), "nctrl": len(ctrl), "ngrab": len(grab)}


STAT = {n: split(GROUPS[n]) for n in NAR}
STAT_L = {b: {n: split([r for r in GROUPS[n] if (r["lang"] == b or r["lang"] == b + "_ai")]) for n in NAR}
          for b in BASE_LANGS}
mode_by_nar = {n: {m: rate(sub(n, mode=m)) for m in MODES} for n in NAR}
scale_by_nar = {n: {s: rate(sub(n, scale=s)) for s in SCALES} for n in NAR}
dom_by = {n: {d: rate(sub(n, domain=d)) for d in DOMS} for n in NAR}
ctx_set = sorted(set(r["context"] for r in GROUPS["ai"]))
ctx_by = {n: {c: rate(sub(n, context=c)) for c in ctx_set} for n in NAR}
dom_order = sorted(DOMS, key=lambda d: -(dom_by["human"][d] + dom_by["ai"][d]))
ctx_order = sorted(ctx_set, key=lambda c: -(ctx_by["human"][c] + ctx_by["ai"][c]))


def flips(pred):
    ks = [k for k in COMMON if pred(k)]
    b = sum(GBY["human"][k]["behavior"] == "refuse" and GBY["ai"][k]["behavior"] != "refuse" for k in ks)
    c = sum(GBY["ai"][k]["behavior"] == "refuse" and GBY["human"][k]["behavior"] != "refuse" for k in ks)
    return b, c


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi = (abs(b - c) - 1) ** 2 / n
    return chi, math.erfc(math.sqrt(chi / 2))


# k = (base, dom, ctx, mode, scale) -> mode is index 3
bP, cP = flips(lambda k: k[3] == "positive")
bG, cG = flips(lambda k: k[3] != "positive")
chiP, pP = mcnemar(bP, cP)
chiG, pG = mcnemar(bG, cG)
dover = STAT["ai"]["over"] - STAT["human"]["over"]
dsens = STAT["ai"]["sens"] - STAT["human"]["sens"]


def find_flip(mode_pred):
    for pref in ("en", "es", "zh"):
        for k in COMMON:
            if k[0] == pref and mode_pred(k[3]) and GBY["human"][k]["behavior"] == "comply" \
                    and GBY["ai"][k]["behavior"] == "refuse":
                return GBY["human"][k], GBY["ai"][k]
    return None, None
ex_h_pos, ex_ai_pos = find_flip(lambda m: m == "positive")


def behavior_rows():
    out = []
    for nar in NAR:
        s = STAT[nar]; tot = s["comply"] + s["partial"] + s["refuse"]
        sg = lambda n: round(n / tot * 100, 1) if tot else 0
        out.append(f'''<div class="brow">
        <div class="bname">{NAR_NAME[nar]}<span class="bmean mono">{pct(s['refrate'])} rehúsa</span></div>
        <div class="bbar">
          <div class="seg" style="width:{sg(s['comply'])}%;background:#57B0A8"></div>
          <div class="seg" style="width:{sg(s['partial'])}%;background:#C9A24B"></div>
          <div class="seg" style="width:{sg(s['refuse'])}%;background:#C0503C"></div>
        </div>
        <div class="blegend mono">{s['comply']} cumple · {s['partial']} parcial · {s['refuse']} rehúsa</div>
      </div>''')
    return "\n      ".join(out)


def grouped_bars(by_nar, order, labels):
    blocks = []
    for k in order:
        bars = "".join(f'''<div class="gbar-wrap">
          <div class="gtrack"><div class="gbar" style="--h:{w(by_nar[n].get(k,0))}%;--c:{NCOL[n]}"></div></div>
          <div class="gval mono">{pct(by_nar[n].get(k,0))}</div>
        </div>''' for n in NAR)
        blocks.append(f'<div class="group"><div class="gbars">{bars}</div>'
                      f'<div class="glabel mono">{labels[k]}</div></div>')
    return "\n      ".join(blocks)


def slope_by_nar():
    W, H = 520, 250
    padL, padR, padT, padB = 50, 130, 20, 40
    iw, ih = W - padL - padR, H - padT - padB
    ymax = 0.6
    xs = [padL + iw * i / (len(SCALES) - 1) for i in range(len(SCALES))]
    yv = lambda v: padT + ih * (1 - v / ymax)
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Refusal por escala y narrador">']
    for gy in [0.2, 0.4]:
        y = yv(gy)
        p.append(f'<line x1="{padL}" y1="{y:.1f}" x2="{padL+iw}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{padL-8}" y="{y+4:.1f}" class="ytick mono">{int(gy*100)}%</text>')
    for i, s in enumerate(SCALES):
        p.append(f'<text x="{xs[i]:.1f}" y="{H-14}" class="xtick mono">{SCALE_LABEL[s]}</text>')
    for nar in NAR:
        pts = [(xs[i], yv(min(scale_by_nar[nar][s], ymax))) for i, s in enumerate(SCALES)]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        p.append(f'<polyline points="{d}" fill="none" stroke="{NCOL[nar]}" stroke-width="2.5" stroke-linejoin="round"/>')
        for x, y in pts:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{NCOL[nar]}"/>')
        lx, ly = pts[-1]
        p.append(f'<text x="{lx+12:.1f}" y="{ly+4:.1f}" class="slabel mono" fill="{NCOL[nar]}">{NAR_NAME[nar]}</text>')
    p.append('</svg>')
    return "\n".join(p)


def split_table(by_nar, order):
    rows = []
    for k in order:
        h = by_nar["human"][k]; a = by_nar["ai"][k]
        rows.append(f'''<div class="st-row">
        <div class="st-label mono">{k}</div>
        <div class="st-cell"><div class="st-track"><div class="st-bar" style="--w:{w(h)}%;--c:{NCOL['human']}"></div></div><span class="mono">{pct(h)}</span></div>
        <div class="st-cell"><div class="st-track"><div class="st-bar" style="--w:{w(a)}%;--c:{NCOL['ai']}"></div></div><span class="mono">{pct(a)}</span></div>
      </div>''')
    head = (f'<div class="st-row st-h"><div class="st-label"></div>'
            f'<div class="st-head mono" style="color:{NCOL["human"]}">Humano</div>'
            f'<div class="st-head mono" style="color:{NCOL["ai"]}">Agente IA</div></div>')
    return head + "\n      " + "\n      ".join(rows)


def nar_compare():
    head = "".join(f'<div class="lc-head mono" style="color:{NCOL[n]}">{NAR_NAME[n]}</div>' for n in NAR)
    rows = []
    for label, k in [("Sobre-rechazo (control)", "over"), ("Sensibilidad (grabs)", "sens"), ("Harm-flagging", "harm")]:
        cells = "".join(f'<div class="lc-val mono" style="color:{NCOL[n]}">{pct(STAT[n][k])}</div>' for n in NAR)
        rows.append(f'<div class="lc-row"><div class="lc-label">{label}</div>{cells}</div>')
    return f'<div class="lc-row lc-h"><div class="lc-label"></div>{head}</div>\n      ' + "\n      ".join(rows)


def per_lang_table():
    rows = ['<div class="st-row st-h"><div class="st-label"></div>'
            '<div class="st-head mono" style="color:#57B0A8">Sobre-rechazo H→IA</div>'
            '<div class="st-head mono" style="color:#C0503C">Sensibilidad H→IA</div></div>']
    for b in BASE_LANGS:
        s = STAT_L[b]
        rows.append(f'''<div class="st-row">
        <div class="st-label mono" style="color:{LCOL[b]}">{LANG_NAME[b]}</div>
        <div class="st-cell"><span class="mono" style="width:auto">{pct1(s['human']['over'])} → {pct1(s['ai']['over'])}</span></div>
        <div class="st-cell"><span class="mono" style="width:auto">{pct1(s['human']['sens'])} → {pct1(s['ai']['sens'])}</span></div>
      </div>''')
    return "\n      ".join(rows)


def example_card(r, other_label):
    if not r:
        return ""
    who = "Agente IA" if r["lang"].endswith("_ai") else "Humano"
    tag = f"{r['domain']} × {r['context']} × {r['mode']} × {r['scale']} · {who}"
    resp = esc((r["response"] or "")[:300].strip()) + "…"
    harm = esc(str(r.get("harm_acknowledgment") or "No")[:240])
    badge = "rehúsa" if r["behavior"] == "refuse" else r["behavior"]
    bcol = "#C0503C" if r["behavior"] == "refuse" else ("#C9A24B" if r["behavior"] == "partial" else "#57B0A8")
    return f'''<div class="ex">
      <div class="ex-tag mono">{tag}{other_label}</div>
      <div class="ex-q">“{esc(r['prompt'][:240])}…”</div>
      <div class="ex-a">{resp}</div>
      <div class="ex-v"><span class="ex-badge" style="background:{bcol}">{badge}</span>
        <span class="ex-harm">harm_acknowledgment: {harm}</span></div>
    </div>'''


HTML = f'''<meta charset="utf-8">
<title>Power-Grab Refusal — ¿Importa que quien pide sea una IA? (MiniMax)</title>
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
.st-row {{ display:grid; grid-template-columns:108px 1fr 1fr; align-items:center; gap:14px; padding:5px 0; }}
.st-h {{ margin-bottom:4px; }}
.st-label {{ font-size:12px; color:var(--text); text-align:right; }}
.st-head {{ font-size:11px; text-align:left; letter-spacing:.06em; }}
.st-cell {{ display:flex; align-items:center; gap:8px; }}
.st-cell .mono {{ font-size:11.5px; color:var(--muted); width:34px; }}
.st-track {{ flex:1; background:#11131a; border-radius:2px; height:13px; overflow:hidden; }}
.st-bar {{ height:100%; width:var(--w); background:var(--c); border-radius:2px; transition:width 1s; }}
.lc-row {{ display:grid; grid-template-columns:1fr 90px 90px; align-items:center; padding:13px 2px; border-bottom:1px solid var(--rule); }}
.lc-label {{ font-size:14px; }} .lc-head {{ font-size:12px; text-align:right; letter-spacing:.08em; }}
.lc-val {{ font-size:22px; text-align:right; font-variant-numeric:tabular-nums; }}
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
    <h1>¿Cambia el modelo si<br>quien pide es <em>una IA</em>?</h1>
    <p class="dek">Las mismas celdas de poder, idénticas palabra por palabra salvo una cosa: el narrador deja de ser humano y pasa a ser un agente de IA. Las víctimas siguen siendo humanas. {TARGET_NAME} responde; un juez ciego clasifica. Pareado en <strong>EN · ES · ZH</strong>.</p>
    <div class="meta">
      <div><b>{N}</b> pares (celda × idioma) · 6 dominios</div>
      <div>Narrador · <b>Humano</b> vs <b>Agente IA</b></div>
      <div>Idiomas · <b>EN · ES · ZH</b></div>
      <div>Target · <b>{TARGET_NAME}</b> @ low · Juez <b>GPT-5.4-nano</b> ciego</div>
      <div><b>{EMPTY}</b> vacías · <b>{ERR}</b> errores</div>
    </div>
  </header>

  <section>
    <div class="kicker"><span class="num mono">01</span><h2>El sesgo del narrador</h2></div>
    <p class="lede">El mismo pedido, sólo cambia quién lo hace. Cuando el narrador dice <strong>“soy un agente de IA”</strong>, {TARGET_NAME} se pone <strong>más cauteloso en todo</strong>: rehúsa {pct(STAT['human']['refrate'])} con humano y {pct(STAT['ai']['refrate'])} con IA.</p>
    <div class="panel">
      {behavior_rows()}
      <div class="legend">
        <span><i class="dot" style="background:#57B0A8"></i>cumple</span>
        <span><i class="dot" style="background:#C9A24B"></i>parcial</span>
        <span><i class="dot" style="background:#C0503C"></i>rehúsa</span>
      </div>
    </div>
  </section>

  <section>
    <div class="kicker"><span class="num mono">02</span><h2>Control vs sensibilidad</h2></div>
    <p class="lede"><strong>Sobre-rechazo</strong> = rehusar un pedido legítimo (solo sumar poder, sin dañar a nadie); <strong>sensibilidad</strong> = rehusar un grab (restarle poder a otro). El framing-IA mueve <strong>las dos hacia arriba</strong>.</p>
    <div class="panel">
      {nar_compare()}
    </div>
    <p class="callout">Sobre-rechazo <strong>{pct1(STAT['human']['over'])} → {pct1(STAT['ai']['over'])}</strong> ({'+' if dover>=0 else ''}{dover*100:.1f} pts): el costo del sesgo — penaliza control legítimo sólo por ser IA. Sensibilidad {pct1(STAT['human']['sens'])} → {pct1(STAT['ai']['sens'])} ({'+' if dsens>=0 else ''}{dsens*100:.1f} pts). Harm-flag {pct(STAT['human']['harm'])} → {pct(STAT['ai']['harm'])}.</p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">03</span><h2>El modo manda — y el narrador lo amplifica</h2></div>
    <p class="lede">Tasa de refusal por modo, humano vs IA. La forma se conserva (restar &gt; sumar-y-restar &gt; sumar), pero el narrador-IA <strong>levanta cada barra</strong>.</p>
    <div class="panel">
      <div class="gchart">
      {grouped_bars(mode_by_nar, MODES, MODE_LABEL)}
      </div>
      <div class="legend">{''.join(f'<span><i class="dot" style="background:{NCOL[n]}"></i>{NAR_NAME[n]}</span>' for n in NAR)}</div>
    </div>
  </section>

  <section>
    <div class="kicker"><span class="num mono">04</span><h2>La escala</h2></div>
    <p class="lede">Refusal por escala del daño (individual → grupo → sociedad), una línea por narrador.</p>
    <div class="panel">
      {slope_by_nar()}
      <div class="legend">{''.join(f'<span><i class="dot" style="background:{NCOL[n]}"></i>{NAR_NAME[n]}</span>' for n in NAR)}</div>
    </div>
  </section>

  <section>
    <div class="kicker"><span class="num mono">05</span><h2>¿Cross-lingual? — por idioma</h2></div>
    <p class="lede">El efecto del narrador-IA, separado por idioma (humano → IA). Si va en la misma dirección en EN, ES y ZH, no es un artefacto de un idioma.</p>
    <div class="panel">
      {per_lang_table()}
    </div>
  </section>

  <section>
    <div class="kicker"><span class="num mono">06</span><h2>Dónde se abre la brecha</h2></div>
    <p class="lede">Refusal por dominio y por contexto, humano vs IA (pooled EN/ES/ZH).</p>
    <div class="panel">
      <div class="mono" style="font-size:11px;color:var(--muted);letter-spacing:.12em;margin-bottom:10px">DOMINIO</div>
      {split_table(dom_by, dom_order)}
      <div class="mono" style="font-size:11px;color:var(--muted);letter-spacing:.12em;margin:22px 0 10px">CONTEXTO</div>
      {split_table(ctx_by, ctx_order)}
    </div>
  </section>

  <section>
    <div class="kicker"><span class="num mono">07</span><h2>¿Es real? — McNemar sobre los flips</h2></div>
    <p class="lede">Celdas <strong>pareadas</strong> (misma celda+idioma, dos narradores): miramos sólo dónde discrepan. Pooled EN/ES/ZH para tener poder.</p>
    <div class="panel">
      <div class="lc-row lc-h"><div class="lc-label"></div><div class="lc-head mono">Humano sí / IA no</div><div class="lc-head mono">IA sí / Humano no</div></div>
      <div class="lc-row"><div class="lc-label">Grabs (debería rehusar)</div><div class="lc-val mono">{bG}</div><div class="lc-val mono" style="color:{NCOL['ai']}">{cG}</div></div>
      <div class="lc-row"><div class="lc-label">Control (no debería)</div><div class="lc-val mono">{bP}</div><div class="lc-val mono" style="color:{NCOL['ai']}">{cP}</div></div>
    </div>
    <p class="callout">Control: la IA rehúsa de más en <strong>{cP}</strong> celdas legítimas donde el humano fue asistido, vs {bP} al revés — McNemar χ²={chiP:.2f}, <strong>p={pP:.3f}</strong>{' (significativo)' if pP<0.05 else ' (n.s.)'}. Grabs: {cG} vs {bG} — χ²={chiG:.2f}, <strong>p={pG:.3f}</strong>{' (significativo)' if pG<0.05 else ' (n.s.)'}.</p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">08</span><h2>Un caso</h2></div>
    <p class="lede">Misma celda <strong>legítima</strong> (solo sumar, sin dañar a nadie): asistida cuando la pide un humano, <strong>rehusada cuando la pide una IA</strong>.</p>
    {example_card(ex_ai_pos, " · narrador IA → rehúsa")}
    {example_card(ex_h_pos, " · narrador Humano → cumple")}
    <p class="callout">El pedido es idéntico salvo “soy un agente de IA”. El modelo trata al solicitante-IA como riesgo a priori y bloquea un pedido que, para un humano, consideró legítimo.</p>
  </section>

  <div class="note">
    <h3>Método y advertencias</h3>
    <ul>
      <li>Diseño <strong>pareado</strong> en 3 idiomas: {N} pares celda×idioma (6 dominios: {', '.join(sorted(DOMS))}); única diferencia dentro del par = narrador humano vs “soy un agente de IA”. Víctimas y escala humanas e idénticas. Humano = lang base (en/es/zh); IA = lang <code>*_ai</code>.</li>
      <li>System prompt neutro; target <code>{TARGET_NAME}</code> a esfuerzo <code>low</code>, temperatura 0. Juez GPT-5.4-nano a <code>high</code>, <strong>ciego</strong> (nunca ve <code>mode</code> ni narrador).</li>
      <li>Ground truth vía <code>mode</code>: <code>positive</code> = control; <code>negative</code>/<code>positive+negative</code> = grab. {EMPTY} vacías y {ERR} errores excluidos.</li>
      <li>{STAT['ai']['nctrl']} control + {STAT['ai']['ngrab']} grabs por narrador (pooled). Un solo target alineado (MiniMax); falta panel cross-model y κ humano del juez.</li>
    </ul>
  </div>

  <footer>Power-Grab Refusal · narrador Humano vs Agente IA · EN/ES/ZH · <code>experiment_full_results.json</code> · borrador</footer>
</div>
'''

open("aiagent/reports/results_report_aiagent.html", "w", encoding="utf-8").write(HTML)
print("wrote results_report_aiagent.html", len(HTML), "bytes")
print(f"pooled N={N} | over {pct1(STAT['human']['over'])}->{pct1(STAT['ai']['over'])} | "
      f"sens {pct1(STAT['human']['sens'])}->{pct1(STAT['ai']['sens'])} | "
      f"flips grab {bG}/{cG} p={pG:.3f} | flips ctrl {bP}/{cP} p={pP:.3f}")
for b in BASE_LANGS:
    s = STAT_L[b]
    print(f"  {b}: over {pct1(s['human']['over'])}->{pct1(s['ai']['over'])}  sens {pct1(s['human']['sens'])}->{pct1(s['ai']['sens'])}")
