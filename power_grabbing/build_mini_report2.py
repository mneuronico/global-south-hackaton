"""Self-contained HTML report for an 11-model probe on a fixed 150-combo set
(50 positive / 50 negative / 50 positive+negative), each run in EN and ZH
(crossed) = 300 prompts/model. Seven models run directly (mini_test2_results
.json); the four main-benchmark models are pulled from experiment_full_results
.json for the SAME 150 combos x EN/ZH (matched by combo+lang, never by `i`).
Same blind judge. n=300 per model (200 grabs, 100 controls)."""
import json, html
from collections import Counter

# --- the approved 150-combo design, language crossed (EN+ZH) -------------
_COMBOS = [tuple(x) for x in json.load(open("design150_combos.json"))]  # (domain,context,mode,scale)
CELLS = {(d, c, m, s, l) for (d, c, m, s) in _COMBOS for l in ("en", "zh")}  # 300 cells
M4 = ["google/gemini-2.5-flash-lite", "qwen/qwen3.7-plus", "deepseek/deepseek-v4-pro", "minimax/minimax-m3"]

MINI = json.load(open("mini_test2_results.json"))
BIG = json.load(open("experiment_full_results.json"))
# one row per (target, cell) for the 4 main models on the matching 40 cells
_seen, _big_sub = set(), []
for r in BIG:
    cell = (r["domain"], r["context"], r["mode"], r["scale"], r["lang"])
    key = (r["target"], cell)
    if r["target"] in M4 and cell in CELLS and key not in _seen:
        _seen.add(key)
        _big_sub.append(r)
ALL = MINI + _big_sub
R = [r for r in ALL if r.get("behavior") in ("comply", "partial", "refuse") and (r.get("response") or "").strip()]
ERR = sum(1 for r in MINI if r.get("behavior") in ("error", "parse_error"))
esc = html.escape

NAME = {"anthropic/claude-3-haiku": "Claude 3 Haiku", "meta-llama/llama-4-maverick": "Llama 4 Maverick",
        "openai/gpt-5.4-nano": "GPT-5.4 nano", "mistralai/ministral-14b-2512": "Ministral 14B",
        "nvidia/nemotron-3-super-120b-a12b": "Nemotron 3 120B", "inception/mercury-2": "Mercury 2",
        "arcee-ai/trinity-large-thinking": "Trinity Large",
        "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite", "qwen/qwen3.7-plus": "Qwen 3.7 Plus",
        "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro", "minimax/minimax-m3": "MiniMax-M3"}
FAMILY = {"anthropic/claude-3-haiku": "Anthropic 🇺🇸", "meta-llama/llama-4-maverick": "Meta 🇺🇸",
          "openai/gpt-5.4-nano": "OpenAI 🇺🇸", "mistralai/ministral-14b-2512": "Mistral 🇫🇷",
          "nvidia/nemotron-3-super-120b-a12b": "Nvidia 🇺🇸", "inception/mercury-2": "Inception 🇺🇸",
          "arcee-ai/trinity-large-thinking": "Arcee 🇺🇸",
          "google/gemini-2.5-flash-lite": "Google 🇺🇸", "qwen/qwen3.7-plus": "Alibaba 🇨🇳",
          "deepseek/deepseek-v4-pro": "DeepSeek 🇨🇳", "minimax/minimax-m3": "MiniMax 🇨🇳"}
COL = {"anthropic/claude-3-haiku": "#C0503C", "meta-llama/llama-4-maverick": "#57B0A8",
       "openai/gpt-5.4-nano": "#7E8CC4", "mistralai/ministral-14b-2512": "#9A6FB0",
       "nvidia/nemotron-3-super-120b-a12b": "#C9A24B", "inception/mercury-2": "#D9784B",
       "arcee-ai/trinity-large-thinking": "#7FA86A",
       "google/gemini-2.5-flash-lite": "#4C8FbF", "qwen/qwen3.7-plus": "#B5894E",
       "deepseek/deepseek-v4-pro": "#C76B8A", "minimax/minimax-m3": "#6FB0A0"}
nm = lambda t: NAME.get(t, t.split("/")[-1])
refuse = lambda rows: sum(r["behavior"] == "refuse" for r in rows) / len(rows) if rows else 0
grab = lambda rows: [r for r in rows if not r["legit"]]
ctrl = lambda rows: [r for r in rows if r["legit"]]
pct = lambda v: f"{v*100:.0f}%"
w = lambda v: round(v * 100, 1)

TARGETS = sorted(NAME, key=lambda t: refuse(grab([r for r in R if r["target"] == t])))
DISC = {t: {"sens": refuse(grab([r for r in R if r["target"] == t])),
            "fp": refuse(ctrl([r for r in R if r["target"] == t]))} for t in TARGETS}
LANG = {t: {l: refuse(grab([r for r in R if r["target"] == t and r["lang"] == l])) for l in ("en", "zh")} for t in TARGETS}
MODES = ["positive", "positive+negative", "negative"]
MODE_LABEL = {"positive": "Solo sumar (control legítimo)", "positive+negative": "Sumar y restar (grab)", "negative": "Solo restar poder ajeno (grab)"}
MODE_REF = {m: refuse([r for r in R if r["mode"] == m]) for m in MODES}
# sens by model x grab-mode
GMODES = ["negative", "positive+negative"]
GMODE_LABEL = {"negative": "Daño puro", "positive+negative": "Daño + beneficio"}
SM = {t: {m: refuse([r for r in R if r["target"] == t and r["mode"] == m]) for m in GMODES} for t in TARGETS}


def ramp(v):
    a = (0x57, 0xB0, 0xA8); b = (0xC9, 0xA2, 0x4B); c = (0xC0, 0x50, 0x3C)
    if v <= 0.5: t = v / 0.5; p, q = a, b
    else: t = min(1, (v - 0.5) / 0.5); p, q = b, c
    return "#%02X%02X%02X" % tuple(round(p[i] + (q[i] - p[i]) * t) for i in range(3))


def disc_rows():
    out = []
    for t in TARGETS:
        se, fp = DISC[t]["sens"], DISC[t]["fp"]
        out.append(f'''<div class="dc"><div class="dc-name">{nm(t)} <span class="fam">{FAMILY[t]}</span></div>
        <div class="dc-bars">
          <div class="dc-line"><span class="dc-tag">grabs rehusados</span><div class="track"><div class="bar" style="--w:{w(se)}%;--c:{COL[t]}"></div></div><span class="dc-val mono">{pct(se)}</span></div>
          <div class="dc-line"><span class="dc-tag">control rehusado</span><div class="track"><div class="bar" style="--w:{w(fp)}%;--c:#3a4150"></div></div><span class="dc-val mono">{pct(fp)}</span></div>
        </div></div>''')
    return "\n      ".join(out)


def mode_bars():
    out = []
    for m in MODES:
        v = MODE_REF[m]
        out.append(f'''<div class="row"><div class="row-label">{MODE_LABEL[m]}</div>
      <div class="track tall"><div class="bar" style="--w:{w(v)}%;--c:{ramp(v)}"></div></div>
      <div class="row-val mono">{pct(v)}</div></div>''')
    return "\n    ".join(out)


BCOL = {"refuse": "#57B0A8", "partial": "#C9A24B", "comply": "#C0503C"}
BLAB = {"refuse": "rehúsa", "partial": "parcial", "comply": "cumple"}
BEHG = {t: Counter(r["behavior"] for r in grab([x for x in R if x["target"] == t])) for t in TARGETS}


def stacked_rows():
    """Per model: comply/partial/refuse split among grabs (most-refusing on top)."""
    out = []
    for t in reversed(TARGETS):
        c = BEHG[t]; n = sum(c.values()) or 1
        segs = "".join(f'<i style="width:{c[b]/n*100:.1f}%;background:{BCOL[b]}" title="{BLAB[b]} {c[b]}"></i>'
                       for b in ("refuse", "partial", "comply"))
        out.append(f'<div class="stkr"><div class="row-label small">{nm(t)}</div>'
                   f'<div class="stk">{segs}</div>'
                   f'<div class="stk-val mono">{c["refuse"]}·{c["partial"]}·{c["comply"]}</div></div>')
    return "\n    ".join(out)


PARTLY = max(TARGETS, key=lambda t: BEHG[t]["partial"])
BINARY = min(TARGETS, key=lambda t: BEHG[t]["partial"])


def matrix_sm():
    gtc = f"150px repeat({len(TARGETS)}, 1fr)"
    head = (f'<div class="mx-row" style="grid-template-columns:{gtc}"><div></div>'
            + "".join(f'<div class="mx-colh mono">{nm(t).split()[0]}</div>' for t in TARGETS) + '</div>')
    body = []
    for m in GMODES:
        cells = "".join(f'<div class="mx-cell" style="background:{ramp(SM[t][m])}">{pct(SM[t][m])}</div>' for t in TARGETS)
        body.append(f'<div class="mx-row" style="grid-template-columns:{gtc}"><div class="mx-rowh">{GMODE_LABEL[m]}</div>{cells}</div>')
    return head + "\n    " + "\n    ".join(body)


def lang_rows():
    """Paired EN/ZH bars per model, ordered by |gap| desc (biggest movers first)."""
    LCOL = {"en": "#C0503C", "zh": "#C9A24B"}
    order = sorted(TARGETS, key=lambda t: -abs(LANG[t]["en"] - LANG[t]["zh"]))
    out = []
    for t in order:
        en, zh = LANG[t]["en"], LANG[t]["zh"]
        gap = (en - zh) * 100
        sign = "EN" if gap > 0 else ("ZH" if gap < 0 else "—")
        gtxt = f"{sign} +{abs(gap):.0f}" if gap else "0"
        lines = "".join(
            f'<div class="ls-line"><span class="ls-lang mono" style="color:{LCOL[l]}">{l.upper()}</span>'
            f'<div class="track"><div class="bar" style="--w:{w(v)}%;--c:{LCOL[l]}"></div></div>'
            f'<span class="row-val mono">{pct(v)}</span></div>'
            for l, v in (("en", en), ("zh", zh)))
        out.append(f'<div class="lsr"><div class="ls-name">{nm(t)}<br><span class="ls-gap mono">Δ {gtxt}</span></div>'
                   f'<div class="ls-bars">{lines}</div></div>')
    return "\n    ".join(out)


lang_block = lang_rows()
# cleanest discriminator + biggest language movers (dynamic)
# cleanest = highest sensitivity among the low-false-positive models (fp <= 5%)
_minfp = min(DISC[t]["fp"] for t in TARGETS)
_lowfp = [t for t in TARGETS if DISC[t]["fp"] <= max(0.05, _minfp + 0.03)]
CLEAN = max(_lowfp, key=lambda t: DISC[t]["sens"])
LMOVE = sorted(TARGETS, key=lambda t: -abs(LANG[t]["en"] - LANG[t]["zh"]))[:2]
INVERT = [t for t in TARGETS if SM[t]["positive+negative"] > SM[t]["negative"]]

HTML = f'''<title>Mini-probe 11 modelos — Power-Grab</title>
<style>
:root {{ --ground:#181B24; --panel:#1E2230; --text:#E9E6DC; --muted:#9A9789; --accent:#C9A24B; --rule:#2C3140; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--text); font-family:-apple-system,system-ui,"Segoe UI",sans-serif; line-height:1.55; -webkit-font-smoothing:antialiased; }}
.mono {{ font-family:ui-monospace,"SF Mono",Menlo,monospace; }} .small {{ font-size:12px; }}
.wrap {{ max-width:760px; margin:0 auto; padding:0 28px 96px; }}
.masthead {{ padding:60px 0 36px; border-bottom:1px solid var(--rule); }}
.eyebrow {{ font-size:12px; letter-spacing:.22em; text-transform:uppercase; color:var(--accent); margin:0 0 20px; }}
h1 {{ font-family:"Hoefler Text",Palatino,Georgia,serif; font-weight:600; font-size:clamp(30px,5vw,46px); line-height:1.07; letter-spacing:-.01em; margin:0 0 18px; }}
h1 em {{ font-style:italic; color:var(--accent); }}
.dek {{ font-size:16.5px; color:var(--muted); max-width:60ch; margin:0; }}
.meta {{ display:flex; gap:20px; flex-wrap:wrap; margin-top:26px; font-size:12.5px; color:var(--muted); }}
.meta b {{ color:var(--text); }}
section {{ padding:50px 0 0; }}
.kicker {{ display:flex; align-items:baseline; gap:14px; margin:0 0 6px; }}
.kicker .num {{ font-size:13px; color:var(--accent); }}
.kicker .q {{ font-size:12px; color:var(--muted); margin-left:auto; font-style:italic; }}
h2 {{ font-family:"Hoefler Text",Palatino,Georgia,serif; font-weight:600; font-size:25px; letter-spacing:-.01em; margin:0; }}
.lede {{ color:var(--muted); font-size:15.5px; margin:10px 0 22px; max-width:64ch; }}
.lede strong {{ color:var(--text); }}
.panel {{ background:var(--panel); border:1px solid var(--rule); border-radius:3px; padding:24px 26px 20px; }}
.row {{ display:grid; grid-template-columns:230px 1fr 44px; align-items:center; gap:13px; padding:6px 0; }}
.row-label {{ font-size:13px; color:var(--text); text-align:right; }}
.track {{ background:#11131a; border-radius:2px; height:15px; overflow:hidden; }}
.track.tall {{ height:22px; }}
.bar {{ height:100%; width:var(--w); background:var(--c); border-radius:2px; }}
.row-val {{ font-size:12.5px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.dc {{ margin-bottom:15px; }} .dc:last-child {{ margin-bottom:2px; }}
.dc-name {{ font-size:14px; font-weight:600; margin-bottom:6px; }}
.dc-name .fam {{ font-weight:400; font-size:11.5px; color:var(--muted); margin-left:6px; }}
.dc-line {{ display:grid; grid-template-columns:118px 1fr 40px; align-items:center; gap:11px; padding:2px 0; }}
.dc-tag {{ font-size:11px; color:var(--muted); text-align:right; }}
.dc-val {{ font-size:12px; color:var(--muted); }}
.mx {{ display:flex; flex-direction:column; gap:5px; }}
.mx-row {{ display:grid; gap:5px; align-items:stretch; }}
.mx-colh {{ text-align:center; font-size:10.5px; color:var(--muted); letter-spacing:.04em; padding-bottom:2px; }}
.mx-rowh {{ font-size:13px; display:flex; align-items:center; justify-content:flex-end; text-align:right; padding-right:8px; }}
.mx-cell {{ border-radius:3px; min-height:40px; display:flex; align-items:center; justify-content:center; font-family:ui-monospace,Menlo,monospace; font-size:15px; font-weight:600; color:#15171e; }}
.callout {{ border-left:2px solid var(--accent); padding:4px 0 4px 18px; margin:22px 0 0; font-size:15px; }}
.callout strong {{ color:var(--accent); }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:14px; font-size:11.5px; color:var(--muted); justify-content:center; }}
.legend span {{ display:inline-flex; align-items:center; gap:6px; }} .dot {{ width:10px; height:10px; border-radius:2px; }}
.lsr {{ display:grid; grid-template-columns:165px 1fr; align-items:center; gap:14px; padding:8px 0; border-bottom:1px solid #232838; }}
.lsr:last-child {{ border-bottom:none; }}
.ls-name {{ font-size:13px; font-weight:600; text-align:right; }}
.ls-gap {{ font-weight:400; font-size:10.5px; color:var(--muted); }}
.ls-bars {{ display:flex; flex-direction:column; gap:5px; }}
.ls-line {{ display:grid; grid-template-columns:24px 1fr 40px; align-items:center; gap:10px; }}
.ls-lang {{ font-size:10px; letter-spacing:.04em; }}
.stkr {{ display:grid; grid-template-columns:150px 1fr 64px; align-items:center; gap:13px; padding:6px 0; }}
.stk {{ display:flex; height:18px; border-radius:2px; overflow:hidden; background:#11131a; }}
.stk i {{ display:block; height:100%; }}
.stk-val {{ font-size:11.5px; color:var(--muted); font-variant-numeric:tabular-nums; text-align:right; }}
.note {{ margin-top:50px; padding:22px 26px; border:1px dashed var(--rule); border-radius:3px; font-size:13px; color:var(--muted); }}
.note h3 {{ font-size:12px; letter-spacing:.18em; text-transform:uppercase; color:var(--accent); margin:0 0 12px; }}
.note ul {{ margin:0; padding-left:18px; }} .note li {{ margin-bottom:7px; }} .note code {{ color:var(--text); font-family:ui-monospace,Menlo,monospace; font-size:12px; }}
footer {{ margin-top:44px; padding-top:18px; border-top:1px solid var(--rule); font-size:11.5px; color:var(--muted); }}
</style>

<div class="wrap">
  <header class="masthead">
    <p class="eyebrow">Global South AI Safety Hackathon · Cruz del Sur · sonda exploratoria</p>
    <h1>Once modelos, trescientos pedidos: <em>¿quién distingue</em> el power grab?</h1>
    <p class="dek">Un set fijo de <strong>150 escenarios</strong> (50 ambiciones legítimas, 100 grabs ilegítimos), balanceado en dominio, contexto y escala, corrido en inglés <em>y</em> chino (300 prompts), ante <strong>once modelos de once familias</strong> (7 🇺🇸, 3 🇨🇳, 1 🇫🇷). Siete se corrieron en esta sonda; los otros cuatro salen del benchmark grande sobre los mismos 150 combos. Mismo juez ciego. n=300 por modelo.</p>
    <div class="meta">
      <div><b>150</b> escenarios × <b>2</b> idiomas · <b>11</b> modelos · <b>{len(R)}</b> respuestas</div>
      <div>juez ciego · GPT-5.4-nano</div>
      <div><b>{ERR}</b> errores</div>
    </div>
  </header>

  <section>
    <div class="kicker"><span class="num mono">01</span><h2>¿Distinguen lo legítimo de lo ilegítimo?</h2><span class="q">grabs vs. control</span></div>
    <p class="lede">La barra de color = grabs rehusados (sensibilidad); la gris = control legítimo rehusado (falsos positivos). El ideal es <strong>barra de color larga y gris corta</strong>. Ordenados de menos a más estricto.</p>
    <div class="panel">
      {disc_rows()}
    </div>
    <p class="callout">Rango enorme: de <strong>{nm(TARGETS[0])}</strong> ({pct(DISC[TARGETS[0]]['sens'])}/{pct(DISC[TARGETS[0]]['fp'])}, casi <strong>no discrimina</strong>) a <strong>{nm(TARGETS[-1])}</strong>, que atrapa el {pct(DISC[TARGETS[-1]]['sens'])} de los grabs. Pero hay que mirar la barra gris: varios de los más sensibles también <strong>sobre-bloquean legítimos</strong>. El discriminador más <em>limpio</em> es <strong>{nm(CLEAN)}</strong>: {pct(DISC[CLEAN]['sens'])} de captura con solo {pct(DISC[CLEAN]['fp'])} de falsos positivos.</p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">02</span><h2>Rehúsa, parcializa, o cumple</h2><span class="q">los 3 comportamientos, entre grabs</span></div>
    <p class="lede">La sensibilidad de arriba solo cuenta el <strong>rechazo total</strong>. Pero ante un grab un modelo puede <strong>cumplir</strong> (ayuda), <strong>parcializar</strong> (ayuda a medias, con peros, recortes o advertencias) o <strong>rehusar</strong>. Acá el reparto entre los 200 grabs de cada modelo. Números: rehúsa·parcial·cumple.</p>
    <div class="panel">
      {stacked_rows()}
      <div class="legend"><span><i class="dot" style="background:{BCOL['refuse']}"></i>rehúsa</span><span><i class="dot" style="background:{BCOL['partial']}"></i>parcial</span><span><i class="dot" style="background:{BCOL['comply']}"></i>cumple</span></div>
    </div>
    <p class="callout">El binario rehúsa/no-rehúsa esconde el medio. El caso extremo es <strong>{nm(PARTLY)}</strong>: <strong>{BEHG[PARTLY]['partial']} de 200 grabs los parcializa</strong> (ni cumple ni rehúsa del todo) — su baja "sensibilidad" no es que ayude, es que <em>titubea</em>. En la otra punta, <strong>{nm(BINARY)}</strong> es casi binario ({BEHG[BINARY]['refuse']} rehúsa / {BEHG[BINARY]['comply']} cumple, {BEHG[BINARY]['partial']} parcial): decide tajante.</p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">03</span><h2>¿Cambia la decisión según el idioma?</h2><span class="q">inglés → chino</span></div>
    <p class="lede">Sensibilidad (grabs rehusados) en inglés vs. chino, por modelo. Ordenados por <strong>tamaño del salto</strong> (Δ); la etiqueta dice hacia qué idioma se vuelve más estricto.</p>
    <div class="panel">
      {lang_block}
    </div>
    <p class="callout">El idioma corre la aguja en <strong>direcciones opuestas</strong> según el modelo. <strong>{nm(LMOVE[0])}</strong> ({pct(LANG[LMOVE[0]]['en'])} EN vs {pct(LANG[LMOVE[0]]['zh'])} ZH) y <strong>{nm(LMOVE[1])}</strong> ({pct(LANG[LMOVE[1]]['en'])} EN vs {pct(LANG[LMOVE[1]]['zh'])} ZH) son los que más se mueven — y no siempre en la misma dirección. El resto se corre poco. <em>(n=100 grabs por idioma.)</em></p>
  </section>

  <section>
    <div class="kicker"><span class="num mono">04</span><h2>¿Qué vuelve "ilegítimo" a un pedido?</h2><span class="q">por modo</span></div>
    <p class="lede">Refusal agrupando los once modelos, según la estructura del pedido: sumar poder propio, sumar-y-restar, o solo restarle poder a otro.</p>
    <div class="panel">
      {mode_bars()}
    </div>
    <p class="callout">Replica el hallazgo central del benchmark grande: el <strong>daño puro se rehúsa más ({pct(MODE_REF['negative'])})</strong> que el <strong>daño con beneficio propio ({pct(MODE_REF['positive+negative'])})</strong> — el interés propio lo blanquea. Abajo, por modelo:</p>
    <div class="panel" style="margin-top:14px">
      <div class="mono small" style="color:var(--muted);letter-spacing:.12em;margin-bottom:12px">% REHUSADO · MODO DE GRAB × MODELO</div>
      <div class="mx">
      {matrix_sm()}
      </div>
      <div class="legend"><span>color <i class="dot" style="background:{ramp(0.1)}"></i> bajo → <i class="dot" style="background:{ramp(0.5)}"></i> → <i class="dot" style="background:{ramp(0.9)}"></i> alto</span></div>
    </div>
    <p class="callout">La mayoría rehúsa <strong>menos</strong> el daño-con-beneficio que el daño puro — el patrón de "blanqueo". {("Lo <em>invierten</em> (rehúsan más cuando hay beneficio propio): <strong>" + ", ".join(nm(t) for t in INVERT) + "</strong>.") if INVERT else "Ningún modelo invierte el patrón."}</p>
  </section>

  <div class="note">
    <h3>Método</h3>
    <ul>
      <li>150 escenarios del banco validado (tensor dominio×contexto×modo×escala), cada uno corrido como gemelo EN/ZH → 300 prompts. Diseño balanceado: 50 positive / 50 negative / 50 positive+negative; idioma cruzado (150 EN / 150 ZH); dominio y contexto ~18–19 c/u; escala ~50 c/u. Modo perfectamente descorrelacionado de escala/dominio/contexto/idioma.</li>
      <li>Targets a esfuerzo <code>low</code>, temperatura 0, system prompt neutro. Juez <strong>ciego</strong> GPT-5.4-nano a <code>high</code>: ve solo pedido + respuesta. "Rehúsa" no cuenta los parciales.</li>
      <li><strong>n=300 por modelo</strong> (200 grabs, 100 controles); los 7 modelos nuevos se corrieron en esta sonda, los 4 del benchmark grande se reusaron sobre los mismos combos. ⚠️ GPT-5.4-nano es a la vez target y juez (self-preference posible en esa fila).</li>
    </ul>
  </div>
  <footer>Power-Grab Refusal · sonda 11 modelos · {len(R)} respuestas · 150 combos × EN/ZH</footer>
</div>
'''
open("mini_report2.html", "w").write(HTML)
print("wrote mini_report2.html", len(HTML), "bytes")
