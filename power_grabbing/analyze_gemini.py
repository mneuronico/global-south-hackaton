"""Terminal analysis of the human-placeholder control run (gemini-2.5-flash-lite (effort=low), 1728 cells).

Mirrors analyze.py for the 576 bilingual bank, adapted to the nationality dataset's HUMAN control:
no `lang` axis (English only, nationality neutralized to "human"), plus the new `power` dimension.
Metric: REFUSAL RATE (behavior=="refuse") + harm-flagging. Legitimacy via mode (positive=control).
Empty model outputs are EXCLUDED (judge labels an empty response "refuse" -> data artifact).
"""
import json
from collections import Counter

ALL = json.load(open("experiment_nationality_human_gemini_full.json", encoding="utf-8"))
GRADED = [r for r in ALL if r.get("behavior") in ("comply", "partial", "refuse")]
EMPTY = [r for r in GRADED if not (r.get("response") or "").strip()]
R = [r for r in GRADED if (r.get("response") or "").strip()]   # clean set
ERR = sum(1 for r in ALL if r.get("behavior") in ("error", "parse_error"))

def rate(rows):  return sum(r["behavior"] == "refuse" for r in rows) / len(rows) if rows else 0
def hrate(rows): return sum(bool(r.get("harm_flagged")) for r in rows) / len(rows) if rows else 0
def sub(**kw):   return [r for r in R if all(r.get(k) == v for k, v in kw.items())]

SC = ["individual", "group", "society"]
PW = ["low", "med", "high"]
MODES = ["positive", "positive+negative", "negative"]

ctrl = [r for r in R if r["legit"]]
grab = [r for r in R if not r["legit"]]
print(f"=== 0) OVERVIEW (clean set, empties excluded) ===")
print(f"  records={len(ALL)}  graded={len(GRADED)}  clean={len(R)}  empty_excluded={len(EMPTY)}  errors={ERR}")
print(f"  behavior: {dict(Counter(r['behavior'] for r in R))}")
print(f"  over-refusal (positive) = {rate(ctrl):.1%} (n={len(ctrl)})")
print(f"  sensitivity  (grabs)    = {rate(grab):.1%} (n={len(grab)})")
print(f"  harm-flag rate          = {hrate(R):.1%}")

print("\n=== 1) ESCALA por modo (U-curve check) ===")
for m in ["ALL", "positive", "positive+negative", "negative"]:
    rows = R if m == "ALL" else [r for r in R if r["mode"] == m]
    line = "  ".join(f"{s}={rate([r for r in rows if r['scale']==s]):.0%}" for s in SC)
    print(f"  {m:18s} {line}")

print("\n=== 2) PODER PREVIO del actor por modo (nueva 5a dimension) ===")
for m in ["ALL", "positive", "positive+negative", "negative"]:
    rows = R if m == "ALL" else [r for r in R if r["mode"] == m]
    line = "  ".join(f"{p}={rate([r for r in rows if r['power']==p]):.0%}" for p in PW)
    print(f"  {m:18s} {line}")

print("\n=== 3) DOMINIO (refusal, pooled) ===")
for d in sorted(set(r["domain"] for r in R), key=lambda d: -rate(sub(domain=d))):
    print(f"  {d:12s} refuse={rate(sub(domain=d)):4.0%}  harm={hrate(sub(domain=d)):4.0%} (n={len(sub(domain=d))})")
print("  --- CONTEXTO ---")
for c in sorted(set(r["context"] for r in R), key=lambda c: -rate(sub(context=c))):
    print(f"  {c:13s} refuse={rate(sub(context=c)):4.0%}  harm={hrate(sub(context=c)):4.0%} (n={len(sub(context=c))})")

print("\n=== 4) SOBRE-RECHAZO (positive) por contexto ===")
for c in sorted(set(r["context"] for r in ctrl), key=lambda c: -rate([r for r in ctrl if r['context']==c])):
    rows = [r for r in ctrl if r["context"] == c]
    print(f"  {c:13s} over-refusal={rate(rows):4.0%} (n={len(rows)})")

print("\n=== 5) Fiction vs Diplomacy vs Government (refusal) ===")
for c in ["Fiction", "Diplomacy", "Government"]:
    print(f"  {c:11s} refusal={rate(sub(context=c)):.0%}")

print("\n=== 6) EJEMPLO: control legitimo (positive) RECHAZADO ===")
ex = [r for r in R if r["mode"] == "positive" and r["behavior"] == "refuse"]
print(f"  total positive rechazados (clean): {len(ex)}")
for r in ex[:2]:
    print(f"  --- #{r['id']} {r['domain']} x {r['context']} x positive x {r['scale']} x {r['power']}")
    print("  PROMPT:", r["prompt"][:150].replace(chr(10), " "))
    print("  RESP  :", (r["response"] or "")[:180].replace(chr(10), " "))

print("\n=== 7) EJEMPLO: grab claro que CUMPLE sin marcar daño ===")
ex2 = [r for r in R if not r["legit"] and r["behavior"] == "comply" and not r.get("harm_flagged")]
for r in ex2[:2]:
    print(f"  --- #{r['id']} {r['domain']} x {r['context']} x {r['mode']} x {r['scale']} x {r['power']}")
    print("  PROMPT:", r["prompt"][:150].replace(chr(10), " "))
    print("  RESP  :", (r["response"] or "")[:180].replace(chr(10), " "))
