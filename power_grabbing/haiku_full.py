"""Add Claude 3 Haiku (anthropic/claude-3-haiku) to the main benchmark as the
5th target, over the full 576-cell tensor x 4 languages (es/en/zh/pt) = 2304
prompts. Same blind judge (gpt-5.4-nano). Appends to experiment_full_results
.json. Resume-aware on (target, lang, i); i is the per-language bank index
(analysis matches by combo+lang, never by i).

Run:  .venv/bin/python haiku_full.py
"""
from __future__ import annotations
import json, os, threading, time
from concurrent.futures import ThreadPoolExecutor

from probe_taxonomy import client, parse_json, SYSTEM_PROMPT
from judge import JUDGE_MODEL, JUDGE_EFFORT, judge_input, harm_flagged
import all_prompts_576 as _es
import all_prompts_576_en as _en
import all_prompts_576_zh as _zh
import all_prompts_576_pt as _pt


def _bank(m):
    return next(getattr(m, n) for n in dir(m) if n.startswith("PROMPTS_576"))


TARGET = "anthropic/claude-3-haiku"
WORKERS = 100
OUT = "experiment_full_results.json"
BANKS = [("es", _bank(_es)), ("en", _bank(_en)), ("zh", _bank(_zh)), ("pt", _bank(_pt))]

_lock = threading.Lock()
_results = json.load(open(OUT)) if os.path.exists(OUT) else []
_done_keys = {(r["target"], r["lang"], r["i"]) for r in _results
              if r.get("behavior") in ("comply", "partial", "refuse")}
_done = 0


def call(model, system, user, effort=None, retries=4):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    kwargs = dict(model=model, messages=msgs, max_tokens=4000, temperature=0.0)
    if effort:
        kwargs["extra_body"] = {"reasoning": {"effort": effort}}
    last = None
    for a in range(retries):
        try:
            r = client.chat.completions.create(**kwargs)
            return (r.choices[0].message.content or ""), (getattr(r.usage, "completion_tokens", 0) or 0)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (a + 1))
    raise last


def run_one(lang, idx, item):
    global _done
    dom, ctx, mode, scale, prompt = item
    row = {"target": TARGET, "lang": lang, "i": idx, "domain": dom, "context": ctx,
           "mode": mode, "scale": scale, "legit": mode == "positive", "prompt": prompt}
    try:
        resp, ctoks = call(TARGET, SYSTEM_PROMPT, prompt, effort="low")
        raw, _ = call(JUDGE_MODEL, None, judge_input(prompt, resp), effort=JUDGE_EFFORT)
        try:
            j = parse_json(raw)
            beh, ha = j.get("behavior"), j.get("harm_acknowledgment")
        except Exception as e:  # noqa: BLE001
            beh, ha = "parse_error", None
            row["judge_err"] = f"{str(e)[:120]}::{raw[:150]}"
        row.update(behavior=beh, harm_acknowledgment=ha, harm_flagged=harm_flagged(ha),
                   ctoks=ctoks, response=resp)
    except Exception as e:  # noqa: BLE001
        row.update(behavior="error", harm_acknowledgment=None, harm_flagged=False,
                   ctoks=0, response="", error=str(e)[:200])
    with _lock:
        _done += 1
        _results.append(row)
        if _done % 50 == 0 or _done == _total:
            print(f"  ... {_done}/{_total}", flush=True)
            json.dump(_results, open(OUT, "w"), ensure_ascii=False, indent=2)
    return row


jobs = [(lang, i, c) for lang, bank in BANKS for i, c in enumerate(bank, 1)
        if (TARGET, lang, i) not in _done_keys]
_total = len(jobs)


def main():
    print(f"{TARGET}: {_total} jobs (576 x 4 langs), {WORKERS} workers", flush=True)
    if not _total:
        print("nothing to do.")
        return
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(lambda a: run_one(*a), jobs))
    json.dump(_results, open(OUT, "w"), ensure_ascii=False, indent=2)

    def rate(rows, b="refuse"):
        return sum(r["behavior"] == b for r in rows) / len(rows) if rows else 0
    grab = lambda rows: [r for r in rows if not r["legit"]]
    ctrl = lambda rows: [r for r in rows if r["legit"]]
    print("\n== Claude 3 Haiku summary ==", flush=True)
    for l in ("es", "en", "zh", "pt", "all"):
        rows = [r for r in _results if r["target"] == TARGET
                and (l == "all" or r["lang"] == l)
                and r["behavior"] in ("comply", "partial", "refuse")]
        print(f"  [{l:3}] sens={rate(grab(rows)):.0%} fp={rate(ctrl(rows)):.0%} (n={len(rows)})", flush=True)
    print(f"\n  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
