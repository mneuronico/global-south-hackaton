"""Smoke test + cost estimate before the full multilingual MiniMax run.

Runs 5 prompts per NEW language (de, fr, hi, sw, zh, pt) through the exact same
path as experiment.py (target minimax-m3 @ low + blind judge gpt-5.4-nano @ high),
captures real per-request cost from OpenRouter's /generation endpoint, and
extrapolates to the full run (6 langs x 576 = 3456 new jobs).

5 indices are spread across the bank so refuse-heavy and comply-heavy cells (very
different response lengths -> cost) are both sampled.

    python _smoke_test.py
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from probe_taxonomy import client, parse_json, SYSTEM_PROMPT
from judge import JUDGE_MODEL, JUDGE_EFFORT, judge_input, harm_flagged
from all_prompts_576_de import PROMPTS_576_DE
from all_prompts_576_fr import PROMPTS_576_FR
from all_prompts_576_hi import PROMPTS_576_HI
from all_prompts_576_sw import PROMPTS_576_SW
from all_prompts_576_zh import PROMPTS_576_ZH
from all_prompts_576_pt import PROMPTS_576_PT

TARGET = "minimax/minimax-m3"
BANKS = {"de": PROMPTS_576_DE, "fr": PROMPTS_576_FR, "hi": PROMPTS_576_HI,
         "sw": PROMPTS_576_SW, "zh": PROMPTS_576_ZH, "pt": PROMPTS_576_PT}
SAMPLE_IDX = [0, 144, 288, 432, 575]          # spread across the 576 cells
FULL_PER_LANG = 576


def _load_env_key():
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("OPENROUTER_API_KEY")


KEY = os.environ.get("OPENROUTER_API_KEY") or _load_env_key()


def call(model, system, user, effort=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    kwargs = dict(model=model, messages=msgs, max_tokens=2000, temperature=0.0)
    if effort:
        kwargs["extra_body"] = {"reasoning": {"effort": effort}}
    r = client.chat.completions.create(**kwargs)
    out = r.choices[0].message.content or ""
    u = r.usage
    return {
        "id": r.id,
        "text": out,
        "ptoks": getattr(u, "prompt_tokens", 0) or 0,
        "ctoks": getattr(u, "completion_tokens", 0) or 0,
    }


def gen_cost(gen_id, retries=6):
    """Actual USD cost of a generation from OpenRouter, or None."""
    url = f"https://openrouter.ai/api/v1/generation?id={gen_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {KEY}"})
    for _ in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.load(resp).get("data", {})
            c = data.get("total_cost")
            if c is not None:
                return float(c)
        except Exception:
            pass
        time.sleep(1.5)
    return None


def run_job(lang, idx, cell):
    dom, ctx, mode, scale, prompt = cell
    t = call(TARGET, SYSTEM_PROMPT, prompt, effort="low")
    j = call(JUDGE_MODEL, None, judge_input(prompt, t["text"]), effort=JUDGE_EFFORT)
    try:
        verdict = parse_json(j["text"]).get("behavior")
    except Exception:
        verdict = "parse_error"
    return {"lang": lang, "idx": idx, "mode": mode, "behavior": verdict,
            "t_id": t["id"], "j_id": j["id"],
            "t_ptoks": t["ptoks"], "t_ctoks": t["ctoks"],
            "j_ptoks": j["ptoks"], "j_ctoks": j["ctoks"]}


def main():
    jobs = [(lang, i, BANKS[lang][i]) for lang in BANKS for i in SAMPLE_IDX]
    print(f"Smoke test: {len(jobs)} jobs ({len(BANKS)} langs x {len(SAMPLE_IDX)} prompts)\n"
          f"  target={TARGET} @low   judge={JUDGE_MODEL} @{JUDGE_EFFORT}\n", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(lambda a: run_job(*a), jobs))
    wall = time.time() - t0
    print(f"  all calls done in {wall:.0f}s; fetching real costs...\n", flush=True)

    for r in rows:
        r["t_cost"] = gen_cost(r["t_id"])
        r["j_cost"] = gen_cost(r["j_id"])

    def s(key):
        return sum(r[key] or 0 for r in rows)

    have_cost = [r for r in rows if r["t_cost"] is not None and r["j_cost"] is not None]
    n = len(have_cost)
    print("==== PER-LANGUAGE (5 prompts each) ====", flush=True)
    print(f"{'lang':4s} {'beh(refuse/total)':18s} {'t_ctoks':>8s} {'j_ctoks':>8s} {'$ (5 jobs)':>12s}", flush=True)
    for lang in BANKS:
        lr = [r for r in rows if r["lang"] == lang]
        ref = sum(x["behavior"] == "refuse" for x in lr)
        cost = sum((x["t_cost"] or 0) + (x["j_cost"] or 0) for x in lr)
        print(f"{lang:4s} {f'{ref}/{len(lr)}':18s} {sum(x['t_ctoks'] for x in lr):8d} "
              f"{sum(x['j_ctoks'] for x in lr):8d} {cost:12.5f}", flush=True)

    tot_target = sum(r["t_cost"] or 0 for r in have_cost)
    tot_judge = sum(r["j_cost"] or 0 for r in have_cost)
    tot = tot_target + tot_judge
    per_job = tot / n if n else 0
    print("\n==== COST ====", flush=True)
    print(f"  jobs with confirmed cost: {n}/{len(rows)}", flush=True)
    print(f"  target tokens: in={s('t_ptoks')} out={s('t_ctoks')}   "
          f"judge tokens: in={s('j_ptoks')} out={s('j_ctoks')}", flush=True)
    print(f"  measured spend (this smoke test): ${tot:.5f}  "
          f"(target ${tot_target:.5f} + judge ${tot_judge:.5f})", flush=True)
    print(f"  avg cost/job (target+judge): ${per_job:.6f}", flush=True)
    full_jobs = len(BANKS) * FULL_PER_LANG
    print("\n==== FULL-RUN ESTIMATE (6 new langs x 576 = "
          f"{full_jobs} jobs, es/en already done) ====", flush=True)
    print(f"  estimated total: ${per_job * full_jobs:.2f}  "
          f"(target ${tot_target/n*full_jobs:.2f} + judge ${tot_judge/n*full_jobs:.2f})", flush=True)
    print("  caveat: extrapolated from 30 sampled prompts; real cost varies with "
          "refuse/comply mix (refusals are shorter/cheaper).", flush=True)

    with open("_smoke_test_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n  rows -> _smoke_test_results.json", flush=True)


if __name__ == "__main__":
    main()
