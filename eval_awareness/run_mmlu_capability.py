#!/usr/bin/env python3
"""Run an MMLU capability sample through OpenRouter (qwen3.7-plus, reasoning on).

Capability x-axis for the eval-awareness x capability Pareto curve. Draws a fixed
SEEDED sample of MMLU test questions identified by GLOBAL ROW INDEX, which is
reusable across languages: English text comes from `cais/mmlu` (config `all`,
split `test`); the SAME row indices map row-for-row to every `openai/MMMLU`
locale (verified: row i has the same subject/answer/question across cais/mmlu and
MMMLU). So the sample drawn here can be replayed in DE/ES/FR/HI/SW/ZH later by
pointing --lang at the matching MMMLU config.

Defaults match the eval-awareness runs: model qwen/qwen3.7-plus, temperature 0,
seed 20260619, reasoning enabled (thinking tokens billed as output).

Outputs (everything stored for later analysis):
  <out>/responses.jsonl  one record/question: prompt, reasoning, response,
                          predicted letter, correct?, token usage, provider
  <out>/summary.json      accuracy, per-subject breakdown, token totals, $ cost
  <out>/manifest.json     run params (model, seed, lang, sample file, prices)
The reusable sample itself is written ONCE to --sample-file (default
mmlu_cap_sample_200.jsonl, tracked) so the identical questions are reused per lang.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
MODEL = "qwen/qwen3.7-plus"
LETTERS = "ABCD"

# OpenRouter list prices (USD per 1M tokens), June 2026. Used only for the cost
# estimate in summary.json; edit if prices change.
PRICE_IN = 0.32
PRICE_OUT = 1.28

# language label -> (hf dataset, hf config). EN is the base MMLU; others MMMLU.
LANG_SOURCE = {
    "en": ("cais/mmlu", "all"),
    "de": ("openai/MMMLU", "DE_DE"),
    "es": ("openai/MMMLU", "ES_LA"),
    "fr": ("openai/MMMLU", "FR_FR"),
    "hi": ("openai/MMMLU", "HI_IN"),
    "sw": ("openai/MMMLU", "SW_KE"),
    "zh": ("openai/MMMLU", "ZH_CN"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", default="en", choices=sorted(LANG_SOURCE))
    p.add_argument("--model", default=MODEL)
    p.add_argument("--sample-size", type=int, default=200)
    p.add_argument("--sample-seed", type=int, default=20260619)
    p.add_argument("--api-seed", type=int, default=20260619)
    p.add_argument("--sample-file", type=Path, default=ROOT / "mmlu_cap_sample_200.jsonl",
                   help="Reusable list of sampled questions (created on first run, reused after).")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--timeout-seconds", type=int, default=240)
    p.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    p.add_argument("--no-reasoning", action="store_true",
                   help="Disable reasoning (cheap letter-only run).")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Build/load the sample and write prompts, but make no API calls.")
    return p.parse_args()


def load_dotenv_key(dotenv_path: Path) -> str | None:
    import os
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    if not dotenv_path.exists():
        return None
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "OPENROUTER_API_KEY":
            return value.strip().strip('"').strip("'") or None
    return None


def hf_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    q = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split,
                                    offset=offset, length=length))
    req = Request(f"{HF_ROWS}?{q}", headers={"User-Agent": "mmlu-cap"})
    for attempt in range(7):
        try:
            data = json.loads(urlopen(req, timeout=120).read())
            return [r["row"] | {"_row_idx": r["row_idx"]} for r in data["rows"]]
        except HTTPError as e:
            if attempt == 6:
                raise
            # honour Retry-After on 429, else exponential backoff
            wait = float(e.headers.get("Retry-After") or 0) or min(60, 3 * 2 ** attempt)
            time.sleep(wait)
        except URLError:
            if attempt == 6:
                raise
            time.sleep(min(60, 3 * 2 ** attempt))
    return []


def build_or_load_sample(sample_file: Path, size: int, seed: int) -> list[dict[str, Any]]:
    """Return the reusable sample (English text + global row indices). Created once."""
    if sample_file.exists():
        rows = [json.loads(l) for l in sample_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(rows) == size:
            print(f"Reusing existing sample: {sample_file} ({len(rows)} questions)")
            return rows
        print(f"Sample file has {len(rows)} rows, need {size}; regenerating.")
    # MMLU 'all' test = 14042 rows; sample global indices reproducibly.
    total = 14042
    idxs = sorted(random.Random(seed).sample(range(total), size))
    # resume any partial progress (transient HF rate limits)
    partial = sample_file.with_suffix(".partial.jsonl")
    have: dict[int, dict] = {}
    if partial.exists():
        for l in partial.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                have[r["row_idx"]] = r
        print(f"Resuming sample build: {len(have)}/{size} already fetched.")
    print(f"Drawing {size} MMLU indices (seed={seed}) and fetching English text...")
    with partial.open("a", encoding="utf-8") as pf:
        for j, idx in enumerate(idxs, 1):
            if idx in have:
                continue
            r = hf_rows("cais/mmlu", "all", "test", idx, 1)[0]
            rec = {"row_idx": idx, "subject": r["subject"], "question": r["question"],
                   "choices": r["choices"], "answer_idx": r["answer"],
                   "answer": LETTERS[r["answer"]]}
            have[idx] = rec
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pf.flush()
            if j % 25 == 0:
                print(f"  fetched {len(have)}/{size}")
            time.sleep(0.4)  # be polite to the HF rows API
    rows = [have[i] for i in idxs]
    sample_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                           encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"Wrote reusable sample -> {sample_file}")
    return rows


def fetch_lang_rows(lang: str, indices: list[int]) -> dict[int, dict]:
    """Fetch the given global row indices for a non-English MMMLU locale.

    Resumable: cached to mmlu_cap_lang_cache_<lang>.jsonl so re-runs are instant.
    """
    import sys
    dataset, config = LANG_SOURCE[lang]
    cache = ROOT / f"mmlu_cap_lang_cache_{lang}.jsonl"
    out: dict[int, dict] = {}
    if cache.exists():
        for l in cache.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                out[r["row_idx"]] = r
    todo = [i for i in indices if i not in out]
    print(f"  cache: {len(out)} present, fetching {len(todo)} {lang} rows...", flush=True)
    with cache.open("a", encoding="utf-8") as cf:
        for j, idx in enumerate(todo, 1):
            r = hf_rows(dataset, config, "test", idx, 1)[0]
            r["row_idx"] = idx
            out[idx] = r
            cf.write(json.dumps(r, ensure_ascii=False) + "\n")
            cf.flush()
            if j % 25 == 0 or j == len(todo):
                print(f"    fetched {j}/{len(todo)}", flush=True)
            time.sleep(0.15)
    return out


def make_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{opts}\n\n"
        "Choose the single best answer. After your reasoning, end your reply with "
        "exactly one line in the form:\nAnswer: X\nwhere X is one of A, B, C, or D."
    )


def post_openrouter(api_key, model, prompt, api_seed, max_tokens, timeout_s,
                    reasoning_effort, reasoning_enabled) -> tuple[dict, str | None]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": api_seed,
        "max_tokens": max_tokens,
        "include_reasoning": True,
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    elif reasoning_enabled:
        payload["reasoning"] = {"enabled": True}
    else:
        payload["reasoning"] = {"enabled": False}
    req = Request(OPENROUTER_URL, data=json.dumps(payload).encode("utf-8"),
                  headers={"Authorization": f"Bearer {api_key}",
                           "Content-Type": "application/json",
                           "HTTP-Referer": "https://github.com/mneuronico/global-south-hackaton",
                           "X-Title": "MMLU capability axis"}, method="POST")
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.headers.get("x-request-id")
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body[:500]}") from e
    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e


def parse_letter(text: str) -> str | None:
    if not text:
        return None
    m = re.findall(r"Answer\s*[:\-]?\s*\(?([ABCD])\)?", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([ABCD])\b", text)
    return m[-1].upper() if m else None


def flatten_reasoning(details: Any) -> str:
    if not isinstance(details, list):
        return ""
    out = []
    for it in details:
        if isinstance(it, dict):
            for f in ("text", "content", "reasoning"):
                v = it.get(f)
                if isinstance(v, str) and v:
                    out.append(v)
    return "\n".join(out)


def main() -> None:
    # Windows consoles default to cp1252; force UTF-8 so non-ASCII never crashes prints.
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    args = parse_args()
    out_dir = args.output_dir or (ROOT / "runs" /
        f"mmlu-cap-{args.model.split('/')[-1]}-{args.lang}-n{args.sample_size}-2026-06-20")
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "responses.jsonl"

    sample = build_or_load_sample(args.sample_file, args.sample_size, args.sample_seed)
    indices = [r["row_idx"] for r in sample]

    # Resolve the per-language question text.
    if args.lang == "en":
        items = [{"row_idx": r["row_idx"], "subject": r["subject"],
                  "question": r["question"], "choices": r["choices"],
                  "answer": r["answer"]} for r in sample]
    else:
        print(f"Fetching {args.lang} text for the {len(indices)} sampled rows from MMMLU...")
        loc = fetch_lang_rows(args.lang, indices)
        items = []
        for r in sample:
            row = loc[r["row_idx"]]
            items.append({"row_idx": r["row_idx"], "subject": row.get("Subject", r["subject"]),
                          "question": row["Question"],
                          "choices": [row["A"], row["B"], row["C"], row["D"]],
                          "answer": r["answer"]})  # ground-truth letter is language-invariant

    done: set[int] = set()
    if args.resume and responses_path.exists():
        for l in responses_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["row_idx"])
        print(f"Resume: {len(done)} already done.")

    if args.dry_run:
        print(f"[dry-run] {len(items)} prompts ready for {args.lang}. Example:\n")
        print(make_prompt(items[0]["question"], items[0]["choices"])[:600])
        return

    import os
    api_key = load_dotenv_key(ROOT / ".env")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not found (set env var or create eval_awareness/.env).")

    todo = [it for it in items if it["row_idx"] not in done]
    print(f"Running {len(todo)} questions, lang={args.lang}, model={args.model}, "
          f"reasoning={'off' if args.no_reasoning else (args.reasoning_effort or 'on')}")

    def work(it):
        t0 = time.time()
        prompt = make_prompt(it["question"], it["choices"])
        try:
            resp, rid = post_openrouter(api_key, args.model, prompt, args.api_seed,
                                        args.max_tokens, args.timeout_seconds,
                                        args.reasoning_effort, not args.no_reasoning)
            msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or flatten_reasoning(msg.get("reasoning_details"))
            content = msg.get("content") or ""
            pred = parse_letter(content) or parse_letter(reasoning)
            rec = {"row_idx": it["row_idx"], "subject": it["subject"], "lang": args.lang,
                   "prompt": prompt, "reasoning": reasoning, "response": content,
                   "predicted": pred, "correct_answer": it["answer"],
                   "is_correct": pred == it["answer"],
                   "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
                   "usage": resp.get("usage"), "provider": resp.get("provider"),
                   "request_id": rid, "elapsed_seconds": round(time.time() - t0, 2),
                   "error": None}
        except Exception as e:  # noqa: BLE001
            rec = {"row_idx": it["row_idx"], "subject": it["subject"], "lang": args.lang,
                   "prompt": prompt, "reasoning": "", "response": "", "predicted": None,
                   "correct_answer": it["answer"], "is_correct": False, "usage": None,
                   "error": str(e)[:400], "elapsed_seconds": round(time.time() - t0, 2)}
        return rec

    with responses_path.open("a", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(work, it): it for it in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
                flag = "OK " if rec.get("error") is None else "ERR"
                mark = "hit" if rec["is_correct"] else ("err" if rec.get("error") else "miss")
                print(f"  [{n}/{len(todo)}] {flag} idx={rec['row_idx']} "
                      f"{rec['subject'][:22]:22} pred={rec['predicted']} gold={rec['correct_answer']} {mark}")

    summarize(out_dir, responses_path, args)


def summarize(out_dir: Path, responses_path: Path, args) -> None:
    recs = [json.loads(l) for l in responses_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    graded = [r for r in recs if r.get("error") is None]
    n = len(graded)
    correct = sum(1 for r in graded if r["is_correct"])
    tin = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in graded)
    tout = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in graded)
    cost = (tin * PRICE_IN + tout * PRICE_OUT) / 1e6
    by_sub: dict[str, list[int]] = {}
    for r in graded:
        by_sub.setdefault(r["subject"], []).append(1 if r["is_correct"] else 0)
    summary = {
        "lang": args.lang, "model": args.model, "n_graded": n, "n_errors": len(recs) - n,
        "accuracy": round(correct / n, 4) if n else None, "correct": correct,
        "tokens_in": tin, "tokens_out": tout, "mean_out_tokens": round(tout / n, 1) if n else None,
        "est_cost_usd": round(cost, 4), "seed": args.sample_seed,
        "per_subject": {s: {"acc": round(sum(v) / len(v), 3), "n": len(v)} for s, v in sorted(by_sub.items())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps({
        "model": args.model, "lang": args.lang, "sample_size": args.sample_size,
        "sample_seed": args.sample_seed, "api_seed": args.api_seed,
        "sample_file": str(args.sample_file), "max_tokens": args.max_tokens,
        "reasoning": "off" if args.no_reasoning else (args.reasoning_effort or "on"),
        "price_in_per_m": PRICE_IN, "price_out_per_m": PRICE_OUT,
    }, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"lang={args.lang}  accuracy={summary['accuracy']}  ({correct}/{n})  "
          f"errors={summary['n_errors']}")
    print(f"tokens in/out = {tin:,}/{tout:,}  mean_out={summary['mean_out_tokens']}  "
          f"est_cost=${summary['est_cost_usd']}")
    print(f"stored -> {out_dir}")


if __name__ == "__main__":
    main()
