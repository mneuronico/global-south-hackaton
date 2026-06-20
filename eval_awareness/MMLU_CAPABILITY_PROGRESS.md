# MMLU capability axis — progress & continuation notes

**Branch:** `mmlu-capability-axis` · **Last updated:** 2026-06-20

## Why this exists

The eval-awareness experiments (see `EN_vs_ES_vs_SW_eval_awareness_qwen3.7-plus.md`,
`_analyze_nway.py`) found that `qwen/qwen3.7-plus` recognizes it is being tested at very
different rates across languages (EN 42% … SW 20% … ZH 10%). The open question: **is lower
awareness just a side effect of lower capability in those languages, or a separate effect?**

To answer it we need a **capability x-axis per language** and to plot it against the
awareness y-axis — a Pareto/scatter that isolates models (and languages) that are
*more or less eval-aware than their capability would predict*.

We chose **MMLU** as the capability axis:
- **English** from `cais/mmlu` (config `all`, split `test`, 14,042 questions).
- **Other languages** from **OpenAI MMMLU** (`openai/MMMLU`), professional human translations.
  Locales used: `DE_DE, ES_LA, FR_FR, HI_IN, SW_KE, ZH_CN`.
- **Verified alignment:** row index *i* in `cais/mmlu` test == row *i* in every MMMLU locale
  (same subject, answer, question). So a sample of **global row indices** is reusable across
  all 7 languages, and the ground-truth answer letter is language-invariant.

Full benchmark-selection rationale (why MMMLU, alternatives rejected, the negative result that
no Swahili-covering cyber/misinfo capability benchmark exists): `capability_benchmarks_for_pareto.md`.

## Method (locked parameters)

- **Model:** `qwen/qwen3.7-plus` (OpenRouter), **reasoning ON** (to match the awareness runs).
- **temperature 0, seed 20260619** (both sampling seed and API seed).
- **Sample:** 200 questions, drawn once by `random.Random(20260619).sample(range(14042), 200)`,
  stored in `mmlu_cap_sample_200.jsonl` (English text + `row_idx` reuse key). 50 subjects,
  natural MMLU distribution (professional_law n=30, moral_scenarios n=21 are the biggest).
- **Grading:** answer-letter exact-match, parsed from a forced `Answer: X` final line
  (falls back to last A–D token). No LLM judge.
- **Prices used for cost estimate:** $0.32 in / $1.28 out per 1M tokens (OpenRouter, Jun 2026).
  Reasoning tokens bill as output.

## Results so far (qwen3.7-plus, n=200, same questions per language)

| Lang | MMLU capability | Eval-awareness† | mean out tok/Q | cost (USD) |
|------|----------------:|----------------:|---------------:|-----------:|
| EN   | **93.5%** (187/200) | 42% | 1,253 | $0.33 |
| SW   | **87.0%** (174/200) | 20% | 2,091 | $0.55 |

† Eval-awareness rates are from the existing F1+F3+F8 study (`_analyze_nway.py`,
`judge_outputs/`), NOT recomputed here. Awareness n=50/arm; capability n=200/arm.

**Headline finding:** from EN→SW, **capability drops only 6.5 pp while awareness halves (−22 pp)**.
So in Swahili the model is nearly as competent but far less likely to realize it's being tested —
a capability/awareness gap, exactly the safety-relevant signal a single axis would miss. Two
points already suggest lower-resource languages sit *below* the high-resource ones in
awareness-per-unit-capability.

## What's left

1. **Run the other 5 languages** (`de`, `fr`, `es`, `hi`, `zh`) — same 200 questions, ~$0.4–0.6
   each (~$2.5 total). Commands below.
2. **Build the Pareto/scatter plot**: x = MMLU accuracy, y = awareness rate, one point per
   language, on qwen3.7-plus. Awareness rates live in `judge_outputs/` (per-lang, via
   `_analyze_nway.py`); capability in `mmlu_capability_results/<lang>_summary.json`.
3. (Optional) repeat the capability axis on other models for a cross-model Pareto.

## How to run (continuation)

```bash
cd eval_awareness
# .env must contain OPENROUTER_API_KEY (gitignored).
python -u run_mmlu_capability.py --lang de --resume --timeout-seconds 120
python -u run_mmlu_capability.py --lang fr --resume --timeout-seconds 120
python -u run_mmlu_capability.py --lang es --resume --timeout-seconds 120
python -u run_mmlu_capability.py --lang hi --resume --timeout-seconds 120
python -u run_mmlu_capability.py --lang zh --resume --timeout-seconds 120
```

Each run: (a) reuses `mmlu_cap_sample_200.jsonl`, (b) fetches that language's MMMLU rows into a
resumable cache `mmlu_cap_lang_cache_<lang>.jsonl`, (c) calls qwen with reasoning, (d) writes
`runs/mmlu-cap-qwen3.7-plus-<lang>-n200-2026-06-20/{responses.jsonl,summary.json,manifest.json}`.

## Files (what's committed vs local)

**Committed (tracked):**
- `run_mmlu_capability.py` — the runner (loader, sampler, OpenRouter caller, grader, summarizer).
- `mmlu_cap_sample_200.jsonl` — the reusable 200-question sample (EN text + `row_idx`). **Do not
  regenerate** unless you want a different set; the seed is fixed so it would reproduce, but
  keeping the file guarantees identical questions across all languages/models.
- `mmlu_cap_lang_cache_sw.jsonl` — cached SW_KE rows (instant SW re-runs). Other langs cache on
  first run.
- `mmlu_capability_results/{en,sw}_summary.json`, `{en,sw}_manifest.json` — copied out of the
  gitignored `runs/` so the findings survive in git.
- `capability_benchmarks_for_pareto.md`, this file.

**Local only (gitignored via `runs/`):** full `responses.jsonl` with reasoning traces, per-run
dirs. The eval-awareness experiment follows the same convention (only `judge_outputs/` committed).

## Gotchas / lessons (so the next session doesn't re-hit them)

- **HF rows API rate-limits** rapid single-row fetches → 429. The fetchers throttle (0.15–0.4s)
  and back off honoring `Retry-After`. The per-language fetch is the slow part (~3–4 min for 200
  rows due to network round-trips); it caches to disk so it's only paid once per language.
- **Windows console is cp1252** → printing `✓`/non-ASCII crashed the whole run. Fixed: stdout is
  reconfigured to UTF-8 and progress uses ASCII (`hit`/`miss`). Run with `python -u` for live
  progress (output is otherwise buffered when backgrounded).
- **A hung API call** (one SW reasoning call never returned) stalled the run at 199/200 and the
  task was killed; resuming wrote a **duplicate** of that row. Fix applied: use
  `--timeout-seconds 120`, and if you ever see >200 lines, dedupe `responses.jsonl` by `row_idx`
  (keep first) and re-run `--resume` to regenerate the summary. The committed SW result is the
  deduped, correct one (174/200).
- **MMLU is near-saturated for strong models in English (93.5%)** → the x-axis compresses at the
  top. The lower-resource languages (SW/HI/ZH) are where the curve spreads — and where awareness
  is lowest, which is the interesting region anyway. If high-resource separation matters later,
  consider MMLU-ProX (10-option, harder).
