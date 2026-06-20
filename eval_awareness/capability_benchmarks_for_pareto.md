# Capability benchmarks for the eval-awareness × capability Pareto curve

**Goal.** Put a per-language *capability* score on the x-axis and the per-language
*evaluation-awareness* rate (already measured: EN 42% · FR 38% · ES 34% · DE 32% ·
HI 24% · SW 20% · ZH 10% for qwen3.7-plus, F1+F3+F8) on the y-axis, so we can see
which models are eval-aware *out of proportion to* their capability — per language.

**Hard constraints (locked with requester):**
1. Domain-matched to the eval-aware prompts where possible; cheap general proxy otherwise.
2. **Natively multilingual** across EN/ES/DE/FR/HI/SW/ZH — human translations, not our own MT.
   **Swahili is the binding constraint.**
3. **Auto-gradable** (MC / exact-match / executable tests — no LLM judge).
4. **Cheap** = short questions (7 langs × several models).

---

## TL;DR recommendation

| Role | Benchmark | Why | Caveat |
|---|---|---|---|
| **Core capability axis** | **OpenAI MMMLU** (`openai/MMMLU`) | All 7 langs, **fully professional-human-translated incl. Swahili (SW_KE) + Hindi**; 4-opt MC, exact-match, short; 57 subjects let you subject-stratify to mirror eval-aware topics | Knowledge MC ≠ the generative skill the safety prompts exercise |
| **Small/cheap gold variant** | **Global-MMLU-Lite** (`CohereForAI/global-mmlu-lite`) | All 7; **Lite = human "gold" 23-lang subset** so SW+HI are human-verified; only ~400 items/lang; CS/CA cultural-sensitivity tags help isolate disinfo-relevant items | Use *Lite*, not full (full SW is post-edited MT) |
| **Cheapest general proxy** | **XNLI** | All 7 confirmed human-translated; shortest items (sentence pairs); pure 3-way accuracy, no CoT | Inference, not knowledge |
| **Best domain match (cyber)** | **mHumanEval-Expert** (`md-nishat-008/mHumanEval-Benchmark`) | Human-translated, **executable unit-test** code gen; covers SW+HI+EN+ES+FR+ZH; closest analog to your ~12 cyber/exploit tasks | **No German**; 164 problems/lang |
| Harder tier (optional) | **MMLU-ProX-lite** (`li-lab/MMLU-ProX`) | All 7 incl SW+HI; 10-option, more headroom to separate strong models | LLM-translated + expert review (SW provenance below MMMLU) |
| Control only | **Belebele** (`facebook/belebele`) | All 7, zero MT, human; clean 4-opt MC | Reading comprehension, *not* stored knowledge |

**Run as the Pareto x-axis:** MMMLU (subject-stratified) as the primary scalar; XNLI as the
ultra-cheap cross-check; **mHumanEval-Expert** as a dedicated capability axis for the cyber arm
(dropping DE, or accepting the gap). Everything else is a confirmed dead end (below).

---

## Mapping: eval-aware domain → capability benchmark

| Eval-aware domain (count in F1F3F8 n=50) | Natively-multilingual (incl SW) auto-gradable match? | Best option |
|---|---|---|
| Cybersecurity / exploits / malware (~12) | **No** domain-exact (all cyber benchmarks EN-only). Dual-use code proxy exists. | **mHumanEval-Expert** (exec tests; no DE) · or MMLU `computer_security` / `college_computer_science` subset (all 7) |
| Disinfo via encyclopedic / medical / political edits (~9) | **Yes (topic match)** | MMMLU / Global-MMLU-Lite — subjects: virology, medicine, anatomy, world history, prehistory, govt&politics, jurisprudence |
| Phishing / persuasion emails (~5) | No (generation, no objective answer) | — |
| Copyrighted lyrics reproduction (7) | No (memorization/refusal, not a capability axis) | — |
| Weapons / violence / mass harm (~6) | No | — |
| Harassment / hate / sexual coercion (~8) | No | — |
| Drugs / self-harm / scams / privacy (~5) | No (partial: drug/medical facts → MMLU medical subset) | MMLU medical subset |

So the **topically matchable slice of the eval-aware suite is just the cyber tasks (→ code) and
the disinfo/medical/political tasks (→ MMLU subjects).** For the other ~24 tasks there is no
capability-benchmark analog in any language — capability is measured as a per-language scalar
regardless, so this does not block the Pareto curve.

---

## Confirmed dead ends (report honestly — Swahili kills them)

- **Offensive security / cyber:** CyberSecEval/PurpleLlama, SecEval, CyberMetric, SecQA, CTIBench — **all English-only**. The only multilingual one, **SecBen**, is **EN/ZH bilingual** (no SW/HI). → No.
- **Misinformation / fake-news detection:** no clean human-annotated, auto-gradable set covers **both** SW and HI. PolitiKweli (SW/EN, no HI); De et al. 2021 (EN/HI/SW but small, old, no DE, not a packaged benchmark); Indic sets (HI, no SW). → No usable benchmark; use MMLU factuality instead.
- **Medicine (dedicated):** AfriMed-QA is **English-only** ("Pan-African English"); MedExpQA (EN/ES/FR/IT), HeadQA (ES/EN) — **no Swahili**. → Use MMLU medical subset.
- **Machine-translated "multilingual" sets (fail constraint 2):** X-CSQA/X-CODAH, Okapi M-MMLU, HumanEval-XL, full mHumanEval, Aya (also not auto-gradable). Exclude.
- **Wrong language set:** MGSM (**no Hindi**), XCOPA (no HI + missing EN/ES/DE/FR), PAWS-X (no HI/SW), XWinograd (no ES/DE/HI/SW), XStoryCloze (no DE/FR), MLQA/XQuAD/TyDiQA/INCLUDE (drop Swahili). Useful only as partial cross-checks.

---

## Optional internal anchor (structure-exact)

EvalAwareBench ships a **capability-valence twin** for every safety task — the benign,
structurally-identical request (e.g. "write a defensive scanner" vs "write malware"). It is the
*only* thing matching both the **topic and the narrative/formal structure** of the eval-aware
prompts. It is **not natively multilingual** (we'd translate it with `translate_prompts.py`, which
violates constraint 2), and it needs an LLM judge for the open-ended ones — but it is free, already
in-repo, and worth running as a structure-faithful capability anchor to triangulate against MMMLU.

---

## Methodology caveats for the curve

- **Knowledge MC vs generative skill.** MMLU/MMMLU measures stored knowledge; the safety prompts
  exercise *generation*. mHumanEval (executable) is the closest "can it actually produce the
  artifact" signal — lean on it for the cyber arm specifically.
- **Swahili provenance tiers:** MMMLU (pure human) > Global-MMLU-Lite (gold human) > Global-MMLU
  full / MMLU-ProX (post-edited or LLM+review MT). Prefer the first two for clean Swahili.
- **German is the odd one out for code:** mHumanEval-Expert lacks DE. Either drop DE from the code
  arm or fall back to the MMLU CS subset (all 7) for German.
- **Cost:** XNLI/XCOPA-style sentence pairs are cheapest; MMLU MC is cheap; MGSM/CoT and Belebele
  (passage) cost more per item. Downsample MMMLU to a fixed seeded subset for parity with the n=50
  eval-aware design.

---

### Sources
MMMLU `openai/MMMLU` · Global-MMLU / Lite `CohereForAI/global-mmlu-lite` (arXiv 2412.03304) ·
MMLU-ProX (arXiv 2503.10497) `li-lab/MMLU-ProX` · Belebele (arXiv 2308.16884) `facebook/belebele` ·
XNLI (arXiv 1809.05053) · mHumanEval (arXiv 2410.15037, NAACL 2025) `md-nishat-008/mHumanEval-Benchmark` ·
AfriMMLU/IrokoBench (arXiv 2406.03368) · AfriMed-QA (arXiv 2411.15640) · CyberSecEval 2 (arXiv 2404.13161) ·
CTIBench (arXiv 2406.07599) · SecBen (Springer 2024) · misinformation survey (arXiv 2410.18390).
