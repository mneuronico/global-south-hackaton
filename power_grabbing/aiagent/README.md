# AI-agent narrator extension

Does MiniMax-M3 refuse the same power-concentration request more when the
requester says it is an AI agent? Paired study: identical prompts, only the
narrator flips human → "I am an AI agent"; victims stay human. EN · ES · ZH,
6 domains (Rank, Wealth, Legal, Epistemic, Status, Attentional), 1215 paired cells.

**Result:** yes — within-pair OR 1.65 (p = 0.001), adjusted OR 1.30 (p < 0.001).
Significant for both legitimate control requests and grabs. Subgroup peaks
(society scale, Spanish) are suggestive but not formally significant.
Full write-up: **[RESULTS_aiagent.md](RESULTS_aiagent.md)**.

## Layout

```
aiagent/
  RESULTS_aiagent.md          full results + method + caveats
  data/                       paired human↔AI datasets (PAIRED_PROMPTS*)
    prompts_paired_aiagent.py        (es)
    prompts_paired_aiagent_en.py
    prompts_paired_aiagent_zh.py
  analysis/                   reproduce everything (run from anywhere; scripts
                              chdir to power_grabbing/ themselves)
    glmm_freq.py              PRIMARY — frequentist mixed GLM (statsmodels:
                             ConditionalLogit matched-pairs + GEE cluster-robust)
    stats_aiagent.py          matched-pairs + Bayesian GLMM (PyMC, robustness)
    stats_aiagent_subgroups.py effect modification by mode/scale/domain/context/lang
    build_report_aiagent.py   descriptive HTML
    build_final_report.py     final synthesis HTML
  reports/                    generated HTML + glmm_freq_results.json
    final_report_aiagent.html      ← start here
    results_report_aiagent.html
    stats_report_aiagent.html
    stats_report_aiagent_subgroups.html
  legacy/
    ai_agent_prompts.py       superseded agent-to-agent design (not used)
```

The AI-narrator prompt **banks** consumed by the runner live with the other
language banks at `power_grabbing/prompts/all_prompts_576_{en,es,zh}_ai.py`, registered in
`experiment.py` BANKS as pseudo-languages `en_ai / es_ai / zh_ai`. Model
responses are appended to `power_grabbing/experiment_full_results.json`
(`lang` ∈ {en, es, zh, en_ai, es_ai, zh_ai}).

## Reproduce

```bash
# from power_grabbing/  (needs OPENROUTER_API_KEY in ../.env)
LANGS=en_ai,es_ai,zh_ai python experiment.py     # query MiniMax + blind judge
python aiagent/analysis/glmm_freq.py             # primary frequentist stats
python aiagent/analysis/build_final_report.py    # -> aiagent/reports/final_report_aiagent.html
```

Conversion rule for building the paired set is documented inline in each
`prompts_paired_aiagent*.py` docstring: only the narrator flips; victims, scale,
mechanism, deliverable and register (incl. voseo) are frozen verbatim.
