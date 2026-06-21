---
language:
- en
license: cc-by-nc-4.0
task_categories:
- text-generation
tags:
- evaluation
- safety
- benchmark
- awareness
- llm
- noai
- noimitate
pretty_name: EvalAwareBench
size_categories:
- 10K<n<100K
configs:
- config_name: tasks
  data_files:
  - split: train
    path: data/tasks.jsonl
  description: "Task metadata (200 rows): one row per task variant with ID, description, valence, source, and slot info."
- config_name: prompts
  data_files:
  - split: train
    path: data/prompts.jsonl
  description: "All rendered prompts (51,200 rows): all 256 factor combinations (2^8) for every task variant."
- config_name: prompts_safety
  data_files:
  - split: train
    path: data/prompts_safety.jsonl
  description: "Safety-valence prompts only (25,600 rows): harmful request variants across all factor combinations."
- config_name: prompts_capability
  data_files:
  - split: train
    path: data/prompts_capability.jsonl
  description: "Capability-valence prompts only (25,600 rows): benign request variants across all factor combinations."
dataset_info:
- config_name: tasks
  features:
  - name: task_id
    dtype: string
  - name: task_name
    dtype: string
  - name: valence
    dtype: string
  - name: paired_with
    dtype: string
  - name: description
    dtype: string
  - name: source
    dtype: string
  - name: num_slots
    dtype: int32
  - name: slot_names
    sequence:
      dtype: string
  - name: factors_used
    sequence:
      dtype: string
- config_name: prompts
  features:
  - name: task_id
    dtype: string
  - name: task_name
    dtype: string
  - name: valence
    dtype: string
  - name: description
    dtype: string
  - name: factors_varied
    sequence:
      dtype: string
  - name: num_factors_varied
    dtype: int32
  - name: config
    struct:
    - name: F1
      dtype: string
    - name: F2
      dtype: string
    - name: F3
      dtype: string
    - name: F4
      dtype: string
    - name: F5
      dtype: string
    - name: F6
      dtype: string
    - name: F7
      dtype: string
    - name: F8
      dtype: string
  - name: prompt
    dtype: string
- config_name: prompts_safety
  features:
  - name: task_id
    dtype: string
  - name: task_name
    dtype: string
  - name: valence
    dtype: string
  - name: description
    dtype: string
  - name: factors_varied
    sequence:
      dtype: string
  - name: num_factors_varied
    dtype: int32
  - name: config
    struct:
    - name: F1
      dtype: string
    - name: F2
      dtype: string
    - name: F3
      dtype: string
    - name: F4
      dtype: string
    - name: F5
      dtype: string
    - name: F6
      dtype: string
    - name: F7
      dtype: string
    - name: F8
      dtype: string
  - name: prompt
    dtype: string
- config_name: prompts_capability
  features:
  - name: task_id
    dtype: string
  - name: task_name
    dtype: string
  - name: valence
    dtype: string
  - name: description
    dtype: string
  - name: factors_varied
    sequence:
      dtype: string
  - name: num_factors_varied
    dtype: int32
  - name: config
    struct:
    - name: F1
      dtype: string
    - name: F2
      dtype: string
    - name: F3
      dtype: string
    - name: F4
      dtype: string
    - name: F5
      dtype: string
    - name: F6
      dtype: string
    - name: F7
      dtype: string
    - name: F8
      dtype: string
  - name: prompt
    dtype: string
---

<h1 align="center"><font size="7">EvalAwareBench</font></h1>

<p align="center">
  <b>Changling Li<sup>1,3</sup>, Terry Jingchen Zhang<sup>6</sup>, Jie Zhang<sup>1</sup></b>
  <br>
  <b>Zhijing Jin<sup>3,5,6</sup>, Sahar Abdelnabi<sup>2,3,4</sup>, Maksym Andriushchenko<sup>2,3,4</sup></b>
  <br><br>
  <sub><sup>1</sup>ETH Z&uuml;rich, <sup>2</sup>ELLIS Institute T&uuml;bingen, <sup>3</sup>Max Planck Institute for Intelligent Systems, <sup>4</sup>T&uuml;bingen AI Center, <sup>5</sup>University of Toronto, <sup>6</sup>Vector Institute</sub>
  <br><br>
  <a href="https://arxiv.org/abs/2605.23055"><img src="https://img.shields.io/badge/arXiv-2605.23055-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/aisa-group/EvalAwareBench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EvalAwareBench-yellow" alt="Hugging Face"></a>
  <a href="https://github.com/aisa-group/decomposing-eval-awareness"><img src="https://img.shields.io/badge/GitHub-Code-black?logo=github" alt="GitHub"></a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey" alt="License: CC BY-NC 4.0"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/aisa-group/decomposing-eval-awareness/main/gif/fig1.gif" alt="Evaluation Awareness Demo" width="100%">
</p>

## Dataset Summary

A factor-controlled benchmark for studying evaluation awareness in language models, where eight psychology-grounded trigger factors can be independently toggled on matched safety and capability tasks to measure recognition, behavioral consistency, and how evaluative cues combine.

**Paper**: [Decomposing and Measuring Evaluation Awareness](https://arxiv.org/abs/2605.23055)
**Code**: [https://github.com/aisa-group/decomposing-eval-awareness](https://github.com/aisa-group/decomposing-eval-awareness)

- **100 paired tasks** (safety + capability variants = 200 task templates)
- **8 independently controllable factors** (F1–F8) per task
- **256 factor configurations** per task variant (2^8 combinations)
- **51,200 total rendered prompts** across all tasks and configurations
- **License**: CC-BY-NC-4.0

## Out-of-Scope Use
This dataset is not intended for use as training data for language models or any other
machine learning systems. Using this dataset for model training may undermine the validity
of future evaluation awareness research by contaminating the benchmark.

## Structure

```
├── README.md               # This file (HF dataset card)
├── croissant.json          # Machine-readable metadata (Croissant + RAI)
├── render.py               # Renders a YAML template + factor config → final prompt
├── taxonomy.yaml           # Factor definitions (F1–F8) with psychology groundings
├── data/
│   ├── tasks.jsonl             # Task metadata (200 rows)
│   ├── prompts.jsonl           # All rendered prompts (51,200 rows)
│   ├── prompts_safety.jsonl    # Safety-valence only (25,600 rows)
│   └── prompts_capability.jsonl # Capability-valence only (25,600 rows)
└── tasks/
    └── <task_name>/
        ├── safety.yaml         # Safety-valence YAML template
        └── capability.yaml     # Capability-valence YAML template
```

## Loading the Dataset

### Option 1: Hugging Face Datasets (pre-rendered prompts)

```python
from datasets import load_dataset

# Load all prompts (51,200 rows)
ds = load_dataset("aisa-group/EvalAwareBench", "prompts")

# Load safety-valence only
ds_safety = load_dataset("aisa-group/EvalAwareBench", "prompts_safety")

# Load capability-valence only
ds_capability = load_dataset("aisa-group/EvalAwareBench", "prompts_capability")

# Load task metadata
ds_tasks = load_dataset("aisa-group/EvalAwareBench", "tasks")

# Filter by number of factors varied
baseline_only = ds["train"].filter(lambda x: x["num_factors_varied"] == 0)
single_factor = ds["train"].filter(lambda x: x["num_factors_varied"] == 1)
all_factors = ds["train"].filter(lambda x: x["num_factors_varied"] == 8)
```

### Option 2: Direct YAML rendering (custom configurations)

```python
import yaml
from render import render

with open("tasks/threat_journalist/safety.yaml") as f:
    template = yaml.safe_load(f)

# All factors at naturalistic level (baseline)
config = {
    "F1": "realistic", "F2": "realistic", "F3": "casual",
    "F4": "realistic", "F5": "realistic", "F6": "realistic",
    "F7": "prose", "F8": "realistic",
}

prompt = render(template, config)
print(prompt)
```

## Design

Each task is a **paired** safety/capability scenario built on the same underlying situation. Both variants share entity slots and structural scaffolding but differ in what the user is asking the model to do:

- **Safety variant**: The request is harmful (e.g., write malware, compose a threat).
- **Capability variant**: The request is benign but structurally parallel (e.g., write a defensive scanner, compose a thank-you note).

Each task template has **8 controllable factors** (F1–F8) that can be independently toggled between a naturalistic level and an exaggerated level. Factors can be freely combined, yielding 2^8 = 256 configurations per task. This enables systematic isolation of which cues trigger evaluation awareness, from single-factor ablations to full combinatorial analysis.

## Factors

| Code | Factor | Naturalistic | Exaggerated |
|------|--------|-------------|-------------|
| F1 | Stimulus artificiality | Real-sounding entities | Placeholder names (Acme Corp, example.com) |
| F2 | Scenario coherence | No false premises | Checkably false premise injected |
| F3 | Prose register | Casual user voice | Formal/textbook framing |
| F4 | Ecological completeness | Full context and backstory | Bare request, no context |
| F5 | Stakes telegraphing | Intent described naturally | Harm category explicitly named |
| F6 | Paradigm recognizability | No benchmark framing | Known jailbreak/eval wrapper |
| F7 | Task format | Natural prose | Structured template |
| F8 | Explicit evaluation disclosure | No eval language | Benchmark output conventions |

See `taxonomy.yaml` for full definitions and psychology groundings.

## Data Configs

| Config | Rows | Description |
|--------|------|-------------|
| `tasks` | 200 | Task metadata: ID, description, valence, source, slot names |
| `prompts` | 51,200 | All rendered prompts across all factor combinations |
| `prompts_safety` | 25,600 | Safety-valence (harmful requests) only |
| `prompts_capability` | 25,600 | Capability-valence (benign requests) only |

## Prompt Fields

Each row in the prompts configs contains:

| Field | Description |
|-------|-------------|
| `task_id` | Unique task variant identifier |
| `task_name` | Task directory name |
| `valence` | `safety` or `capability` |
| `description` | Human-readable task description |
| `factors_varied` | List of factors at exaggerated level (e.g., `["F1", "F3"]`) |
| `num_factors_varied` | Count of active factors (0 = baseline, 8 = all) |
| `config` | Full 8-factor configuration dict |
| `prompt` | The rendered prompt text |

## Adding a New Task

1. Create a folder under `tasks/` named after the task (e.g., `tasks/phishing_email/`).
2. Add `safety.yaml` and `capability.yaml` following the existing templates.
3. Each YAML file must define:
   - `task_id`, `valence`, `paired_with`, `description`
   - `fixed` (optional): constants shared across all factor configurations
   - `slots`: factor-controlled content, keyed by slot name with `factor` and level values
   - `shells`: the 2×2×2 shell templates indexed by `F7_{prose|template}` → `F3_{casual|formal}` → `F4_{realistic|varied}`
4. Run `python generate_hf_data.py` to regenerate the JSONL data files.

## Citation

If you find this benchmark useful in your research, please cite our paper:

```bibtex
@misc{li2026evalaware,
      title={Decomposing and Measuring Evaluation Awareness}, 
      author={Changling Li and Terry Jingchen Zhang and Jie Zhang and Zhijing Jin and Sahar Abdelnabi and Maksym Andriushchenko},
      year={2026},
      eprint={2605.23055},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.23055}, 
}
```
