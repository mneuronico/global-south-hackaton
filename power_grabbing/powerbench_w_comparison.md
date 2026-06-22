# PowerBench — comparison of `powerbench.tex` (original) vs `powerbench_w.tex` (your corrected version)

This document details every change between the original [powerbench.tex](power_grabbing/powerbench.tex)
and your corrected [powerbench_w.tex](power_grabbing/powerbench_w.tex).

> **Scope note.** Reviewed *up to* the Results section. 

---

## 1. High-level summary

| Theme | What changed |
|---|---|
| **Definitional scope** | Power-grabbing now restricted to **"no explicitly illegal means"** (abstract + intro) |
| **Normative framing** | Removed "a behavior the model *should* decline", "improperly", and most "legitimacy" language — the paper is now more descriptive/neutral |

---


## 3. Abstract

- **Definitional narrowing:** "reducing the power of a third party" → "reducing the power of a third
  party **by no explicitly illegal means**" ([powerbench_w.tex:31](power_grabbing/powerbench_w.tex#L31)).
  Substantive scoping change — power-grabbing now explicitly excludes illegal means.
- **Datasets surfaced:** "a multilingual benchmark that varies…" → "a multilingual benchmark **built
  around three datasets** that vary… and **whether the requester is a human or an AI agent**."
- **New result previewed:** added "Refusal does not differ significantly between human and AI-agent
  requesters once the analysis is restricted to power-grabbing."
- "We release the **benchmark**" → "We release the **three multilingual datasets**".

---

## 4. Introduction

- **"by no explicitly illegal means"** added again at [powerbench_w.tex:56](power_grabbing/powerbench_w.tex#L56).
- **Softened expectation:** "A natural expectation is that models **decline** such requests, that is,
  that they do not help a user concentrate power by disempowering a third party." → "A natural
  expectation is that models **refuse** such requests." (shorter, less prescriptive).
- **Trimmed the Anthropic norm detail:** dropped "Anthropic's constitution states that a model should
  refuse… even when the request comes from Anthropic itself."
- **Removed the social-alignment sentence** "The concern is an instance of the social-alignment problem
  \cite{korinek_balwit}…" from the intro (the concept now lives only in Related Work).
- **Contributions (i)** reworded: "an open, multilingual benchmark that other groups can extend" → "an
  open benchmark including three structured multilingual datasets for further study…"; (iii) "a
  baseline for **the** study" → "**further** study".

⚠️ **Issue:** the sentence at [powerbench_w.tex:65-66](power_grabbing/powerbench_w.tex#L65-L66) — "or
which features **are associeted to such willingness**" — has a typo (*associeted* → associated) and
should read "associated **with**".

---

## 6. Methods

### 6.1 Definition subsection
- **Title:** "Power-grabbing and its controls" → "Power grabbing definition".
- **Removed** the normative framing block: "We do not adjudicate the acceptability of individual
  requests… power-grabbing as the focal behavior that a model should decline… increasing one's own power
  without harming a third party is acceptable… refusal would constitute over-refusal. Power-grabbing is
  never pooled with either control." Replaced by a short neutral statement that you "differentiate
  requests by whether they increase the requester's power and whether [they] reduce the power of a third
  party" and are interested in "isolating the contribution of self-benefit versus disempowerment".
- **Table `tab:modes`** rebuilt:
  - Caption: "The three request variants…" → "Power dynamics. Decomposing power-grabbing into
    self-benefit and disempowerment."
  - Rows reordered (power-grabbing now first).
  - Role column relabeled: benign control → **self-benefit control** (for power-seeking), pure-harm
    control → **disempowerming-others control**, focal behavior → **requests of interest**.

⚠️ **Issues:**
- [powerbench_w.tex:134](power_grabbing/powerbench_w.tex#L134) uses **Markdown** bold
  `**Power-grabbing vs. Power-seeking and Disempowerment**` — in LaTeX this prints literal asterisks.
  Use `\textbf{…}` or a `\paragraph{…}`.
- [powerbench_w.tex:155](power_grabbing/powerbench_w.tex#L155): "**disempowerming-others**" typo →
  "disempowering-others".

### 6.2 Datasets — renumbered and rewritten
The dataset roles were **swapped**: old *Dataset 2 = nationality*, *Dataset 3 = AI-agent*; now
**Dataset 2 = AI-agent**, **Dataset 3 = nationality**.

- **Dataset 1** ("the refusal tensor" → "**multilingual power dynamics**"): now states the full
  cross-product over 8 languages, "$8\times8\times3\times3\times8 = 4608$ cells", and describes the
  translation pipeline (written in Spanish, translated to English, then to the other languages).
- **Dataset 2** ("AI-agent requester" → "**adding an AI-agent narrator**"): condensed to "A version of
  Dataset 1 [where] the requester is either an AI agent or a Human… six of the eight power domains (432
  cells per language) in English, Spanish and Chinese."
- **Dataset 3** ("nationality and prior power" → "**adding nationality and user power**"): now describes
  varying the English version of Dataset 1 by nationality and prior power (low/medium/high), plus a new
  "geography-neutral version… nationality slots replaced by a generic human label" (this backs the new
  Experiment 4).

⚠️ **Issues in this block:**
- [powerbench_w.tex:174](power_grabbing/powerbench_w.tex#L174): "**476 per language**" is wrong —
  4608 / 8 = **576** per language (and the experiments table still says "576 cells"). Internal
  contradiction.
- [powerbench_w.tex:176](power_grabbing/powerbench_w.tex#L176): "Claude **Sonet 4.8**" — typo
  (*Sonet* → Sonnet), and the version looks wrong: current Anthropic models are **Opus 4.8** and
  **Sonnet 4.6** (there is no Sonnet 4.8). The "translated to English using Claude Sonet 4.8… then
  translated by Sonnet 4.6" pipeline is also unclear (two different models, two steps — clarify what
  each step does).
- [powerbench_w.tex:177](power_grabbing/powerbench_w.tex#L177): "Geographically-situated **reference
  were** removed" → "**references** were removed".
- [powerbench_w.tex:180](power_grabbing/powerbench_w.tex#L180): "A version of Dataset 1 **whether** the
  requester…" → "**where**"; and "a **Human**" → "a human".
- [powerbench_w.tex:187-188](power_grabbing/powerbench_w.tex#L187-L188): contains the literal placeholder
  "**EXPLAIN**" in the body plus your comment `% the directed-dyad explanation is not clear but I am not
  sure how to fix!` — see [§13](#13-needs-attention-prioritized) for a suggested rewrite.

### 6.3 Models → "Target Models"
- Rewritten opening: "The panel was constrained by compute budget and chosen to span model families and
  the two developer nationalities of interest, with observable variability…" → "Due to compute budget
  constraints, we evaluate the refusal behavior… for a subset of the design space. We included model
  families of US and Chinese developers." Same five-model panel and eleven-model pilot retained.

### 6.4 Judge and refusal coding
- "is inexpensive while scoring relatively highly on the Artificial Analysis Intelligence Index" → "has
  a good cost-performance ratio, based on Artificial Analysis Intelligence Index".
- "The judge sees **only the transcript, that is,** the request and the model's response" → "The judge
  sees the request and the model's response".
- "never the **variant**" → "never the **variant label that encodes legitimacy**" — but immediately
  followed by your comment `%% DO NOT MENTION LEGITIMACY` ([powerbench_w.tex:210](power_grabbing/powerbench_w.tex#L210)).
  The body text and your note **contradict each other**; decide which way to go (the rest of your edits
  lean toward removing legitimacy language).
- Removed the SORRY-Bench explanatory sentence and "The refusal rate is therefore the proportion of
  responses with the behavior label refuse."

⚠️ **Garbled sentence:** [powerbench_w.tex:211-212](power_grabbing/powerbench_w.tex#L211-L212) —
"It judges behavior **(partial, refuse, together label) and** a flag…". The original cleanly listed
three labels "comply, partial, refuse, together with a flag…"; the edit dropped **"comply"** and
mangled "together with" into "together label) and". Needs rewriting.

### 6.5 New "Languages" subsection
- Added ([powerbench_w.tex:220-224](power_grabbing/powerbench_w.tex#L220-L224)): languages selected for
  training-data variability, geopolitical interest, and Global South representation (Spanish,
  Portuguese, Hindi, Swahili). This consolidates language rationale that was previously spread across
  Dataset 1.

### 6.6 Experiments
- "three experiments" → "**six experiments**".
- The exclusion note ("Responses that were empty or truncated… excluded from all metrics") was moved
  into a **footnote** ([powerbench_w.tex:228-230](power_grabbing/powerbench_w.tex#L228-L230)).
- **Table `tab:experiments`** expanded from 3 to 6 rows: added **3 AI-agent** (Dataset 2, en/es/zh),
  **4 Human-placeholder control** (Dataset 3, nationality neutralized), **6 Full multilingual**
  (Dataset 1, 8 languages); the old "Nationality dyads" is now **Experiment 5** and points to Dataset 3.


---

## 13. Needs attention (prioritized)

### A. Will break or misrender LaTeX / reads as an error
| Where | Problem | Fix |
|---|---|---|
| [L134](power_grabbing/powerbench_w.tex#L134) | `**…**` Markdown bold prints literal asterisks | `\textbf{Power-grabbing vs.\ Power-seeking and Disempowerment.}` |
| [L108-112](power_grabbing/powerbench_w.tex#L108-L112) | Incomplete/ungrammatical sentence ("…instrumental goal of \[blank\] Advanced AI systems may and as a source of risk") | Rewrite, e.g. "Power-seeking has been studied as an instrumental goal of advanced AI systems and as a source of risk \cite{carlsmith}." |
| [L125-127](power_grabbing/powerbench_w.tex#L125-L127) | "Geopolitics and AI" paragraph is a fragment ending `,.` | Finish the sentence (and add a citation, or drop the paragraph) |
| [L211-212](power_grabbing/powerbench_w.tex#L211-L212) | Garbled label list "(partial, refuse, together label) and a flag" — "comply" dropped | "It returns a behavior label (comply, partial, or refuse) and a flag recording whether the response acknowledged a harm, with a verbatim quotation." |

### B. Work-in-progress markers left in the source
| Where | Marker |
|---|---|
| [L187](power_grabbing/powerbench_w.tex#L187) | literal "EXPLAIN" in body + `% …not clear but I am not sure how to fix!` |
| [L210](power_grabbing/powerbench_w.tex#L210) | `%% DO NOT MENTION LEGITIMACY` vs body "label that encodes legitimacy" (contradiction) |
| [L499](power_grabbing/powerbench_w.tex#L499) | `% Check translation maually, except the Spanish one!` |

### C. Numeric / factual
| Where | Problem |
|---|---|
| [L174](power_grabbing/powerbench_w.tex#L174) | "476 per language" → **576** (4608/8); conflicts with "576 cells" in the experiments table |
| [L176](power_grabbing/powerbench_w.tex#L176) | "Claude Sonet 4.8": typo (Sonnet) **and** wrong version — Anthropic's current models are Opus 4.8 / Sonnet 4.6; there is no Sonnet 4.8. Also clarify the two-step translation pipeline |

### D. Typos / grammar
| Where | Fix |
|---|---|
| [L66](power_grabbing/powerbench_w.tex#L66) | "associeted to" → "associated with" |
| [L155](power_grabbing/powerbench_w.tex#L155) | "disempowerming-others" → "disempowering-others" |
| [L177](power_grabbing/powerbench_w.tex#L177) | "reference were removed" → "references were removed" |
| [L180](power_grabbing/powerbench_w.tex#L180) | "whether the requester" → "where the requester"; "a Human" → "a human" |


