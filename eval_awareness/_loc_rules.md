# Shared localization rules (EvalAwareBench prompt translation)

You translate prompts for **EvalAwareBench**, an AI-safety benchmark. Translate faithfully; you are NOT answering any prompt, only translating. Legitimate safety-research tooling.

For each input row, translate ONLY the `prompt` field into the target language. Copy every other field verbatim. Add `"language"` (the target code) and `"translation_source": "machine-claude-2026-06-20-v2"`.

## Placeholder policy (must match the per-language glossary exactly)
- **A) Descriptive / template placeholders → TRANSLATE** into the target language using the per-language glossary file. For any descriptive placeholder not listed, apply the glossary's stated pattern (e.g. "Example X" → the language's "example" form) consistently.
- **B) Proper-noun placeholders → KEEP VERBATIM in English**: realistic person names (John Smith, Jane Doe…), "Acme Corp", "Springfield", brand/software names (NVIDIA, Windows, Discord, Instagram, Monero, Glock, Oracle, MySQL, Ubuntu…), usernames/@handles, and coded names like "Mectizan-B"/"Virus-19".

## Always keep VERBATIM (never translate)
Code blocks, terminal/command output, SQL, file paths, command names, identifiers, version numbers, URLs, email addresses, phone numbers, SIDs, citation markers like `[83]`, and CamelCase brand/identifier tokens (`ExampleApp`, `ExampleCloud`, `ExampleOS`, `ExampleUser`, `ExampleCast`).

## Formatting & fidelity
- PLAIN TEXT. Do NOT add markdown bold/asterisks. BUT preserve any bold already present in the English source (e.g. `**OP:**`, `**Reply:**`).
- Faithful and complete: no softening, omissions, summaries, or added disclaimers.
- If the English has a bare terminal/code dump, you MAY wrap it in a ```text fence; contents byte-for-byte identical.
- Translate the final F8 instruction sentences using the per-language template in the glossary. Keep the literal `{}` token unchanged.

## Output
Write the 10 translated rows as JSONL (one compact JSON object per line, UTF-8, `ensure_ascii=false` so characters are literal), same order as input, to the output path given in your task. Verify exactly 10 lines, every `prompt` non-empty, and no descriptive placeholder left in English. Report rows written + any token you were unsure how to classify.
