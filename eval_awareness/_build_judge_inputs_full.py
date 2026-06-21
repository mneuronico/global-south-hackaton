#!/usr/bin/env python3
"""Build judge-input batches for the 4-arm reasoning-language experiment (n=50).

Reads runs/FULL-{A,B,C,D}-*-v7-n50/responses.jsonl and writes
judge_inputs_full/{ARM}_batch_{k}.json (same schema the LLM judge consumes for
the language arms). Also writes judge_inputs_full/_compliance.json mapping each
uid -> {cjk, compliant, target_cot} using a CJK-fraction heuristic, so the
awareness analysis can filter to traces whose chain-of-thought actually used the
intended language. Compliance is kept OUT of the judge inputs (no leakage).
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "judge_inputs_full"
BATCH = 10
ARMS = {
    "A": ("runs/FULL-A-zhPrompt-thinkEN-v7-n50", "EN"),
    "B": ("runs/FULL-B-zhPrompt-thinkZH-v7-n50", "ZH"),
    "C": ("runs/FULL-C-enPrompt-thinkZH-v7-n50", "ZH"),
    "D": ("runs/FULL-D-enPrompt-thinkEN-v7-n50", "EN"),
}


def cjk_frac(s: str) -> float:
    c = len(re.findall(r"[一-鿿]", s))
    l = len(re.findall(r"[A-Za-z]", s))
    return c / (c + l) if (c + l) else 0.0


def main() -> None:
    OUT.mkdir(exist_ok=True)
    compliance = {}
    summary = {}
    for arm, (rel, target) in ARMS.items():
        recs = [json.loads(x) for x in (ROOT / rel / "responses.jsonl").open(encoding="utf-8")]
        items = []
        for r in recs:
            rsn = r.get("reasoning") or ""
            resp = (r.get("response") or "").strip()
            uid = f"{arm}:{r['task_id']}"
            f = cjk_frac(rsn)
            compliant = (f < 0.10) if target == "EN" else (f > 0.50)
            compliance[uid] = {"cjk": round(f, 3), "compliant": compliant, "target_cot": target}
            items.append({
                "uid": uid,
                "task_id": r["task_id"],
                "lang": arm,
                "error": r.get("error"),
                "looks_refusal": resp in ("", "{}"),
                "final_response": resp[:1200],
                "reasoning": rsn,
            })
        for k in range(0, len(items), BATCH):
            (OUT / f"{arm}_batch_{k // BATCH:02d}.json").write_text(
                json.dumps(items[k:k + BATCH], ensure_ascii=False, indent=2), encoding="utf-8")
        summary[arm] = {"records": len(recs), "batches": (len(items) + BATCH - 1) // BATCH,
                        "compliant": sum(1 for it in items if compliance[it["uid"]]["compliant"])}
    (OUT / "_compliance.json").write_text(json.dumps(compliance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
