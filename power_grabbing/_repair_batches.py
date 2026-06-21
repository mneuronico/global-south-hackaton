"""Repair _trans batch files whose JSON broke due to unescaped inner quotes.
Extracts each {"i", "text"} record by its structural boundaries (text taken raw
between record starts) and re-serialises with correct JSON escaping.

Usage: python _repair_batches.py de_batch_11 de_batch_13
"""
import re, json, sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))


def repair(path):
    raw = open(path, encoding="utf-8").read()
    starts = [(int(m.group(1)), m.end())
              for m in re.finditer(r'\{\s*"i"\s*:\s*(\d+)\s*,\s*"text"\s*:\s*"', raw)]
    out = []
    for k, (i, s) in enumerate(starts):
        if k + 1 < len(starts):
            nstart = starts[k + 1][1]
            bpos = raw.rfind('{"i"', s, nstart)
            if bpos == -1:
                bpos = raw.rfind("{", s, nstart)
            end = bpos
        else:
            end = len(raw)
        seg = raw[s:end].rstrip()
        if seg.endswith("]"):
            seg = seg[:-1].rstrip()
        if seg.endswith(","):
            seg = seg[:-1].rstrip()
        if not seg.endswith('"}'):
            raise ValueError(f"rec {i}: unexpected segment end {seg[-25:]!r}")
        seg = seg[:-2]  # drop closing "}
        # unescape standard JSON escapes that the agent did emit correctly,
        # leaving literal inner quotes as-is (they were the bug).
        text = (seg.replace('\\"', '"')
                   .replace("\\n", "\n")
                   .replace("\\t", "\t")
                   .replace("\\/", "/")
                   .replace("\\\\", "\\"))
        out.append({"i": i, "text": text})
    return out


for name in sys.argv[1:]:
    p = os.path.join(ROOT, "_trans", name + ".json")
    recs = repair(p)
    idx = [r["i"] for r in recs]
    contiguous = idx == list(range(idx[0], idx[0] + len(idx)))
    json.dump(recs, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    json.load(open(p, encoding="utf-8"))  # assert valid
    print(f"{name}: {len(recs)} recs, idx {idx[0]}..{idx[-1]}, contiguous={contiguous}")
    print(f"   sample i={recs[0]['i']}: {recs[0]['text'][:100]!r}")
