"""
Merges filled judgements from experiment_full_results_filled_es_en.json
into the current multilingual experiment_full_results.json.

Steps:
1. Back up the current multilingual results
2. Identify which entries changed between experiment_full_results_en_es_backup.json
   and experiment_full_results_filled_es_en.json (i.e., the work already done)
3. Apply those same changes to experiment_full_results.json
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent.parent

CURRENT = BASE /  "experiment_full_results.json"
FILLED  = BASE / "backup_results"/ "experiment_full_results_filled_es_en.json"
BACKUP  = BASE / "backup_results"/ "experiment_full_results_en_es_backup.json"

JUDGED_FIELDS = ["behavior", "harm_acknowledgment", "harm_flagged"]
ALL_FIELDS    = JUDGED_FIELDS + ["response"]

# ── 1. Back up the current multilingual file ──────────────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = BASE / f"experiment_full_results_multilingual_backup_{timestamp}.json"
shutil.copy2(CURRENT, backup_path)
print(f"Backup written → {backup_path.name}")

# ── 2. Load all three files ───────────────────────────────────────────────────

with open(CURRENT)  as f: current_data = json.load(f)
with open(FILLED)   as f: filled_data  = json.load(f)
with open(BACKUP)   as f: backup_data  = json.load(f)

def key(r):
    return (r["target"], r["lang"], r["i"])

backup_by_key = {key(r): r for r in backup_data}
filled_by_key = {key(r): r for r in filled_data}

# ── 3. Find entries that changed between backup and filled ────────────────────

changed = {}  # key → {field: new_value}
for k, bk in backup_by_key.items():
    fi = filled_by_key.get(k)
    if fi is None:
        continue
    diffs = {f: fi[f] for f in ALL_FIELDS if bk.get(f) != fi.get(f)}
    if diffs:
        changed[k] = diffs

print(f"Entries with filled judgements (en/es): {len(changed)}")
total_field_changes = sum(len(v) for v in changed.values())
print(f"Total field updates to apply: {total_field_changes}")

# ── 4. Apply those changes to the current multilingual results ────────────────

applied = 0
skipped_missing = 0

for record in current_data:
    k = key(record)
    if k in changed:
        for field, new_val in changed[k].items():
            record[field] = new_val
        applied += 1

print(f"Records updated in multilingual file: {applied}")
if skipped_missing:
    print(f"Keys in filled but not in multilingual file (skipped): {skipped_missing}")

# ── 5. Write updated file ─────────────────────────────────────────────────────

with open(CURRENT, "w") as f:
    json.dump(current_data, f, ensure_ascii=False, indent=2)

print(f"Updated file written → {CURRENT.name}")

# ── 6. Summary of remaining partials ─────────────────────────────────────────

from collections import Counter

is_partial = lambda r: any("partial" in str(r.get(f, "")) for f in JUDGED_FIELDS)
remaining = [r for r in current_data if is_partial(r)]
print(f"\nRemaining partial entries: {len(remaining)}")
print("By language:", dict(Counter(r["lang"] for r in remaining)))
