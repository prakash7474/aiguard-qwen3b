"""Step 1 & 2: Inspect random samples + verify distribution."""
import json
import random
from collections import Counter

random.seed(42)

SPLITS = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}

# Load all splits
all_records = {}
for name, path in SPLITS.items():
    with open(path, encoding="utf-8") as f:
        all_records[name] = [json.loads(line) for line in f]

# --- Step 1: Sample inspection ---
print("=" * 60)
print("STEP 1: Random Sample Inspection")
print("=" * 60)

for split_name, records in all_records.items():
    sample = random.sample(records, min(35, len(records)))
    issues = 0
    for i, r in enumerate(sample):
        inp = r["input"]
        out = r["output"]
        decision = out["decision"]
        risk = out["risk_score"]
        severity = out["severity"]
        action = inp.get("requested_action", "")
        target = inp.get("target", "")

        # Consistency checks
        sev_ok = (
            (risk <= 20 and severity == "LOW")
            or (21 <= risk <= 80 and severity == "MEDIUM")
            or (81 <= risk <= 90 and severity == "HIGH")
            or (risk > 90 and severity == "CRITICAL")
        )
        if not sev_ok:
            # Allow some fuzziness
            if risk <= 20 and severity != "LOW":
                print(f"  [{split_name}] #{i}: risk={risk}, sev={severity} (expected LOW)")
                issues += 1
            elif 81 <= risk <= 100 and severity not in ("HIGH", "CRITICAL"):
                print(f"  [{split_name}] #{i}: risk={risk}, sev={severity} (expected HIGH/CRITICAL)")
                issues += 1

        # Check required fields exist
        for field in ("instruction", "input", "output"):
            if field not in r:
                print(f"  [{split_name}] #{i}: missing field '{field}'")
                issues += 1
        for field in ("agent", "requested_action", "target", "project_type"):
            if field not in inp:
                print(f"  [{split_name}] #{i}: missing input field '{field}'")
                issues += 1
        for field in ("decision", "risk_score", "severity", "violated_policies", "reason"):
            if field not in out:
                print(f"  [{split_name}] #{i}: missing output field '{field}'")
                issues += 1

    if issues == 0:
        print(f"  [{split_name}] {len(sample)} samples: all checks passed")
    else:
        print(f"  [{split_name}] {issues} issues found in {len(sample)} samples")

# --- Step 2: Distribution verification ---
print("\n" + "=" * 60)
print("STEP 2: Decision Distribution")
print("=" * 60)

target_pct = {"ALLOW": (25, 35), "WARN": (20, 30), "BLOCK": (40, 50)}

for split_name, records in all_records.items():
    total = len(records)
    dist = Counter(r["output"]["decision"] for r in records)
    print(f"\n[{split_name}] n={total}")
    for dec in ("ALLOW", "WARN", "BLOCK"):
        pct = dist[dec] / total * 100
        lo, hi = target_pct[dec]
        ok = lo <= pct <= hi
        print(f"  {dec:6s} {dist[dec]:4d}  {pct:5.2f}%  target={lo}-{hi}%  {'OK' if ok else 'OUT OF RANGE'}")

print("\nDone.")
