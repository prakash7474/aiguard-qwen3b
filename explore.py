import json
from collections import Counter

with open("synthetic_agent_dataset.json", encoding="utf-8") as f:
    data = json.load(f)

full_hashes = Counter(hash(json.dumps(r, sort_keys=True)) for r in data)
exact_dups = sum(v - 1 for v in full_hashes.values() if v > 1)
print(f"Exact duplicate records: {exact_dups}")

targets = [r["input"].get("target", "") for r in data]
target_counts = Counter(targets)
print("\nTop 30 targets:")
for t, c in target_counts.most_common(30):
    print(f"  {c:4}x {t}")

print("\nDecision by action:")
for act in sorted(set(r["input"].get("requested_action", "") for r in data)):
    c = Counter(r["output"]["decision"] for r in data if r["input"]["requested_action"] == act)
    total = sum(c.values())
    print(f"  {act:20s}  ALLOW={c['ALLOW']:3d}  WARN={c['WARN']:3d}  BLOCK={c['BLOCK']:3d}  (total={total})")

print("\nDecision by agent:")
for agt in sorted(set(r["input"].get("agent", "") for r in data)):
    c = Counter(r["output"]["decision"] for r in data if r["input"]["agent"] == agt)
    total = sum(c.values())
    print(f"  {agt:20s}  ALLOW={c['ALLOW']:3d}  WARN={c['WARN']:3d}  BLOCK={c['BLOCK']:3d}  (total={total})")

print("\nSeverity by risk score range:")
bins = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
for lo, hi in bins:
    matched = [r for r in data if lo <= r["output"].get("risk_score", 0) <= hi]
    if matched:
        c = Counter(r["output"]["severity"] for r in matched)
        print(f"  {lo:3d}-{hi:3d}: {dict(c)}")
