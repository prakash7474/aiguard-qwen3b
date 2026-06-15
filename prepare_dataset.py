import json
import hashlib
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

INPUT_FILE = "synthetic_agent_dataset.json"

TRAIN_FILE = "train.jsonl"
VALID_FILE = "validation.jsonl"
TEST_FILE = "test.jsonl"

RANDOM_SEED = 42


def normalize_target(target: str) -> str:
    """Generalize common paths and secrets."""
    if not target:
        return "<UNKNOWN>"

    target = target.lower().strip()

    replacements = {
        ".env.production": ".env",
        ".env.local": ".env",
        ".env.development": ".env",
        "~/.aws/credentials": "<AWS_CREDENTIALS>",
        "~/.ssh/id_rsa": "<SSH_PRIVATE_KEY>",
        "id_rsa": "<SSH_PRIVATE_KEY>",
    }

    for old, new in replacements.items():
        target = target.replace(old, new)

    if target.startswith("src/"):
        return "src/<SOURCE_FILE>"

    if target.startswith("tests/"):
        return "tests/<TEST_FILE>"

    return target


def create_dedup_key(record: dict) -> str:
    """Generate a semantic deduplication key."""

    inp = record["input"]
    out = record["output"]

    fields = [
        inp.get("requested_action", ""),
        normalize_target(inp.get("target", "")),
        inp.get("project_type", ""),
        out.get("decision", ""),
        ",".join(sorted(out.get("violated_policies", []))),
    ]

    return hashlib.md5("|".join(fields).encode()).hexdigest()


def is_valid(record: dict) -> bool:
    """Validate required fields."""

    try:
        decision = record["output"]["decision"]

        return (
            "instruction" in record
            and "input" in record
            and "output" in record
            and decision in {"ALLOW", "WARN", "BLOCK"}
        )

    except KeyError:
        return False


def load_jsonl(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            data = json.load(f)
            if isinstance(data, list):
                yield from data
                return

        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {line_number}")


records = []
seen = set()

for record in load_jsonl(INPUT_FILE):

    if not is_valid(record):
        continue

    key = create_dedup_key(record)

    if key in seen:
        continue

    seen.add(key)
    records.append(record)

print(f"Valid unique records: {len(records):,}")

distribution = Counter(
    record["output"]["decision"]
    for record in records
)

print("\nDecision distribution:")

for decision, count in distribution.items():
    percentage = count / len(records) * 100

    print(f"{decision:6} {count:8,} ({percentage:.2f}%)")

labels = [
    record["output"]["decision"]
    for record in records
]

train_records, temp_records = train_test_split(
    records,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=labels,
)

temp_labels = [
    record["output"]["decision"]
    for record in temp_records
]

validation_records, test_records = train_test_split(
    temp_records,
    test_size=0.5,
    random_state=RANDOM_SEED,
    stratify=temp_labels,
)


def save_jsonl(filepath: str, data: list):
    with open(filepath, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


save_jsonl(TRAIN_FILE, train_records)
save_jsonl(VALID_FILE, validation_records)
save_jsonl(TEST_FILE, test_records)

print("\nSaved datasets:")
print(f"Train:      {len(train_records):,}")
print(f"Validation: {len(validation_records):,}")
print(f"Test:       {len(test_records):,}")
