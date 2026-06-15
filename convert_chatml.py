import json

SYSTEM_PROMPT = """
You are AI Guard CLI, a security assistant for AI coding agents.

Analyze requests and return JSON with:
- decision
- risk_score
- severity
- reason
- violated_policies
- recommendations
- audit_log
""".strip()

SPLITS = {
    "train": ("train.jsonl", "train_chatml.jsonl"),
    "validation": ("validation.jsonl", "validation_chatml.jsonl"),
    "test": ("test.jsonl", "test_chatml.jsonl"),
}

for input_name, output_name in SPLITS.values():
    with open(input_name, "r", encoding="utf-8") as fin, \
         open(output_name, "w", encoding="utf-8") as fout:

        count = 0
        for line in fin:
            record = json.loads(line)

            user_content = f"""
AGENT: {record['input'].get('agent', 'unknown')}
ACTION: {record['input'].get('requested_action', '')}
TARGET: {record['input'].get('target', '')}
PROJECT_TYPE: {record['input'].get('project_type', '')}
""".strip()

            assistant_content = json.dumps(
                record["output"],
                ensure_ascii=False
            )

            chat = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }

            fout.write(json.dumps(chat, ensure_ascii=False) + "\n")
            count += 1

        print(f"Saved {count} records to {output_name}")
