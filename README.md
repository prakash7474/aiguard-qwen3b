# AI Guard CLI — Qwen3B

Fine-tuned security guard for AI coding agents. Classifies agent requests as **ALLOW**, **WARN**, or **BLOCK** based on operational risk.

Built on [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) using LoRA + 4-bit QLoRA.

## Pipeline

```
synthetic_agent_dataset.json  →  prepare_dataset.py  →  train/val/test .jsonl
                                                              ↓
                                                       convert_chatml.py
                                                              ↓
                                                      train_chatml.jsonl
                                                              ↓
                                                       train_qlora.py  →  aiguard-qwen3b/  (LoRA adapter)
                                                              ↓
                                                  ┌─────────────────┐
                                                  │   test_model.py  │  quick smoke test
                                                  │  run_evaluation.py│  full eval on 150 held-out cases
                                                  └─────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `prepare_dataset.py` | Deduplicates, normalizes targets, stratifies split into train/validation/test |
| `verify.py` | Inspects random samples and verifies decision distribution meets targets |
| `convert_chatml.py` | Converts raw JSONL to ChatML format for SFT |
| `train_qlora.py` | Trains LoRA adapter with 4-bit QLoRA on Qwen2.5-3B-Instruct |
| `test_model.py` | Quick smoke test on 5 example cases |
| `create_evaluation.py` | Generates 150 held-out evaluation examples with unseen agents/projects |
| `run_evaluation.py` | Runs full evaluation, prints accuracy + confusion matrix |
| `explore.py` | Exploratory analysis of the synthetic dataset |

## Usage

### 1. Prepare dataset

```bash
python prepare_dataset.py
python verify.py
python convert_chatml.py
```

### 2. Train

```bash
python train_qlora.py
```

Requires a GPU with ~8GB+ VRAM (QLoRA 4-bit).

### 3. Test & evaluate

```bash
python test_model.py
python create_evaluation.py
python run_evaluation.py
```

## Model

The LoRA adapter is saved in `aiguard-qwen3b/`. See [its README](aiguard-qwen3b/README.md) for model card details.

**Adapter config:** rank=8, alpha=16, dropout=0.05, targets q_proj/k_proj/v_proj/o_proj.

**Training:** 1 epoch, batch size 1, gradient accumulation 8, learning rate 2e-4, bf16.

## Requirements

- Python 3.10+
- PyTorch 2.12+
- transformers, peft, trl, datasets, bitsandbytes, scikit-learn
