"""Small shared helpers: prompt template, IO, seeding, devices, checkpoints."""

from __future__ import annotations

import contextlib
import json
import random
from pathlib import Path

import numpy as np
import torch

# Qwen ChatML markup without a system turn. Used identically for SFT, DPO,
# and generation from the Qwen base/SFT/DPO checkpoints. The response is
# followed by the tokenizer EOS (<|endoftext|> for Qwen2.5 base).
PROMPT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"


def format_prompt(instruction):
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)  # atomic: a disconnect never leaves a half-written file


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(row, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device):
    """fp16 autocast on CUDA (T4 has no fast bf16); full fp32 elsewhere."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def save_checkpoint(model, tokenizer, path, **info):
    path = Path(path)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    write_json(info, path / "checkpoint_info.json")
