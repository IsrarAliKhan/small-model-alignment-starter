"""Task 5: TRL DPOTrainer + PEFT/LoRA persona tuning on the Ramsay-style preference subset.

Pinned to trl==0.24.0: trl 1.x always uses a Triton fused log-prob kernel and crashes
where Triton is unavailable (e.g. macOS), so local smoke tests could not cover it.

Rows are converted to TRL's conversational format:
  prompt   = [{"role": "user", "content": ...}]
  chosen   = [{"role": "assistant", "content": ...}]
  rejected = [{"role": "assistant", "content": ...}]
With a PEFT model and ref_model=None, TRL uses the base model (adapter disabled)
as the frozen reference.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

# Single GPU: with 2 visible GPUs HF Trainer wraps the PEFT model in nn.DataParallel,
# which crashes ("Expected all tensors to be on the same device"). Must be set before torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from alignment.utils import read_json, set_seed, write_json, write_jsonl


def to_conversational(row):
    return {"prompt": [{"role": "user", "content": row["prompt"]}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}]}


def normalize_log(record):
    """TRL log keys -> the metrics.jsonl names used by the from-scratch scripts (for plot_metrics.py)."""
    split = "val" if "eval_loss" in record else "train"
    renamed = {"split": split}
    for key, value in record.items():
        key = key.removeprefix("eval_")
        renamed[{"rewards/accuracies": "reward_accuracy", "rewards/margins": "reward_margin"}.get(key, key)] = value
    return renamed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--data", default="data/persona_lora/ramsay_style_preferences.json")
    ap.add_argument("--output_dir", default="outputs/lora_persona")
    ap.add_argument("--val_size", type=int, default=20)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_prompt_length", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=-1, help="cap steps (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(vars(args), out / "config.json")
    set_seed(args.seed)

    rows = read_json(args.data)
    random.Random(args.seed).shuffle(rows)
    val_rows, train_rows = rows[: args.val_size], rows[args.val_size :]
    write_json(val_rows, out / "val_rows.json")
    train_ds = Dataset.from_list([to_conversational(r) for r in train_rows])
    val_ds = Dataset.from_list([to_conversational(r) for r in val_rows])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","), bias="none", task_type="CAUSAL_LM",
    )
    config = DPOConfig(
        output_dir=str(out / "trainer"), beta=args.beta, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
        max_length=args.max_length, max_prompt_length=args.max_prompt_length, lr_scheduler_type="cosine", warmup_ratio=0.1,
        logging_steps=5, eval_strategy="epoch", save_strategy="no", report_to="none", seed=args.seed,
        fp16=torch.cuda.is_available(), gradient_checkpointing=False,
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=config, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tokenizer, peft_config=peft_config)
    trainer.model.print_trainable_parameters()
    trainer.train()

    trainer.model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    write_jsonl([normalize_log(r) for r in trainer.state.log_history], out / "metrics.jsonl")
    print(f"saved adapter to {out / 'adapter'}")


if __name__ == "__main__":
    main()
