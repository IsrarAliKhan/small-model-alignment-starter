"""From-scratch SFT loop (no HF Trainer): response-only loss, grad accumulation,
fp16 autocast with fp32 master weights + GradScaler, clipping, cosine schedule,
validation loss, and best/final checkpoints.

Metrics go to {output_dir}/metrics.jsonl (one JSON object per log/eval step).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from alignment.core import batch_response_logprobs, build_sft_batch, sft_loss_from_batch
from alignment.utils import append_jsonl, autocast, format_prompt, pick_device, read_jsonl, save_checkpoint, set_seed, write_json


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--train_file", default="data/processed/sft_train.jsonl")
    ap.add_argument("--val_file", default="data/processed/sft_val.jsonl")
    ap.add_argument("--output_dir", default="outputs/sft")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None, help="cap optimizer steps (smoke tests)")
    ap.add_argument("--max_train_examples", type=int, default=None)
    ap.add_argument("--max_val_examples", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=100, help="optimizer steps")
    ap.add_argument("--log_every", type=int, default=10, help="optimizer steps")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def param_groups(model, weight_decay):
    """No weight decay on biases and norm weights (1-D params)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_logp, total_tokens = 0.0, 0
    for batch in loader:
        with autocast(device):
            logps, counts = batch_response_logprobs(model, batch)
        total_logp += logps.sum().item()
        total_tokens += counts.sum().item()
    model.train()
    return -total_logp / max(total_tokens, 1)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(vars(args), out / "config.json")
    metrics_path = out / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # fp32 master weights; autocast does fp16 compute on CUDA.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()

    train_rows = read_jsonl(args.train_file)[: args.max_train_examples]
    val_rows = read_jsonl(args.val_file)[: args.max_val_examples]

    def collate(rows):
        return build_sft_batch(
            tokenizer, [format_prompt(r["prompt"]) for r in rows], [r["response"] for r in rows],
            args.max_length, pad_to_max_length=False,
        )

    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_rows, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    total_steps = args.epochs * (len(train_loader) // args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(args.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    print(f"train={len(train_rows)} val={len(val_rows)} optimizer_steps={total_steps} device={device}")

    step, best_val, running, start = 0, float("inf"), [], time.time()
    while step < total_steps:
        optimizer.zero_grad(set_to_none=True)  # drop leftover partial accumulation from the previous epoch
        running = []
        for micro, batch in enumerate(train_loader):
            with autocast(device):
                loss = sft_loss_from_batch(model, batch)
            scaler.scale(loss / args.grad_accum).backward()
            running.append(loss.item())
            if (micro + 1) % args.grad_accum:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == total_steps:
                record = {"split": "train", "step": step, "loss": sum(running) / len(running),
                          "lr": scheduler.get_last_lr()[0], "grad_norm": grad_norm,
                          "elapsed_min": (time.time() - start) / 60}
                append_jsonl(record, metrics_path)
                print(record, flush=True)
            running = []

            if step % args.eval_every == 0 or step == total_steps:
                val_loss = evaluate(model, val_loader, device)
                record = {"split": "val", "step": step, "loss": val_loss}
                append_jsonl(record, metrics_path)
                print(record, flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(model, tokenizer, out / "best", step=step, val_loss=val_loss)
            if step >= total_steps:
                break

    save_checkpoint(model, tokenizer, out / "final", step=step, best_val_loss=best_val)
    print(f"done: steps={step} best_val_loss={best_val:.4f} -> {out}")


if __name__ == "__main__":
    main()
