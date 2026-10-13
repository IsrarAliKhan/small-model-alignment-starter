"""From-scratch DPO loop (no TRL). Same code for HH-RLHF and UltraFeedback runs.

Policy: trainable fp32 copy of the SFT checkpoint on cuda:0 (fp16 autocast).
Reference: frozen copy of the same checkpoint on cuda:1 when available. It is
kept in fp32 and run under the same autocast so that at step 0 policy and
reference produce identical log-probs (rewards exactly 0).

Validation reference log-probs are computed once and cached (the reference
never changes). Best checkpoint = highest validation reward accuracy.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup

from alignment.core import batch_response_logprobs, build_sft_batch, dpo_loss_from_logps, dpo_logps
from alignment.utils import append_jsonl, autocast, format_prompt, pick_device, read_jsonl, save_checkpoint, set_seed, write_json


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_checkpoint", required=True, help="initial policy AND frozen reference")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--warmup_steps", type=int, default=0, help="0 = fixed LR (guide recipe)")
    ap.add_argument("--optimizer", choices=["adamw", "rmsprop"], default="adamw",
                    help="adamw: bias-corrected first step ~lr; rmsprop (no warmup) first step is ~10x lr")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1, help="pairs per micro-batch")
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--max_train_examples", type=int, default=None)
    ap.add_argument("--max_val_examples", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reference_device", default="auto", help="auto: cuda:1 if present, else policy device")
    return ap.parse_args()


def summarize(losses, chosen_rewards, rejected_rewards):
    chosen, rejected = torch.cat(chosen_rewards), torch.cat(rejected_rewards)
    return {
        "loss": sum(losses) / len(losses),
        "reward_accuracy": (chosen > rejected).float().mean().item(),  # strict: ties count as wrong
        "reward_margin": (chosen - rejected).mean().item(),
        "chosen_reward": chosen.mean().item(),
        "rejected_reward": rejected.mean().item(),
    }


def make_batch(tokenizer, rows, max_length):
    prompts = [format_prompt(r["prompt"]) for r in rows]
    return build_sft_batch(tokenizer, prompts + prompts, [r["chosen"] for r in rows] + [r["rejected"] for r in rows],
                           max_length, pad_to_max_length=False)


@torch.no_grad()
def cached_reference_logps(reference, tokenizer, val_batches, max_length, device):
    cache = []
    for rows in val_batches:
        with autocast(device):
            logps, _ = batch_response_logprobs(reference, make_batch(tokenizer, rows, max_length))
        cache.append(logps.float().cpu())
    return cache


@torch.no_grad()
def evaluate(policy, tokenizer, val_batches, ref_cache, args, device):
    policy.eval()
    losses, chosen_rewards, rejected_rewards = [], [], []
    for rows, ref in zip(val_batches, ref_cache):
        with autocast(device):
            pol, _ = batch_response_logprobs(policy, make_batch(tokenizer, rows, args.max_length))
        pol, ref, n = pol.float().cpu(), ref, len(rows)
        loss, cr, rr = dpo_loss_from_logps(pol[:n], pol[n:], ref[:n], ref[n:], args.beta)
        losses.append(loss.item())
        chosen_rewards.append(cr)
        rejected_rewards.append(rr)
    policy.train()
    return summarize(losses, chosen_rewards, rejected_rewards)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(vars(args), out / "config.json")
    metrics_path = out / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    set_seed(args.seed)

    device = pick_device(args.device)
    if args.reference_device == "auto":
        ref_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else device
    else:
        ref_device = torch.device(args.reference_device)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    policy = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, torch_dtype=torch.float32).to(device)
    reference = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, torch_dtype=torch.float32).to(ref_device)
    reference.eval().requires_grad_(False)
    policy.train()

    train_rows = read_jsonl(args.train_file)[: args.max_train_examples]
    val_rows = read_jsonl(args.val_file)[: args.max_val_examples]
    val_batches = [val_rows[i : i + args.eval_batch_size] for i in range(0, len(val_rows), args.eval_batch_size)]
    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=lambda rows: rows,
                              generator=torch.Generator().manual_seed(args.seed))

    total_steps = args.epochs * (len(train_loader) // args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(policy.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    print(f"train={len(train_rows)} val={len(val_rows)} optimizer_steps={total_steps} "
          f"policy={device} reference={ref_device}")

    print("caching validation reference log-probs ...", flush=True)
    ref_cache = cached_reference_logps(reference, tokenizer, val_batches, args.max_length, ref_device)
    record = {"split": "val", "step": 0, **evaluate(policy, tokenizer, val_batches, ref_cache, args, device)}
    append_jsonl(record, metrics_path)
    print(record, flush=True)  # sanity: loss ~= log 2, accuracy 0, margin 0

    step, best_acc, start = 0, -1.0, time.time()
    while step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        losses, chosen_rewards, rejected_rewards = [], [], []
        for micro, rows in enumerate(train_loader):
            prompts = [format_prompt(r["prompt"]) for r in rows]
            with autocast(device):
                logps = dpo_logps(policy, reference, tokenizer, prompts, [r["chosen"] for r in rows],
                                  [r["rejected"] for r in rows], args.max_length, pad_to_max_length=False)
                loss, cr, rr = dpo_loss_from_logps(*logps, args.beta)
            scaler.scale(loss / args.grad_accum).backward()
            losses.append(loss.item())
            chosen_rewards.append(cr.float().cpu())
            rejected_rewards.append(rr.float().cpu())
            if (micro + 1) % args.grad_accum:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm).item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == total_steps:
                record = {"split": "train", "step": step, **summarize(losses, chosen_rewards, rejected_rewards),
                          "grad_norm": grad_norm, "lr": scheduler.get_last_lr()[0],
                          "elapsed_min": (time.time() - start) / 60}
                append_jsonl(record, metrics_path)
                print(record, flush=True)
                losses, chosen_rewards, rejected_rewards = [], [], []

            if step % args.eval_every == 0 or step == total_steps:
                metrics = evaluate(policy, tokenizer, val_batches, ref_cache, args, device)
                append_jsonl({"split": "val", "step": step, **metrics}, metrics_path)
                print({"split": "val", "step": step, **metrics}, flush=True)
                if metrics["reward_accuracy"] > best_acc:
                    best_acc = metrics["reward_accuracy"]
                    save_checkpoint(policy, tokenizer, out / "best", step=step, **metrics)
            if step >= total_steps:
                break

    save_checkpoint(policy, tokenizer, out / "final", step=step, best_val_reward_accuracy=best_acc)
    print(f"done: steps={step} best_val_reward_accuracy={best_acc:.4f} -> {out}")


if __name__ == "__main__":
    main()
