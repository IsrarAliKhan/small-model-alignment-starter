"""Batched greedy generation on a fixed eval set. Resumable: rows already in
--output are skipped, and the file is rewritten after every batch.

Output: JSON list of {id, instruction, output, generator, dataset}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.utils import format_prompt, pick_device, read_json, write_json


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or checkpoint dir (base model when --adapter is set)")
    ap.add_argument("--adapter", default=None, help="optional PEFT/LoRA adapter dir")
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--generator", required=True, help="model name stored in each row")
    ap.add_argument("--chat_template", action="store_true",
                    help="use the tokenizer's chat template (instruct models); default: alignment.utils.PROMPT_TEMPLATE")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (sanity check)")
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model)
    tokenizer.padding_side = "left"  # decoder-only batched generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    stop_ids = {tokenizer.eos_token_id}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
        stop_ids.add(im_end)

    def render(instruction):
        if args.chat_template:
            return tokenizer.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False,
                                                 add_generation_prompt=True)
        return format_prompt(instruction)

    rows = read_json(args.eval_set)[: args.limit]
    results = read_json(args.output) if Path(args.output).exists() else []
    done = {r["id"] for r in results}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to generate -> {args.output}")

    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i : i + args.batch_size]
        enc = tokenizer([render(r["instruction"]) for r in chunk], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=False, temperature=None, top_p=None, top_k=None,
                max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size, eos_token_id=sorted(stop_ids),
                pad_token_id=tokenizer.pad_token_id,
            )
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for row, text in zip(chunk, texts):
            results.append({"id": row["id"], "instruction": row["instruction"], "output": text.strip(),
                            "generator": args.generator, "dataset": row.get("dataset", Path(args.eval_set).stem)})
        order = {r["id"]: k for k, r in enumerate(rows)}
        results.sort(key=lambda r: order.get(r["id"], len(order)))
        write_json(results, args.output)
        print(f"{len(results)}/{len(rows)}", flush=True)

    write_json(vars(args), Path(args.output).with_suffix(".config.json"))


if __name__ == "__main__":
    main()
