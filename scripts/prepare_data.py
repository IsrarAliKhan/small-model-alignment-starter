"""Download and convert SFT / preference / AlpacaEval data into fixed local files.

Outputs (all single-turn, raw instruction text; the prompt template is applied
at training time):
  sft: {out_dir}/sft_{train,val}.jsonl   rows: id, prompt, response
  hh:  {out_dir}/hh_{train,val}.jsonl    rows: id, prompt, chosen, rejected
  uf:  {out_dir}/uf_{train,val}.jsonl    rows: id, prompt, chosen, rejected
  alpaca: eval_sets/alpaca_eval_300_seed42.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from alignment.utils import format_prompt, write_json, write_jsonl

HH_HUMAN, HH_ASSISTANT = "\n\nHuman:", "\n\nAssistant:"


def parse_hh(chosen_text, rejected_text):
    """Split a single-turn HH-RLHF pair into (prompt, chosen, rejected); None if multi-turn."""
    for text in (chosen_text, rejected_text):
        if text.count(HH_HUMAN) != 1 or text.count(HH_ASSISTANT) != 1:
            return None
    c_split, r_split = chosen_text.rfind(HH_ASSISTANT), rejected_text.rfind(HH_ASSISTANT)
    if chosen_text[:c_split] != rejected_text[:r_split]:
        return None
    prompt = chosen_text[:c_split].replace(HH_HUMAN, "", 1).strip()
    chosen = chosen_text[c_split + len(HH_ASSISTANT):].strip()
    rejected = rejected_text[r_split + len(HH_ASSISTANT):].strip()
    return prompt, chosen, rejected


def sft_rows(split, seed):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split).shuffle(seed=seed)
    for ex in ds:
        msgs = ex["messages"]
        if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
            yield {"id": ex["prompt_id"], "prompt": msgs[0]["content"], "response": msgs[1]["content"]}


def hh_rows(split, seed):
    ds = load_dataset("Anthropic/hh-rlhf", split=split).shuffle(seed=seed)
    for i, ex in enumerate(ds):
        parsed = parse_hh(ex["chosen"], ex["rejected"])
        if parsed:
            yield {"id": f"hh-{split}-{i}", "prompt": parsed[0], "chosen": parsed[1], "rejected": parsed[2]}


def uf_rows(split, seed):
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split).shuffle(seed=seed)
    for ex in ds:
        if len(ex["chosen"]) != 2 or len(ex["rejected"]) != 2 or ex["score_chosen"] <= ex["score_rejected"]:
            continue  # multi-turn or tied scores
        yield {"id": ex["prompt_id"], "prompt": ex["prompt"],
               "chosen": ex["chosen"][-1]["content"], "rejected": ex["rejected"][-1]["content"]}


def collect(rows, n, tokenizer, max_prompt_tokens):
    """Take the first n rows with a non-empty, distinct answer and a short enough prompt."""
    kept, seen = [], set()
    for row in rows:
        answers = [row[k] for k in ("response", "chosen", "rejected") if k in row]
        if not row["prompt"].strip() or not all(a.strip() for a in answers) or len(set(answers)) != len(answers):
            continue
        if row["prompt"] in seen:
            continue
        if len(tokenizer.encode(format_prompt(row["prompt"]), add_special_tokens=False)) > max_prompt_tokens:
            continue
        seen.add(row["prompt"])
        kept.append(row)
        if len(kept) == n:
            return kept
    print(f"warning: only {len(kept)} rows available (wanted {n})")
    return kept


def make_alpaca_subset(n, seed, path):
    rows = json.load(open(hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")))
    picked = random.Random(seed).sample(range(len(rows)), n)
    subset = [{"id": f"alpaca-eval-{seed}-{i:04d}", "instruction": rows[i]["instruction"], "dataset": "alpaca_eval"}
              for i in picked]
    write_json(subset, path)
    print(f"wrote {len(subset)} prompts to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sft", "hh", "uf", "alpaca"], required=True)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--n_train", type=int, default=None, help="default: 30000 for sft, 15000 for hh/uf")
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--max_prompt_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--alpaca_n", type=int, default=300)
    ap.add_argument("--alpaca_path", default="eval_sets/alpaca_eval_300_seed42.json")
    args = ap.parse_args()

    if args.task == "alpaca":
        return make_alpaca_subset(args.alpaca_n, args.seed, args.alpaca_path)

    source, train_split, val_split, n_train = {
        "sft": (sft_rows, "train_sft", "test_sft", 30000),
        "hh": (hh_rows, "train", "test", 15000),
        "uf": (uf_rows, "train_prefs", "test_prefs", 15000),
    }[args.task]
    n_train = args.n_train or n_train
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    out = Path(args.out_dir)
    for split, name, n in ((train_split, "train", n_train), (val_split, "val", args.n_val)):
        rows = collect(source(split, args.seed), n, tokenizer, args.max_prompt_tokens)
        write_jsonl(rows, out / f"{args.task}_{name}.jsonl")
        print(f"wrote {len(rows)} rows to {out / f'{args.task}_{name}.jsonl'}")


if __name__ == "__main__":
    main()
