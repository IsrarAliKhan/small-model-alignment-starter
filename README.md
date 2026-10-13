# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B

SFT, two from-scratch DPO runs (HH-RLHF, UltraFeedback), a TRL DPOTrainer + LoRA persona run, generation, and blind pairwise judging with Kaggle models. See `docs/assignment_manual.pdf` for the full assignment.

## Layout

```text
submission_adapter.py        grading interface -> thin wrappers over alignment/core.py
alignment/core.py            batching, response-only log-probs, SFT loss, DPO loss (the graded math)
alignment/utils.py           prompt template, IO, seeding, device/autocast, checkpoint saving
scripts/prepare_data.py      download + filter + fix train/val subsets (sft | hh | uf | alpaca)
scripts/train_sft.py         from-scratch SFT loop (Task 3)
scripts/train_dpo.py         from-scratch DPO loop, used for both HH and UF (Task 4)
scripts/train_lora_dpo.py    TRL DPOTrainer + LoRA persona run (Task 5)
scripts/generate.py          resumable batched greedy generation (Tasks 2, 6)
scripts/build_pairs.py       blind A/B pair files with hidden model identities (Task 6)
scripts/judge_pairwise_kaggle.py  resumable batched judging via kaggle_benchmarks (Task 7)
scripts/plot_metrics.py      train/val curves from metrics.jsonl
notebooks/01..06_*.ipynb     Kaggle notebooks, one per stage
eval_sets/                   fixed eval prompts: HH synthetic (300), AlpacaEval seed 42 (300), persona style (40)
tests/                       public smoke tests + stricter core/judge tests (offline, CPU)
```

## Kaggle workflow

Each notebook clones this repo, installs dependencies without touching Kaggle's PyTorch, and writes everything under `/kaggle/working/small-model-alignment-starter/outputs/<run>/`. After each notebook finishes, click **Save Version** and attach its output as an input to the next notebook. Later notebooks find earlier checkpoints automatically with a glob over `/kaggle/input`.

| Notebook | Accelerator | What it does | Time |
|---|---|---|---|
| `01_setup_baseline_sft` | T4 x2 | env check, tests, base-model generations, SFT data + training | ~2 h |
| `02_dpo_hh` | T4 x2 | HH-RLHF DPO from `outputs/sft/best` | ~2-2.5 h |
| `03_dpo_uf` | T4 x2 | UltraFeedback DPO from `outputs/sft/best` | ~2-2.5 h |
| `04_lora_persona` | T4 | TRL + LoRA on SmolLM2-360M-Instruct | ~15 min |
| `05_generate_eval_outputs` | T4 x2 | SFT/DPO/persona generations + blind pair files | ~0.5-1 h |
| `06_judge_kaggle` | none (utility benchmark) | Gemini judging, summaries | ~0.5 h |

If the repo is private, make it public or clone it with a token from Kaggle Secrets.

## Local development

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch -r requirements.txt
uv pip install --python .venv/bin/python -e . --no-deps
PYTHONPATH=. .venv/bin/python -m pytest -q tests
```

Every script runs on CPU/MPS for tiny smoke runs, using flags such as `--max_train_examples`, `--max_steps` and `--limit`. Autocast and GradScaler are only enabled on CUDA. Judging can be tested offline with `--backend dry_run`.

## Implementation choices (report these)

- **Prompt format** (SFT, DPO, generation for Qwen models): `<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n`, with no system turn. Each response ends with the tokenizer EOS (`<|endoftext|>`). Generation stops at EOS or `<|im_end|>`.
- **Batching**: prompt and response are tokenized separately. EOS is appended to the response, then the full sequence is right-truncated to `max_length`, so EOS can be cut off. Padding is on the right. Labels are `-100` on prompt and padding positions. Training pads to the longest row; the adapter pads to `max_length`. The values are identical either way.
- **SFT loss reduction**: token-level mean NLL over all unmasked response tokens in the micro-batch.
- **Data filters**: single-turn only (UltraChat SFT: first user/assistant turn of each conversation), non-empty and distinct answers, deduplicated prompts, formatted prompt ≤ 256 tokens. UltraFeedback pairs with tied scores are dropped. Seed 42.
- **SFT** (single GPU, cuda:0): fp32 weights, fp16 autocast + GradScaler, AdamW (wd 0.1, none on 1-D params), LR 2e-5, cosine schedule with 3% warmup, clip 1.0, batch 2 × 16, 1 epoch. `best` is chosen by validation loss.
- **DPO**: β = 0.1, LR 1e-6 fixed, AdamW (wd 0), clip 1.0, batch 1 pair × 32. The policy is on cuda:0. The frozen reference is on cuda:1, also in fp32 under the same autocast, so step-0 rewards are exactly 0. Validation reference log-probs are cached. `best` is chosen by validation reward accuracy.
- **LoRA persona run** (`trl==0.24.0`, because trl 1.x needs Triton for its fused log-prob kernel): LoRA r 16, alpha 32, dropout 0.05 on all attention and MLP projections, LR 5e-5 with cosine schedule and 10% warmup, β 0.1, batch 4 × 2, 3 epochs, 200 train / 20 val. TRL uses the base model with the adapter disabled as the reference.
- **Decoding**: greedy, repetition_penalty 1.1, max_new_tokens 512. These settings are identical for every model in a comparison.
- **Judging**: 5 pairs per request, sleep 5 s, 3 attempts per batch, resumable. Unresolved batches go to `*_failures.jsonl`; summaries go to `*_summary.json`, which includes A/B position counts.
