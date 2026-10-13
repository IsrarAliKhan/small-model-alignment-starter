"""Core SFT/DPO math: batching, response-only log-probabilities, and losses.

Conventions (shared by training scripts and ``submission_adapter.py``):

* Prompt and response are tokenized separately (no special tokens added), the
  tokenizer's EOS id is appended to the response, and the concatenation is
  right-truncated to ``max_length``. A truncated response may therefore lose
  its EOS token.
* Labels are un-shifted copies of ``input_ids`` on response positions and
  ``-100`` on prompt and padding positions. Sequences are right-padded.
* Token ``t`` is scored by the logits at position ``t - 1``.
* SFT loss reduction: mean negative log-probability over all unmasked
  response tokens in the batch (token-level mean).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _pad_token_id(tokenizer):
    for token_id in (getattr(tokenizer, "pad_token_id", None), getattr(tokenizer, "eos_token_id", None)):
        if token_id is not None:
            return token_id
    return 0


def _model_device(model):
    return next(model.parameters()).device


def build_sft_batch(tokenizer, prompts, responses, max_length, pad_to_max_length=True):
    """Tokenize prompt/response pairs into a right-padded, response-masked LM batch.

    ``pad_to_max_length=False`` pads only to the longest row (used in training
    to save compute); values on non-padding positions are identical either way.
    """

    prompts, responses = list(prompts), list(responses)
    if len(prompts) != len(responses):
        raise ValueError(f"got {len(prompts)} prompts but {len(responses)} responses")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    rows = []
    for prompt, response in zip(prompts, responses):
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        response_ids = list(tokenizer.encode(response, add_special_tokens=False))
        if eos_id is not None:
            response_ids.append(eos_id)
        input_ids = (prompt_ids + response_ids)[:max_length]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
        rows.append((input_ids, labels))

    width = max_length if pad_to_max_length else max((len(ids) for ids, _ in rows), default=0)
    pad_id = _pad_token_id(tokenizer)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, row_labels) in enumerate(rows):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
        labels[row, : len(row_labels)] = torch.tensor(row_labels, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def batch_response_logprobs(model, batch):
    """Return (summed response-token log-probs, response-token counts), one per row."""

    device = _model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"][:, 1:].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    mask = labels.ne(IGNORE_INDEX)
    targets = labels.masked_fill(~mask, 0).unsqueeze(-1)
    # log_softmax(x)[y] = x[y] - logsumexp(x); avoids materializing a full log-softmax copy.
    token_logps = torch.gather(logits, -1, targets).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    token_logps = torch.where(mask, token_logps, torch.zeros_like(token_logps))
    return token_logps.sum(-1), mask.sum(-1)


def sft_loss_from_batch(model, batch):
    """Token-level mean negative log-likelihood over unmasked response tokens."""

    logps, counts = batch_response_logprobs(model, batch)
    return -logps.sum() / counts.sum().clamp_min(1)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length, pad_to_max_length=True):
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length, pad_to_max_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length, pad_to_max_length),
    }


def dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta):
    """Return (mean DPO loss, chosen implicit rewards, rejected implicit rewards).

    loss = -log sigmoid(beta * [(pi_c - pi_r) - (ref_c - ref_r)])
    rewards r = beta * (pi - ref), detached (diagnostics only).
    """

    logits = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    loss = -F.logsigmoid(logits).mean()
    chosen_rewards = (beta * (policy_chosen - reference_chosen)).detach()
    rejected_rewards = (beta * (policy_rejected - reference_rejected)).detach()
    return loss, chosen_rewards, rejected_rewards


def dpo_logps(policy_model, reference_model, tokenizer, prompts, chosen, rejected, max_length, pad_to_max_length=True):
    """Policy/reference summed response log-probs for chosen and rejected rows.

    Chosen and rejected rows go through one forward pass per model. The
    reference runs under ``no_grad`` and its outputs are moved to the policy's
    device (the reference may live on another GPU).
    """

    prompts = list(prompts)
    n = len(prompts)
    batch = build_sft_batch(tokenizer, prompts + prompts, list(chosen) + list(rejected), max_length, pad_to_max_length)
    policy_logps, _ = batch_response_logprobs(policy_model, batch)
    with torch.no_grad():
        reference_logps, _ = batch_response_logprobs(reference_model, batch)
    reference_logps = reference_logps.to(policy_logps.device)
    return policy_logps[:n], policy_logps[n:], reference_logps[:n], reference_logps[n:]
