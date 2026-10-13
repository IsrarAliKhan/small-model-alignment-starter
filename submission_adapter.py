"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.
"""

from __future__ import annotations

from alignment import core


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return core.build_sft_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    batch = core.build_sft_batch(tokenizer, prompts, responses, max_length)
    return core.batch_response_logprobs(model, batch)[0]


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    batch = core.build_sft_batch(tokenizer, prompts, responses, max_length)
    return core.sft_loss_from_batch(model, batch)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return core.build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return core.dpo_loss_from_logps(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta
    )[0]


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    logps = core.dpo_logps(
        policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, max_length
    )
    return core.dpo_loss_from_logps(*logps, beta)[0]
