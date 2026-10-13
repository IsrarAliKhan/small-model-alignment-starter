"""Stricter checks than the public smoke tests: truncation, ordering, numerics, padding."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import submission_adapter as adapter
from alignment import core
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def manual_logp(model, tokenizer, prompt, response, max_length):
    """Reference implementation: score each kept response token one by one."""
    prompt_ids = tokenizer.encode(prompt)
    ids = (prompt_ids + tokenizer.encode(response) + [tokenizer.eos_token_id])[:max_length]
    logits = model(torch.tensor([ids])).logits[0].float()
    logps = F.log_softmax(logits, -1)
    return sum((logps[t - 1, ids[t]] for t in range(max(len(prompt_ids), 1), len(ids))), torch.tensor(0.0))


def tiny_qwen(vocab_size):
    """Randomly initialized 2-layer transformer: position-dependent, unlike ToyCausalLM."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(config).eval()


def test_truncation_drops_eos_and_keeps_exact_length():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab: "], ["hello"], max_length=6)
    assert batch["input_ids"].shape == (1, 6)
    assert batch["input_ids"][0].tolist() == tok.encode("ab: he")
    assert batch["labels"][0].tolist() == [-100] * 4 + tok.encode("he")
    assert tok.eos_token_id not in batch["labels"][0].tolist()
    assert batch["attention_mask"][0].tolist() == [1] * 6


def test_eos_kept_when_it_exactly_fits():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["a"], ["bc"], max_length=4)
    assert batch["labels"][0].tolist() == [-100] + tok.encode("bc") + [tok.eos_token_id]


def test_prompt_longer_than_max_length_gives_zero_logp_and_finite_loss():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    batch = adapter.build_sft_batch(tok, ["long prompt here"], ["x"], max_length=5)
    assert batch["labels"].eq(-100).all()
    logps = adapter.compute_response_logprobs(model, tok, ["long prompt here"], ["x"], max_length=5)
    assert logps.tolist() == [0.0]
    loss = adapter.compute_sft_loss(model, tok, ["long prompt here"], ["x"], max_length=5)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("max_length", [3, 7, 12, 40])
def test_logprobs_match_manual_scoring_on_position_dependent_model(max_length):
    tok = ToyTokenizer()
    model = tiny_qwen(tok.vocab_size)
    prompts = ["Q: hi ", "Question: what is two plus two? ", "x"]
    responses = ["yes", "four, obviously", "a much longer response than the others"]
    with torch.no_grad():
        got = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length)
        want = torch.stack([torch.as_tensor(manual_logp(model, tok, p, r, max_length)) for p, r in zip(prompts, responses)])
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_row_order_preserved_and_padding_mode_invariant():
    tok = ToyTokenizer()
    model = tiny_qwen(tok.vocab_size)
    prompts = ["A: ", "Bee: ", "C: "]
    responses = ["short", "medium one", "z"]
    with torch.no_grad():
        forward = core.batch_response_logprobs(model, core.build_sft_batch(tok, prompts, responses, 32))[0]
        dynamic = core.batch_response_logprobs(
            model, core.build_sft_batch(tok, prompts, responses, 32, pad_to_max_length=False)
        )[0]
        reversed_ = adapter.compute_response_logprobs(model, tok, prompts[::-1], responses[::-1], 32)
    torch.testing.assert_close(forward, dynamic)
    torch.testing.assert_close(forward, reversed_.flip(0))


def test_tokenizer_without_pad_or_eos():
    tok = ToyTokenizer()
    tok.pad_token_id = None
    batch = adapter.build_sft_batch(tok, ["a", "bbb"], ["c", "d"], max_length=8)
    assert batch["input_ids"][0, 3:].eq(tok.eos_token_id).all()  # pad falls back to EOS id
    assert batch["attention_mask"][0].tolist() == [1, 1, 1] + [0] * 5

    tok.eos_token_id = None
    batch = adapter.build_sft_batch(tok, ["a"], ["c"], max_length=8)
    assert batch["labels"][0].ne(-100).sum() == 1  # no EOS appended


def test_sft_loss_is_token_mean():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.3)
    prompts, responses = ["P: ", "Q: "], ["a", "bcdef"]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, 16)
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, 16)
    torch.testing.assert_close(loss, -logps.sum() / (2 + 6))


def test_dpo_loss_matches_formula_and_beta_scaling():
    pc, pr = torch.tensor([-1.0, -3.0]), torch.tensor([-2.0, -1.0])
    rc, rr = torch.tensor([-1.5, -2.0]), torch.tensor([-1.5, -2.0])
    for beta in (0.1, 0.5):
        want = -F.logsigmoid(beta * ((pc - pr) - (rc - rr))).mean()
        torch.testing.assert_close(adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta), want)


def test_dpo_batch_rows_match_sft_batches():
    tok = ToyTokenizer()
    batch = adapter.build_dpo_batch(tok, ["P: ", "Q: "], ["good", "yes"], ["bad", "nope"], 10)
    for side, responses in (("chosen", ["good", "yes"]), ("rejected", ["bad", "nope"])):
        expected = adapter.build_sft_batch(tok, ["P: ", "Q: "], responses, 10)
        for key in expected:
            assert torch.equal(batch[side][key], expected[key])


def test_dpo_gradient_step_increases_preference_and_leaves_reference_untouched():
    tok = ToyTokenizer()
    policy, reference = tiny_qwen(tok.vocab_size), tiny_qwen(tok.vocab_size)
    reference.load_state_dict(policy.state_dict())
    ref_before = {k: v.clone() for k, v in reference.state_dict().items()}
    prompts, chosen, rejected = ["Pick: ", "Say: "], ["good", "yes"], ["bad", "no"]
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)

    loss0 = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=0.5, max_length=16)
    torch.testing.assert_close(loss0, torch.log(torch.tensor(2.0)), rtol=1e-5, atol=1e-5)
    loss0.backward()
    assert all(p.grad is None for p in reference.parameters())
    optimizer.step()

    with torch.no_grad():
        loss1 = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=0.5, max_length=16)
    assert loss1 < loss0
    assert all(torch.equal(v, ref_before[k]) for k, v in reference.state_dict().items())
