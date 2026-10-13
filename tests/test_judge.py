"""Offline checks for judge reply parsing, A/B -> model mapping, and resume."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("judge", ROOT / "scripts/judge_pairwise_kaggle.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def test_parse_verdicts_handles_fences_case_and_missing_items():
    reply = '```json\n[{"item": 1, "winner": "a", "reason": "x"}, {"item": "3", "winner": "Tie"}, {"item": 4, "winner": "C"}]\n```'
    got = [v for v, _ in judge.parse_verdicts(reply, 4)]
    assert got == ["A", "unknown", "tie", "unknown"]


def test_parse_verdicts_raises_without_json():
    with pytest.raises(ValueError):
        judge.parse_verdicts("I think A is better.", 2)


def test_outcome_maps_position_back_to_model():
    pair = {"meta": {"A": "sft", "B": "dpo", "candidate": "dpo", "baseline": "sft"}}
    assert judge.outcome_for(pair, "B") == "win"
    assert judge.outcome_for(pair, "A") == "loss"
    assert judge.outcome_for(pair, "tie") == "tie"


def test_prompt_hides_model_identity():
    pair = {"instruction": "hi", "response_a": "x", "response_b": "y", "dataset": "alpaca_eval",
            "meta": {"A": "secret-sft", "B": "secret-dpo", "candidate": "secret-dpo", "baseline": "secret-sft"}}
    assert "secret" not in judge.build_prompt([pair], "general")


def test_dry_run_resume_judges_each_pair_once(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    with open(pairs, "w") as f:
        for i in range(7):
            f.write(json.dumps({"pair_id": f"p{i}", "dataset": "alpaca_eval", "instruction": "q", "response_a": "a",
                                "response_b": "b", "meta": {"A": "m1", "B": "m2", "candidate": "m1", "baseline": "m2"}}) + "\n")
    out = tmp_path / "judged.jsonl"
    cmd = [sys.executable, str(ROOT / "scripts/judge_pairwise_kaggle.py"), "--pairs_path", str(pairs),
           "--output_path", str(out), "--backend", "dry_run", "--sleep_seconds", "0", "--reuse_judgments"]
    subprocess.run(cmd + ["--limit", "4"], check=True, capture_output=True, cwd=ROOT)
    subprocess.run(cmd, check=True, capture_output=True, cwd=ROOT)
    ids = [json.loads(line)["pair_id"] for line in open(out)]
    assert sorted(ids) == [f"p{i}" for i in range(7)]
    summary = json.load(open(tmp_path / "judged_summary.json"))
    assert summary["judged"] == 7 and summary["win"] + summary["loss"] + summary["tie"] + summary["unknown"] == 7
