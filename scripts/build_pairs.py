"""Join two generation files by id into a blind pairwise JSONL file.

--candidate is the model under test (DPO), --baseline the comparison (SFT).
A/B order is randomized per pair (seeded); the A/B -> model mapping lives
only in the "meta" field, which the judge never sees.
"""

from __future__ import annotations

import argparse
import random

from alignment.utils import read_json, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="generation JSON of the model under test")
    ap.add_argument("--baseline", required=True, help="generation JSON of the reference model")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    candidate = read_json(args.candidate)
    baseline = {r["id"]: r for r in read_json(args.baseline)}
    rng = random.Random(args.seed)
    pairs = []
    for cand in candidate:
        base = baseline.get(cand["id"])
        if base is None:
            continue
        assert cand["instruction"] == base["instruction"], f"prompt mismatch for {cand['id']}"
        a, b = (cand, base) if rng.random() < 0.5 else (base, cand)
        pairs.append({
            "pair_id": cand["id"], "dataset": cand["dataset"], "instruction": cand["instruction"],
            "response_a": a["output"], "response_b": b["output"],
            "meta": {"A": a["generator"], "B": b["generator"],
                     "candidate": cand["generator"], "baseline": base["generator"]},
        })
    write_jsonl(pairs, args.output)
    print(f"wrote {len(pairs)} pairs ({len(candidate) - len(pairs)} unmatched) to {args.output}")


if __name__ == "__main__":
    main()
