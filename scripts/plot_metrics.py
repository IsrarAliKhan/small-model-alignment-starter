"""Plot train/val curves from one or more metrics.jsonl files.

    python scripts/plot_metrics.py --runs outputs/sft --keys loss --output outputs/plots/sft.png
    python scripts/plot_metrics.py --runs outputs/dpo_hh outputs/dpo_uf \
        --keys loss reward_accuracy reward_margin --output outputs/plots/dpo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alignment.utils import read_jsonl  # noqa: E402


def ema(values, alpha):
    out, avg = [], None
    for v in values:
        avg = v if avg is None else alpha * avg + (1 - alpha) * v
        out.append(avg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs containing metrics.jsonl")
    ap.add_argument("--keys", nargs="+", default=["loss"])
    ap.add_argument("--smoothing", type=float, default=0.6, help="EMA factor for train curves (0 = raw)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(args.keys), figsize=(5.5 * len(args.keys), 4), squeeze=False)
    for run in args.runs:
        records = read_jsonl(Path(run) / "metrics.jsonl")
        name = Path(run).name
        for ax, key in zip(axes[0], args.keys):
            for split, style in (("train", "-"), ("val", "o--")):
                points = [(r["step"], r[key]) for r in records if r.get("split") == split and key in r and "step" in r]
                if not points:
                    continue
                steps, values = zip(*points)
                if split == "train" and args.smoothing:
                    values = ema(values, args.smoothing)
                label = f"{name} {split}" + (f" (EMA {args.smoothing})" if split == "train" and args.smoothing else "")
                ax.plot(steps, values, style, label=label, markersize=3)
            ax.set_xlabel("optimizer step")
            ax.set_ylabel(key)
            ax.set_title(key.replace("_", " "))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
