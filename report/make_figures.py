"""Report figures from results/*/metrics.jsonl.

    PYTHONPATH=. .venv/bin/python report/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alignment.utils import read_jsonl  # noqa: E402

RES, OUT = Path("results"), Path("report/figures")
EMA = 0.9
BEST = {"dpo_hh": 400, "dpo_uf": 450}
COLORS = {"dpo_hh": "tab:red", "dpo_uf": "tab:blue"}
LABELS = {"dpo_hh": "HH-RLHF DPO", "dpo_uf": "UltraFeedback DPO"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7})


def ema(values, alpha=EMA):
    out, avg = [], None
    for v in values:
        avg = v if avg is None else alpha * avg + (1 - alpha) * v
        out.append(avg)
    return out


def series(records, split, key):
    pts = [(r["step"], r[key]) for r in records if r.get("split") == split and key in r and "step" in r]
    return zip(*pts) if pts else ([], [])


def sft_lora():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    sft = read_jsonl(RES / "sft/metrics.jsonl")
    s, v = series(sft, "train", "loss")
    s, v = list(s), list(v)
    axes[0].plot(s, v, color="0.8", lw=0.6, label="train (raw)")
    axes[0].plot(s, ema(v), color="tab:green", lw=1, label=f"train (EMA {EMA})")
    s, v = series(sft, "val", "loss")
    axes[0].plot(list(s), list(v), "o--", color="k", ms=3, lw=0.8, label="validation")
    axes[0].axvline(937, color="0.5", ls=":", lw=0.8)
    axes[0].annotate("best = final (step 937)", (925, 1.18), ha="right", fontsize=7)
    axes[0].set(title="SFT loss (Qwen2.5-0.5B)", xlabel="optimizer step", ylabel="token-mean NLL", ylim=(1.15, 1.6))

    lora = read_jsonl(RES / "lora_persona/metrics.jsonl")
    for ax, key, title in ((axes[1], "loss", "LoRA-DPO loss"), (axes[2], "reward_margin", "LoRA-DPO reward margin")):
        s, v = series(lora, "train", key)
        ax.plot(list(s), list(v), "-", color="tab:purple", lw=1, label="train (raw, every 5 steps)")
        s, v = series(lora, "val", key)
        ax.plot(list(s), list(v), "o--", color="k", ms=3, lw=0.8, label="validation (per epoch)")
        ax.set(title=title, xlabel="optimizer step (25 = 1 epoch)", ylabel=key.replace("_", " "))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sft_lora.pdf")


def dpo():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for run in ("dpo_hh", "dpo_uf"):
        rec = read_jsonl(RES / f"{run}/metrics.jsonl")
        for ax, key in zip(axes, ("loss", "reward_accuracy", "reward_margin")):
            s, v = series(rec, "train", key)
            ax.plot(list(s), ema(list(v)), color=COLORS[run], alpha=0.35, lw=0.8)
            s, v = series(rec, "val", key)
            s, v = list(s), list(v)
            ax.plot(s, v, "o-", color=COLORS[run], ms=2.5, lw=1, label=f"{LABELS[run]} (val)")
            b = s.index(BEST[run])
            ax.plot(s[b], v[b], "*", color=COLORS[run], ms=9, mec="k", mew=0.5)
    titles = ("DPO loss", "reward accuracy ($r_w > r_l$)", "reward margin ($r_w - r_l$)")
    for ax, t in zip(axes, titles):
        ax.set(title=t, xlabel="optimizer step (32 pairs/step)")
        ax.grid(alpha=0.3)
    axes[1].legend(loc="lower right", title=f"faint = train (EMA {EMA})\n$\\star$ = selected best checkpoint", title_fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "dpo.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sft_lora()
    dpo()
    print("saved", sorted(p.name for p in OUT.iterdir()))
