"""Summarize asr_eval.py output (JSON lines on stdin): WER / CER per source and prompt."""
import collections
import json
import sys

import numpy as np


def source(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    return "torch" if parts[1].startswith("ref") else parts[1]


def main():
    rows = [json.loads(l) for l in sys.stdin if l.startswith("{")]
    grp = collections.defaultdict(list)
    for r in rows:
        grp[(source(r["file"]), r["prompt"])].append(r.get("wer", r.get("cer")))
    order = ["torch", "litert_gpu", "litert_cpu_i8", "litert_cpu_i4b32", "litert_cpu_i4"]
    found = sorted({k[0] for k in grp}, key=lambda s: order.index(s) if s in order else len(order))
    srcs = found
    prompts = [p for p in ["fox", "tavern", "cold", "zh_old", "long"] if any(k[1] == p for k in grp)]
    print("| prompt | " + " | ".join(srcs) + " |")
    print("|---" * (len(srcs) + 1) + "|")
    for p in prompts:
        print(f"| {p} | " + " | ".join(" ".join(f"{v:.2f}" for v in grp[(s, p)]) for s in srcs) + " |")
    en = [p for p in prompts if p != "zh_old"]
    print("| mean en WER | " + " | ".join(f"{np.mean([v for p in en for v in grp[(s, p)]]):.3f}" for s in srcs) + " |")
    if "zh_old" in prompts:
        print("| mean zh CER | " + " | ".join(f"{np.mean(grp[(s, 'zh_old')]):.3f}" for s in srcs) + " |")


if __name__ == "__main__":
    main()
