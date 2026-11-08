"""Backbone step / prefill timing on the CPU (teacher-forced rows of a reference run).

  .venv\\Scripts\\python tests\\bench_step.py models\\lm_i8.tflite 4,8,16 [name]
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from moss_litert import Backbone  # noqa: E402


def main():
    path = sys.argv[1]
    threads = [int(t) for t in (sys.argv[2] if len(sys.argv) > 2 else "8").split(",")]
    name = sys.argv[3] if len(sys.argv) > 3 else "cold"
    d = os.path.join(ROOT, "out", "ref", name)
    prompt = np.load(os.path.join(d, "prompt_ids.npy"))
    gen = np.load(os.path.join(d, "gen_ids.npy"))
    for th in threads:
        lm = Backbone(path, "cpu", th)
        pre = []
        for _ in range(3):
            t0 = time.perf_counter()
            lm.prefill(prompt)
            pre.append(time.perf_counter() - t0)
        ts = []
        for r in gen[:60]:
            t0 = time.perf_counter()
            lm.step(r)
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts[5:]) * 1e3
        print(f"threads {th:2d}: P={lm.P} prefill {min(pre) * 1e3:7.1f} ms ({prompt.shape[0]} rows)  "
              f"step median {np.median(ts):6.1f} ms  p10 {np.percentile(ts, 10):6.1f}  p90 {np.percentile(ts, 90):6.1f}",
              flush=True)
        del lm


if __name__ == "__main__":
    main()
