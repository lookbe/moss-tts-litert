"""Where a GPU backbone step goes: input writes, the run, read-back (GpuBackbone internals).

  .venv\\Scripts\\python tests\\bench_gpu_step.py models\\lm_gpu_i8.tflite [P]
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from moss_litert import GpuBackbone, NEG, rope_qwen  # noqa: E402


def med(ts):
    return f"{np.median(ts) * 1e3:6.3f} ms"


def main():
    path = sys.argv[1]
    lm = GpuBackbone(path)
    d = os.path.join(ROOT, "out", "ref", "cold")
    prompt = np.load(os.path.join(d, "prompt_ids.npy"))
    gen = np.load(os.path.join(d, "gen_ids.npy"))
    lm.prefill(prompt)
    for r in gen[:20]:
        lm.step(r)
    sig = lm.steps[lm.P]
    ins, outs = lm._small_io(sig)
    x = lm.emb(gen[20][None])[None]
    cs, sn = rope_qwen(np.array([lm.slot - lm.pad], np.float64), lm.HD)
    j = np.arange(lm.P)
    mask = np.tile(np.where((j >= lm.pad) & (j <= lm.slot), 0.0, NEG).astype(np.float32), (2, 1))
    idx = np.array([0, 0, lm.slot, 0], np.int32)
    imap = dict(ins)
    imap.update(zip(lm.names, lm.banks[lm.cur]))
    omap = dict(outs)
    omap.update(zip(["n" + n for n in lm.names], lm.banks[1 - lm.cur]))
    N = 40
    tw, tr, trd, tfull = [], [], [], []
    for _ in range(N):
        t0 = time.perf_counter()
        ins["x"].write(x.ravel()); ins["cos"].write(cs.ravel()); ins["sin"].write(sn.ravel())
        ins["mask"].write(mask.ravel()); ins["idx"].write(idx)
        t1 = time.perf_counter()
        sig.run(imap, omap)               # same banks every time: timing only
        t2 = time.perf_counter()
        outs["logits"].read(lm.NO, np.float32); outs["hn"].read(lm.D, np.float32)
        t3 = time.perf_counter()
        tw.append(t1 - t0); tr.append(t2 - t1); trd.append(t3 - t2); tfull.append(t3 - t0)
    print(f"P={lm.P}: writes {med(tw)}  run {med(tr)}  reads {med(trd)}  total {med(tfull)}")
    # run + read only (the read waits for the GPU): true GPU time per step
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        sig.run(imap, omap)
        outs["logits"].read(lm.NO, np.float32)
        ts.append(time.perf_counter() - t0)
    print(f"run + 1 read: {med(ts)}")
    # 5 queued runs then one read: does the run call itself block?
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        for _ in range(5):
            sig.run(imap, omap)
        outs["logits"].read(lm.NO, np.float32)
        ts.append((time.perf_counter() - t0) / 5)
    print(f"5 runs per read: {med(ts)} per run")


if __name__ == "__main__":
    main()
