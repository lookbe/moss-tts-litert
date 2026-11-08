"""Weight-streaming floor of the backbone on XNNPACK: 28 layers x 7 FCs at the real shapes, 1 row, no attention."""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tflb  # noqa: E402
from moss_litert import aligned_zeros, litert_model, Sig  # noqa: E402

L, D, KVD, FF = 28, 2048, 1024, 6144


def build(path, fmt, layers=L, rows=1):
    rng = np.random.default_rng(0)
    m = tflb.Model()
    g = m.graph("main")
    x = g.input(tflb.F32, [1, rows, D], "x")
    for i in range(layers):
        mk = lambda o, n: (lambda: rng.standard_normal((o, n)).astype(np.float32) * 0.02)  # noqa: E731
        q = g.fc(x, g.weight(f"q{i}", [D, D], mk(D, D), fmt))
        k = g.fc(x, g.weight(f"k{i}", [KVD, D], mk(KVD, D), fmt))
        v = g.fc(x, g.weight(f"v{i}", [KVD, D], mk(KVD, D), fmt))
        o = g.fc(q, g.weight(f"o{i}", [D, D], mk(D, D), fmt))
        a = g.fc(o, g.weight(f"g{i}", [FF, D], mk(FF, D), fmt))
        b = g.fc(o, g.weight(f"u{i}", [FF, D], mk(FF, D), fmt))
        x = g.add(x, g.add(g.fc(g.mul(a, b), g.weight(f"d{i}", [D, FF], mk(D, FF), fmt)),
                           g.add(g.slice(k, [0, 0, 0], [1, rows, KVD]), g.slice(v, [0, 0, 0], [1, rows, KVD]))
                           if False else o))
    g.output(x, "y")
    m.signature("main", g)
    return m.save(path)


def main():
    td = tempfile.mkdtemp()
    for fmt in sys.argv[1:] or ["i8", "f16"]:
        p = os.path.join(td, f"fc_{fmt}.tflite")
        size = build(p, fmt)
        for th in (2, 4, 8, 16):
            cm, io = litert_model(p, "cpu", th)
            sig = Sig(cm, io, "main")
            xa, ya = aligned_zeros((1, 1, D)), aligned_zeros((1, 1, D))
            xa[:] = 0.1
            ins, outs = {"x": sig.host(xa)}, {"y": sig.host(ya)}
            ts = []
            for _ in range(15):
                t0 = time.perf_counter()
                sig.run(ins, outs)
                ts.append(time.perf_counter() - t0)
            ms = np.median(ts[3:]) * 1e3
            print(f"{fmt} threads {th:2d}: {ms:7.2f} ms/step  ({size / 1e9 / (ms / 1e3):5.1f} GB/s of weights)",
                  flush=True)
            del cm


if __name__ == "__main__":
    main()
