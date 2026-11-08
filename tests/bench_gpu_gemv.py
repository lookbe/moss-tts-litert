"""WebGPU floor of the backbone's FCs for one row: 28 layers of FCs only, separate vs fused projections, weight
formats. Not the real model (random weights); timing only.

  .venv\\Scripts\\python tests\\bench_gpu_gemv.py
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tflb  # noqa: E402
from ai_edge_litert.compiled_model import CompiledModel  # noqa: E402
from ai_edge_litert.gpu_options import GpuOptions  # noqa: E402
from ai_edge_litert.hardware_accelerator import HardwareAccelerator as HA  # noqa: E402
from ai_edge_litert.options import Options  # noqa: E402

L, D, KVD, FF = 28, 2048, 1024, 6144


def build(path, fmt, fused):
    rng = np.random.default_rng(0)
    m = tflb.Model(flat_io=False)
    g = m.graph("main")
    x = g.input(tflb.F32, [1, 1, D], "x")
    mk = lambda o, i: (lambda: rng.standard_normal((o, i)).astype(np.float32) * 0.02)  # noqa: E731
    wbytes = 0
    for l in range(L):
        if fused:
            qkv = g.fc(x, g.weight(f"qkv{l}", [D + 2 * KVD, D], mk(D + 2 * KVD, D), fmt))
            q = g.slice(qkv, [0, 0, 0], [1, 1, D])
            k = g.slice(qkv, [0, 0, D], [1, 1, KVD])
            v = g.slice(qkv, [0, 0, D + KVD], [1, 1, KVD])
        else:
            q = g.fc(x, g.weight(f"q{l}", [D, D], mk(D, D), fmt))
            k = g.fc(x, g.weight(f"k{l}", [KVD, D], mk(KVD, D), fmt))
            v = g.fc(x, g.weight(f"v{l}", [KVD, D], mk(KVD, D), fmt))
        a = g.add(q, g.concat([k, v], -1))
        x = g.add(x, g.fc(a, g.weight(f"o{l}", [D, D], mk(D, D), fmt)))
        if fused:
            gu = g.fc(x, g.weight(f"gu{l}", [2 * FF, D], mk(2 * FF, D), fmt))
            gt = g.slice(gu, [0, 0, 0], [1, 1, FF])
            up = g.slice(gu, [0, 0, FF], [1, 1, FF])
        else:
            gt = g.fc(x, g.weight(f"g{l}", [FF, D], mk(FF, D), fmt))
            up = g.fc(x, g.weight(f"u{l}", [FF, D], mk(FF, D), fmt))
        x = g.add(x, g.fc(g.mul(g.mul(gt, g.logistic(gt)), up), g.weight(f"d{l}", [D, FF], mk(D, FF), fmt)))
        wbytes += (D * D * 2 + 2 * KVD * D + 3 * FF * D)
    g.output(x, "y")
    m.signature("main", g)
    m.save(path)
    return wbytes


def bench(path, opts):
    cm = CompiledModel.from_file(path, options=opts)
    ib = cm.create_input_buffer_by_name("main", "x")
    ob = cm.create_output_buffer_by_name("main", "y")
    ib.write(np.full(D, 0.1, np.float32))
    for _ in range(3):
        cm.run_by_name("main", {"x": ib}, {"y": ob})
    ob.read(1, np.float32)
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        for _ in range(4):
            cm.run_by_name("main", {"x": ib}, {"y": ob})
        ob.read(1, np.float32)
        ts.append((time.perf_counter() - t0) / 4)
    return float(np.median(ts))


def main():
    td = tempfile.mkdtemp()
    for fmt in ("i8", "i4", "f16"):
        for fused in (False, True):
            p = os.path.join(td, f"g_{fmt}_{fused}.tflite")
            n = build(p, fmt, fused)
            bpw = {"i8": 1, "i4": 0.5, "f16": 2}[fmt]
            for name, opts in [("default", Options(HA.GPU)),
                               ("src_quant", Options(HA.GPU, gpu_options=GpuOptions(allow_src_quantized_fc_conv_ops=True)))]:
                if name == "src_quant" and fmt != "i8":
                    continue
                try:
                    t = bench(p, opts)
                    print(f"{fmt:4s} {'fused' if fused else 'separate':8s} {name:9s}: {t * 1e3:6.2f} ms per step "
                          f"({n * bpw / t / 1e9:5.0f} GB/s of weights)", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"{fmt} fused={fused} {name}: ERROR {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
