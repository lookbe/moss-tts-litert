"""Cost of binding the KV caches as graph I/O on WebGPU: 28 layers x (K, V) [1,8,P,128], one attention-like
read per layer, optionally written back through DYNAMIC_UPDATE_SLICE; fp32 vs fp16 tensors at the graph boundary.

  .venv\\Scripts\\python tests\\bench_gpu_cacheio.py
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tflb  # noqa: E402
from ai_edge_litert.compiled_model import CompiledModel  # noqa: E402
from ai_edge_litert.hardware_accelerator import HardwareAccelerator as HA  # noqa: E402
from ai_edge_litert.options import Options  # noqa: E402

L, KV, HD = 28, 8, 128


def build(path, P, io_dt, write):
    m = tflb.Model(flat_io=False)
    g = m.graph("main")
    q = g.input(tflb.F32, [1, KV, 2, HD], "q")
    kn = g.input(tflb.F32, [1, KV, 1, HD], "kn")
    idx = g.input(tflb.I32, [4], "idx")
    acc = None
    outs = []
    for l in range(L):
        kc = g.input(io_dt, [1, KV, P, HD], f"k{l}")
        k32 = kc if io_dt == tflb.F32 else g.cast(kc, tflb.F32)
        if write:
            k32 = g.dynamic_update_slice(k32, kn, idx)
            outs.append(k32 if io_dt == tflb.F32 else g.cast(k32, io_dt))
        s = g.bmm(q, k32, adj_y=True)
        acc = s if acc is None else g.add(acc, s)
    g.output(acc, "y")
    for l, t in enumerate(outs):
        g.output(t, f"nk{l}")
    m.signature("main", g)
    m.save(path)


def bench(path):
    cm = CompiledModel.from_file(path, options=Options(HA.GPU))
    sl = cm.get_signature_list()["main"]
    ins = {n: cm.create_input_buffer_by_name("main", n) for n in sl["inputs"]}
    outs = {n: cm.create_output_buffer_by_name("main", n) for n in sl["outputs"]}
    for n, b in ins.items():
        d = cm.get_input_tensor_details("main")[n]
        sz = int(np.prod(d["shape"]))
        dt = np.int32 if n == "idx" else (np.float16 if "16" in str(d["dtype"]) else np.float32)
        b.write(np.zeros(sz, dt))
    for _ in range(3):
        cm.run_by_name("main", ins, outs)
    outs["y"].read(1, np.float32)
    ts = []
    for _ in range(15):
        t0 = time.perf_counter()
        cm.run_by_name("main", ins, outs)
        outs["y"].read(1, np.float32)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    td = tempfile.mkdtemp()
    for write in (False, True):
        for io_dt in (tflb.F32, tflb.F16):
            row = []
            for P in (256, 1024, 2048):
                p = os.path.join(td, f"c_{P}_{io_dt}_{write}.tflite")
                build(p, P, io_dt, write)
                try:
                    row.append(f"P={P}: {bench(p) * 1e3:6.2f} ms")
                except Exception as e:  # noqa: BLE001
                    row.append(f"P={P}: ERROR {str(e)[:60]}")
            print(f"{'write-back' if write else 'read-only '} {io_dt} I/O: " + "  ".join(row), flush=True)


if __name__ == "__main__":
    main()
