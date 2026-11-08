"""WebGPU throughput of single ops at the model's shapes (FC with M rows, batch matmul), fp32 / fp16 arithmetic.

  .venv\\Scripts\\python tests\\bench_gpu_ops.py
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

REPS = 8      # ops chained in one graph, so dispatch overhead is visible as well


def run_graph(build, f32, iters=10):
    m = tflb.Model(flat_io=False)
    g = m.graph("main")
    flops = build(g)
    m.signature("main", g)
    p = os.path.join(tempfile.mkdtemp(), "op.tflite")
    m.save(p)
    cm = CompiledModel.from_file(p, options=Options(HA.GPU, gpu_options=GpuOptions(enforce_f32=f32)))
    ins = {n: cm.create_input_buffer_by_name("main", n) for n in cm.get_signature_list()["main"]["inputs"]}
    outs = {n: cm.create_output_buffer_by_name("main", n) for n in cm.get_signature_list()["main"]["outputs"]}
    for n, b in ins.items():
        sz = int(np.prod(cm.get_input_tensor_details("main")[n]["shape"]))
        b.write(np.random.default_rng(0).standard_normal(sz).astype(np.float32) * 0.1)
    on = next(iter(outs))
    cm.run_by_name("main", ins, outs)
    outs[on].read(1, np.float32)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(4):
            cm.run_by_name("main", ins, outs)
        outs[on].read(1, np.float32)
        ts.append((time.perf_counter() - t0) / 4)
    t = float(np.median(ts))
    return t, flops / t / 1e12


def fc_case(M, K, N, wfmt):
    def build(g):
        x = g.input(tflb.F32, [1, M, K], "x")
        w = g.weight("w", [N, K], lambda: np.random.default_rng(1).standard_normal((N, K)).astype(np.float32) * 0.02,
                     wfmt)
        w2 = g.weight("w2", [K, N], lambda: np.random.default_rng(2).standard_normal((K, N)).astype(np.float32) * 0.02,
                      wfmt)
        y = x
        for _ in range(REPS // 2):
            y = g.fc(g.fc(y, w), w2)
        g.output(y, "y")
        return 2.0 * M * K * N * REPS
    return build


def bmm_case(H, M, Kk, D):
    def build(g):
        q = g.input(tflb.F32, [1, H, M, D], "q")
        k = g.input(tflb.F32, [1, H, Kk, D], "k")
        y = q
        for _ in range(REPS // 2):
            s = g.softmax(g.bmm(y, k, adj_y=True))      # [1,H,M,Kk]
            y = g.bmm(s, k)                              # [1,H,M,D]
        g.output(y, "y")
        return 2.0 * 2 * H * M * Kk * D * (REPS // 2)
    return build


def main():
    cases = [
        ("FC LM 1 x 2048 -> 6144 (i8)", fc_case(1, 2048, 6144, "i8")),
        ("FC codec 128 x 1280 -> 5120 (f16)", fc_case(128, 1280, 5120, "f16")),
        ("FC codec 1024 x 768 -> 3072 (f16)", fc_case(1024, 768, 3072, "f16")),
        ("FC codec 1024 x 768 -> 3072 (i8)", fc_case(1024, 768, 3072, "i8")),
        ("FC codec 1024 x 768 -> 3072 (f32)", fc_case(1024, 768, 3072, "f32")),
        ("BMM attn 12h 1024 x 2024 x 64", bmm_case(12, 1024, 2024, 64)),
        ("BMM attn 20h 128 x 253 x 128? (64)", bmm_case(20, 128, 253, 64)),
    ]
    for name, build in cases:
        for f32 in (True, False):
            try:
                t, tf = run_graph(build, f32)
                print(f"{name:40s} {'fp32' if f32 else 'fp16'}: {t * 1e3 / REPS:7.3f} ms per op, {tf:6.2f} TFLOPS",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{name:40s} {'fp32' if f32 else 'fp16'}: ERROR {str(e)[:100]}", flush=True)


if __name__ == "__main__":
    main()
