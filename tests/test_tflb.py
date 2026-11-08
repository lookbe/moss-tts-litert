"""Builder smoke test: every op the exporters use, external (offset) buffers, CPU and GPU vs numpy."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tflb  # noqa: E402
from ai_edge_litert.compiled_model import CompiledModel  # noqa: E402
from ai_edge_litert.hardware_accelerator import HardwareAccelerator as HA  # noqa: E402
from ai_edge_litert.options import Options, CpuOptions  # noqa: E402

rng = np.random.default_rng(0)
D, O, T = 256, 512, 4
w = rng.standard_normal((O, D)).astype(np.float32) * 0.05
x = rng.standard_normal((1, T, D)).astype(np.float32)
nw = rng.standard_normal(D).astype(np.float32)


def build(path, fmt):
    m = tflb.Model()
    g = m.graph("main")
    xi = g.input(tflb.F32, [1, T, D], "x")
    ms = g.mean(g.mul(xi, xi), [-1])
    xn = g.mul(g.mul(xi, g.rsqrt(g.add(ms, g.scalar(1e-6)))), g.weight("nw", [D], lambda: nw, tflb.F32))
    y = g.fc(xn, g.weight("w", [O, D], lambda: w, fmt))                 # [1,T,O]
    a = g.reshape(y, [1, T, 4, 128])
    a = g.transpose(a, [0, 2, 1, 3])                                      # [1,4,T,128]
    h1 = g.slice(a, [0, 0, 0, 0], [1, 4, T, 64])
    h2 = g.slice(a, [0, 0, 0, 64], [1, 4, T, 64])
    r = g.concat([g.sub(h1, h2), g.add(h1, h2)], -1)
    s = g.softmax(g.bmm(r, a, adj_y=True))                                # [1,4,T,T]
    o = g.bmm(s, a)                                                       # [1,4,T,128]
    gl = g.gelu(o, False)
    sm = g.mul(gl, g.logistic(gl))
    mx = g.reduce_max(g.abs(sm), [-1])
    g.output(g.div(sm, g.maximum(mx, g.scalar(1e-4))))
    m.signature("main", g, ["x"], ["y"])
    return m.save(path)


def ref(fmt):
    import math
    ww = w
    if fmt == "i8":
        q, sc = tflb.quantize_rows_i8(w)
        ww = q.astype(np.float32) * sc[:, None]
    elif fmt == "f16":
        ww = w.astype(np.float16).astype(np.float32)
    xn = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * nw
    y = xn @ ww.T
    a = y.reshape(1, T, 4, 128).transpose(0, 2, 1, 3)
    r = np.concatenate([a[..., :64] - a[..., 64:], a[..., :64] + a[..., 64:]], -1)
    sc = r @ a.transpose(0, 1, 3, 2)
    sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
    o = sc @ a
    erf = np.vectorize(math.erf)
    gl = 0.5 * o * (1 + erf(o / math.sqrt(2)))
    sm = gl / (1 + np.exp(-gl))
    return sm / np.maximum(np.abs(sm).max(-1, keepdims=True), 1e-4)


def run(path, accel):
    opts = Options(HA.CPU, cpu_options=CpuOptions(num_threads=4)) if accel == "cpu" else Options(HA.GPU)
    cm = CompiledModel.from_file(path, options=opts)
    ib = {"x": cm.create_input_buffer_by_name("main", "x")}
    ob = {"y": cm.create_output_buffer_by_name("main", "y")}
    ib["x"].write(x.ravel())
    cm.run_by_name("main", ib, ob)
    full = cm.is_fully_accelerated() if accel == "gpu" else None
    return np.array(ob["y"].read(4 * T * 128, np.float32)).reshape(1, 4, T, 128), full


ok = True
with tempfile.TemporaryDirectory() as td:
    for fmt in ("f32", "f16", "i8"):
        p = os.path.join(td, f"t_{fmt}.tflite")
        size = build(p, fmt)
        r = ref(fmt)
        for accel in ("cpu", "gpu"):
            try:
                got, full = run(p, accel)
                err = np.abs(got - r).max()
                tol = 2e-3 if accel == "cpu" else 3e-2
                if fmt == "i8" and accel == "cpu":
                    tol = 6e-2     # dynamic int8 activations
                print(f"{fmt:4s} {accel}: size {size} B, max|d| {err:.2e}" + (f", fully accelerated {full}" if full is not None else "")
                      + ("" if err < tol else "   <-- FAIL"))
                ok &= err < tol
            except Exception as e:  # noqa: BLE001
                print(f"{fmt:4s} {accel}: ERROR {e}")
                ok = False
print("OK" if ok else "FAILED")
