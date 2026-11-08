"""Does this LiteRT build run blockwise int4 FCs (BlockwiseQuantization, fp16 scales)? CPU and GPU vs numpy."""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tflb  # noqa: E402
from moss_litert import aligned_zeros, litert_model, Sig  # noqa: E402

O, I = 6144, 2048


def dequant(w, bs):
    q, s = tflb.quantize_blocks_i4(w, bs)
    n = np.empty(q.size * 2, np.int8)
    n[0::2] = (q & 0xF).astype(np.int8)
    n[1::2] = (q >> 4).astype(np.int8)
    n = np.where(n > 7, n - 16, n).reshape(O, I // bs, bs).astype(np.float32)
    return (n * s.astype(np.float16).astype(np.float32)[:, :, None]).reshape(O, I)


def main():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((O, I)).astype(np.float32) * 0.02
    x = rng.standard_normal((1, 1, I)).astype(np.float32)
    td = tempfile.mkdtemp()
    for bs in (32, 128):
        m = tflb.Model()
        g = m.graph("main")
        xi = g.input(tflb.F32, [1, 1, I], "x")
        g.output(g.fc(xi, g.weight("w", [O, I], lambda: w, f"i4b{bs}")), "y")
        m.signature("main", g)
        p = os.path.join(td, f"b{bs}.tflite")
        m.save(p)
        ref = (x.reshape(-1) @ dequant(w, bs).T)
        for acc in os.environ.get("PROBE_ACC", "cpu,gpu").split(","):
            try:
                if acc == "cpu":
                    os.environ["MOSS_XNNPACK_CACHE"] = "off"
                    cm, io = litert_model(p, "cpu", 4)
                    sig = Sig(cm, io, "main")
                    xa, ya = aligned_zeros((1, 1, I)), aligned_zeros((1, 1, O))
                    xa[:] = x
                    ins, outs = {"x": sig.host(xa)}, {"y": sig.host(ya)}
                    run = lambda: sig.run(ins, outs)  # noqa: E731
                    get = lambda: ya.reshape(-1)  # noqa: E731
                else:
                    from ai_edge_litert.compiled_model import CompiledModel
                    from ai_edge_litert.hardware_accelerator import HardwareAccelerator as HA
                    from ai_edge_litert.options import Options
                    cm = CompiledModel.from_file(p.replace(".tflite", ".tflite"), options=Options(HA.GPU))
                    ib = cm.create_input_buffer_by_name("main", "x")
                    ob = cm.create_output_buffer_by_name("main", "y")
                    ib.write(x.ravel())
                    run = lambda: cm.run_by_name("main", {"x": ib}, {"y": ob})  # noqa: E731
                    get = lambda: np.array(ob.read(O, np.float32))  # noqa: E731
                run()
                y = get()
                ts = []
                for _ in range(20):
                    t0 = time.perf_counter()
                    run()
                    get()
                    ts.append(time.perf_counter() - t0)
                err = np.abs(y - ref).max() / np.abs(ref).max()
                print(f"i4b{bs} {acc}: rel err vs dequantized {err:.2e}, {np.median(ts) * 1e3:.3f} ms per FC", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"i4b{bs} {acc}: ERROR {str(e)[:160]}", flush=True)


if __name__ == "__main__":
    main()
