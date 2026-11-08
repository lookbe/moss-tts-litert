"""Which compact op forms does LiteRT 2.3.0 WebGPU accept, and are they fp16-safe at the backbone's magnitudes?

  rms_l2     RMSNorm as L2_NORMALIZATION * (sqrt(D) * w)            2 ops
  rms_comp   odml.rms_norm composite                                  1 op
  rope_fc    RoPE with rotate_half as an FC by a 128x128 matrix       4 ops
  sdpa_comp  odml.scaled_dot_product_attention composite              1 op
Each runs in its own process (a WebGPU crash kills the process): `probe_gpu_fused.py <case>`; no argument runs all.
"""
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import tflb  # noqa: E402

D, HD, H, KV = 2048, 128, 16, 8


def rms_ref(x, w, eps=1e-6):
    return x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps) * w


def case_rms_l2(g, rng):
    x = rng.standard_normal((1, 1, D)).astype(np.float32) * 3.0
    x[0, 0, 7] = 21000.0                                  # the scaled massive activation
    w = rng.standard_normal(D).astype(np.float32)
    xi = g.input(tflb.F32, [1, 1, D], "x")
    y = g.mul(g.l2_norm(xi), g.const_f32(w * math.sqrt(D)))
    g.output(y, "y")
    return {"x": x}, rms_ref(x, w)


def case_rms_comp(g, rng):
    x = rng.standard_normal((1, 1, D)).astype(np.float32) * 3.0
    x[0, 0, 7] = 21000.0
    w = rng.standard_normal(D).astype(np.float32)
    xi = g.input(tflb.F32, [1, 1, D], "x")
    body = g.model.graph("odml.rms_norm.impl")
    bx = body.input(tflb.F32, [1, 1, D], "x")
    bw = body.input(tflb.F32, [D], "w")
    m = body.maximum(body.reduce_max(body.abs(bx), [-1]), body.scalar(1e-4))
    xs = body.div(bx, m)
    r = body.rsqrt(body.add(body.mean(body.mul(xs, xs), [-1]), body.div(body.scalar(1e-6), body.mul(m, m))))
    body.output(body.mul(body.mul(xs, r), bw), "y")
    y = g.composite("odml.rms_norm", {"epsilon": 1e-6}, [xi, g.const_f32(w)], body)
    g.output(y, "y")
    return {"x": x}, rms_ref(x, w)


def rot_matrix():
    R = np.zeros((HD, HD), np.float32)        # y = x @ R.T = rotate_half(x) = [-x2, x1]
    h = HD // 2
    for i in range(h):
        R[i, i + h] = -1.0
        R[i + h, i] = 1.0
    return R


def case_rope_fc(g, rng):
    x = rng.standard_normal((1, H, 1, HD)).astype(np.float32) * 4
    ang = 37.0 / 1e6 ** (np.arange(0, HD, 2) / HD)
    c = np.cos(np.concatenate([ang, ang])).astype(np.float32)
    s = np.sin(np.concatenate([ang, ang])).astype(np.float32)
    xi = g.input(tflb.F32, [1, H, 1, HD], "x")
    ci = g.input(tflb.F32, [1, 1, 1, HD], "c")
    si = g.input(tflb.F32, [1, 1, 1, HD], "s")
    rot = g.fc(xi, g.const_f32(rot_matrix()))
    g.output(g.add(g.mul(xi, ci), g.mul(rot, si)), "y")
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    ref = x * c + np.concatenate([-x2, x1], -1) * s
    return {"x": x, "c": c.reshape(1, 1, 1, HD), "s": s.reshape(1, 1, 1, HD)}, ref


def case_sdpa_comp(g, rng):
    T, P = 1, 64
    q = rng.standard_normal((1, T, H, HD)).astype(np.float32)
    k = rng.standard_normal((1, P, H, HD)).astype(np.float32)
    v = rng.standard_normal((1, P, H, HD)).astype(np.float32)
    mk = np.zeros((1, 1, T, P), np.float32)
    mk[..., 40:] = -30000.0
    qi, ki, vi, mi = (g.input(tflb.F32, list(a.shape), n) for a, n in ((q, "q"), (k, "k"), (v, "v"), (mk, "m")))
    sc = 1 / math.sqrt(HD)
    body = g.model.graph("odml.scaled_dot_product_attention.impl")
    bq, bk, bv, bm = (body.input(tflb.F32, list(a.shape), n) for a, n in ((q, "q"), (k, "k"), (v, "v"), (mk, "m")))
    rs = body.scalar(math.sqrt(sc))
    qh = body.transpose(body.mul(bq, rs), [0, 2, 1, 3])
    kt = body.transpose(body.mul(bk, rs), [0, 2, 3, 1])
    vh = body.transpose(bv, [0, 2, 1, 3])
    p = body.softmax(body.add(body.bmm(qh, kt), bm))
    body.output(body.transpose(body.bmm(p, vh), [0, 2, 1, 3]), "y")
    g.output(g.composite("odml.scaled_dot_product_attention", {"scale": sc}, [qi, ki, vi, mi], body), "y")
    s = np.einsum("tnh,snh->nts", q[0], k[0]) * sc + mk[0]
    s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
    ref = np.einsum("nts,snh->tnh", s, v[0])[None]
    return {"q": q, "k": k, "v": v, "m": mk}, ref


CASES = {"rms_l2": case_rms_l2, "rms_comp": case_rms_comp, "rope_fc": case_rope_fc, "sdpa_comp": case_sdpa_comp}


def run_case(name):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.gpu_options import GpuOptions
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator as HA
    from ai_edge_litert.options import Options, CpuOptions
    rng = np.random.default_rng(0)
    m = tflb.Model(flat_io=False)
    g = m.graph("main")
    feeds, ref = CASES[name](g, rng)
    m.signature("main", g)
    p = os.path.join(tempfile.mkdtemp(), f"{name}.tflite")
    m.save(p)
    for acc, opts in (("cpu", Options(HA.CPU, cpu_options=CpuOptions(num_threads=2))),
                      ("gpu fp16", Options(HA.GPU)),
                      ("gpu fp32", Options(HA.GPU, gpu_options=GpuOptions(enforce_f32=True)))):
        try:
            cm = CompiledModel.from_file(p, options=opts)
            ins = {n: cm.create_input_buffer_by_name("main", n) for n in feeds}
            out = cm.create_output_buffer_by_name("main", "y")
            for n, a in feeds.items():
                ins[n].write(a.ravel())
            cm.run_by_name("main", ins, {"y": out})
            y = np.array(out.read(ref.size, np.float32)).reshape(ref.shape)
            err = np.abs(y - ref).max() / np.abs(ref).max()
            full = cm.is_fully_accelerated() if acc != "cpu" else "-"
            print(f"{name:10s} {acc:8s}: rel err {err:.2e}  fully accelerated {full}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name:10s} {acc:8s}: ERROR {str(e)[:120]}", flush=True)


def main():
    if len(sys.argv) > 1:
        run_case(sys.argv[1])
        return
    for name in CASES:
        r = subprocess.run([sys.executable, __file__, name], capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        lines = [l for l in r.stdout.splitlines() if l.startswith(name)]
        print("\n".join(lines) if lines else f"{name:10s}: no output")
        if r.returncode not in (0,):
            print(f"{name:10s}: process exit code {r.returncode} (crash)")


if __name__ == "__main__":
    main()
