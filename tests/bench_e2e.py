"""End-to-end generation benchmark: load once, warm up, generate every prompt for several seeds.

  .venv\\Scripts\\python tests\\bench_e2e.py --accel gpu --lm models\\lm_gpu_i8.tflite --dec models\\dec_gpu_f16.tflite
      --out out\\litert_gpu --seeds 0,1,2

Writes <out>/<name>_s<seed>.wav and <out>/results.jsonl (per-run stats). RTFx = audio seconds / wall seconds
from the prompt to the last sample (prefill + steps + sampling + full decode, no streaming overlap).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from moss_litert import VoiceGenerator  # noqa: E402


def peak_rss_gb() -> float:
    """Peak working set of this process (Windows), GB."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    gpmi = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    gpmi(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", required=True)
    ap.add_argument("--dec", required=True)
    ap.add_argument("--accel", default="cpu")
    ap.add_argument("--dec-accel", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--lm-f32", action="store_true")
    ap.add_argument("--dec-fp16", action="store_true")
    ap.add_argument("--names", default="fox,tavern,cold,zh_old")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--stream", action="store_true", help="decode while generating (TTFA)")
    ap.add_argument("--first-block", type=int, default=4)
    ap.add_argument("--check-stream", action="store_true",
                    help="also run the non-streaming path per prompt and report the SNR between the two")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(os.path.join(ROOT, "prompts.json"), encoding="utf-8") as f:
        prompts = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    vg = VoiceGenerator(args.lm, args.dec, args.accel, args.dec_accel, args.threads, args.block, args.lm_f32,
                        not args.dec_fp16)
    print(f"load {time.time() - t0:.1f} s (lm {vg.lm.load_s:.1f}, dec {vg.dec.load_s:.1f}), "
          f"peak RSS {peak_rss_gb():.2f} GB", flush=True)
    p = prompts["fox"]
    vg.generate(p["text"], p["instruction"], seed=123, stream=args.stream, first_block=args.first_block)  # warm-up
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for name in args.names.split(","):
            p = prompts[name]
            wav, gen, codes, st = vg.generate(p["text"], p["instruction"], seed=seed, stream=args.stream,
                                              first_block=args.first_block)
            sf.write(os.path.join(args.out, f"{name}_s{seed}.wav"), wav, 24000)
            np.save(os.path.join(args.out, f"{name}_s{seed}_gen.npy"), gen)
            st.update(name=name, seed=seed, ms_per_step=1e3 * st["lm_s"] / max(st["steps"] - 1, 1),
                      dec_rtfx=st["audio_s"] / max(st["decode_s"], 1e-9))
            if args.check_stream:
                w2, g2, _, _ = vg.generate(p["text"], p["instruction"], seed=seed, stream=not args.stream)
                n = min(len(w2), len(wav))
                same_rows = g2.shape == gen.shape and bool((g2 == gen).all())
                err = float(((w2[:n] - wav[:n]) ** 2).sum()) if same_rows else float("nan")
                st["stream_vs_batch_snr"] = 10 * np.log10(float((w2[:n] ** 2).sum()) / max(err, 1e-30)) \
                    if same_rows else None
                st["same_rows"] = same_rows
            rows.append(st)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}), flush=True)
    with open(os.path.join(args.out, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rt = np.array([r["rtfx"] for r in rows])
    ms = np.array([r["ms_per_step"] for r in rows])
    tt = np.array([r["ttfa_s"] for r in rows])
    print(f"SUMMARY accel={args.accel} stream={args.stream} runs={len(rows)} rtfx median {np.median(rt):.2f} "
          f"(min {rt.min():.2f}, max {rt.max():.2f}); ms/step median {np.median(ms):.1f}; ttfa median "
          f"{np.median(tt):.3f} s; audio {sum(r['audio_s'] for r in rows):.1f} s in "
          f"{sum(r['total_s'] for r in rows):.1f} s; peak RSS {peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
