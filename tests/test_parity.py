"""Teacher-forced parity of the LiteRT graphs against the PyTorch reference (out/ref/<name>, from ref_torch.py).

LM: the reference's prompt + generated rows go through prefill + step; every step's logits are compared with the
torch logits of the same row. Reported per prompt:
  rel     max |d| / max |ref| over all logits
  kl      mean KL(ref || litert) of the audio heads' sampling distributions (softmax(logits / 1.5))
  top1    fraction of (step, audio head) pairs with the same argmax
  slot    the 2 text-slot logits: same argmax
  text0   full text head at the first row: max |d| / max |ref|, same argmax
Codec: the reference codes through the LiteRT decoder vs the codec's non-streaming torch decode (wav_full.wav) and
upstream's chunked decode (wav_upstream.wav): SNR in dB.

  .venv\\Scripts\\python tests\\test_parity.py --lm models\\lm_f32.tflite --dec models\\dec_f32.tflite
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
from moss_litert import Backbone, GpuBackbone, CodecDecoder, N_VQ, AV  # noqa: E402


def snr(ref, x):
    n = min(len(ref), len(x))
    ref, x = ref[:n], x[:n]
    return 10 * np.log10((ref ** 2).sum() / max(((ref - x) ** 2).sum(), 1e-30))


def lsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def lm_parity(lm: Backbone, d: str):
    prompt = np.load(os.path.join(d, "prompt_ids.npy"))
    gen = np.load(os.path.join(d, "gen_ids.npy"))
    ref = np.load(os.path.join(d, "tf_logits.npy"))
    ref_text0 = np.load(os.path.join(d, "tf_text0.npy"))
    t0 = time.perf_counter()
    lg, hn = lm.prefill(prompt)
    t_pre = time.perf_counter() - t0
    text0 = lm.text_logits(hn) if lm.text_head is not None else ref_text0
    outs = [lg]
    t1 = time.perf_counter()
    for r in gen[:-1]:
        lg, hn = lm.step(r)
        outs.append(lg)
    t_step = (time.perf_counter() - t1) / max(len(gen) - 1, 1)
    got = np.stack(outs)
    finite = np.isfinite(ref)
    rel = np.abs(got - ref)[finite].max() / np.abs(ref[finite]).max()
    ga = got[:, 2:].reshape(-1, N_VQ, AV)[..., :-1] / 1.5
    ra = ref[:, 2:].reshape(-1, N_VQ, AV)[..., :-1] / 1.5
    lp, lq = lsm(ra), lsm(ga)
    kl = (np.exp(lp) * (lp - lq)).sum(-1).mean()
    top1 = (ga.argmax(-1) == ra.argmax(-1)).mean()
    slot = (got[:, :2].argmax(-1) == ref[:, :2].argmax(-1)).mean()
    t_rel = np.abs(text0 - ref_text0).max() / np.abs(ref_text0).max()
    t_top = int(text0.argmax() == ref_text0.argmax())
    return dict(rel=float(rel), kl=float(kl), top1=float(top1), slot=float(slot), text0=float(t_rel),
                text0_top1=t_top, prefill_ms=t_pre * 1e3, step_ms=t_step * 1e3, steps=len(gen))


def dec_parity(dec: CodecDecoder, d: str, blocks):
    codes = np.load(os.path.join(d, "codes.npy"))
    full, _ = sf.read(os.path.join(d, "wav_full.wav"), dtype="float32")
    up, _ = sf.read(os.path.join(d, "wav_upstream.wav"), dtype="float32")
    res = {}
    for B in blocks:
        t0 = time.perf_counter()
        w = dec.decode(codes, B)
        dt = time.perf_counter() - t0
        res[f"b{B}"] = dict(snr_full=float(snr(full, w)), snr_upstream=float(snr(up, w)),
                            rtfx=float(len(w) / 24000 / dt))
    res["upstream_vs_full_snr"] = float(snr(full, up))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="")
    ap.add_argument("--dec", default="")
    ap.add_argument("--accel", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--names", default="fox,tavern,cold,zh_old")
    ap.add_argument("--blocks", default="")
    ap.add_argument("--f32", action="store_true", help="GPU: enforce fp32 arithmetic")
    ap.add_argument("--dec-accel", default="")
    ap.add_argument("--codec-chunk", type=int, default=0)
    args = ap.parse_args()
    names = args.names.split(",")
    results = {}
    if args.lm:
        if args.accel == "gpu":
            lm = GpuBackbone(args.lm, args.threads, enforce_f32=args.f32)
            print(f"lm fully accelerated: {lm.fully}")
        else:
            lm = Backbone(args.lm, args.accel, args.threads)
        print(f"lm load {lm.load_s:.1f} s")
        for n in names:
            r = lm_parity(lm, os.path.join(ROOT, "out", "ref", n))
            results.setdefault(n, {})["lm"] = r
            print(n, "LM", json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}),
                  flush=True)
        del lm
    if args.dec:
        dec = CodecDecoder(args.dec, args.dec_accel or args.accel, args.threads, enforce_f32=args.f32)
        dec.chunk = args.codec_chunk
        print(f"dec load {dec.load_s:.1f} s")
        blocks = [int(b) for b in args.blocks.split(",") if b] or sorted(dec.sigs)
        for n in names:
            r = dec_parity(dec, os.path.join(ROOT, "out", "ref", n), blocks)
            results.setdefault(n, {})["dec"] = r
            print(n, "DEC", json.dumps(r), flush=True)
    return results


if __name__ == "__main__":
    main()
