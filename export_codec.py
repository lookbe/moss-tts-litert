"""MOSS-Audio-Tokenizer decoder -> streaming LiteRT graphs dec_b{B} (B = 12.5 Hz frames per call).

The decoder is four causal transformer stages (12.5 / 25 / 50 / 100 Hz) joined by patch reshapes (x2, x2, x2, x240
samples). Each attention layer keeps a sliding K/V window of exactly the codec's context (int(rate * 10 s) rows:
125 / 250 / 500 / 1000), shifted inside the graph:

    K_all = concat(K_state, k_new)  ->  scores over K_all (mask from the host)  ->  K_state' = K_all[T:]

With a full window kept, chunked decoding equals the codec's non-streaming decode (attention limited to the last
`context` positions) for any block size; upstream's chunk_duration=8 streaming decode is slightly different (its
ring cache loses context at chunk starts).

  dec_b{B}  in : zq [1,B,768], per stage s: cos_s / sin_s [1,T_s,1,hd/2], mask_s [1,1,T_s,W_s+T_s], then per stage
                 and layer k / v state [1,H_s,W_s,hd]
            out: audio [1,1920*B], then the new states (same order)

Host-side assets: the residual-LFQ dequantizer is linear, so it folds into one [1024, 768] table per codebook
(codec_tables.npy, codec_bias.npy): zq[t] = bias + sum_i table_i[code_i[t]].

  .venv\\Scripts\\python export_codec.py --wfmt f32 --blocks 4,16 --out models\\dec_f32.tflite
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

import tflb
from st import SafeTensors

HERE = os.path.dirname(os.path.abspath(__file__))
CODEC = os.path.join(HERE, "..", "MOSS-Audio-Tokenizer")


def stage_specs(cfg: dict) -> list[dict]:
    """Decoder transformer stages with their context window and the patch size that follows."""
    rate = float(cfg["sampling_rate"])
    for e in cfg["encoder_kwargs"]:
        if e["module_type"] == "PatchedPretransform":
            rate /= e["patch_size"]
    specs = []
    for i, d in enumerate(cfg["decoder_kwargs"]):
        if d["module_type"] == "Transformer":
            specs.append(dict(d, index=i, context=int(rate * cfg["causal_transformer_context_duration"]),
                              rate=rate, patch=1))
        else:
            rate *= d["patch_size"]
            specs[-1]["patch"] = d["patch_size"]
    return specs


def wn_weight(st: SafeTensors, prefix: str) -> np.ndarray:
    """weight_norm(dim=0) of a 1x1 Conv1d -> [out, in]."""
    g = st.get(prefix + ".parametrizations.weight.original0").astype(np.float64)
    v = st.get(prefix + ".parametrizations.weight.original1").astype(np.float64)
    n = np.sqrt((v.reshape(v.shape[0], -1) ** 2).sum(1)).reshape(-1, 1, 1)
    return (g * v / n)[:, :, 0]


def export_tables(st: SafeTensors, out_dir: str, n_vq: int):
    wop = wn_weight(st, "quantizer.output_proj")                     # [768, 512]
    bop = st.get("quantizer.output_proj.bias").astype(np.float64)
    tabs = []
    for i in range(n_vq):
        p = f"quantizer.quantizers.{i}"
        cb = st.get(p + ".codebook.weight").astype(np.float64)        # [1024, 8]
        w = wn_weight(st, p + ".out_proj")                            # [512, 8]
        b = st.get(p + ".out_proj.bias").astype(np.float64)
        tabs.append((cb @ w.T + b) @ wop.T)                           # [1024, 768]
    np.save(os.path.join(out_dir, "codec_tables.npy"), np.stack(tabs).astype(np.float32))
    np.save(os.path.join(out_dir, "codec_bias.npy"), bop.astype(np.float32))


def deinterleave_perm(heads: int, hd: int) -> np.ndarray:
    """Row order that turns interleaved RoPE pairs (2i, 2i+1) into half-split (i, i + hd/2) per head."""
    within = np.concatenate([np.arange(0, hd, 2), np.arange(1, hd, 2)])
    return (np.arange(heads)[:, None] * hd + within[None, :]).reshape(-1)


class DecExporter:
    def __init__(self, codec_dir: str, wfmt: str, safe_norm: bool, gelu_approx: bool, flat_io: bool = True):
        with open(os.path.join(codec_dir, "config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.specs = stage_specs(self.cfg)
        self.st = SafeTensors(codec_dir)
        self.wfmt = wfmt
        self.safe_norm = safe_norm
        self.gelu_approx = gelu_approx
        self.ablate: set[str] = set()       # debug timing: ln, rope, state, attn, ffn, gelu (wrong numerics)
        self.model = tflb.Model("moss-audio-tokenizer decoder", flat_io=flat_io)
        self.code_dim = self.cfg["code_dim"]

    def ln(self, g, x, key):
        D = g.shape(x)[-1]
        if "ln" in self.ablate:
            return x
        if self.safe_norm:
            m = g.maximum(g.reduce_max(g.abs(x), [-1]), g.scalar(1e-4))
            x = g.div(x, m)
            eps = g.div(g.scalar(1e-5), g.mul(m, m))
        else:
            eps = g.scalar(1e-5)
        xc = g.sub(x, g.mean(x, [-1]))
        y = g.mul(xc, g.rsqrt(g.add(g.mean(g.mul(xc, xc), [-1]), eps)))
        y = g.mul(y, g.weight(key + ".weight", [D], lambda: self.st.get(key + ".weight"), tflb.F32))
        return g.add(y, g.weight(key + ".bias", [D], lambda: self.st.get(key + ".bias"), tflb.F32))

    @staticmethod
    def rope(g, x, cos, sin):
        s = g.shape(x)
        h = s[-1] // 2
        x1 = g.slice(x, [0] * len(s), s[:-1] + [h])
        x2 = g.slice(x, [0] * (len(s) - 1) + [h], s[:-1] + [h])
        return g.concat([g.sub(g.mul(x1, cos), g.mul(x2, sin)), g.add(g.mul(x2, cos), g.mul(x1, sin))], -1)

    def fc(self, g, x, key, shape, make):
        return g.fc(x, g.weight(key, shape, make, self.wfmt))

    def layer(self, g, sp, li, x, cos, sin, mask, kc, vc, T):
        d, H = sp["d_model"], sp["num_heads"]
        hd = d // H
        W = sp["context"]
        p = f"decoder.{sp['index']}.transformer.layers.{li}."
        st = self.st
        xn = self.ln(g, x, p + "norm1")
        wkey = p + "self_attn.in_projs.0.weight"
        perm = deinterleave_perm(H, hd)
        q = self.fc(g, xn, wkey + "/q", [d, d], lambda: st.get(wkey)[:d][perm])
        k = self.fc(g, xn, wkey + "/k", [d, d], lambda: st.get(wkey)[d:2 * d][perm])
        v = self.fc(g, xn, wkey + "/v", [d, d], lambda: st.get(wkey)[2 * d:])

        def heads(t):
            t = g.reshape(t, [1, T, H, hd])
            return t

        if "rope" in self.ablate:
            q, k = heads(q), heads(k)
        else:
            q = self.rope(g, heads(q), cos, sin)
            k = self.rope(g, heads(k), cos, sin)
        v = heads(v)
        if T == 1:
            q, k, v = (g.reshape(t, [1, H, 1, hd]) for t in (q, k, v))
        else:
            q, k, v = (g.transpose(t, [0, 2, 1, 3]) for t in (q, k, v))
        q = g.mul(q, g.scalar(1.0 / math.sqrt(hd)))
        if "state" in self.ablate:                     # attention over the block only, states passed through
            ka, va = k, v
            m = g.slice(mask, [0, 0, 0, W], [1, 1, T, T])
        else:
            ka = g.concat([kc, k], 2)                  # [1,H,W+T,hd]
            va = g.concat([vc, v], 2)
            m = mask
        if "attn" in self.ablate:
            o = g.add(v, g.mul(q, g.scalar(0.0)))
        else:
            o = g.bmm(g.softmax(g.add(g.bmm(q, ka, adj_y=True), m)), va)   # [1,H,T,hd]
        o = g.reshape(o, [1, 1, d]) if T == 1 else g.reshape(g.transpose(o, [0, 2, 1, 3]), [1, T, d])
        okey = p + "self_attn.out_projs.0.weight"
        s1 = p + "layer_scale_1.scale"
        x = g.add(x, self.fc(g, o, okey, [d, d], lambda: st.get(okey) * st.get(s1)[:, None]))
        if "ffn" not in self.ablate:
            xn = self.ln(g, x, p + "norm2")
            ff = sp["dim_feedforward"]
            h = self.fc(g, xn, p + "linear1.weight", [ff, d], lambda: st.get(p + "linear1.weight"))
            h = h if "gelu" in self.ablate else g.gelu(h, self.gelu_approx)
            s2 = p + "layer_scale_2.scale"
            x = g.add(x, self.fc(g, h, p + "linear2.weight", [d, ff],
                                 lambda: st.get(p + "linear2.weight") * st.get(s2)[:, None]))
        if "state" in self.ablate:
            return x, kc, vc
        knew = g.slice(ka, [0, 0, T, 0], [1, H, W, hd])
        vnew = g.slice(va, [0, 0, T, 0], [1, H, W, hd])
        return x, knew, vnew

    def block(self, B: int):
        name = f"dec_b{B}"
        g = self.model.graph(name)
        in_names = []

        def inp(dt, shape, n):
            in_names.append(n)
            return g.input(dt, shape, n)

        x = inp(tflb.F32, [1, B, self.code_dim], "zq")
        rows = []
        T = B
        per_stage = []
        for s, sp in enumerate(self.specs):
            hd = sp["d_model"] // sp["num_heads"]
            W = sp["context"]
            cos = inp(tflb.F32, [1, T, 1, hd // 2], f"cos{s}")
            sin = inp(tflb.F32, [1, T, 1, hd // 2], f"sin{s}")
            mask = inp(tflb.F32, [1, 1, T, W + T], f"mask{s}")
            per_stage.append((cos, sin, mask, T))
            rows.append(T)
            T *= sp["patch"]
        states = []
        for s, sp in enumerate(self.specs):
            hd = sp["d_model"] // sp["num_heads"]
            for li in range(sp["num_layers"]):
                kc = inp(tflb.F32, [1, sp["num_heads"], sp["context"], hd], f"s{s}k{li}")
                vc = inp(tflb.F32, [1, sp["num_heads"], sp["context"], hd], f"s{s}v{li}")
                states.append((kc, vc))
        new_states = []
        si = 0
        for s, sp in enumerate(self.specs):
            cos, sin, mask, T = per_stage[s]
            d = sp["d_model"]
            pre = f"decoder.{sp['index']}."
            if sp["input_dimension"] != d:
                x = self.fc(g, x, pre + "input_proj.weight", [d, sp["input_dimension"]],
                            lambda k=pre + "input_proj.weight": self.st.get(k))
            for li in range(sp["num_layers"]):
                kc, vc = states[si]
                si += 1
                x, kn, vn = self.layer(g, sp, li, x, cos, sin, mask, kc, vc, T)
                new_states += [kn, vn]
            if sp["output_dimension"] != d:
                x = self.fc(g, x, pre + "output_proj.weight", [sp["output_dimension"], d],
                            lambda k=pre + "output_proj.weight": self.st.get(k))
            h = sp["patch"]
            if h > 1:
                c = sp["output_dimension"] // h
                if c == 1:
                    x = g.reshape(x, [1, T * h, 1])
                else:
                    x = g.reshape(g.transpose(g.reshape(x, [1, T, c, h]), [0, 1, 3, 2]), [1, T * h, c])
        audio = g.reshape(x, [1, -1])
        g.output(audio, "audio")
        for i, t in enumerate(new_states):
            g.output(t, "n" + in_names[len(in_names) - len(new_states) + i])
        self.model.signature(name, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default=CODEC)
    ap.add_argument("--out", default=os.path.join(HERE, "models", "dec_f32.tflite"))
    ap.add_argument("--wfmt", default="f32", choices=["i4", "i4b32", "i4b64", "i4b128", "i8", "f16", "f32"])
    ap.add_argument("--blocks", default="4,16")
    ap.add_argument("--n-vq", type=int, default=16)
    ap.add_argument("--safe-norm", action="store_true")
    ap.add_argument("--gelu-approx", action="store_true", help="tanh GELU instead of erf")
    ap.add_argument("--ranked-io", action="store_true", help="ranked I/O (GPU managed buffers)")
    ap.add_argument("--ablate", default="", help="debug timing only: comma list of ln,rope,state,attn,ffn,gelu")
    args = ap.parse_args()
    t0 = time.time()
    ex = DecExporter(args.codec, args.wfmt, args.safe_norm, args.gelu_approx, not args.ranked_io)
    ex.ablate = {a for a in args.ablate.split(",") if a}
    for B in [int(b) for b in args.blocks.split(",") if b]:
        ex.block(B)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    size = ex.model.save(args.out)
    export_tables(ex.st, out_dir, args.n_vq)
    with open(os.path.join(out_dir, "codec_stages.json"), "w", encoding="utf-8") as f:
        json.dump([{k: sp[k] for k in ("d_model", "num_heads", "num_layers", "context", "rate", "patch")}
                   for sp in ex.specs], f, indent=1)
    print(f"wrote {args.out}: {size / 1e6:.1f} MB, {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
