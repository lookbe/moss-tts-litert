"""MOSS-VoiceGenerator backbone (Qwen3-1.7B + 17 heads) -> one multi-signature LiteRT model.

Signatures (weights stored once in the file):
  step_p{P}           1 row
  prefill_t{T}_p{P}   T rows (chunked prompt prefill)
      in : x [1,T,2048] (summed embeddings), cos / sin [1,T,1,64], mask [1,1,2T,P+T] (additive), k0..k27, v0..v27
           [1,8,P,128] (read only: the host writes the returned rows into them)
      out: logits [1,1,2+16*1025] (last row: the 2 slot rows of the text head | 16 audio heads), hn [1,1,2048]
           (last row after the final norm), k_new / v_new [28,8,T,128]
  text_head           hn [1,1,2048] -> full text logits [1,1,155648] (only needed while not inside audio)

GQA: the 16 query heads are reshaped to [8 groups, 2T rows] (head 2g+h, row t -> row h*T+t), so each KV head is
used as is and the mask has 2T rows. The 1/sqrt(head_dim) scale is folded into the q_norm weight.

  .venv\\Scripts\\python export_lm.py --wfmt i8 --out models\\lm_i8.tflite
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import tflb
from st import SafeTensors

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "..", "MOSS-VoiceGenerator")

GEN_SLOT, DELAY_SLOT = 151656, 151662


class LMExporter:
    def __init__(self, ckpt: str, wfmt: str, safe_norm: bool, kv: str = "host", flat_io: bool = True,
                 resid_scale: float = 1.0, mlp_scale: float = 1.0):
        with open(os.path.join(ckpt, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        lc = cfg["language_config"]
        self.cfg = cfg
        self.D = lc["hidden_size"]
        self.H = lc["num_attention_heads"]
        self.KV = lc["num_key_value_heads"]
        self.HD = lc["head_dim"]
        self.L = lc["num_hidden_layers"]
        self.FF = lc["intermediate_size"]
        self.eps = lc["rms_norm_eps"]
        self.n_vq = cfg["n_vq"]
        self.AV = cfg["audio_vocab_size"] + 1
        self.st = SafeTensors(os.path.join(ckpt, "model.safetensors"))
        self.wfmt = wfmt
        # Output heads (logits go straight to sampling) stay int8 when the transformer is int4.
        self.head_fmt = "i8" if wfmt.startswith("i4") else wfmt
        self.safe_norm = safe_norm
        self.kv = kv
        # fp16 GPUs: layer 2's MLP writes ~1.7e5 into one residual channel (and its input reaches ~8e4), past the
        # fp16 max. The residual stream runs scaled by c (input, o_proj and down_proj rows) and up_proj by c2
        # (undone in down_proj); RMSNorm is scale invariant (eps scaled by c^2).
        self.c, self.c2 = float(resid_scale), float(mlp_scale)
        self.lean = False
        self.fold_norm = False
        self.ablate: set[str] = set()   # debug timing (wrong numerics): norm, rope, attn
        self.lean_rope = "fc"           # fc: rotate_half as a 128x128 FC (4 ops); slices: 9 ops
        self.lean_qknorm = "safe"       # safe: the 7-op max-normalized core; plain: 4 ops on pre-scaled q|k (KL x3-8)
        self.model = tflb.Model("moss-voicegenerator backbone", flat_io=flat_io)
        self.n_heads_out = 2 + self.n_vq * self.AV

    # --------------------------------------------------------------- pieces
    def w(self, g: tflb.Graph, key: str, fmt=None, make=None):
        shape = self.st.shape(key) if make is None else None
        make = make or (lambda: self.st.get(key))
        if shape is None:
            shape = list(np.asarray(make()).shape)
        return g.weight(key, shape, make, fmt or self.wfmt)

    def rms(self, g: tflb.Graph, x: int, key: str, scale: float = 1.0, resid: bool = False) -> int:
        """RMSNorm; `resid` norms read the (possibly scaled) residual stream, so their eps scales by c^2."""
        D = g.shape(x)[-1]
        eps = self.eps * (self.c * self.c if resid else 1.0)
        if self.safe_norm:
            # Normalize by max|x| first: fp16 GPUs overflow x*x for the large activations Qwen carries.
            m = g.maximum(g.reduce_max(g.abs(x), [-1]), g.scalar(1e-4))
            inv = g.div(g.scalar(1.0), m)
            xs = g.mul(x, inv)
            ms = g.mean(g.mul(xs, xs), [-1])
            y = g.mul(xs, g.rsqrt(g.add(ms, g.mul(g.mul(inv, inv), g.scalar(eps)))))
        else:
            ms = g.mean(g.mul(x, x), [-1])
            y = g.mul(x, g.rsqrt(g.add(ms, g.scalar(eps))))
        w = g.weight(key + (f"*{scale}" if scale != 1.0 else ""), [D], lambda: self.st.get(key) * scale, tflb.F32)
        return g.mul(y, w)

    @staticmethod
    def rope(g: tflb.Graph, x: int, cos: int, sin: int) -> int:
        """Half-split (rotate_half) RoPE on the last axis; cos/sin hold the first half of the angles."""
        s = g.shape(x)
        h = s[-1] // 2
        x1 = g.slice(x, [0] * len(s), s[:-1] + [h])
        x2 = g.slice(x, [0] * (len(s) - 1) + [h], s[:-1] + [h])
        o1 = g.sub(g.mul(x1, cos), g.mul(x2, sin))
        o2 = g.add(g.mul(x2, cos), g.mul(x1, sin))
        return g.concat([o1, o2], -1)

    def heads_t(self, g: tflb.Graph, x: int, T: int, n: int) -> int:
        """[1,T,n*HD] -> [1,n,T,HD]."""
        x = g.reshape(x, [1, T, n, self.HD])
        return g.reshape(x, [1, n, T, self.HD]) if T == 1 else g.transpose(x, [0, 2, 1, 3])

    def layer(self, g, i, x, cos, sin, mask, kc, vc, T, P, idx=-1):
        p = f"language_model.layers.{i}."
        D, H, KV, HD = self.D, self.H, self.KV, self.HD
        c, c2 = self.c, self.c2
        xn = self.rms(g, x, p + "input_layernorm.weight", resid=True)
        q = g.reshape(g.fc(xn, self.w(g, p + "self_attn.q_proj.weight")), [1, T, H, HD])
        k = g.reshape(g.fc(xn, self.w(g, p + "self_attn.k_proj.weight")), [1, T, KV, HD])
        v = g.fc(xn, self.w(g, p + "self_attn.v_proj.weight"))
        q = self.rope(g, self.rms(g, q, p + "self_attn.q_norm.weight", HD ** -0.5), cos, sin)
        k = self.rope(g, self.rms(g, k, p + "self_attn.k_norm.weight"), cos, sin)
        q = g.reshape(q, [1, H, T, HD]) if T == 1 else g.transpose(q, [0, 2, 1, 3])
        k = g.reshape(k, [1, KV, T, HD]) if T == 1 else g.transpose(k, [0, 2, 1, 3])
        v = self.heads_t(g, v, T, KV)
        rep = H // KV
        q8 = g.reshape(q, [1, KV, rep * T, HD])
        if idx >= 0:
            # GPU: the new rows are written into the resident cache in-graph; attention runs over the P slots.
            kc = g.dynamic_update_slice(kc, k, idx)
            vc = g.dynamic_update_slice(vc, v, idx)
            pr = g.softmax(g.add(g.bmm(q8, kc, adj_y=True), mask))
            o = g.bmm(pr, vc)
            k, v = kc, vc
        else:
            # CPU: the cache is read only; one softmax over cache slots | own rows.
            sc = g.bmm(q8, kc, adj_y=True)                 # [1,KV,2T,P]
            sn = g.bmm(q8, k, adj_y=True)                  # [1,KV,2T,T]
            s = g.add(g.concat([sc, sn], -1), mask)
            pr = g.softmax(s)
            pc = g.slice(pr, [0, 0, 0, 0], [1, KV, rep * T, P])
            pn = g.slice(pr, [0, 0, 0, P], [1, KV, rep * T, T])
            o = g.add(g.bmm(pc, vc), g.bmm(pn, v))         # [1,KV,2T,HD]
        o = g.reshape(o, [1, H, T, HD])
        o = g.reshape(o, [1, T, D]) if T == 1 else g.reshape(g.transpose(o, [0, 2, 1, 3]), [1, T, D])
        x = g.add(x, g.fc(o, self.ws(g, p + "self_attn.o_proj.weight", c)))
        xn = self.rms(g, x, p + "post_attention_layernorm.weight", resid=True)
        gate = g.fc(xn, self.w(g, p + "mlp.gate_proj.weight"))
        up = g.fc(xn, self.ws(g, p + "mlp.up_proj.weight", c2))
        h = g.mul(g.mul(gate, g.logistic(gate)), up)
        x = g.add(x, g.fc(h, self.ws(g, p + "mlp.down_proj.weight", c / c2)))
        return x, k, v

    # ------------------------------------------------------------------ lean (ggml-style) layer
    # Fewer, cheaper ops for dispatch-bound GPUs (see EXPERIMENTS.md section 10):
    #  * RMSNorm weights folded into the columns of the FCs that read the norm (q/k/v, gate/up, the heads), so a
    #    residual norm is the 7-op fp16-safe core x/m * rsqrt(mean((x/m)^2)) (eps dropped: the mean is >= 1/D);
    #  * q and k from one FC whose rows are pre-scaled (q 1/32, k 1/128) so a plain 4-op RMSNorm stays inside fp16,
    #    one norm and one RoPE over all 24 heads, RoPE's rotate_half as an FC by a constant 128x128 matrix.
    SQ, SK = 1.0 / 32, 1.0 / 128

    def rms_core(self, g: tflb.Graph, x: int) -> int:
        xs = g.div(x, g.reduce_max(g.abs(x), [-1]))
        return g.mul(xs, g.rsqrt(g.mean(g.mul(xs, xs), [-1])))

    def norm_lean(self, g: tflb.Graph, x: int, key: str):
        """Residual RMSNorm for the lean layer -> (columns to fold into the next FCs, normed tensor). With fold_norm
        the weight goes into those columns (1 op less, but int8 per-row quantization of the folded weights costs
        ~4x the logit KL, measured); otherwise it is one MUL and the columns are ones."""
        w = self.st.get(key)
        core = x if "norm" in self.ablate else self.rms_core(g, x)
        if self.fold_norm:
            return w, core
        return np.ones_like(w), g.mul(core, g.weight(key, [w.size], lambda: w, tflb.F32))

    def colw(self, g: tflb.Graph, key: str, cols: np.ndarray, row_scale: float = 1.0, fmt=None):
        """FC weight `key` with the preceding norm's weight folded into its columns, rows scaled by row_scale."""
        return g.weight(f"{key}|fold{int(self.fold_norm)}*{row_scale}", self.st.shape(key),
                        lambda: self.st.get(key) * cols[None, :] * np.float32(row_scale), fmt or self.wfmt)

    @staticmethod
    def rot_matrix(hd: int) -> np.ndarray:
        """R with x @ R.T == rotate_half(x) == [-x2, x1]."""
        R = np.zeros((hd, hd), np.float32)
        h = hd // 2
        R[np.arange(h), np.arange(h) + h] = -1.0
        R[np.arange(h) + h, np.arange(h)] = 1.0
        return R

    def layer_lean(self, g, i, x, cosf, sinf, mask, kc, vc, T, P, idx, R):
        p = f"language_model.layers.{i}."
        D, H, KV, HD = self.D, self.H, self.KV, self.HD
        c, c2 = self.c, self.c2
        st = self.st
        w_in, xn = self.norm_lean(g, x, p + "input_layernorm.weight")
        fold = f"|fold{int(self.fold_norm)}"

        def make_qk():
            wq = st.get(p + "self_attn.q_proj.weight") * np.float32(self.SQ)
            wk = st.get(p + "self_attn.k_proj.weight") * np.float32(self.SK)
            return np.concatenate([wq, wk], 0) * w_in[None, :]

        qk = g.fc(xn, g.weight(p + "qk|lean" + fold, [(H + KV) * HD, D], make_qk, self.wfmt))
        v = g.fc(xn, self.colw(g, p + "self_attn.v_proj.weight", w_in))
        qk = g.reshape(qk, [1, T, H + KV, HD])
        wqk = np.concatenate([np.tile(st.get(p + "self_attn.q_norm.weight") * HD ** -0.5, (H, 1)),
                              np.tile(st.get(p + "self_attn.k_norm.weight"), (KV, 1))], 0)
        qkn = self.rms_core(g, qk) if self.lean_qknorm == "safe" else \
            g.mul(qk, g.rsqrt(g.mean(g.mul(qk, qk), [-1])))
        qk = g.mul(qkn, g.weight(p + "qk_norm|lean", [H + KV, HD], lambda: wqk, tflb.F32))
        if "rope" in self.ablate:
            pass
        elif self.lean_rope == "fc":
            qk = g.add(g.mul(qk, cosf[0]), g.mul(g.fc(qk, R), sinf[0]))
        else:
            qk = self.rope(g, qk, cosf[1], sinf[1])
        q = g.slice(qk, [0, 0, 0, 0], [1, T, H, HD])
        k = g.slice(qk, [0, 0, H, 0], [1, T, KV, HD])
        rep = H // KV
        if T == 1:
            q8 = g.reshape(q, [1, KV, rep, HD])
            k = g.reshape(k, [1, KV, 1, HD])
            v = g.reshape(v, [1, KV, 1, HD])
        else:
            q8 = g.reshape(g.transpose(q, [0, 2, 1, 3]), [1, KV, rep * T, HD])
            k = g.transpose(k, [0, 2, 1, 3])
            v = g.transpose(g.reshape(v, [1, T, KV, HD]), [0, 2, 1, 3])
        if "attn" in self.ablate:                       # timing only: no cache traffic, no attention
            o = g.add(q8, g.mul(v, g.scalar(0.0))) if T == 1 else q8
            k, v = g.add(kc, g.scalar(0.0)), g.add(vc, g.scalar(0.0))
        elif idx >= 0:
            kc = g.dynamic_update_slice(kc, k, idx)
            vc = g.dynamic_update_slice(vc, v, idx)
            o = g.bmm(g.softmax(g.add(g.bmm(q8, kc, adj_y=True), mask)), vc)
            k, v = kc, vc
        else:
            s = g.add(g.concat([g.bmm(q8, kc, adj_y=True), g.bmm(q8, k, adj_y=True)], -1), mask)
            pr = g.softmax(s)
            o = g.add(g.bmm(g.slice(pr, [0, 0, 0, 0], [1, KV, rep * T, P]), vc),
                      g.bmm(g.slice(pr, [0, 0, 0, P], [1, KV, rep * T, T]), v))
        o = g.reshape(o, [1, T, D]) if T == 1 else g.reshape(g.transpose(g.reshape(o, [1, H, T, HD]), [0, 2, 1, 3]),
                                                               [1, T, D])
        x = g.add(x, g.fc(o, self.ws(g, p + "self_attn.o_proj.weight", c)))
        w_post, xn = self.norm_lean(g, x, p + "post_attention_layernorm.weight")
        gate = g.fc(xn, self.colw(g, p + "mlp.gate_proj.weight", w_post))
        up = g.fc(xn, self.colw(g, p + "mlp.up_proj.weight", w_post, c2))
        h = g.mul(g.mul(gate, g.logistic(gate)), up)
        x = g.add(x, g.fc(h, self.ws(g, p + "mlp.down_proj.weight", c / c2)))
        return x, k, v

    def ws(self, g: tflb.Graph, key: str, s: float):
        """Weight with every row scaled by s (exact under per-row int8 quantization)."""
        if s == 1.0:
            return self.w(g, key)
        return g.weight(f"{key}*{s}", self.st.shape(key), lambda: self.st.get(key) * np.float32(s), self.wfmt)

    def heads_weight(self) -> np.ndarray:
        t = self.st.raw("lm_heads.0.weight")
        rows = [(t[[GEN_SLOT, DELAY_SLOT]].astype(np.uint32) << 16).view(np.float32)]
        rows += [self.st.get(f"lm_heads.{i + 1}.weight") for i in range(self.n_vq)]
        return np.concatenate(rows, 0)

    def body(self, name: str, T: int, P: int) -> tflb.Graph:
        g = self.model.graph(name)
        D, KV, HD, L = self.D, self.KV, self.HD, self.L
        rep = self.H // KV
        dus = self.kv == "dus"
        x = g.input(tflb.F32, [1, T, D], "x")
        cos = g.input(tflb.F32, [1, T, 1, HD // 2], "cos")
        sin = g.input(tflb.F32, [1, T, 1, HD // 2], "sin")
        mask = g.input(tflb.F32, [1, 1, rep * T, P if dus else P + T], "mask")
        idx = g.input(tflb.I32, [4], "idx") if dus else -1      # DUS start [0, 0, slot, 0]
        kcs = [g.input(tflb.F32, [1, KV, P, HD], f"k{i}") for i in range(L)]
        vcs = [g.input(tflb.F32, [1, KV, P, HD], f"v{i}") for i in range(L)]
        if self.c != 1.0:
            x = g.mul(x, g.scalar(self.c))
        ks, vs = [], []
        if self.lean:
            cosf, sinf = (g.concat([cos, cos], -1), cos), (g.concat([sin, sin], -1), sin)   # (full width, half)
            R = g.weight("rope_rot", [HD, HD], lambda: self.rot_matrix(HD), tflb.F32)
        for i in range(L):
            if self.lean:
                x, k, v = self.layer_lean(g, i, x, cosf, sinf, mask, kcs[i], vcs[i], T, P, idx, R)
            else:
                x, k, v = self.layer(g, i, x, cos, sin, mask, kcs[i], vcs[i], T, P, idx)
            ks.append(k)
            vs.append(v)
        if T > 1:
            x = g.slice(x, [0, T - 1, 0], [1, 1, D])
        if self.lean:
            # With fold_norm, hn is the unweighted core and the final norm's weight sits in the heads' and the text
            # head's columns.
            wf, hn = self.norm_lean(g, x, "language_model.norm.weight")
            hw = g.weight(f"heads|fold{int(self.fold_norm)}", [self.n_heads_out, D],
                          lambda: self.heads_weight() * wf[None, :], self.head_fmt)
        else:
            hn = self.rms(g, x, "language_model.norm.weight", resid=True)
            hw = g.weight("heads", [self.n_heads_out, D], self.heads_weight, self.head_fmt)
        logits = g.fc(hn, hw)
        g.output(logits, "logits")
        g.output(hn, "hn")
        if dus:
            for i in range(L):
                g.output(ks[i], f"nk{i}")
            for i in range(L):
                g.output(vs[i], f"nv{i}")
        else:
            g.output(g.concat(ks, 0), "k_new")      # [L,KV,T,HD]
            g.output(g.concat(vs, 0), "v_new")
        self.model.signature(name, g)
        return g

    def text_head(self):
        g = self.model.graph("text_head")
        hn = g.input(tflb.F32, [1, 1, self.D], "hn")
        if self.lean and self.fold_norm:
            w = self.colw(g, "lm_heads.0.weight", self.st.get("language_model.norm.weight"), fmt=self.head_fmt)
        else:
            w = self.w(g, "lm_heads.0.weight", self.head_fmt)
        g.output(g.fc(hn, w), "logits")
        self.model.signature("text_head", g)


def export_assets(st: SafeTensors, out_dir: str, n_vq: int):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "emb_text_bf16.npy")
    if not os.path.exists(p):
        np.save(p, np.asarray(st.raw("language_model.embed_tokens.weight")))
    p = os.path.join(out_dir, "emb_audio_f32.npy")
    if not os.path.exists(p):
        np.save(p, np.stack([st.get(f"emb_ext.{i}.weight") for i in range(n_vq)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default=os.path.join(HERE, "models", "lm_i8.tflite"))
    ap.add_argument("--wfmt", default="i8", choices=["i4", "i4b32", "i4b64", "i4b128", "i8", "f16", "f32"])
    ap.add_argument("--caps", default="512,1024,2048")
    ap.add_argument("--prefill", default="64", help="prefill row counts")
    ap.add_argument("--prefill-caps", default="", help="capacities that get prefill graphs (default: all)")
    ap.add_argument("--safe-norm", action="store_true", help="fp16-safe RMSNorm (GPU)")
    ap.add_argument("--kv", default="host", choices=["host", "dus"],
                    help="host: read-only caches, new rows returned (CPU); dus: caches updated in-graph (GPU)")
    ap.add_argument("--ranked-io", action="store_true",
                    help="ranked signature I/O (GPU managed buffers) instead of flat 1-D (CPU zero-copy)")
    ap.add_argument("--resid-scale", type=float, default=1.0, help="residual stream scale (fp16 GPU: 0.125)")
    ap.add_argument("--mlp-scale", type=float, default=1.0, help="up_proj scale (fp16 GPU: 0.125)")
    ap.add_argument("--layers", type=int, default=0, help="debug: only the first N layers")
    ap.add_argument("--no-text-head", action="store_true")
    ap.add_argument("--lean", action="store_true", help="ggml-style lean layers (fused q|k norm + RoPE, cheaper norms)")
    ap.add_argument("--lean-rope", default="fc", choices=["fc", "slices"])
    ap.add_argument("--lean-qknorm", default="safe", choices=["plain", "safe"])
    ap.add_argument("--ablate", default="", help="debug timing only (lean): comma list of norm,rope,attn")
    args = ap.parse_args()

    t0 = time.time()
    ex = LMExporter(args.ckpt, args.wfmt, args.safe_norm, args.kv, not args.ranked_io, args.resid_scale,
                    args.mlp_scale)
    ex.lean = args.lean
    ex.lean_rope, ex.lean_qknorm = args.lean_rope, args.lean_qknorm
    ex.ablate = {a for a in args.ablate.split(",") if a}
    if args.layers:
        ex.L = args.layers
    caps = [int(c) for c in args.caps.split(",") if c]
    pre = [int(t) for t in args.prefill.split(",") if t]
    for P in caps:
        ex.body(f"step_p{P}", 1, P)
    for P in [int(c) for c in args.prefill_caps.split(",") if c] or caps:
        for T in pre:
            ex.body(f"prefill_t{T}_p{P}", T, P)
    if not args.no_text_head:
        ex.text_head()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    size = ex.model.save(args.out)
    export_assets(ex.st, os.path.dirname(os.path.abspath(args.out)), ex.n_vq)
    print(f"wrote {args.out}: {size / 1e6:.1f} MB, {len(ex.model.graphs)} signatures, "
          f"{len(ex.model.buffers)} buffers, {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
