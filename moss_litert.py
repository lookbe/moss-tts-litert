"""MOSS-VoiceGenerator on LiteRT: tokenizer + prompt, backbone (prefill / step / text head), delay-pattern sampling
and the streaming codec decoder. Torch-free (numpy + tokenizers + ai_edge_litert).

  .venv\\Scripts\\python moss_litert.py --name fox --out out\\fox.wav
  .venv\\Scripts\\python moss_litert.py --text "..." --instruction "..." --out x.wav [--accel gpu]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VG = os.path.join(HERE, "..", "MOSS-VoiceGenerator")
MODELS = os.path.join(HERE, "models")

# Token ids (MOSS-VoiceGenerator/config.json, configuration_moss_tts.py)
PAD, IM_START, IM_END = 151643, 151644, 151645
AUDIO_START, AUDIO_END = 151652, 151653
GEN_SLOT, DELAY_SLOT = 151656, 151662
AUDIO_PAD = 1024
N_VQ = 16
AV = 1025
NEG = -30000.0          # additive mask value (fp16-safe)


# ======================================================================== prompt
def normalize_instruction(s: str) -> str:
    if not s:
        return s
    s = s.replace("\n", " ")
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"\{.*?\}", "", s)
    for ch in "【】《》（）『』「」～-_":
        s = s.replace(ch, "，")
    s = re.sub(r'([，。！？,.!?;；])+', r'\1', s)
    if re.search(r'[一-鿿]', s):
        s = s.replace(',', '，')
    return s.strip()


def normalize_text(s: str) -> str:
    if not s:
        return s
    s = s.replace("\n", " ")
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"\{.*?\}", "", s)
    for ch in "【】《》（）『』「」～":
        s = s.replace(ch, "，")
    for q in ['"', '"', '"']:
        s = s.replace(q, "")
    s = re.sub(r'([，。！？,.!?;；])+', r'\1', s)
    return s.strip()


USER_TEMPLATE = """<user_inst>
- Reference(s):
None
- Instruction:
{instruction}
- Tokens:
None
- Quality:
None
- Sound Event:
None
- Ambient Sound:
None
- Language:
None
- Text:
{text}
</user_inst>"""


class Prompter:
    def __init__(self, vg_dir: str = VG):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(vg_dir, "tokenizer.json"))

    def build(self, text: str, instruction: str, normalize: bool = False) -> np.ndarray:
        """[S,17] int64 like MossTTSDelayProcessor(mode='generation') for one user message.

        normalize defaults to False like upstream's actual path: README and web app build the message with
        processor.build_user_message() (normalize=False), and the processor passes built messages through
        unchanged, so its normalize_inputs=True never applies. (Normalizing turns "middle-aged" into
        "middle，aged".)"""
        if normalize:
            text, instruction = normalize_text(text), normalize_instruction(instruction)
        content = USER_TEMPLATE.replace("{instruction}", str(instruction)).replace("{text}", str(text))
        chat = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        ids = np.array(self.tok.encode(chat, add_special_tokens=True).ids, np.int64)
        rows = np.full((len(ids), 1 + N_VQ), AUDIO_PAD, np.int64)
        rows[:, 0] = ids
        return rows


# ======================================================================== litert helpers
_ENV = None


def shared_env():
    """One LiteRT environment for every compiled model of the process. Each CompiledModel.from_file without one
    creates its own environment (on the GPU its own WebGPU device); alternating two devices made every codec call
    3-8x slower while the backbone ran between them (EXPERIMENTS.md section 10)."""
    global _ENV
    if _ENV is None:
        import ai_edge_litert
        from ai_edge_litert.environment import Environment, EnvironmentOptions
        _ENV = Environment.create(options=EnvironmentOptions(
            runtime_path=os.path.dirname(os.path.abspath(ai_edge_litert.__file__))))
    return _ENV


def litert_model(path: str, accel: str, threads: int):
    """CompiledModel + the exporter's sidecar JSON (logical I/O shapes; the graphs take flat 1-D I/O)."""
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator as HA
    from ai_edge_litert.options import Options, CpuOptions
    if accel == "gpu":
        opts = Options(HA.GPU)
    else:
        # XNNPACK weight cache: without it every signature packs its own copy of the weights (backbone 11 GB RSS,
        # fp16 codec +20 GB); with a cache *file* they share one mmapped copy and reloads take 0.3 s. The name
        # carries the model's size and mtime so a re-export never reuses a stale cache. MOSS_XNNPACK_CACHE=off
        # disables it. Experiment hooks: MOSS_XNNPACK_FLAGS (int), MOSS_YNNPACK=1.
        cache = os.environ.get("MOSS_XNNPACK_CACHE", "")
        if cache != "off":
            st = os.stat(path)
            cache = f"{os.path.splitext(path)[0]}.{st.st_size}_{int(st.st_mtime)}.xnnpack_cache"
        else:
            cache = ""
        flags = os.environ.get("MOSS_XNNPACK_FLAGS")
        opts = Options(HA.CPU, cpu_options=CpuOptions(
            num_threads=threads, xnnpack_flags=int(flags, 0) if flags else None,
            enable_ynnpack=os.environ.get("MOSS_YNNPACK") == "1", xnnpack_weight_cache_path=cache))
    with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
        io = json.load(f)
    return CompiledModel.from_file(path, options=opts, environment=shared_env()), io


def aligned_zeros(shape, dtype=np.float32, align=64) -> np.ndarray:
    n = int(np.prod(shape))
    it = np.dtype(dtype).itemsize
    raw = np.zeros(n + align // it, dtype)
    off = (-raw.ctypes.data % align) // it
    return raw[off:off + n].reshape(shape)


class Sig:
    """One signature: its I/O names, logical shapes (sidecar JSON) and buffer helpers."""

    def __init__(self, cm, io: dict, key: str):
        self.cm, self.key = cm, key
        self.io = io["signatures"][key]
        self.in_names, self.out_names = list(self.io["inputs"]), list(self.io["outputs"])

    def in_shape(self, n):
        return tuple(self.io["inputs"][n])

    def out_shape(self, n):
        return tuple(self.io["outputs"][n])

    def host(self, arr: np.ndarray):
        """Zero-copy 1-D view of a 64-byte aligned array (the graphs reshape inside)."""
        from ai_edge_litert.tensor_buffer import TensorBuffer
        assert arr.ctypes.data % 64 == 0 and arr.flags.c_contiguous
        return TensorBuffer.create_from_host_memory(arr.reshape(-1))

    def run(self, ins: dict, outs: dict):
        self.cm.run_by_name(self.key, ins, outs)


# ======================================================================== backbone
class Embedder:
    def __init__(self, models: str = MODELS):
        self.text = np.load(os.path.join(models, "emb_text_bf16.npy"), mmap_mode="r")
        self.audio = np.load(os.path.join(models, "emb_audio_f32.npy"))          # [16,1025,2048]

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(rows)
        x = (self.text[rows[:, 0]].astype(np.uint32) << 16).view(np.float32)
        for i in range(N_VQ):
            x = x + self.audio[i][rows[:, 1 + i]]
        return x.astype(np.float32)


def rope_qwen(pos: np.ndarray, hd: int = 128, theta: float = 1e6):
    inv = 1.0 / theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.asarray(pos, np.float64)[:, None] * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


class Backbone:
    """CPU path: caches are zero-copy host arrays; the graphs return the new K/V rows and the host writes them."""

    def __init__(self, path: str, accel: str = "cpu", threads: int = 8, models: str = MODELS):
        t0 = time.time()
        self.cm, io = litert_model(path, accel, threads)
        self.load_s = time.time() - t0
        self.accel = accel
        sl = io["signatures"]
        self.steps = {int(k.split("_p")[1]): Sig(self.cm, io, k) for k in sl if k.startswith("step_p")}
        self.prefills = {}
        for k in sl:
            if k.startswith("prefill_t"):
                T, P = (int(v) for v in re.match(r"prefill_t(\d+)_p(\d+)", k).groups())
                self.prefills[(T, P)] = Sig(self.cm, io, k)
        self.text_head = Sig(self.cm, io, "text_head") if "text_head" in sl else None
        any_step = next(iter(self.steps.values()))
        self.L = sum(1 for n in any_step.in_names if re.fullmatch(r"k\d+", n))
        _, self.KV, _, self.HD = any_step.in_shape("k0")
        self.D = any_step.in_shape("x")[-1]
        self.NO = any_step.out_shape("logits")[-1]
        self.emb = Embedder(models)
        self.P = 0
        self.n = 0
        self._bufs: dict = {}

    # ------------------------------------------------------------- caches
    def _alloc(self, P: int):
        old = (self.P, getattr(self, "K", None), getattr(self, "V", None))
        self.K = [aligned_zeros((1, self.KV, P, self.HD)) for _ in range(self.L)]
        self.V = [aligned_zeros((1, self.KV, P, self.HD)) for _ in range(self.L)]
        if old[1] is not None:
            for l in range(self.L):
                self.K[l][:, :, :old[0]] = old[1][l]
                self.V[l][:, :, :old[0]] = old[2][l]
        self.P = P
        self._bufs = {}

    def _ensure(self, need: int):
        if need <= self.P:
            return
        caps = sorted(self.steps)
        fit = [P for P in caps if P >= need]
        if not fit:
            raise RuntimeError(f"context {need} exceeds the largest capacity {caps[-1]}")
        self._alloc(fit[0])

    def reset(self):
        self.P, self.n = 0, 0
        self.K = self.V = None
        self._bufs = {}

    def _io(self, sig: Sig, T: int):
        """Host buffers for a signature at the current capacity (cached)."""
        key = (sig.key, self.P)
        if key in self._bufs:
            return self._bufs[key]
        a = {"x": aligned_zeros(sig.in_shape("x")), "cos": aligned_zeros(sig.in_shape("cos")),
             "sin": aligned_zeros(sig.in_shape("sin")), "mask": aligned_zeros(sig.in_shape("mask")),
             "logits": aligned_zeros(sig.out_shape("logits")), "hn": aligned_zeros(sig.out_shape("hn")),
             "k_new": aligned_zeros(sig.out_shape("k_new")), "v_new": aligned_zeros(sig.out_shape("v_new"))}
        ins = {n: sig.host(a[n]) for n in ("x", "cos", "sin", "mask")}
        for l in range(self.L):
            ins[f"k{l}"] = sig.host(self.K[l])
            ins[f"v{l}"] = sig.host(self.V[l])
        outs = {n: sig.host(a[n]) for n in ("logits", "hn", "k_new", "v_new")}
        self._bufs[key] = (a, ins, outs)
        return self._bufs[key]

    # ------------------------------------------------------------- run
    def prefill(self, rows: np.ndarray, T: int | None = None):
        """Prompt rows [S,17] -> (logits [NO], hn [D]) of the last row."""
        S = rows.shape[0]
        self.reset()
        Ts = sorted({t for t, _ in self.prefills})
        T = T or Ts[-1]
        nch = -(-S // T)
        self._ensure(nch * T + 1)
        caps = sorted(P for t, P in self.prefills if t == T and P >= self.P)
        if not caps:
            raise RuntimeError(f"no prefill graph with T={T} for capacity {self.P}")
        if caps[0] != self.P:
            self._alloc(caps[0])
        sig = self.prefills[(T, self.P)]
        a, ins, outs = self._io(sig, T)
        x_all = self.emb(rows)
        pad = nch * T - S
        rep = sig.in_shape("mask")[2] // T
        for c in range(nch):
            g = np.arange(c * T, (c + 1) * T) - pad           # global prompt row, < 0 = pad
            real = g >= 0
            gi = np.where(real, g, 0)
            a["x"][0] = x_all[gi]
            cs, sn = rope_qwen(gi.astype(np.float64), self.HD)
            a["cos"][0, :, 0] = cs
            a["sin"][0, :, 0] = sn
            m = np.full((T, self.P + T), NEG, np.float32)
            m[:, :self.n] = 0.0
            own = np.tril(np.ones((T, T), bool)) & real[None, :]
            own[~real] = np.eye(T, dtype=bool)[~real]           # pad rows see themselves only
            m[:, self.P:][own] = 0.0
            m[~real, :self.P] = NEG
            a["mask"][0, 0] = np.tile(m, (rep, 1))
            sig.run(ins, outs)
            kn, vn = a["k_new"], a["v_new"]                     # [L,KV,T,HD]
            idx = np.nonzero(real)[0]
            slots = g[idx]
            for l in range(self.L):
                self.K[l][0][:, slots] = kn[l][:, idx]
                self.V[l][0][:, slots] = vn[l][:, idx]
            self.n = int(slots[-1]) + 1 if len(slots) else self.n
        return a["logits"][0, 0].copy(), a["hn"][0, 0].copy()

    def step(self, row: np.ndarray):
        """One generated row [17] at position self.n -> (logits, hn)."""
        self._ensure(self.n + 1)
        sig = self.steps[self.P]
        a, ins, outs = self._io(sig, 1)
        a["x"][0, 0] = self.emb(row[None])[0]
        cs, sn = rope_qwen(np.array([self.n], np.float64), self.HD)
        a["cos"][0, 0, 0] = cs[0]
        a["sin"][0, 0, 0] = sn[0]
        m = a["mask"][0, 0]
        m[:, :self.n] = 0.0
        m[:, self.n:self.P] = NEG
        m[:, self.P] = 0.0
        sig.run(ins, outs)
        kn, vn = a["k_new"], a["v_new"]
        for l in range(self.L):
            self.K[l][0, :, self.n] = kn[l, :, 0]
            self.V[l][0, :, self.n] = vn[l, :, 0]
        self.n += 1
        return a["logits"][0, 0].copy(), a["hn"][0, 0].copy()

    def text_logits(self, hn: np.ndarray) -> np.ndarray:
        sig = self.text_head
        if "text" not in self._bufs:
            h = aligned_zeros(sig.in_shape("hn"))
            o = aligned_zeros(sig.out_shape("logits"))
            self._bufs["text"] = (h, o, {"hn": sig.host(h)}, {"logits": sig.host(o)})
        h, o, ins, outs = self._bufs["text"]
        h[0, 0] = hn
        sig.run(ins, outs)
        return o[0, 0].copy()


class GpuBackbone:
    """GPU path (export --kv dus --ranked-io): caches stay on the device in two banks of managed buffers; each call
    writes its rows in-graph (DYNAMIC_UPDATE_SLICE at `idx`) and outputs the updated caches into the other bank.
    Slots are rows in call order (prefill pad rows included, masked forever); RoPE uses the token position."""

    def __init__(self, path: str, threads: int = 8, models: str = MODELS, enforce_f32: bool = False):
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator as HA
        from ai_edge_litert.options import Options
        from ai_edge_litert.gpu_options import GpuOptions
        with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
            io = json.load(f)
        t0 = time.time()
        self.cm = CompiledModel.from_file(path, options=Options(HA.GPU, gpu_options=GpuOptions(enforce_f32=enforce_f32)),
                                          environment=shared_env())
        self.load_s = time.time() - t0
        self.fully = self.cm.is_fully_accelerated()
        sl = io["signatures"]
        self.steps = {int(k.split("_p")[1]): Sig(self.cm, io, k) for k in sl if k.startswith("step_p")}
        self.prefills = {}
        for k in sl:
            if k.startswith("prefill_t"):
                T, P = (int(v) for v in re.match(r"prefill_t(\d+)_p(\d+)", k).groups())
                self.prefills[(T, P)] = Sig(self.cm, io, k)
        self.text_head = Sig(self.cm, io, "text_head") if "text_head" in sl else None
        s = next(iter(self.steps.values()))
        self.L = sum(1 for n in s.in_names if re.fullmatch(r"k\d+", n))
        _, self.KV, _, self.HD = s.in_shape("k0")
        self.D = s.in_shape("x")[-1]
        self.NO = s.out_shape("logits")[-1]
        self.emb = Embedder(models)
        self.P = 0
        self._small = {}
        self.max_need = 0

    def _small_io(self, sig: Sig):
        if sig.key not in self._small:
            ins = {n: self.cm.create_input_buffer_by_name(sig.key, n) for n in ("x", "cos", "sin", "mask", "idx")}
            outs = {n: self.cm.create_output_buffer_by_name(sig.key, n) for n in ("logits", "hn")}
            self._small[sig.key] = (ins, outs)
        return self._small[sig.key]

    def _alloc(self, P: int):
        old = None
        if self.P:
            old = [np.array(b.read(self.KV * self.P * self.HD, np.float32)).reshape(1, self.KV, self.P, self.HD)
                   for b in self.banks[self.cur]]
        key = f"step_p{P}"
        names = [f"k{l}" for l in range(self.L)] + [f"v{l}" for l in range(self.L)]
        a = [self.cm.create_input_buffer_by_name(key, n) for n in names]
        b = [self.cm.create_output_buffer_by_name(key, "n" + n) for n in names]
        z = np.zeros(self.KV * P * self.HD, np.float32)       # OpenCL does not zero new buffers
        for buf in a + b:
            buf.write(z)
        if old is not None:
            for buf, arr in zip(a, old):
                full = np.zeros((1, self.KV, P, self.HD), np.float32)
                full[:, :, :self.P] = arr
                buf.write(full.ravel())
        self.names = names
        self.banks = [a, b]
        self.cur = 0
        self.P = P

    def _run(self, sig: Sig, x, pos, slot0: int, mask):
        ins, outs = self._small_io(sig)
        ins["x"].write(x.ravel())
        cs, sn = rope_qwen(pos, self.HD)
        ins["cos"].write(cs.ravel())
        ins["sin"].write(sn.ravel())
        ins["mask"].write(mask.ravel())
        ins["idx"].write(np.array([0, 0, slot0, 0], np.int32))
        imap = dict(ins)
        imap.update(zip(self.names, self.banks[self.cur]))
        omap = dict(outs)
        omap.update(zip(["n" + n for n in self.names], self.banks[1 - self.cur]))
        sig.run(imap, omap)
        self.cur = 1 - self.cur
        lg = np.array(outs["logits"].read(self.NO, np.float32))
        hn = np.array(outs["hn"].read(self.D, np.float32))
        return lg, hn

    def _ensure(self, need: int):
        if need <= self.P:
            return
        fit = [P for P in sorted(self.steps) if P >= need]
        if not fit:
            raise RuntimeError(f"context {need} exceeds the largest capacity {max(self.steps)}")
        self._alloc(fit[0])

    def prefill(self, rows: np.ndarray, T: int | None = None):
        S = rows.shape[0]
        Ts = sorted({t for t, _ in self.prefills})
        # One call when a prefill graph fits the prompt (multi-row FCs are far more efficient with more rows),
        # chunks of the largest otherwise.
        T = T or next((t for t in Ts if t >= S), Ts[-1])
        nch = -(-S // T)
        need = max(nch * T + 1, self.max_need)
        caps = sorted(P for t, P in self.prefills if t == T and P >= need)
        if not caps:
            raise RuntimeError(f"no prefill graph with T={T} for {need} slots")
        if self.P != caps[0]:
            self.P = 0                  # fresh buffers, nothing to carry over
            self._alloc(caps[0])
        # else: reuse the device caches as they are; every slot not written by this utterance is masked
        sig = self.prefills[(T, self.P)]
        x_all = self.emb(rows)
        self.pad = nch * T - S
        rep = sig.in_shape("mask")[2] // T
        for c in range(nch):
            slots = np.arange(c * T, (c + 1) * T)
            g = slots - self.pad
            real = g >= 0
            x = x_all[np.where(real, g, 0)][None]
            j = np.arange(self.P)
            m = (j[None, :] >= self.pad) & (j[None, :] <= slots[:, None])
            m[~real] = j[None, :] == slots[~real][:, None]
            mask = np.where(m, 0.0, NEG).astype(np.float32)
            lg, hn = self._run(sig, x, np.where(real, g, 0).astype(np.float64), c * T, np.tile(mask, (rep, 1)))
        self.slot = nch * T
        return lg, hn

    def step(self, row: np.ndarray):
        self._ensure(self.slot + 1)
        sig = self.steps[self.P]
        x = self.emb(row[None])[None]
        j = np.arange(self.P)
        m = np.where((j >= self.pad) & (j <= self.slot), 0.0, NEG).astype(np.float32)
        lg, hn = self._run(sig, x, np.array([self.slot - self.pad], np.float64), self.slot, np.tile(m, (2, 1)))
        self.slot += 1
        return lg, hn

    def text_logits(self, hn: np.ndarray) -> np.ndarray:
        sig = self.text_head
        if "text" not in self._small:
            self._small["text"] = ({"hn": self.cm.create_input_buffer_by_name("text_head", "hn")},
                                   {"logits": self.cm.create_output_buffer_by_name("text_head", "logits")})
        ins, outs = self._small["text"]
        ins["hn"].write(np.asarray(hn, np.float32).ravel())
        sig.run(ins, outs)
        return np.array(outs["logits"].read(int(np.prod(sig.out_shape("logits"))), np.float32))


# ======================================================================== sampling (upstream inference_utils port)
def _softmax(x):
    m = np.max(x, -1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)


def sample_rows(logits: np.ndarray, rng: np.random.Generator, top_k: int, top_p: float,
                seen: np.ndarray | None = None, penalty: float = 1.0, do_sample: bool = True) -> np.ndarray:
    """logits [N,V] (already temperature-scaled) -> [N] tokens. `seen` [V] bool: the repetition-penalty set, one
    for every row, as upstream's 2-D path does. Vectorized over rows: top-k by one argpartition, then softmax,
    top-p and inverse-CDF sampling over the k candidates only (softmax after top-k masking involves only them, so
    this equals upstream's chain on the full row)."""
    l = logits.astype(np.float64, copy=True)
    if seen is not None and penalty != 1.0:
        l = np.where(seen & (l > 0), l / penalty, np.where(seen, l * penalty, l))
    if not do_sample:
        return l.argmax(-1)
    N, V = l.shape
    k = min(top_k, V) if top_k and top_k > 0 else V
    rows = np.arange(N)[:, None]
    cand = np.argpartition(-l, k - 1, axis=1)[:, :k] if k < V else np.broadcast_to(np.arange(V), (N, V))
    cv = l[rows, cand]
    order = np.argsort(-cv, axis=1, kind="stable")
    cand = cand[rows, order]
    cv = cv[rows, order]
    p = np.exp(cv - cv[:, :1])                        # cv[:, 0] is the row max (or -inf when all are masked)
    p /= p.sum(1, keepdims=True)
    if top_p is not None and top_p < 1.0:
        cum = np.cumsum(p, 1)
        remove = np.zeros_like(cum, bool)
        remove[:, 1:] = cum[:, :-1] > top_p           # shift right: keep the first token past top_p
        p = np.where(remove, 0.0, p)
        p /= p.sum(1, keepdims=True)
    c = np.cumsum(p, 1)
    u = rng.random(N)[:, None] * c[:, -1:]
    pick = np.minimum((c <= u).sum(1), k - 1)
    return cand[np.arange(N), pick].astype(np.int64)


class DelaySampler:
    """MossTTSDelayModel.generate() for batch 1, row by row."""

    def __init__(self, prompt: np.ndarray, seed: int, text_temperature=1.5, text_top_p=1.0, text_top_k=50,
                 audio_temperature=1.5, audio_top_p=0.6, audio_top_k=50, audio_repetition_penalty=1.1):
        self.rng = np.random.default_rng(seed)
        self.tt, self.tp, self.tk = text_temperature, text_top_p, text_top_k
        self.at, self.ap, self.ak, self.rp = audio_temperature, audio_top_p, audio_top_k, audio_repetition_penalty
        # Repetition-penalty sets as [V] masks: channel 0 over its own history, channels 1-15 one shared set.
        self.seen0 = np.zeros(AV, bool)
        self.seen0[np.unique(prompt[:, 1])] = True
        self.seen_rest = np.zeros(AV, bool)
        self.seen_rest[np.unique(prompt[:, 2:])] = True
        self.audio_len = 0
        last = int(prompt[-1, 0])
        if last in (AUDIO_START, GEN_SLOT):
            idx = np.nonzero(prompt[:, 0] == AUDIO_START)[0]
            if len(idx):
                self.audio_len = prompt.shape[0] - int(idx[-1])
        self.is_audio = self.audio_len > 0
        self.delayed = None                                   # None = int64 max
        self.stopping = False
        self.t = 0

    def needs_full_text(self) -> bool:
        """True when the next text token is sampled from the whole vocabulary (needs the text head)."""
        forced = self.delayed is not None and self.delayed <= N_VQ
        return not forced and not self.stopping and not self.is_audio

    def next_row(self, logits: np.ndarray, text_logits: np.ndarray | None = None) -> np.ndarray:
        n = N_VQ
        if self.delayed is not None and self.delayed < n:
            text = DELAY_SLOT
        elif self.delayed == n:
            text = AUDIO_END
            self.is_audio = False
        else:
            if self.is_audio:
                # Only gen_slot / delay_slot survive the mask: sample over those two logits.
                l2 = logits[:2] / self.tt
                if self.t == 0:
                    l2 = l2.copy(); l2[1] = -np.inf
                pick = sample_rows(l2[None], self.rng, self.tk, self.tp, do_sample=self.tt > 0)[0]
                text = (GEN_SLOT, DELAY_SLOT)[pick]
            else:
                l = text_logits.astype(np.float64) / self.tt
                l[[PAD, GEN_SLOT, DELAY_SLOT, AUDIO_END]] = -np.inf
                if self.t == 0:
                    l[DELAY_SLOT] = -np.inf
                if self.t <= n:
                    l[IM_END] = -np.inf
                text = int(sample_rows(l[None], self.rng, self.tk, self.tp, do_sample=self.tt > 0)[0])
        if text == AUDIO_START:
            self.is_audio = True
        if text == IM_END:
            self.stopping = True

        audio = np.full(n, AUDIO_PAD, np.int64)
        pre = np.arange(n) < self.audio_len
        post = np.ones(n, bool) if self.delayed is None else np.arange(n) > self.delayed - 1
        mask = pre & post
        if mask.any():
            al = logits[2:].reshape(n, AV) / self.at
            al[:, AUDIO_PAD] = -np.inf
            if mask[0]:
                audio[0] = sample_rows(al[0:1], self.rng, self.ak, self.ap, self.seen0, self.rp)[0]
            rest = np.nonzero(mask[1:])[0] + 1
            if len(rest):
                audio[rest] = sample_rows(al[rest], self.rng, self.ak, self.ap, self.seen_rest, self.rp)
        if text in (AUDIO_START, GEN_SLOT, DELAY_SLOT):
            self.audio_len += 1
        if text == AUDIO_END:
            self.audio_len = 0
        if self.delayed is None and text == DELAY_SLOT:
            self.delayed = 0
        if self.delayed is not None:
            self.delayed += 1
            if self.delayed > n:
                self.delayed = None
        self.t += 1
        self.seen0[audio[0]] = True
        self.seen_rest[audio[1:]] = True
        row = np.empty(1 + n, np.int64)
        row[0] = text
        row[1:] = audio
        return row


def de_delay(gen_rows: np.ndarray) -> np.ndarray:
    """Generated rows [N,17] -> frames [F,16] of the first audio segment (the processor splits segments at all-pad
    frames; upstream's examples use audio_codes_list[0])."""
    d = gen_rows[:, 1:]
    T = d.shape[0] - N_VQ + 1
    if T <= 0:
        return np.zeros((0, N_VQ), np.int64)
    codes = np.stack([d[i:i + T, i] for i in range(N_VQ)], 1)
    real = np.nonzero(~(codes == AUDIO_PAD).all(1))[0]
    if len(real) == 0:
        return np.zeros((0, N_VQ), np.int64)
    breaks = np.nonzero(np.diff(real) != 1)[0]
    end = real[breaks[0]] + 1 if len(breaks) else real[-1] + 1
    return codes[real[0]:end]


# ======================================================================== codec decoder
class CodecDecoder:
    """Streaming decoder: frames [F,16] -> 24 kHz audio, in blocks of B frames, with ping-pong state banks:
    zero-copy host arrays on the CPU (flat-I/O export), managed device buffers on the GPU (--ranked-io export)."""

    def __init__(self, path: str, accel: str = "cpu", threads: int = 8, models: str = MODELS,
                 enforce_f32: bool = False):
        t0 = time.time()
        if accel == "gpu":
            from ai_edge_litert.compiled_model import CompiledModel
            from ai_edge_litert.hardware_accelerator import HardwareAccelerator as HA
            from ai_edge_litert.options import Options
            from ai_edge_litert.gpu_options import GpuOptions
            with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
                io = json.load(f)
            self.cm = CompiledModel.from_file(
                path, options=Options(HA.GPU, gpu_options=GpuOptions(enforce_f32=enforce_f32)),
                environment=shared_env())
        else:
            self.cm, io = litert_model(path, accel, threads)
        self.load_s = time.time() - t0
        self.accel = accel
        self.host_mode = accel == "cpu" and io["flat_io"]
        with open(os.path.join(models, "codec_stages.json"), encoding="utf-8") as f:
            self.stages = json.load(f)
        self.tables = np.load(os.path.join(models, "codec_tables.npy"))      # [16,1024,768]
        self.bias = np.load(os.path.join(models, "codec_bias.npy"))
        self.sigs = {int(k[5:]): Sig(self.cm, io, k) for k in io["signatures"] if k.startswith("dec_b")}
        big = self.sigs[max(self.sigs)]
        self.state_names = [n for n in big.in_names if re.fullmatch(r"s\d+[kv]\d+", n)]
        self.state_shapes = {n: big.in_shape(n) for n in self.state_names}
        if self.host_mode:
            self.banks = [{n: aligned_zeros(s) for n, s in self.state_shapes.items()} for _ in range(2)]
        else:
            self.banks = [{n: self.cm.create_input_buffer_by_name(big.key, n) for n in self.state_names},
                          {n: self.cm.create_output_buffer_by_name(big.key, "n" + n) for n in self.state_names}]
        self._bufs = {}
        self._masks = {}            # (B, stage, min(p0, W)) -> mask; constant once a stage's window is full
        self._written = {}          # (B, stage) -> mask key last uploaded (GPU: skip unchanged uploads)
        self.chunk = 0              # >0: upstream's chunked visibility (chunk_duration=8 s -> 100 frames)
        self._zeroed = False
        self.reset()

    def reset(self):
        """New utterance. The state buffers are zeroed only once: afterwards they hold finite values from the last
        utterance, and every slot before position 0 is masked, so rewriting them (~340 MB on the GPU) is wasted."""
        if not self._zeroed:
            for b in self.banks:
                for n, a in b.items():
                    if self.host_mode:
                        a.fill(0.0)
                    else:
                        a.write(np.zeros(int(np.prod(self.state_shapes[n])), np.float32))
            self._zeroed = True
        self.cur = 0
        self.pos = [0] * len(self.stages)
        self._written = {}

    def _mask(self, B: int, s: int, p0: int, T: int, W: int, total: int | None):
        if self.chunk:
            # Upstream decodes in chunks against ring caches of W slots: key k is visible from query q only if it
            # is still in the ring once q's chunk (of `chunk` frames at 12.5 Hz) is written.
            q = p0 + np.arange(T)
            kpos = np.concatenate([p0 - W + np.arange(W), q])
            c = self.chunk * (T // B)
            cend = (q // c + 1) * c
            if total is not None:
                cend = np.minimum(cend, total * (T // B))
            d = q[:, None] - kpos[None, :]
            ok = (kpos[None, :] >= 0) & (d >= 0) & (d < W) & (kpos[None, :] >= cend[:, None] - W)
            return None, np.where(ok, 0.0, NEG).astype(np.float32)
        key = (B, s, min(p0, W))
        m = self._masks.get(key)
        if m is None:
            q = p0 + np.arange(T)
            kpos = np.concatenate([p0 - W + np.arange(W), q])
            d = q[:, None] - kpos[None, :]
            ok = (kpos[None, :] >= 0) & (d >= 0) & (d < W)
            m = self._masks[key] = np.where(ok, 0.0, NEG).astype(np.float32)
        return key, m

    def zq(self, codes: np.ndarray) -> np.ndarray:
        z = np.repeat(self.bias[None], codes.shape[0], 0).astype(np.float32)
        for i in range(codes.shape[1]):
            z += self.tables[i][codes[:, i]]
        return z

    def _io(self, B: int):
        if B in self._bufs:
            return self._bufs[B]
        sig = self.sigs[B]
        a = {n: aligned_zeros(sig.in_shape(n)) for n in sig.in_names if n not in self.state_shapes}
        a["audio"] = aligned_zeros(sig.out_shape("audio"))
        if self.host_mode:
            fixed_in = {n: sig.host(a[n]) for n in a if n != "audio"}
            banks = [{n: sig.host(b[n]) for n in self.state_names} for b in self.banks]
            out_audio = sig.host(a["audio"])
        else:
            fixed_in = {n: self.cm.create_input_buffer_by_name(sig.key, n) for n in a if n != "audio"}
            banks = self.banks
            out_audio = self.cm.create_output_buffer_by_name(sig.key, "audio")
        self._bufs[B] = (sig, a, fixed_in, banks, out_audio)
        return self._bufs[B]

    def decode_block(self, codes: np.ndarray, total: int | None = None) -> np.ndarray:
        """codes [B,16] for one block size B in self.sigs -> audio [B*1920]. `total` (frames of the utterance)
        only matters with chunked visibility."""
        B = codes.shape[0]
        sig, a, fixed_in, banks, out_audio = self._io(B)
        a["zq"][0] = self.zq(codes)
        T = B
        skip = set()
        for s, st in enumerate(self.stages):
            hd = st["d_model"] // st["num_heads"]
            W = st["context"]
            p0 = self.pos[s]
            q = p0 + np.arange(T)
            freqs = np.exp(np.arange(hd // 2, dtype=np.float64) * (-math.log(10000.0) * 2 / hd))
            ang = q[:, None].astype(np.float64) * freqs[None]
            a[f"cos{s}"][0, :, 0] = np.cos(ang)
            a[f"sin{s}"][0, :, 0] = np.sin(ang)
            key, m = self._mask(B, s, p0, T, W, total)
            if key is not None and self._written.get((B, s)) == key:
                skip.add(f"mask{s}")
            else:
                a[f"mask{s}"][0, 0] = m
                self._written[(B, s)] = key
            self.pos[s] += T
            T *= st["patch"]
        if not self.host_mode:
            for n, buf in fixed_in.items():
                if n not in skip:
                    buf.write(a[n].ravel())
        ins = dict(fixed_in)
        ins.update(banks[self.cur])
        outs = {"audio": out_audio}
        outs.update({"n" + n: buf for n, buf in banks[1 - self.cur].items()})
        sig.run(ins, outs)
        self.cur = 1 - self.cur
        if not self.host_mode:
            return np.array(out_audio.read(a["audio"].size, np.float32))
        return a["audio"][0].copy()

    def decode(self, codes: np.ndarray, block: int | None = None) -> np.ndarray:
        """All frames at once (blocks of `block`, last block padded and trimmed)."""
        self.reset()
        B = block or max(self.sigs)
        out = []
        for i in range(0, codes.shape[0], B):
            c = codes[i:i + B]
            n = c.shape[0]
            if n < B:
                c = np.concatenate([c, np.zeros((B - n, c.shape[1]), c.dtype)], 0)
            out.append(self.decode_block(c, codes.shape[0])[:n * 1920])
        return np.concatenate(out) if out else np.zeros(0, np.float32)


# ======================================================================== pipeline
class VoiceGenerator:
    def __init__(self, lm_path: str, dec_path: str, accel: str = "cpu", dec_accel: str | None = None,
                 threads: int = 8, dec_block: int = 16, lm_f32: bool = False, dec_f32: bool = True):
        """GPU precision: the backbone runs fp16 (its export scales the residual stream to fit), the codec fp32
        (fp16 arithmetic costs it 25-35 dB SNR and fp32 is no slower at 16-frame blocks)."""
        self.prompter = Prompter()
        if accel == "gpu":
            self.lm = GpuBackbone(lm_path, threads, enforce_f32=lm_f32)
        else:
            self.lm = Backbone(lm_path, accel, threads)
        self.dec = CodecDecoder(dec_path, dec_accel or accel, threads, enforce_f32=dec_f32)
        self.dec_block = dec_block

    def generate(self, text: str, instruction: str, seed: int = 0, max_new_tokens: int = 1000,
                 stop_at_audio_end: bool = True, stream: bool = False, first_block: int = 4, on_audio=None,
                 **sampling):
        """stop_at_audio_end: upstream keeps sampling text after <|audio_end|> until <|im_end|> (usually the next
        row, sometimes a few junk tokens and an empty second segment); those rows cannot change the first segment,
        so the default stops there.

        stream: decode while generating. Frame f of the segment (channel i at row a0+f+i, a0 = the audio_start row)
        is complete once row a0+f+15 exists; blocks of `first_block` frames, then of self.dec_block, are decoded as
        soon as they are complete and passed to on_audio(chunk). Same audio as the non-streaming path."""
        stats = {}
        t0 = time.perf_counter()
        prompt = self.prompter.build(text, instruction)
        logits, hn = self.lm.prefill(prompt)
        stats["prefill_s"] = time.perf_counter() - t0
        stats["prompt_rows"] = int(prompt.shape[0])
        sampler = DelaySampler(prompt, seed, **sampling)
        rows = []
        t_lm = t_samp = t_head = t_dec = 0.0
        a0, seg_end, done, chunks, ttfa = None, None, 0, [], None
        if stream:
            self.dec.reset()
            if first_block not in self.dec.sigs:
                first_block = min(b for b in self.dec.sigs if b >= min(first_block, max(self.dec.sigs)))

        def frame(f):
            return np.array([rows[a0 + f + i][1 + i] for i in range(N_VQ)], np.int64)

        def flush(final: bool):
            nonlocal done, ttfa, t_dec
            ready = len(rows) - 1 - a0 - (N_VQ - 1)            # frames 1..ready are complete
            if seg_end is not None:
                ready = min(ready, seg_end)
            while True:
                B = first_block if done == 0 else self.dec_block
                n = min(B, ready - done)
                if n <= 0 or (n < B and not final):
                    return
                ts = time.perf_counter()
                codes = np.stack([frame(f) for f in range(done + 1, done + 1 + n)])
                Bs = min(b for b in self.dec.sigs if b >= n) if n < B else B
                if Bs > n:
                    codes = np.concatenate([codes, np.zeros((Bs - n, N_VQ), np.int64)], 0)
                chunk = self.dec.decode_block(codes)[:n * 1920]
                t_dec += time.perf_counter() - ts
                chunks.append(chunk)
                done += n
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                if on_audio is not None:
                    on_audio(chunk)

        for _ in range(max_new_tokens):
            ts = time.perf_counter()
            tl = None
            if sampler.needs_full_text():
                tl = self.lm.text_logits(hn)
                t_head += time.perf_counter() - ts
            ts2 = time.perf_counter()
            row = sampler.next_row(logits, tl)
            t_samp += time.perf_counter() - ts2
            rows.append(row)
            if a0 is None and row[0] == AUDIO_START:
                a0 = len(rows) - 1
            if stream and a0 is not None:
                if seg_end is None and row[0] == AUDIO_END:
                    seg_end = len(rows) - 1 - a0 - N_VQ          # last frame d: its ch0 is at row a0+d
                flush(False)
            if sampler.stopping or (stop_at_audio_end and row[0] == AUDIO_END):
                break
            ts3 = time.perf_counter()
            logits, hn = self.lm.step(row)
            t_lm += time.perf_counter() - ts3
        gen = np.stack(rows)
        codes = de_delay(gen)
        if stream and a0 is not None:
            if seg_end is None:
                seg_end = codes.shape[0]
            flush(True)
            wav = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        else:
            t1 = time.perf_counter()
            wav = self.dec.decode(codes, self.dec_block)
            t_dec = time.perf_counter() - t1
        stats.update(steps=len(rows), lm_s=t_lm, sample_s=t_samp, text_head_s=t_head, decode_s=t_dec,
                     frames=int(codes.shape[0]), audio_s=len(wav) / 24000, total_s=time.perf_counter() - t0)
        stats["ttfa_s"] = ttfa if ttfa is not None else stats["total_s"]
        stats["rtfx"] = stats["audio_s"] / stats["total_s"] if stats["total_s"] else 0.0
        return wav, gen, codes, stats


def main():
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--instruction", default="")
    ap.add_argument("--lm", default=None, help="default: models/lm_gpu_i8_lean.tflite (gpu), models/lm_i8.tflite (cpu)")
    ap.add_argument("--dec", default=None, help="default: models/dec_gpu_f16_s.tflite (gpu), models/dec_f16.tflite (cpu)")
    ap.add_argument("--accel", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--dec-accel", default=None, choices=["cpu", "gpu"])
    ap.add_argument("--lm-f32", action="store_true", help="GPU: fp32 arithmetic for the backbone")
    ap.add_argument("--dec-fp16", action="store_true", help="GPU: fp16 arithmetic for the codec")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--block", type=int, default=0, help="codec frames per call (default: 128 gpu, 64 cpu)")
    ap.add_argument("--stream", action="store_true", help="decode while generating (default on gpu)")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--first-block", type=int, default=16, help="first streamed codec block (frames)")
    ap.add_argument("--codec-chunk", type=int, default=0,
                    help="reproduce upstream's chunked codec decode (100 = its 8 s chunks); 0 = full context")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "litert.wav"))
    args = ap.parse_args()
    if args.name:
        with open(os.path.join(HERE, "prompts.json"), encoding="utf-8") as f:
            p = json.load(f)[args.name]
        args.text, args.instruction = p["text"], p["instruction"]
    gpu = args.accel == "gpu"
    args.lm = args.lm or os.path.join(MODELS, "lm_gpu_i8_lean.tflite" if gpu else "lm_i8.tflite")
    args.dec = args.dec or os.path.join(MODELS, "dec_gpu_f16_s.tflite" if gpu else "dec_f16.tflite")
    stream = (args.stream or gpu) and not args.no_stream
    vg = VoiceGenerator(args.lm, args.dec, args.accel, args.dec_accel, args.threads,
                        args.block or (128 if gpu else 64), args.lm_f32, not args.dec_fp16)
    vg.dec.chunk = args.codec_chunk
    print(f"load: lm {vg.lm.load_s:.1f} s, dec {vg.dec.load_s:.1f} s")
    wav, gen, codes, stats = vg.generate(args.text, args.instruction, args.seed, args.max_new_tokens, stream=stream,
                                         first_block=args.first_block)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, wav, 24000)
    np.save(os.path.splitext(args.out)[0] + "_gen.npy", gen)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()}))


if __name__ == "__main__":
    main()
