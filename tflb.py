"""Minimal .tflite writer: a Python port of pocket-tts-cpp's litert_export/TfliteBuilder.

A Model holds the buffers, operator codes and signature defs; each Graph is one subgraph (one signature). Weights
are cached per model by key, so every signature that uses a weight points at one buffer (one copy in the file).
Ops do their own shape inference (static shapes only).

Difference from the C++ builder: constant buffers larger than EXTERNAL_MIN bytes are written *after* the flatbuffer
and referenced by Buffer.offset / Buffer.size (the TFLite "buffer offset" layout the converter uses for > 2 GB
models). That keeps the flatbuffer itself a few MB, so models of any size (a 1.7B LM in fp32 is 6 GB) serialize, and
every weight starts 64-byte aligned in the file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Sequence

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as S

F32, F16, I32, I8, I64, I4 = "f32", "f16", "i32", "i8", "i64", "i4"
_TF = {F32: S.TensorType.FLOAT32, F16: S.TensorType.FLOAT16, I32: S.TensorType.INT32, I8: S.TensorType.INT8,
       I64: S.TensorType.INT64, I4: S.TensorType.INT4}
_FROM_TF = {v: k for k, v in _TF.items()}
BO = S.BuiltinOperator
OPT = S.BuiltinOptions

EXTERNAL_MIN = 4096      # constants at least this big go after the flatbuffer
ALIGN = 64


def _broadcast(a: Sequence[int], b: Sequence[int]) -> list[int]:
    n = max(len(a), len(b))
    a = [1] * (n - len(a)) + list(a)
    b = [1] * (n - len(b)) + list(b)
    out = []
    for x, y in zip(a, b):
        if x != y and x != 1 and y != 1:
            raise ValueError(f"broadcast: incompatible shapes {a} and {b}")
        out.append(max(x, y))
    return out


def quantize_rows_i8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 per output channel (rows): scale = max|row| / 127, round half to even."""
    w = np.asarray(w, np.float32).reshape(w.shape[0], -1)
    scale = np.maximum(np.abs(w).max(axis=1), 1e-9).astype(np.float32) / 127.0
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def quantize_rows_i4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int4 per output channel: scale = max|row| / 7, values in [-7, 7], packed two per byte (element 2i
    in the low nibble, 2i+1 in the high nibble, TFLite's INT4 layout)."""
    w = np.asarray(w, np.float32).reshape(w.shape[0], -1)
    scale = np.maximum(np.abs(w).max(axis=1), 1e-9).astype(np.float32) / 7.0
    q = np.clip(np.rint(w / scale[:, None]), -7, 7).astype(np.int8).reshape(-1)
    if q.size % 2:
        q = np.concatenate([q, np.zeros(1, np.int8)])
    lo = (q[0::2] & 0x0F).astype(np.uint8)
    hi = (q[1::2] & 0x0F).astype(np.uint8)
    return (lo | (hi << 4)).astype(np.uint8), scale


def quantize_blocks_i4(w: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int4 per (row, block of `block` inputs): returns packed nibbles and scales [rows, cols/block]."""
    w = np.asarray(w, np.float32)
    o, i = w.shape
    wb = w.reshape(o, i // block, block)
    # The scales are stored as fp16: floor them at its smallest normal (an all-zero block, e.g. the never-predicted
    # pad row of an audio head, would otherwise get scale 0, which XNNPACK rejects).
    scale = np.maximum(np.abs(wb).max(axis=2).astype(np.float32) / 7.0, 2.0 ** -14)
    q = np.clip(np.rint(wb / scale[:, :, None]), -7, 7).astype(np.int8).reshape(-1)
    lo = (q[0::2] & 0x0F).astype(np.uint8)
    hi = (q[1::2] & 0x0F).astype(np.uint8)
    return (lo | (hi << 4)).astype(np.uint8), scale


@dataclass
class SharedWeight:
    buffer: int
    dtype: str
    shape: list[int]
    scales: np.ndarray | None = None
    block: int = 0                      # blockwise int4: block size; `scales_buffer` holds the fp16 scales
    scales_buffer: int = 0


class Graph:
    def __init__(self, model: "Model", index: int, name: str):
        self.model = model
        self.index = index
        self.sg = model.m.subgraphs[index]
        self.inputs: list[int] = []
        self.outputs: list[int] = []
        self.logical_in: list[tuple[str, list[int]]] = []
        self.logical_out: list[tuple[str, list[int]]] = []
        self._keyed: dict[str, int] = {}

    # ------------------------------------------------------------- tensors
    def shape(self, t: int) -> list[int]:
        return list(self.sg.tensors[t].shape)

    def dtype(self, t: int) -> str:
        return _FROM_TF[self.sg.tensors[t].type]

    def _new(self, dt: str, shape: Sequence[int], name: str = "") -> int:
        t = S.TensorT()
        t.shape = [int(d) for d in shape]
        t.type = _TF[dt]
        t.hasRank = True
        t.buffer = 0
        t.name = name or f"t{len(self.sg.tensors)}"
        self.sg.tensors.append(t)
        return len(self.sg.tensors) - 1

    def input(self, dt: str, shape: Sequence[int], name: str | None = None) -> int:
        """Signature input. With model.flat_io the tensor is 1-D and reshaped here (LiteRT's Python API only binds
        zero-copy host memory as 1-D buffers); the logical shape goes to the sidecar JSON either way."""
        name = name or f"args_{len(self.inputs)}"
        shape = [int(d) for d in shape]
        self.logical_in.append((name, shape))
        flat = self.model.flat_io and len(shape) != 1
        t = self._new(dt, [int(np.prod(shape))] if flat else shape, name)
        self.sg.inputs.append(t)
        self.inputs.append(t)
        return self.reshape(t, shape) if flat else t

    def output(self, t: int, name: str | None = None) -> None:
        name = name or f"output_{len(self.outputs)}"
        self.logical_out.append((name, self.shape(t)))
        if self.model.flat_io and len(self.shape(t)) != 1:
            t = self.reshape(t, [-1])
        self.sg.outputs.append(t)
        self.outputs.append(t)

    def const(self, arr: np.ndarray, dt: str | None = None) -> int:
        arr = np.ascontiguousarray(arr)
        dt = dt or {np.dtype(np.float32): F32, np.dtype(np.int32): I32, np.dtype(np.float16): F16,
                    np.dtype(np.int8): I8}[arr.dtype]
        t = self._new(dt, arr.shape)
        self.sg.tensors[t].buffer = self.model.buffer(arr.tobytes())
        return t

    def const_f32(self, v, shape=None) -> int:
        a = np.asarray(v, np.float32)
        return self.const(a.reshape(shape) if shape is not None else a, F32)

    def const_i32(self, v, shape=None) -> int:
        a = np.asarray(v, np.int32)
        return self.const(a.reshape(shape) if shape is not None else a, I32)

    def scalar(self, v: float) -> int:
        return self.const_f32(np.float32(v), [])

    # ------------------------------------------------------------- shared weights
    def _shared_tensor(self, key: str) -> int:
        if key in self._keyed:
            return self._keyed[key]
        w = self.model.shared[key]
        if w.block:
            # Blockwise int4: the scales are a constant FLOAT16 tensor [rows, cols/block] that the weight's
            # BlockwiseQuantization details point at.
            st = self._new(F16, [w.shape[0], w.shape[1] // w.block], key + "/scales")
            self.sg.tensors[st].buffer = w.scales_buffer
            t = self._new(I4, w.shape, key)
            tt = self.sg.tensors[t]
            tt.buffer = w.buffer
            q = S.QuantizationParametersT()
            q.detailsType = S.QuantizationDetails.BlockwiseQuantization
            d = S.BlockwiseQuantizationT()
            d.scales = st
            d.zeroPoints = -1
            d.blockSize = w.block
            q.details = d
            q.quantizedDimension = 0
            tt.quantization = q
            self._keyed[key] = t
            return t
        t = self._new(w.dtype, w.shape, key)
        tt = self.sg.tensors[t]
        tt.buffer = w.buffer
        if w.scales is not None:
            q = S.QuantizationParametersT()
            q.scale = w.scales.astype(np.float32)
            q.zeroPoint = np.zeros(len(w.scales), np.int64)
            q.quantizedDimension = 0
            tt.quantization = q
        self._keyed[key] = t
        return t

    def weight(self, key: str, shape: Sequence[int], make: Callable[[], np.ndarray], fmt: str = F32) -> int:
        """Shared constant. fmt f32: float; i8: int8 per output channel (rows), the dynamic-range format
        FULLY_CONNECTED consumes directly; f16: fp16 constant + DEQUANTIZE (returns the FLOAT32 output)."""
        shape = [int(d) for d in shape]
        k = f"{key}#{fmt}"
        if fmt == F16:
            kd = k + "deq"
            if kd in self._keyed:
                return self._keyed[kd]
        if k not in self.model.shared:
            v = np.asarray(make(), np.float32).reshape(shape)
            if fmt == F32:
                self.model.shared[k] = SharedWeight(self.model.buffer(v.tobytes()), F32, shape)
            elif fmt == F16:
                self.model.shared[k] = SharedWeight(self.model.buffer(v.astype(np.float16).tobytes()), F16, shape)
            elif fmt == I8:
                q, s = quantize_rows_i8(v)
                self.model.shared[k] = SharedWeight(self.model.buffer(q.tobytes()), I8, shape, s)
            elif fmt == I4:
                q, s = quantize_rows_i4(v)
                self.model.shared[k] = SharedWeight(self.model.buffer(q.tobytes()), I4, shape, s)
            elif fmt.startswith("i4b"):                          # i4b32, i4b64, ...: blockwise int4
                bs = int(fmt[3:])
                if v.ndim != 2 or v.shape[1] % bs:
                    return self.weight(key, shape, make, I8)        # small / odd shapes stay int8
                q, s = quantize_blocks_i4(v, bs)
                self.model.shared[k] = SharedWeight(self.model.buffer(q.tobytes()), I4, shape, None, bs,
                                                    self.model.buffer(s.astype(np.float16).tobytes()))
            else:
                raise ValueError(fmt)
        t = self._shared_tensor(k)
        if fmt == F16:
            out = self._new(F32, shape)
            self._op(BO.DEQUANTIZE, [t], [out])
            self._keyed[k + "deq"] = out
            return out
        return t

    # ------------------------------------------------------------- ops
    def _op(self, code: int, ins: Sequence[int], outs: Sequence[int], opts_type: int = 0, opts=None,
            version: int = 1) -> None:
        op = S.OperatorT()
        op.opcodeIndex = self.model.opcode(code, version)
        op.inputs = [int(i) for i in ins]
        op.outputs = [int(o) for o in outs]
        if opts is not None:
            op.builtinOptionsType = opts_type
            op.builtinOptions = opts
        self.sg.operators.append(op)

    def _unary(self, code: int, x: int, opts_type: int = 0, opts=None) -> int:
        y = self._new(self.dtype(x), self.shape(x))
        self._op(code, [x], [y], opts_type, opts)
        return y

    def _binary(self, code: int, a: int, b: int) -> int:
        y = self._new(self.dtype(a), _broadcast(self.shape(a), self.shape(b)))
        if code == BO.ADD:
            o = S.AddOptionsT(); o.potScaleInt16 = True
            self._op(code, [a, b], [y], OPT.AddOptions, o)
        elif code == BO.SUB:
            o = S.SubOptionsT(); o.potScaleInt16 = True
            self._op(code, [a, b], [y], OPT.SubOptions, o)
        elif code == BO.MUL:
            self._op(code, [a, b], [y], OPT.MulOptions, S.MulOptionsT())
        elif code == BO.DIV:
            self._op(code, [a, b], [y], OPT.DivOptions, S.DivOptionsT())
        else:
            self._op(code, [a, b], [y])
        return y

    def add(self, a, b): return self._binary(BO.ADD, a, b)
    def sub(self, a, b): return self._binary(BO.SUB, a, b)
    def mul(self, a, b): return self._binary(BO.MUL, a, b)
    def div(self, a, b): return self._binary(BO.DIV, a, b)
    def maximum(self, a, b): return self._binary(BO.MAXIMUM, a, b)
    def minimum(self, a, b): return self._binary(BO.MINIMUM, a, b)
    def abs(self, x): return self._unary(BO.ABS, x)
    def neg(self, x): return self._unary(BO.NEG, x)
    def rsqrt(self, x): return self._unary(BO.RSQRT, x)
    def sqrt(self, x): return self._unary(BO.SQRT, x)
    def square(self, x): return self._unary(BO.SQUARE, x)
    def logistic(self, x): return self._unary(BO.LOGISTIC, x)
    def tanh(self, x): return self._unary(BO.TANH, x)
    def relu(self, x): return self._unary(BO.RELU, x)

    def exp(self, x):
        return self._unary(BO.EXP, x, OPT.ExpOptions, S.ExpOptionsT())

    def gelu(self, x, approximate: bool):
        o = S.GeluOptionsT(); o.approximate = approximate
        return self._unary(BO.GELU, x, OPT.GeluOptions, o)

    def softmax(self, x, beta: float = 1.0):
        o = S.SoftmaxOptionsT(); o.beta = beta
        return self._unary(BO.SOFTMAX, x, OPT.SoftmaxOptions, o)

    def fc(self, x: int, w: int, bias: int = -1) -> int:
        xs, ws = self.shape(x), self.shape(w)
        if len(ws) != 2 or xs[-1] != ws[1]:
            raise ValueError(f"fc: input {xs} vs weight {ws}")
        y = self._new(F32, xs[:-1] + [ws[0]])
        o = S.FullyConnectedOptionsT(); o.keepNumDims = True
        self._op(BO.FULLY_CONNECTED, [x, w, bias], [y], OPT.FullyConnectedOptions, o, version=5)
        return y

    def bmm(self, a: int, b: int, adj_x: bool = False, adj_y: bool = False) -> int:
        as_, bs = self.shape(a), self.shape(b)
        m = as_[-1] if adj_x else as_[-2]
        ka = as_[-2] if adj_x else as_[-1]
        kb = bs[-1] if adj_y else bs[-2]
        n = bs[-2] if adj_y else bs[-1]
        if ka != kb:
            raise ValueError(f"bmm: K mismatch {as_} x {bs}")
        y = self._new(F32, _broadcast(as_[:-2], bs[:-2]) + [m, n])
        o = S.BatchMatMulOptionsT(); o.adjX = adj_x; o.adjY = adj_y
        self._op(BO.BATCH_MATMUL, [a, b], [y], OPT.BatchMatMulOptions, o)
        return y

    def reshape(self, x: int, shape: Sequence[int]) -> int:
        r = [int(d) for d in shape]
        n = int(np.prod(self.shape(x)))
        if -1 in r:
            i = r.index(-1)
            known = int(np.prod([d for j, d in enumerate(r) if j != i]))
            r[i] = n // known
        if int(np.prod(r)) != n:
            raise ValueError(f"reshape: {self.shape(x)} -> {shape}")
        sh = self.const_i32(r, [len(r)])
        y = self._new(self.dtype(x), r)
        self._op(BO.RESHAPE, [x, sh], [y])
        return y

    def transpose(self, x: int, perm: Sequence[int]) -> int:
        xs = self.shape(x)
        p = self.const_i32(list(perm), [len(perm)])
        y = self._new(self.dtype(x), [xs[i] for i in perm])
        self._op(BO.TRANSPOSE, [x, p], [y])
        return y

    def slice(self, x: int, begin: Sequence[int], size: Sequence[int]) -> int:
        xs = self.shape(x)
        if len(begin) != len(xs) or len(size) != len(xs):
            raise ValueError("slice: rank mismatch")
        ys = [xs[i] - begin[i] if size[i] < 0 else size[i] for i in range(len(xs))]
        for i in range(len(xs)):
            if begin[i] < 0 or begin[i] + ys[i] > xs[i]:
                raise ValueError(f"slice: out of range on {xs}")
        b = self.const_i32(list(begin), [len(begin)])
        s = self.const_i32(ys, [len(ys)])
        y = self._new(self.dtype(x), ys)
        self._op(BO.SLICE, [x, b, s], [y])
        return y

    def concat(self, xs: Sequence[int], axis: int) -> int:
        ys = self.shape(xs[0])
        ax = axis % len(ys)
        for t in xs[1:]:
            ys[ax] += self.shape(t)[ax]
        y = self._new(self.dtype(xs[0]), ys)
        o = S.ConcatenationOptionsT(); o.axis = ax
        self._op(BO.CONCATENATION, list(xs), [y], OPT.ConcatenationOptions, o)
        return y

    def _reduce(self, code: int, x: int, axes: Sequence[int], keep: bool) -> int:
        xs = self.shape(x)
        ax = sorted(a % len(xs) for a in axes)
        ys = [(1 if i in ax else d) for i, d in enumerate(xs) if keep or i not in ax]
        a = self.const_i32(ax, [len(ax)])
        y = self._new(self.dtype(x), ys)
        o = S.ReducerOptionsT(); o.keepDims = keep
        self._op(code, [x, a], [y], OPT.ReducerOptions, o)
        return y

    def mean(self, x, axes, keep=True): return self._reduce(BO.MEAN, x, axes, keep)
    def sum(self, x, axes, keep=True): return self._reduce(BO.SUM, x, axes, keep)
    def reduce_max(self, x, axes, keep=True): return self._reduce(BO.REDUCE_MAX, x, axes, keep)

    def dynamic_update_slice(self, x: int, update: int, indices: int) -> int:
        y = self._new(self.dtype(x), self.shape(x))
        self._op(BO.DYNAMIC_UPDATE_SLICE, [x, update, indices], [y])
        return y

    def cast(self, x: int, dt: str) -> int:
        y = self._new(dt, self.shape(x))
        o = S.CastOptionsT(); o.inDataType = _TF[self.dtype(x)]; o.outDataType = _TF[dt]
        self._op(BO.CAST, [x], [y], OPT.CastOptions, o)
        return y

    def l2_norm(self, x: int) -> int:
        """L2_NORMALIZATION over the last axis: x / sqrt(max(sum(x^2), eps))."""
        o = S.L2NormOptionsT()
        return self._unary(BO.L2_NORMALIZATION, x, OPT.L2NormOptions, o)

    def composite(self, name: str, attrs: dict, ins: Sequence[int], body: "Graph") -> int:
        """StableHLO composite `name` (flexbuffer attributes) over `ins`; `body` (same inputs, one output) is the
        decomposition the runtime falls back to. Port of pocket-tts-cpp's Graph::composite."""
        import flatbuffers.flexbuffers as fx
        if len(body.outputs) != 1:
            raise ValueError("composite: the body needs exactly one output")
        b = fx.Builder()
        with b.Map():
            for k, v in attrs.items():
                if isinstance(v, bool):
                    b.Bool(k, v)
                elif isinstance(v, int):
                    b.Int(k, v)
                else:
                    b.Float(k, float(v))
        y = self._new(body.dtype(body.outputs[0]), body.shape(body.outputs[0]))
        op = S.OperatorT()
        op.opcodeIndex = self.model.opcode(BO.STABLEHLO_COMPOSITE, 1)
        op.inputs = [int(i) for i in ins]
        op.outputs = [y]
        o = S.StableHLOCompositeOptionsT()
        o.name = name
        o.decompositionSubgraphIndex = body.index
        o.compositeAttributes = list(b.Finish())
        o.compositeAttributesFormat = S.CustomOptionsFormat.FLEXBUFFERS
        o.version = 0
        op.builtinOptions2Type = S.BuiltinOptions2.StableHLOCompositeOptions
        op.builtinOptions2 = o
        self.sg.operators.append(op)
        return y

    def gather(self, params: int, indices: int, axis: int = 0) -> int:
        ps, is_ = self.shape(params), self.shape(indices)
        ax = axis % len(ps)
        y = self._new(self.dtype(params), ps[:ax] + is_ + ps[ax + 1:])
        o = S.GatherOptionsT(); o.axis = ax; o.batchDims = 0
        self._op(BO.GATHER, [params, indices], [y], OPT.GatherOptions, o)
        return y


class Model:
    def __init__(self, description: str = "moss-tts-litert tflb", flat_io: bool = True):
        self.flat_io = flat_io
        self.m = S.ModelT()
        self.m.version = 3
        self.m.description = description
        self.m.subgraphs = []
        self.m.operatorCodes = []
        self.m.signatureDefs = []
        self.buffers: list[bytes | None] = [None]   # buffer 0: the empty sentinel
        self._opcodes: dict[tuple[int, int], int] = {}
        self.shared: dict[str, SharedWeight] = {}
        self.graphs: list[Graph] = []
        self.io: dict[str, dict] = {}      # signature -> logical input / output shapes (sidecar JSON)

    def graph(self, name: str) -> Graph:
        sg = S.SubGraphT()
        sg.name = name
        sg.tensors, sg.inputs, sg.outputs, sg.operators = [], [], [], []
        self.m.subgraphs.append(sg)
        g = Graph(self, len(self.graphs), name)
        self.graphs.append(g)
        return g

    def signature(self, key: str, g: Graph, in_names: Sequence[str] | None = None,
                  out_names: Sequence[str] | None = None) -> None:
        """Signature over g's inputs / outputs; names default to the ones given to input() / output()."""
        in_names = list(in_names) if in_names else [n for n, _ in g.logical_in]
        out_names = list(out_names) if out_names else [n for n, _ in g.logical_out]
        sig = S.SignatureDefT()
        sig.signatureKey = key
        sig.subgraphIndex = g.index
        sig.inputs, sig.outputs = [], []
        for i, t in enumerate(g.inputs):
            tm = S.TensorMapT(); tm.name = in_names[i]; tm.tensorIndex = t
            sig.inputs.append(tm)
        for i, t in enumerate(g.outputs):
            tm = S.TensorMapT(); tm.name = out_names[i]; tm.tensorIndex = t
            sig.outputs.append(tm)
        self.m.signatureDefs.append(sig)
        self.io[key] = {"inputs": {n: s for n, (_, s) in zip(in_names, g.logical_in)},
                        "outputs": {n: s for n, (_, s) in zip(out_names, g.logical_out)}}

    def buffer(self, data: bytes) -> int:
        self.buffers.append(bytes(data))
        return len(self.buffers) - 1

    def opcode(self, code: int, version: int) -> int:
        key = (code, version)
        if key not in self._opcodes:
            oc = S.OperatorCodeT()
            oc.builtinCode = code
            oc.deprecatedBuiltinCode = min(code, 127)
            oc.version = version
            self.m.operatorCodes.append(oc)
            self._opcodes[key] = len(self.m.operatorCodes) - 1
        return self._opcodes[key]

    # ------------------------------------------------------------- serialization
    def _pack(self, ext_offsets: dict[int, int]) -> bytes:
        b = flatbuffers.Builder(16 << 20)
        codes = [oc.Pack(b) for oc in self.m.operatorCodes]
        S.ModelStartOperatorCodesVector(b, len(codes))
        for c in reversed(codes):
            b.PrependUOffsetTRelative(c)
        codes_vec = b.EndVector()
        sgs = [sg.Pack(b) for sg in self.m.subgraphs]
        S.ModelStartSubgraphsVector(b, len(sgs))
        for s in reversed(sgs):
            b.PrependUOffsetTRelative(s)
        sgs_vec = b.EndVector()
        desc = b.CreateString(self.m.description)
        bufs = []
        for i, data in enumerate(self.buffers):
            if i == 0:
                S.BufferStart(b)
                bufs.append(S.BufferEnd(b))
            elif i in ext_offsets:
                S.BufferStart(b)
                S.BufferAddOffset(b, ext_offsets[i])
                S.BufferAddSize(b, len(data))
                bufs.append(S.BufferEnd(b))
            else:
                # Inline, 16-byte aligned. Every buffer but the sentinel gets a data vector, even an empty one:
                # without it LiteRT treats a zero-length constant as non-constant (see TfliteBuilder.cpp).
                b.StartVector(1, len(data), 16)
                b.head = b.head - len(data)
                b.Bytes[b.head:b.head + len(data)] = data
                b.vectorNumElems = len(data)
                vec = b.EndVector()
                S.BufferStart(b)
                S.BufferAddData(b, vec)
                bufs.append(S.BufferEnd(b))
        S.ModelStartBuffersVector(b, len(bufs))
        for x in reversed(bufs):
            b.PrependUOffsetTRelative(x)
        bufs_vec = b.EndVector()
        sigs = [s.Pack(b) for s in self.m.signatureDefs]
        S.ModelStartSignatureDefsVector(b, len(sigs))
        for s in reversed(sigs):
            b.PrependUOffsetTRelative(s)
        sigs_vec = b.EndVector()
        S.ModelStart(b)
        S.ModelAddVersion(b, self.m.version)
        S.ModelAddOperatorCodes(b, codes_vec)
        S.ModelAddSubgraphs(b, sgs_vec)
        S.ModelAddDescription(b, desc)
        S.ModelAddBuffers(b, bufs_vec)
        S.ModelAddSignatureDefs(b, sigs_vec)
        root = S.ModelEnd(b)
        b.Finish(root, file_identifier=b"TFL3")
        return bytes(b.Output())

    def save(self, path: str) -> int:
        ext = [i for i, d in enumerate(self.buffers) if i > 0 and len(d) >= EXTERNAL_MIN]
        # Pass 1 with placeholder offsets (non-zero, so the fields are present) to learn the flatbuffer size;
        # pass 2 with the real offsets. Field presence is identical, so the size does not change.
        fb = self._pack({i: 1 for i in ext})
        pos = (len(fb) + ALIGN - 1) // ALIGN * ALIGN
        offsets = {}
        for i in ext:
            offsets[i] = pos
            pos = (pos + len(self.buffers[i]) + ALIGN - 1) // ALIGN * ALIGN
        fb2 = self._pack(offsets)
        if len(fb2) != len(fb):
            raise RuntimeError("flatbuffer size changed between passes")
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(fb2)
            for i in ext:
                f.write(b"\0" * (offsets[i] - f.tell()))
                f.write(self.buffers[i])
        os.replace(tmp, path)
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump({"flat_io": self.flat_io, "signatures": self.io}, f)
        return os.path.getsize(path)
