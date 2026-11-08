"""Torch-free safetensors reader (memory-mapped; bf16 is widened to float32 exactly)."""
from __future__ import annotations

import json
import os
import struct

import numpy as np

_DT = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64, "I32": np.int32,
       "U8": np.uint8, "I8": np.int8}


class SafeTensors:
    """One .safetensors file, or a directory / index.json of shards."""

    def __init__(self, path: str):
        files = []
        if os.path.isdir(path):
            idx = os.path.join(path, "model.safetensors.index.json")
            if os.path.exists(idx):
                path = idx
            else:
                files = [os.path.join(path, "model.safetensors")]
        if path.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                wm = json.load(f)["weight_map"]
            d = os.path.dirname(path)
            files = sorted({os.path.join(d, v) for v in wm.values()})
        elif not files:
            files = [path]
        self.entries: dict[str, tuple[np.memmap, dict]] = {}
        for fn in files:
            with open(fn, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            mm = np.memmap(fn, dtype=np.uint8, mode="r")
            for k, v in header.items():
                if k == "__metadata__":
                    continue
                v = dict(v)
                v["base"] = 8 + n
                self.entries[k] = (mm, v)

    def keys(self):
        return self.entries.keys()

    def __contains__(self, k):
        return k in self.entries

    def shape(self, k) -> list[int]:
        return list(self.entries[k][1]["shape"])

    def raw(self, k) -> np.ndarray:
        mm, v = self.entries[k]
        a, b = v["data_offsets"]
        return mm[v["base"] + a: v["base"] + b].view(_DT[v["dtype"]]).reshape(v["shape"])

    def get(self, k) -> np.ndarray:
        """float32 copy (bf16 widened exactly)."""
        mm, v = self.entries[k]
        r = self.raw(k)
        if v["dtype"] == "BF16":
            return (r.astype(np.uint32) << 16).view(np.float32)
        return np.array(r, dtype=np.float32 if r.dtype.kind == "f" else r.dtype)


def bf16_rows_to_f32(table_u16: np.ndarray, rows) -> np.ndarray:
    return (table_u16[rows].astype(np.uint32) << 16).view(np.float32)
