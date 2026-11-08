"""Whisper large-v3-turbo transcripts + WER (en) / CER (zh) of generated WAVs against the prompt text.

  .venv\\Scripts\\python tests\\asr_eval.py out\\ref\\fox\\wav_full.wav:fox out\\litert\\fox.wav:fox ...
Each argument is WAV:PROMPT_NAME (prompts.json). Prints one JSON line per file.
"""
import json
import os
import re
import sys
import unicodedata

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def norm_en(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return s.split()


def norm_zh(s):
    s = unicodedata.normalize("NFKC", s)
    return [c for c in s if re.match(r"[一-鿿0-9a-zA-Z]", c)]


def edit_distance(a, b):
    d = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return int(d[len(b)])


def main(items):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    name = "openai/whisper-large-v3-turbo"
    proc = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, dtype=torch.float32).eval()
    with open(os.path.join(ROOT, "prompts.json"), encoding="utf-8") as f:
        prompts = json.load(f)
    out = []
    for it in items:
        path, pname = it.rsplit(":", 1)
        p = prompts[pname]
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        wav16 = resample_poly(wav, 2, 3) if sr == 24000 else wav
        feats = proc(wav16, sampling_rate=16000, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = model.generate(feats, language=p["language"], task="transcribe")
        hyp = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if p["language"] == "zh":
            r, h = norm_zh(p["text"]), norm_zh(hyp)
            metric = "cer"
        else:
            r, h = norm_en(p["text"]), norm_en(hyp)
            metric = "wer"
        err = edit_distance(r, h) / max(len(r), 1)
        rec = {"file": os.path.relpath(path, ROOT), "prompt": pname, metric: round(err, 4),
               "dur_s": round(len(wav) / sr, 2), "hyp": hyp}
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
