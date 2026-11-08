# moss-tts-litert: MOSS-VoiceGenerator on LiteRT

Experiment: run [MOSS-VoiceGenerator](../MOSS-VoiceGenerator) (voice design TTS: a free-form voice description +
text -> 24 kHz speech, Qwen3-1.7B backbone with 16 delayed audio heads, MOSS-Audio-Tokenizer codec) end to end on
**LiteRT** (`ai_edge_litert` 2.3.0), on the CPU (XNNPACK) and on the GPU (WebGPU / D3D12), with no PyTorch at run
time. The LiteRT graphs are written op by op from the safetensors by a Python port of pocket-tts-cpp's
`litert_export/TfliteBuilder` (`tflb.py`), following the lessons in `../pocket-tts-cpp/LITERT_EXPERIMENTS.md`.

Results, measurements and everything tried: **[EXPERIMENTS.md](EXPERIMENTS.md)**. In short (Ryzen 9 7950X + RX
7900 XT, Windows 11):

| path | RTFx (audio s / wall s) | first audio | logits vs PyTorch (KL) | memory |
|---|---|---|---|---|
| PyTorch fp32 CPU (upstream code) | ~0.2 | | | |
| LiteRT CPU, int8 backbone + fp16 codec | 0.88-0.95 | (batch) | 0.006-0.008 | 2.6-8.7 GB RAM |
| LiteRT CPU, int4 blockwise-32 backbone + fp16 codec | 1.27 | (batch) | 0.04 | 2.6-7.7 GB RAM |
| LiteRT GPU, first version (int8 backbone fp16 + codec fp32), streaming | 3.6-5.0 | ~0.6-0.7 s | 0.0004-0.0007 | ~15 GB VRAM |
| **LiteRT GPU, lean graphs (ggml-inspired, section 10), streaming** | **6.6 (4.4-7.4)** | **0.37 s** | 0.0004 | ~15 GB VRAM |
| LiteRT GPU, lean graphs, batch decode | 6.9 (4.6-7.6), 7.3 on 29 s | | 0.0004 | |
| for comparison: `../moss-tts-ggml`, Vulkan Q8_0 | 10-17 | (no streaming) | | |

fp32 exports match PyTorch to 2-4e-6 (backbone) and 67-71 dB (codec). Whisper WER of the generated speech is the
same as upstream's (lean GPU streaming: en 0.018, zh 0.137 vs torch 0.052 / 0.137). Not listening-tested.

## What runs where

```
text + instruction --(Prompter: upstream template + normalization, tokenizers)--> prompt rows [S, 17]
   host: embedding sum (text table bf16 + 16 audio tables) ------------------------> x [S, 2048]
   LiteRT backbone  prefill_t{128|256}_p{P}   (one call when the prompt fits, KV cache filled)
   loop:
     LiteRT backbone  step_p{P}  -> 2 text-slot logits | 16 x 1025 audio logits, final hidden
     (LiteRT text_head only when the next text token is free: the first row, after audio_end)
     host: DelaySampler = upstream generate() for batch 1 (delay pattern, top-k / top-p / repetition penalty)
   host: de-delay -> frames [F, 16] -> 16 lookup tables (folded RLFQ dequantizer) -> zq [F, 768]
   LiteRT codec  dec_b{B}  (streaming, 4 causal transformer stages, sliding K/V windows) -> 24 kHz audio
```

* Backbone: `export_lm.py`. CPU flavour: read-only K/V cache inputs bound zero-copy to host arrays, the graph
  returns the new rows and the host writes them in place. GPU flavour (`--kv dus --ranked-io`): caches stay on the
  device and are written in-graph (DYNAMIC_UPDATE_SLICE), two buffer banks swapped per call; fp16-safe RMSNorm and a
  scaled residual stream (`--resid-scale/--mlp-scale 0.125`) so fp16 arithmetic does not overflow.
* Codec: `export_codec.py`. One signature per block size B (12.5 Hz frames per call). Every attention layer keeps the
  codec's full context window (125/250/500/1000 rows per stage) as state, so chunked decoding equals the codec's
  non-streaming decode for any B.
* Runtime: `moss_litert.py` (torch-free: numpy, tokenizers, ai_edge_litert).

## Layout

| path | |
|---|---|
| `tflb.py` | TFLite flatbuffer builder (weights stored after the flatbuffer with Buffer.offset: any size, 64-byte aligned; flat 1-D signature I/O + sidecar JSON of logical shapes; int8 / int4 / blockwise-int4 / fp16 weights) |
| `st.py` | torch-free safetensors reader (bf16 widened exactly) |
| `export_lm.py` | backbone + heads -> `models/lm_*.tflite` (+ `emb_text_bf16.npy`, `emb_audio_f32.npy`) |
| `export_codec.py` | codec decoder -> `models/dec_*.tflite` (+ `codec_tables.npy`, `codec_bias.npy`, `codec_stages.json`) |
| `moss_litert.py` | runtime: Prompter, Backbone (CPU), GpuBackbone, DelaySampler, CodecDecoder, VoiceGenerator (+ CLI) |
| `ref_torch.py` | PyTorch reference (upstream code, fp32 CPU): generation, teacher-forced logits, codec decodes |
| `prompts.json` | test prompts (en / zh, short and long) |
| `tests/test_tflb.py` | builder smoke test (every op used, CPU + GPU vs numpy) |
| `tests/test_parity.py` | teacher-forced LiteRT vs PyTorch: logits (rel error, KL, top-1) and codec SNR |
| `tests/bench_step.py`, `tests/bench_fc.py` | backbone step / FC weight-streaming timings |
| `tests/bench_e2e.py` | end-to-end generation benchmark (RTFx, TTFA, streaming check) |
| `tests/asr_eval.py` | Whisper large-v3-turbo transcripts, WER / CER |
| `models/`, `out/`, `logs/` | exports, generated audio / references, run logs (not source) |

## Setup (Windows, done once)

```
cd D:\tools\moss-tts\moss-tts-litert
C:\Users\PC\miniconda3\python.exe -m venv .venv
.venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install "transformers==5.0.0" safetensors tokenizers ai-edge-litert==2.3.0 numpy soundfile scipy huggingface_hub tqdm einops
```

* torch / transformers are only needed for `ref_torch.py` and `tests/asr_eval.py` (transformers 5.0.0 as upstream
  pins it; 5.20 breaks the processor's `from_pretrained`).
* Codec weights: `../MOSS-Audio-Tokenizer` (OpenMOSS-Team/MOSS-Audio-Tokenizer, 7.1 GB fp32).
* **GPU:** LiteRT 2.3.0's WebGPU accelerator needs a recent `dxcompiler.dll` + `dxil.dll` next to it
  (`.venv\Lib\site-packages\ai_edge_litert\`); without them device creation fails with
  `DynamicLib.Open: dxcompiler.dll Windows Error: 87`. The copies used here came from
  `D:\tools\pocket-tts-ungated\litert-runner\weight_share\runtimes\2.3.0\` (see `../pocket-tts-cpp/WINDOWS_GPU.md`).

## Export

```
:: CPU (XNNPACK): int8 backbone (or int4 blockwise 32 for speed), fp16 codec
.venv\Scripts\python export_lm.py --wfmt i8 --caps 256,512,1024,2048 --prefill 64 --prefill-caps 256,512 --out models\lm_i8.tflite
.venv\Scripts\python export_lm.py --wfmt i4b32 --caps 256,512,1024,2048 --prefill 64 --prefill-caps 256,512 --out models\lm_i4b32.tflite
.venv\Scripts\python export_codec.py --wfmt f16 --blocks 1,4,16,32,64 --out models\dec_f16.tflite

:: GPU (WebGPU): lean int8 backbone with resident caches (fp16-safe), single-call prefills; codec blocks 16 + 128
.venv\Scripts\python export_lm.py --lean --wfmt i8 --kv dus --ranked-io --resid-scale 0.125 --mlp-scale 0.125 --caps 512,1024,2048 --prefill 128,256 --prefill-caps 512 --out models\lm_gpu_i8_lean.tflite
.venv\Scripts\python export_codec.py --wfmt f16 --ranked-io --safe-norm --blocks 16,128 --out models\dec_gpu_f16_s.tflite
:: (first version, before section 10: --safe-norm instead of --lean, --prefill 64, codec --blocks 4,16)

:: parity reference (exact fp32)
.venv\Scripts\python export_lm.py --wfmt f32 --caps 512 --prefill 64 --out models\lm_f32.tflite
.venv\Scripts\python export_codec.py --wfmt f32 --blocks 4,16 --out models\dec_f32.tflite
```

Each export takes 5-25 s (the backbone checkpoint is read with numpy memory maps).

## Run

```
:: CPU (defaults: lm_i8 + dec_f16, 4 threads, 64-frame codec blocks, batch decode)
.venv\Scripts\python moss_litert.py --name tavern --out out\tavern.wav
:: GPU (defaults: lm_gpu_i8_lean + dec_gpu_f16_s, streaming 16 then 128 frames; backbone fp16, codec fp32)
.venv\Scripts\python moss_litert.py --name tavern --accel gpu --out out\tavern.wav
:: any text
.venv\Scripts\python moss_litert.py --accel gpu ^
    --instruction "Little girl, innocent and curious, high-pitched and adorable" --text "Mommy, why is the sky blue?" --out out\girl.wav
```

Options: `--no-stream`, `--block N`, `--first-block N`, `--codec-chunk 100` (reproduce upstream's chunked codec
decode, 8 s chunks; default is the codec's full-context decode, which differs from upstream's on clips over 8 s),
`--lm` / `--dec` for other exports.

Notes:
* Use 4 CPU threads: on this machine 16 threads make the backbone step 3.5x slower (`EXPERIMENTS.md`).
* CPU runs write an XNNPACK weight cache next to each model (`<model>.<size>_<mtime>.xnnpack_cache`). The first load
  writes it; later loads take ~0.3 s for the backbone, and RAM drops from 31 GB to under 9 GB.
  `MOSS_XNNPACK_CACHE=off` disables it.
* The prompt is built without text normalization, like upstream's actual path (`moss_litert.Prompter`).
* The first GPU load compiles shaders for ~40 s.
* All compiled models share one LiteRT environment (`moss_litert.shared_env()`); separate environments put the
  backbone and the codec on two WebGPU devices and made every codec call 3-8x slower.

## Tests

```
.venv\Scripts\python ref_torch.py                       :: PyTorch references -> out\ref\<name>\ (slow: ~0.3 s per step)
.venv\Scripts\python tests\test_tflb.py
.venv\Scripts\python tests\test_parity.py --lm models\lm_f32.tflite --dec models\dec_f32.tflite --threads 4
.venv\Scripts\python tests\test_parity.py --accel gpu --lm models\lm_gpu_i8_lean.tflite
.venv\Scripts\python tests\test_parity.py --accel gpu --f32 --dec models\dec_gpu_f16_b128.tflite --blocks 128 --codec-chunk 100
.venv\Scripts\python tests\bench_e2e.py --accel gpu --lm models\lm_gpu_i8_lean.tflite --dec models\dec_gpu_f16_s.tflite --block 128 --stream --first-block 16 --out out\litert_gpu --seeds 0,1,2
.venv\Scripts\python tests\bench_gpu_step.py models\lm_gpu_i8_lean.tflite   :: where a GPU step goes (writes / run / read)
.venv\Scripts\python tests\bench_gpu_gemv.py          :: FC-only floor of the backbone on WebGPU, weight formats
.venv\Scripts\python tests\bench_gpu_ops.py           :: FC / attention TFLOPS at the codec's shapes
.venv\Scripts\python tests\bench_gpu_cacheio.py       :: cost of KV caches as graph I/O
.venv\Scripts\python tests\probe_gpu_fused.py         :: which compact op forms WebGPU accepts (L2 norm, composites, RoPE-FC)
.venv\Scripts\python tests\opcount.py models\lm_gpu_i8_lean.tflite step_p512
.venv\Scripts\python tests\asr_eval.py out\litert_gpu\tavern_s0.wav:tavern
```
