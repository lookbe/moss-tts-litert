# MOSS-VoiceGenerator on LiteRT: what was tried and what came of it (2026-10-26)

Status words as in `../pocket-tts-cpp/LITERT_EXPERIMENTS.md`: **KEPT** (in the current pipeline), **NO GAIN**,
**WORSE**, **BLOCKED**, **NOT TRIED**. Correctness was checked numerically against the upstream PyTorch code and by
Whisper transcription. **Nobody has done a listening test.**

## 0. Bottom line

* **It works end to end on LiteRT, on both backends, with no PyTorch at run time.** The fp32 graphs reproduce
  upstream's logits to 2-4e-6 and the codec to 67-71 dB, and generated speech transcribes as well as upstream's
  (sections 3, 6).
* **GPU (RX 7900 XT, WebGPU):** int8 backbone in fp16 (after fixing an fp16 overflow, 5.1) plus the codec in fp32.
  First version: RTFx ~3.6-5.0, first audio after ~0.6-0.7 s. **After the ggml-inspired work (section 10): RTFx
  6.6 streaming / 6.9 batch (7.3 on 29 s), first audio after 0.37 s**, ~35x the PyTorch CPU reference and about
  half of the ggml port's speed (10-17x on Vulkan); the rest of that gap is LiteRT's GPU kernels (section 10).
  Costs: ~15 GB of VRAM, ~31-38 GB of host RAM, and ~40 s of shader compile at load.
* **CPU (7950X, XNNPACK, 4 threads):** int8 backbone RTFx 0.88 (0.95 on the long text); int4 blockwise-32 backbone
  RTFx 1.27, i.e. faster than real time, at a measurable logit cost (KL 0.04 vs 0.006) that Whisper does not
  see. 2.6-8.7 GB RAM with the XNNPACK weight cache (31 GB without).
* Recommended configurations:
  * GPU: `lm_gpu_i8_lean` + `dec_gpu_f16_s` (fp32 arithmetic), streaming 16 then 128 frames, one shared
    environment.
  * CPU: `lm_i8` (quality) or `lm_i4b32` (speed), with `dec_f16` and 64-frame blocks.
* Not done: a listening test; a native (C++) runner (the Python loop costs ~1 ms per step, and the codec cannot
  overlap the backbone on the CPU); phone runs. See section 9.

## 1. Setup

| | |
|---|---|
| machine | Ryzen 9 7950X (16 cores), 127 GB RAM, Radeon RX 7900 XT 20 GB, Windows 11 |
| runtime | `ai_edge_litert` 2.3.0 Python wheel: CPU = XNNPACK, GPU = WebGPU accelerator on D3D12 (needs `dxcompiler.dll` / `dxil.dll`, see README) |
| model | MOSS-VoiceGenerator: Qwen3-1.7B backbone (28 layers, d 2048, GQA 16/8, head 128), 16 audio heads x 1025 + text head 155648, bf16 checkpoint |
| codec | MOSS-Audio-Tokenizer decoder (~0.9B params, fp32 checkpoint): RLFQ dequantizer (16 of 32 codebooks used) + 4 causal transformer stages (32 x d1280 @ 12.5 Hz, 3 x 12 x d768 @ 25 / 50 / 100 Hz), patch reshapes to 24 kHz |
| reference | upstream code on PyTorch 2.14 CPU fp32, transformers 5.0.0 (`ref_torch.py`): 18.7-54 s to generate 3.5-10 s of audio, ~0.3 s per step, RTFx ~0.19 |
| prompts | `prompts.json`: fox, tavern, cold (en), zh_old (zh), long (en, ~30 s) |

Frame rate is 12.5 Hz: one backbone step per 80 ms of audio, plus 17 rows of overhead per utterance (audio_start and
the 16-row delay tail).

## 2. Approach

| decision | why | status |
|---|---|---|
| Graphs written op by op by `tflb.py` (Python port of pocket-tts-cpp's `TfliteBuilder`), from the safetensors | full control over the ops each backend sees (the pocket-tts log is mostly about op choice); no litert-torch, which only runs in WSL here (16 GB RAM limit for a 1.7B model); exports take 5-25 s | KEPT |
| Weights stored after the flatbuffer (Buffer.offset / size, 64-byte aligned) | a 1.7B model is 7 GB in fp32; the flatbuffer stays a few MB. Works on XNNPACK and WebGPU, fp32 / fp16 / int8 / int4 | KEPT |
| Host: tokenizer + prompt template, embedding sums (text table bf16, 16 audio tables), sampling, de-delay, RLFQ lookup tables | all cheap; the text table is 637 MB but only 17 rows per step are read | KEPT |
| Text head reduced to 2 rows in the step graph (gen_slot, delay_slot) + a separate `text_head` signature | inside audio the upstream mask leaves only those two tokens; the full 155648-row head is needed only for the first row and after audio_end (~2 calls per utterance) | KEPT (exact) |
| Prompt builder ported (normalization + template + chat template, `tokenizers`) | identical ids to the upstream processor on all 4 prompts | KEPT |
| Sampler = upstream `generate()` for batch 1, in numpy (incl. the quirk that channels 1-15 share one repetition-penalty token set: upstream's 2-D penalty path flattens all their histories) | same distribution; RNG differs (numpy vs torch), so runs are not bit-identical to torch | KEPT |
| Codec decoder streaming with a full-context sliding K/V window per layer | equals the codec's non-streaming decode for any block size (upstream's chunk_duration=8 ring cache drops context at chunk starts; on these clips the two agree to 99-328 dB) | KEPT |
| Stop at audio_end (upstream continues to im_end) | after audio_end upstream samples free text at temperature 1.5; once in 12 runs that gave junk tokens and an empty second segment (34 wasted steps). Later rows cannot change the first segment, which is what upstream's examples use | KEPT |

## 3. Correctness (teacher-forced, `tests/test_parity.py`)

The reference's prompt + generated rows go through the LiteRT graphs; every step's logits are compared with the torch
logits of the same row. 4 prompts, 62-138 steps each.
* `rel` = max|d| / max|ref| over all logits.
* `KL` = mean KL(torch || litert) of the audio heads' sampling distributions (softmax(logits / 1.5)).
* `top-1` = argmax agreement over (step, audio head).
* `slot` = the text decision inside audio. `text0` = the full text head at the first row.

| backbone | rel | KL (nats) | top-1 | slot | text0 rel / argmax | status |
|---|---|---|---|---|---|---|
| CPU fp32 | 2-4e-6 | 0 | 100% | 100% | 3e-6 / same | reference export |
| CPU int8 per channel (XNNPACK quantizes activations to int8 too) | 3.6-5.8% | 0.0059-0.0067 | 73-79% | 100% | 3.4% / same | KEPT (CPU) |
| CPU int4 per channel, round to nearest (heads int4 too) | 51-104% | 0.138-0.153 | 27-32% | 98-100% | 27% / same | WORSE (quality) |
| CPU int4 blockwise 128 (fp16 scales), heads int8 | 10-16% | 0.060-0.064 | 42-49% | 98-100% | 4.0% / same | NO GAIN over block 32 |
| **CPU int4 blockwise 32 (fp16 scales), heads int8** | 7-13% | 0.038-0.044 | 47-54% | 100% | 4.0% / same | KEPT as the fast CPU option |
| GPU int8, fp16 arithmetic, unmodified graph | NaN | NaN | 0% | 76-89% | NaN | BLOCKED (see 5.1) |
| GPU int8, fp32 arithmetic (`enforce_f32`) | 1.0% | 0.00036 | 93% | 100% | 1.3% / same | works, 2.3x slower step |
| **GPU int8, fp16 arithmetic, scaled residual (5.1)** | 1.1% | 0.00042-0.00046 | 91-94% | 100% | 1.6% / same | KEPT (GPU) |

GPU int8 is weight-only (activations stay float), hence its 15x lower KL than XNNPACK's int8.

Codec, reference codes -> audio, SNR against the codec's non-streaming torch decode:

| codec | SNR | status |
|---|---|---|
| CPU fp32 weights | 67-71 dB (blocks 4 and 16 alike) | reference export |
| CPU fp16 weights (expanded to fp32 by XNNPACK) | 58-65 dB | KEPT (CPU) |
| CPU int8 weights (XNNPACK quantizes activations) | 18-25 dB | WORSE (quality) |
| GPU fp16 weights, fp16 arithmetic | 27-41 dB (worse at 16-frame blocks) | WORSE |
| **GPU fp16 weights, fp32 arithmetic** | 60-65 dB, same speed at 16-frame blocks | KEPT (GPU) |

End-to-end: the streamed GPU audio equals the batch-decoded audio of the same rows to 104-116 dB, and a seed gives
identical rows across runs on the GPU.

## 4. CPU (XNNPACK, 7950X)

Models: `lm_i8.tflite` (caps 256/512/1024/2048, prefill 64 at 256/512, text head), `dec_f16.tflite` (blocks
1/4/16/32/64). Flat 1-D signature I/O bound zero-copy to host arrays (4.3).

### 4.1 Threads (backbone int8, P=512, `tests/bench_step.py`)

| threads | step | prefill 105 rows |
|---|---|---|
| 2 | 56.7 ms | 526 ms |
| **4** | **57.5 ms** (55 ms at P=256) | **338 ms** (293 ms at P=256) |
| 8 | 72.6 ms | 404 ms |
| 16 | 205 ms | 1143 ms |

More threads only add synchronization: one row through ~2,800 small ops. 4 threads it is (KEPT). The codec is not
faster with more threads either (16-frame blocks: 2.2x / 1.7x / 1.6x real time at 4 / 8 / 16 threads).

### 4.2 Weight streaming floor (`tests/bench_fc.py`: the 196 backbone FCs alone, 1 row)

| weights | ms per step | effective GB/s |
|---|---|---|
| fp32 | 117-125 | 45-48 |
| int8 per channel (qd8 x qc8w) | 42-48 | 30-34 |
| int4 per channel (qd8 x qc4w) | 31 | 23 |

So the int8 step (55 ms) is ~45 ms of FC weight streaming + ~10 ms of attention, norms and RoPE. Thread count does
not change the floor (memory bound at ~34 GB/s on this desktop).

### 4.3 LiteRT Python API findings

| finding | consequence |
|---|---|
| `TensorBuffer.create_from_host_memory` always makes a 1-D buffer; binding it to a ranked input fails at run ("Failed to register input tensor buffer") | every CPU signature takes and returns flat 1-D tensors and reshapes inside; a sidecar JSON keeps the logical shapes (KEPT). The in-graph reshape of a 117 MB input cost 0.8 ms (7.9 vs 7.1 ms), i.e. no copy |
| zero-copy works: host writes into the bound arrays are seen by the next run | the CPU KV cache is plain numpy; the graph returns only the new rows (KEPT) |
| **XNNPACK packs a separate weight copy per signature** | backbone (7 signatures) peaks at 11.2 GB RSS, + fp16 codec (5 signatures) 31 GB |
| **`CpuOptions(xnnpack_weight_cache_path=<file>)`** (a directory is silently ignored) | one shared mmapped packed copy: backbone 11.2 -> 1.9-2.2 GB, codec ~20 -> 6.1 GB, backbone reload 2.8 -> 0.3 s, step unchanged; the codec even got faster (3.4 -> 5.7x real time at 64-frame blocks). KEPT as the default, cache file named after the model's size + mtime |
| `xnnpack_flags=0` | hybrid int8 FCs fall back to TFLite's builtin kernels: codec 0.38x real time, 13.9 dB. WORSE |
| `enable_ynnpack=True` | not in this Windows wheel ("ERROR: [unknown:0]"). BLOCKED |
| no weight-only int8 (float activations) path for FCs | int8 on the CPU always quantizes activations: fine for the backbone (KL 0.006), not for the codec (18-25 dB). The CPU codec stays fp16 |

### 4.4 Codec block size (fp16 weights, 4 threads, before the weight cache)

| block (frames per call) | real time factor |
|---|---|
| 4 | ~1.0x |
| 16 | 2.6-2.9x |
| 32 | 2.7-3.7x |
| 64 | 3.1-4.0x (5.7-5.8x with the weight cache) |

Each call streams all ~0.9B weights, so small blocks are weight-bound. KEPT: 64-frame blocks for batch decoding on
the CPU.

### 4.5 int4 on the CPU

| variant | step (P=256..512) | KL | status |
|---|---|---|---|
| int8 per channel | 55 ms | 0.006 | KEPT (quality) |
| int4 per channel (XNNPACK qc4w), heads int4 too | 35 ms | 0.14-0.15 | WORSE (quality) |
| int4 blockwise 128, heads int8 | 36 ms | 0.06 | NO GAIN |
| **int4 blockwise 32, heads int8** | **37-38 ms** | **0.04** | KEPT (speed option) |

* Blockwise int4 = TFLite `BlockwiseQuantization` details on an INT4 weight, scales a constant FLOAT16 tensor
  `[rows, cols / block]`, no zero points. XNNPACK takes it (qb4w); the probe (`tests/probe_i4b.py`) matches the
  dequantized weights to 0.8% (the int8 activation quantization).
* **Bug found on the way:** an all-zero block (the never-predicted pad row 1024 of each audio head) got scale
  1e-9 / 7, which is 0 in fp16. XNNPACK then rejects the whole tensor ("unsupported scale value (0.000000) in channel
  16416 for INT4 tensor"), and the run fails. Fixed by flooring the scales at 2^-14, and by keeping the 17 output
  heads and the text head in int8 for int4 exports (their logits go straight into sampling).
* **WebGPU (2.3.0) segfaults** compiling a blockwise int4 FC, even one alone (exit 139), the same as the pocket-tts
  log reports. BLOCKED on this GPU; per-channel int4 runs there (probe: 0.14% error) but was not tried in the
  backbone.

## 5. GPU (WebGPU on the RX 7900 XT)

Models: `lm_gpu_i8.tflite` (`--kv dus --ranked-io --safe-norm --resid-scale 0.125 --mlp-scale 0.125`, caps 512/1024,
prefill 64 at 512), `dec_gpu_f16.tflite` (`--ranked-io --safe-norm`, blocks 4 / 16). Every op is accelerated
(`is_fully_accelerated() == True`) for both.

### 5.1 fp16 overflow in the backbone (found and fixed)

* With the default fp16 arithmetic the unmodified graph outputs NaN from the first prompt chunk.
* Cause (measured with forward hooks in torch, `cold` prompt): a "massive activation". Layer 2's MLP writes
  **169,028** into one residual channel (its input `silu(gate) * up` already reaches 81,949); the value stays in the
  residual stream through all 28 layers (169,219 at the end). fp16 tops out at 65,504. Everything else is small
  (q / k <= 1,543, other residual updates <= 11,401).
* Fix, exact in real arithmetic:
  * the residual stream runs scaled by c = 1/8 (input x c, `o_proj` and `down_proj` rows x c, residual RMSNorm eps x
    c^2; RMSNorm is scale invariant);
  * `up_proj` rows x 1/8, undone in `down_proj` (so the MLP product stays < 10,300);
  * every RMSNorm divides by max|x| first (the "safe" norm from pocket-tts), since x^2 still overflows.

  Row scalings are exact under per-channel int8 quantization.

| GPU backbone | step | KL vs torch |
|---|---|---|
| fp16 arithmetic, unmodified | 8.5 ms | NaN |
| fp32 arithmetic (`GpuOptions(enforce_f32=True)`) | 20 ms | 0.00036 |
| **fp16 arithmetic, scaled** | **8.7-9.6 ms** | 0.00042-0.00046 |

### 5.2 Timings

| | value |
|---|---|
| backbone step (P=512, int8, fp16) | 8.5-10 ms median in the loop (jitter up to 17 ms when codec work shares the GPU queue) |
| prefill (2 chunks of 64 rows) | 0.20-0.28 s |
| codec, 16-frame blocks, fp32 arithmetic | 14-29x real time (fp16 arithmetic: the same speed, 25-35 dB worse) |
| codec, 4-frame blocks | 3-14x real time |
| compile at load | backbone 24-34 s, codec 17-23 s (no program cache used) |
| **end to end, batch decode, 12 runs** | **RTFx median 3.74** (3.08-5.06), 80.2 s of audio in 20.9 s |
| **end to end, streaming (first block 4 frames, then 16)** | RTFx median 3.57 (1.92-4.79), **time to first audio median 0.59 s** (0.48-0.84) |

Where the time goes (tavern seed 0, 10.2 s of audio, batch decode): prefill 0.20 s, 145 backbone steps 1.23 s,
sampling (numpy) 0.16 s, codec 0.42 s, total 2.0 s.

The step is not bandwidth bound: 1.4 GB of int8 weights at 800 GB/s would take ~2 ms. The graph has ~100 ops per
layer (~2,800 per step, safe norms and RoPE slices included), so like pocket-tts on this GPU it is dispatch bound,
and every step ends with a host read-back, because sampling is on the host.

## 6. Intelligibility (Whisper large-v3-turbo, `tests/asr_eval.py`, `tests/asr_summary.py`)

Seeds 0 / 1 / 2 per prompt. WER for English, CER for Chinese. Torch = upstream generate + the processor's decode.
The LiteRT runs use their own RNG, so these are different samples of the same distribution, not the same audio.

| prompt | torch | LiteRT GPU (int8, fp16) | LiteRT CPU int8 | LiteRT CPU int4 per channel | LiteRT CPU int4 block 32 |
|---|---|---|---|---|---|
| fox | 0.00 0.00 0.00 | 0.00 0.00 0.00 | 0.00 0.00 0.00 | 0.00 0.00 0.11 | 0.00 0.00 0.00 |
| tavern | 0.00 0.00 0.00 | 0.05 0.00 0.21 | 0.00 0.00 0.00 | 0.11 0.05 0.00 | 0.00 0.00 0.00 |
| cold | 0.06 0.00 0.41 | 0.06 0.00 0.00 | 0.06 0.00 0.12 | 0.06 0.06 0.00 | 0.00 0.00 0.00 |
| zh_old (CER) | 0.18 0.06 0.18 | 0.18 0.18 0.18 | 0.18 0.18 0.06 | 0.06 0.06 0.06 | 0.18 0.06 0.06 |
| mean en WER | 0.052 | 0.036 | 0.020 | 0.043 | 0.000 |
| mean zh CER | 0.137 | 0.176 | 0.137 | 0.059 | 0.098 |

* All within sample noise. The largest errors are the model ending the text early (torch seed 2 "cold" stops after
  "catching a cold"; GPU seed 2 "tavern" drops "or a tall tale") or Whisper skipping the "Achoo!" / "哎呀"
  interjections. These happen in torch as well.
* WER does not measure timbre, prosody or audio quality. The int4 rows transcribe fine, but their logit error
  (KL 0.04-0.15) is the kind of change a listening test should judge. **Not listening-tested.**

## 7. End to end (`tests/bench_e2e.py`, 4 prompts x seeds 0-2, load once, warm-up run excluded)

RTFx = seconds of audio / wall seconds from text to the last sample (prefill + steps + sampling + decode).

| path | backbone step | RTFx median (min-max) | time to first audio | load | peak RSS |
|---|---|---|---|---|---|
| PyTorch fp32 CPU, upstream code (reference) | ~300 ms | ~0.19 | (not streaming) | | |
| LiteRT CPU int8 + fp16 codec (64-frame blocks), 4 threads | 54 ms | 0.88 (0.73-1.08) | (batch) | 8.7 s | 8.7 GB |
| LiteRT CPU int4 per channel + fp16 codec | 35 ms | 1.18 (0.97-1.41) | (batch) | 9.9 s | 7.7 GB |
| **LiteRT CPU int4 blockwise 32 + fp16 codec** | **35 ms** | **1.27 (1.06-1.51)** | (batch) | 10.1 s | 7.7 GB |
| **LiteRT GPU int8 fp16 + codec fp32, batch decode** | **9.6 ms** | **3.74 (3.08-5.06)** | 1.1-3.4 s | 52 s | |
| LiteRT GPU, streaming (first block 4, then 16) | 10.3 ms | 3.57 (1.92-4.79) | **0.59 s** (0.48-0.84) | 57 s | |

* CPU loads are with the XNNPACK weight cache already written (the first load writes it: +1-2 s; without the cache
  the same runs peaked at 31 GB).
* A single early CPU int8 run measured 0.57 / 82 ms per step; per-step timing inside `generate()` and a repeat gave
  55 ms, i.e. that run was disturbed. Compare numbers within one table.
* The GPU load is the WebGPU shader compile of every signature (no program cache tried).
* The CPU time per audio second: backbone ~0.7 s (int8) or ~0.45 s (int4), codec ~0.15-0.18 s at 64-frame blocks,
  prefill ~0.3-0.5 s per utterance. The CPU is sequential here (the LiteRT Python call holds the GIL, as pocket-tts
  found), so overlapping codec and backbone would need threads in native code.

### 7.1 Long text (`long`, ~29 s of speech, 195 prompt rows + ~380 generated)

| path | runs | audio | RTFx | first audio | WER |
|---|---|---|---|---|---|
| PyTorch fp32 CPU (reference) | 2 | 28.8 / 30.2 s | 0.25 (113 s for 28.8 s) | | 0.011 / 0.200 (seed 1 drops the last sentence) |
| LiteRT GPU streaming | 3 | 28.9-29.4 s | 4.96-5.08 | 0.66-0.76 s | 0.000 / 0.011 / 0.011 |
| LiteRT CPU int8 | 2 | 27.8 / 29.3 s | 0.95 | (batch) | 0.011 / 0.011 |

* The KV caches grow during these runs: CPU 256 -> 512 slots (host copy), GPU 512 -> 1024 (read back, re-upload).
  Teacher-forced over the torch reference's 378 rows the logits stay as close as on the short prompts (CPU int8 KL
  0.0077, GPU 0.00066; slot decisions 100%), so growth is exact.
* Longer utterances are faster per second of audio on the GPU (prefill and decoder ramp amortized), and capacities
  1024 / 2048 cover ~60 / ~140 s.

## 8. Other findings

| finding | consequence |
|---|---|
| **Upstream does not normalize prompts in practice.** `processor.build_user_message()` defaults to `normalize=False`, and the processor passes already-built messages through unchanged, so `normalize_inputs=True` (README, web app) has no effect | `Prompter.build` defaults to no normalization (it had normalized: "middle-aged" became "middle，aged", 1 token off on `long`, which changed the generated speech). Ids now match the processor on all 5 prompts |
| After audio_end upstream keeps sampling free text at temperature 1.5 | stop at audio_end (section 2) |
| Upstream's repetition penalty for audio channels 1-15 uses one token set (the union of all 15 channels' histories) | ported as is |
| GPU memory: each signature holds its own weight copy (backbone 6.1 GB for 4 signatures; codec 8.1 GB for 2 block sizes in fp32, 4.4 GB for one, 2.6 GB for one in fp16) | for a 20 GB card, use one codec block size or fp16 codec arithmetic. `GpuOptions(constant_tensor_sharing=True)` segfaults on WebGPU per the pocket-tts log; NOT TRIED here |
| GPU process host memory peaks at ~28 GB during load, and shared GPU memory reaches 13 GB during compile (3 GB after) | the WebGPU accelerator keeps host copies of the weights (pocket-tts measured ~1.5x the VRAM) |

## 9. Not tried / open

1. **Listening test** of int8 vs int4-b32 (CPU) and of the GPU fp16 backbone. The WAVs are in `out/litert_*`.
2. **Native runner** (C++ on the LiteRT C API, as pocket-tts-cpp's `src/litert`): per-step Python and numpy
   sampling cost ~1 ms on the GPU path (~10%), and on the CPU the codec could run on other cores while the backbone
   steps.
3. **In-graph sampling** (Gumbel-max with top-k / top-p) plus in-graph embedding lookup, so the GPU can queue several
   steps per sync. The delay pattern makes the text token mostly forced, but the audio sampling needs top-p and the
   repetition penalty in-graph.
4. **GPU op count:** the step is dispatch bound (~2,800 ops). Candidates: q/k pre-scaled so their RMSNorms need no
   safe form (5 fewer ops per norm), fused q/k/v and gate/up FCs, RoPE as one op pair.
5. **GPTQ / AWQ for int4** (pocket-tts: GPTQ halved the int4 error). Blockwise int4 on the GPU is blocked by the
   WebGPU crash; per-channel int4 on the GPU was only probed (one FC).
6. **Codec on the CPU with int8 weights and float activations** (no such XNNPACK path exposed; int8 activations
   cost 35+ dB).
7. **Phones** (OpenCL / Adreno / Mali). The backbone alone is 1.4 GB in int8 and the codec 0.9B parameters, much
   larger than pocket-tts; the per-signature weight copies would not fit in phone RAM as exported here.
8. GPU program cache (`serialization_dir`) to cut the ~55 s compile.

## 10. Learning from the ggml port (`../moss-tts-ggml`, 2026-10-26)

`moss-tts-ggml` is a GGML port of the same model (Q8_0 LM, F16 / F32 codec, Vulkan). Same machine, same prompts:

| | ggml Vulkan Q8_0 + F16 codec | LiteRT GPU before | LiteRT GPU after this section |
|---|---|---|---|
| long (29 s) | 1.72 s, **RTFx 17** | ~5.8 s, RTFx 5.0 | ~4.0 s, RTFx 7.2-7.3 |
| tavern (4.6-9 s) | 0.30-0.39 s, RTFx 10-12 | RTFx 3.8-5.1 | RTFx 6.9-7.6 |
| LM step | 4.1 ms (3.78 compute + 0.36 host) | 9.6-10.4 ms | 7.5 ms (9.2 past 512 slots) |
| prefill | 41-46 ms | 0.20-0.45 s | 87-89 ms (112 ms for 195 rows) |
| codec, 29 s clip | 87-94 ms | ~1.0 s | ~300 ms |
| sampling, 29 s clip | (C++, in the 0.23 ms/step host time) | ~0.6 s | ~0.12 s |
| first audio | (no streaming: whole clip) | 0.59 s | **0.37 s** |
| load | ~2.5 s | ~55 s | ~40 s |

What ggml does structurally, and what was tried here:

| ggml structure | LiteRT experiment | result | status |
|---|---|---|---|
| ~20 fused kernels per layer (`rms_norm`, `rope_ext`, `swiglu`, `flash_attn_ext`) | our step graph had 2,731 ops (97 per layer: the fp16-safe RMSNorm 13 ops x 4, RoPE 9 x 2, ...). Measured: the GPU step is 3.7 ms of FC weight streaming (`tests/bench_gpu_gemv.py`, 380 GB/s) + ~4 ms for the other ~2,500 ops; the `run()` call alone spends 4 ms encoding dispatches (`tests/bench_gpu_step.py`) | see the next rows | |
| fused RMSNorm | `L2_NORMALIZATION` (2-op RMSNorm): **rejected by WebGPU**. `odml.rms_norm` composite: **crashes WebGPU** (0xC0000005). Norm weights folded into the next FC's columns: 1 op less, but int8 per-row quantization of folded weights costs **~4x the KL** | 7-op safe core + 1 MUL kept | BLOCKED / WORSE |
| fused RoPE | rotate_half as an FC by a constant 128x128 matrix: 4 ops instead of 9, exact in fp32, 1e-3 in fp16; q and k through one FC, one norm, one RoPE over 24 heads | **`--lean`**: 1,468 ops, KL unchanged (0.00042) | KEPT |
| plain q/k RMSNorm | q / k rows pre-scaled (1/32, 1/128) so plain squares fit fp16: 4 ops instead of 7, but KL x3-8 (fp16 subnormals in small squares) | safe core kept | WORSE |
| `flash_attn_ext` | `odml.scaled_dot_product_attention` composite: **crashes WebGPU** (as in the pocket-tts log) | | BLOCKED |
| fused projections (q/k/v, gate/up) | GEMV floor identical fused or separate (3.68 vs 3.72 ms) | | NO GAIN |
| Q4 / smaller weights | per-channel int4 GEMV = int8 time on WebGPU (3.73 ms); fp16 weights 7.7 ms; `allow_src_quantized_fc_conv_ops` no change; blockwise int4 crashes (4.5) | int8 stays | NO GAIN |
| Lean step, all of the above | 8.3 -> **6.6 ms** GPU time per step, 7.5 ms in the loop. Ablating every norm, RoPE and the attention only reaches 6.2 ms: the rest is the FC kernels (3.7 ms) and the cache I/O | | KEPT |
| KV cache written in place (`set_rows`), attention over `pad(T, 256)` | ours: caches are graph I/O. Step at P = 256 / 512 / 1024 / 2048: 7.2 / 7.6 / 10.3 / 19.1 ms. Read-only caches: 7.4 / 6.9 / 8.7 / 12.2 ms. `tests/bench_gpu_cacheio.py`: bound graph I/O is converted every run, the write-back costs most (K alone 5.3 ms at 2048); fp16-typed cache I/O is slower (casts) | two-level cache would help past 512 slots only | NOT TRIED (open) |
| one prefill call | prefill graphs for 128 and 256 rows, the smallest that fits is used: 85-105 rows 133 -> 87 ms, 195 rows 274 -> 112 ms | | KEPT |
| codec: one graph over the whole clip | GPU codec blocks of 16 / 64 / 128 / 384 frames: 22 / 37 / 48 / 112x real time (warm). The ablation (`export_codec.py --ablate`) shows the codec is FC-bound: removing LayerNorm, RoPE, attention or GELU changes nothing, removing the FFN doubles the speed; WebGPU FCs reach 5 (128 rows) to 13 (1,024 rows) TFLOPS, ggml's Vulkan coopmat ~18 | 128-frame blocks (one signature, ~100 ms per 10 s) | KEPT |
| codec: no per-utterance state reset | our reset rewrote ~340 MB of zeros into the GPU state banks per utterance; stale states are finite and masked, so zero once | -100+ ms per utterance | KEPT |
| codec: masks | steady-state masks are identical from call to call: cached and not re-uploaded (-17 ms of numpy per call) | | KEPT |
| codec: the official chunked semantics | the official decode is chunked (8 s); our full-context decode differs on the 29 s clip by 16 dB (ggml found 22 dB). `--codec-chunk 100` reproduces it through the mask: **63 dB** vs upstream's waveform | full context stays the default | KEPT (option) |
| sampling in C++ | numpy sampling vectorized over the 16 rows: 1.6 -> 0.115 ms per step, same distribution (max deviation 0.015 over 4,000 draws, same support) | | KEPT |
| one backend context | **every Python `CompiledModel` created its own environment = its own WebGPU device**; alternating the LM and the codec then made each codec call 3-8x slower (16-frame block 300-800 ms instead of 40). One shared `Environment` for all models fixes it | streaming RTFx 3.6 -> 6.6, first audio 0.87 -> 0.37 s | KEPT |
| CPU | the lean graph on XNNPACK: 47.6 vs 48.6 ms per step (FC-bound at ~42 ms). ggml CPU Q8_0 at 16 threads: 42 ms per step, RTFx 1.47; ours 0.88 (int8) / 1.27 (int4 b32) | | NO GAIN |

End to end after this section (`tests/bench_e2e.py`, 5 prompts x 3 seeds, warm):

| GPU mode | RTFx median (min-max) | first audio | LM step |
|---|---|---|---|
| batch, codec 128-frame blocks | **6.88** (4.63-7.64) | 0.68-1.19 s (short), ~4 s (long) | 7.5 ms |
| streaming, first block 16 then 128 | **6.63** (4.36-7.35) | **0.37 s** (0.40-0.42 long) | 7.5 ms |

Where the rest of the gap to ggml is, measured: the WebGPU GEMV (3.7 ms vs ggml's whole 3.8 ms step) and multi-row FC throughput (5-13 vs ~18 TFLOPS), and the cache-as-graph-I/O cost. Neither can be fixed from the graph side on LiteRT 2.3.0; a newer runtime with working composites / in-place caches, or a native runner around the C API, are the next steps.

Intelligibility of the lean GPU streaming output (Whisper, 5 prompts x 3 seeds): en WER 0.018 (fox / tavern 0.00,
cold 0 / 0 / 0.18, long 0.01-0.02), zh CER 0.137, the same as torch (0.052 / 0.137) and the first GPU version.
