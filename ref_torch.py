"""PyTorch reference for the LiteRT port (CPU, fp32, the upstream code from ../MOSS-VoiceGenerator).

For each prompt in prompts.json writes out/ref/<name>/:
  prompt_ids.npy   [S,17] the processor's input ids (text channel + 16 audio channels)
  gen_ids.npy      [N,17] generated rows (upstream model.generate, seeded)
  codes.npy        [F,16] de-delayed audio codes of the (first) audio segment
  tf_logits.npy    [N,16402] teacher-forced logits for every generated row, laid out like the LiteRT graph output:
                   text-head rows [gen_slot, delay_slot] | 16 audio heads x 1025
  tf_text0.npy     [155648] full text logits of the first generated row
  wav_upstream.wav processor.decode (the codec's chunk_duration=8 streaming decode), as upstream returns it
  wav_full.wav     the codec's non-streaming decode of the same codes (what the LiteRT decoder reproduces)
  meta.json        timings and sizes

  .venv\\Scripts\\python ref_torch.py [--names fox,tavern] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VG = os.path.abspath(os.path.join(HERE, "..", "MOSS-VoiceGenerator"))
CODEC = os.path.abspath(os.path.join(HERE, "..", "MOSS-Audio-Tokenizer"))
GEN_SLOT, DELAY_SLOT = 151656, 151662


def load(threads: int):
    from transformers import AutoModel, AutoProcessor
    torch.set_num_threads(threads)
    processor = AutoProcessor.from_pretrained(VG, trust_remote_code=True, normalize_inputs=True, codec_path=CODEC)
    processor.audio_tokenizer = processor.audio_tokenizer.eval()
    model = AutoModel.from_pretrained(VG, trust_remote_code=True, dtype=torch.float32, attn_implementation="sdpa")
    return processor, model.eval()


@torch.inference_mode()
def teacher_forced(model, ids: torch.Tensor, S: int):
    """Logits for rows S-1 .. end-1 (the rows that predicted each generated row)."""
    out = model(input_ids=ids[None, :-1], use_cache=False)
    lg = [l[0, S - 1:].float() for l in out.logits]
    text = lg[0]
    audio = torch.stack(lg[1:], 1)                                   # [N,16,1025]
    tf = torch.cat([text[:, [GEN_SLOT, DELAY_SLOT]], audio.reshape(audio.shape[0], -1)], 1)
    return tf.numpy(), text[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "ref"))
    args = ap.parse_args()
    with open(os.path.join(HERE, "prompts.json"), encoding="utf-8") as f:
        prompts = json.load(f)
    names = [n for n in args.names.split(",") if n] or list(prompts)

    t0 = time.time()
    processor, model = load(args.threads)
    print(f"loaded in {time.time() - t0:.1f} s", flush=True)
    codec = processor.audio_tokenizer
    sr = processor.model_config.sampling_rate

    for name in names:
        p = prompts[name]
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        conv = [[processor.build_user_message(text=p["text"], instruction=p["instruction"])]]
        batch = processor(conv, mode="generation")
        ids = batch["input_ids"][0]
        S = ids.shape[0]
        np.save(os.path.join(d, "prompt_ids.npy"), ids.numpy())

        torch.manual_seed(args.seed)
        t1 = time.time()
        outputs = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                 max_new_tokens=args.max_new_tokens)
        t_gen = time.time() - t1
        # generate() returns (start_length, rows from 3 rows after the last <|im_start|> on). The prompt ends with
        # "<|im_start|>assistant\n", so start_length is 0 and the rows are exactly the generated ones.
        start, rows = outputs[0]
        gen_rows = rows[int(start):]
        n_gen = gen_rows.shape[0]
        np.save(os.path.join(d, "gen_ids.npy"), gen_rows.numpy())

        t2 = time.time()
        msgs = processor.decode(outputs)
        t_dec_up = time.time() - t2
        wav_up = msgs[0].audio_codes_list[0].numpy() if msgs[0] is not None else np.zeros(0, np.float32)
        sf.write(os.path.join(d, "wav_upstream.wav"), wav_up, sr)

        codes = processor.apply_de_delay_pattern(gen_rows[:, 1:])
        keep = ~(codes == processor.model_config.audio_pad_code).all(1)
        codes = codes[keep]
        np.save(os.path.join(d, "codes.npy"), codes.numpy())

        t3 = time.time()
        with torch.inference_mode():
            dec = codec.decode(codes.T.contiguous().long()[:, None, :], return_dict=True)
        t_dec_full = time.time() - t3
        wav_full = dec.audio[0, 0].float().numpy()
        sf.write(os.path.join(d, "wav_full.wav"), wav_full, sr)

        t4 = time.time()
        tf, text0 = teacher_forced(model, torch.cat([ids, gen_rows], 0), S)
        t_tf = time.time() - t4
        np.save(os.path.join(d, "tf_logits.npy"), tf.astype(np.float32))
        np.save(os.path.join(d, "tf_text0.npy"), text0.astype(np.float32))

        meta = dict(name=name, prompt_rows=int(S), gen_rows=int(n_gen), frames=int(codes.shape[0]),
                    audio_s=float(len(wav_full) / sr), t_generate=t_gen, t_decode_upstream=t_dec_up,
                    t_decode_full=t_dec_full, t_teacher_forced=t_tf, seed=args.seed,
                    first_text=int(gen_rows[0, 0]), last_text=int(gen_rows[-1, 0]))
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    sys.exit(main())
