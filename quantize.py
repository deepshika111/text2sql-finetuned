"""Make a 4-bit version of the fine-tuned model.

First merges the LoRA adapter into the base model, then quantizes it
to 4-bit GPTQ using Spider prompts for calibration.

  python quantize.py --adapter runs/sft_s1/adapter --out models/sft_s1
  -> models/sft_s1/merged   (16-bit)
  -> models/sft_s1/gptq4    (4-bit)
"""

import argparse
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load
from prompts import build_messages

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CALIBRATION = 256


def merge(base, adapter, out):
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)


def calibration_texts(tok, n):
    rows = random.Random(0).sample(load("spider_train"), n)
    texts = []
    for row in rows:
        messages = build_messages(row) + [
            {"role": "assistant", "content": row["gold"]}]
        texts.append(tok.apply_chat_template(messages, tokenize=False))
    return texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    merged, gptq = Path(args.out) / "merged", Path(args.out) / "gptq4"
    if not merged.exists():
        merge(args.model, args.adapter, merged)

    from gptqmodel import GPTQModel, QuantizeConfig

    tok = AutoTokenizer.from_pretrained(merged)
    model = GPTQModel.load(str(merged),
                           QuantizeConfig(bits=4, group_size=128))
    model.quantize(calibration_texts(tok, CALIBRATION), batch_size=4)
    model.save(str(gptq))
    tok.save_pretrained(gptq)

    size = lambda d: sum(f.stat().st_size for f in d.rglob("*")) / 1e9
    print(f"merged 16-bit: {size(merged):.2f} GB")
    print(f"gptq 4-bit:    {size(gptq):.2f} GB")


if __name__ == "__main__":
    main()
