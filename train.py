"""Fine-tune the model on Spider with QLoRA.

Run it once per seed:
  python train.py --seed 1 --out runs/sft_s1
  python train.py --seed 2 --out runs/sft_s2
  python train.py --seed 3 --out runs/sft_s3

It saves a checkpoint every 100 steps. If Colab disconnects, run the
same command again and it picks up from the last checkpoint.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from data import load
from gpu import has_bf16
from prompts import REFUSE, build_messages, count_tokens
from safety import make_unanswerable

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def make_examples(rows, tok, max_len):
    examples, skipped = [], 0
    for row in rows:
        prompt = build_messages(row)
        answer = REFUSE if row.get("unanswerable") else row["gold"]
        completion = [{"role": "assistant", "content": answer}]
        n_tok = count_tokens(tok, prompt + completion,
                             add_generation_prompt=False)
        if n_tok > max_len:  # skip examples that would get cut off
            skipped += 1
            continue
        examples.append({"prompt": prompt, "completion": completion})
    print(f"{len(examples)} examples, {skipped} too long and skipped")
    return examples


def find_checkpoint(args):
    """Find a checkpoint to resume from (local, or the copy on the hub)."""
    out = Path(args.out)
    if out.exists() and get_last_checkpoint(out):
        return get_last_checkpoint(out)
    if args.hub:
        try:
            snapshot_download(args.hub, local_dir=out,
                              allow_patterns="last-checkpoint/*")
        except Exception:
            return None  # nothing saved yet
        if (out / "last-checkpoint").exists():
            print("resuming from hub checkpoint")
            return str(out / "last-checkpoint")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--unanswerable", type=float, default=0.05,
                   help="share of extra training rows that should refuse")
    p.add_argument("--limit", type=int, help="small smoke-test run")
    p.add_argument("--hub", help="HF repo to save checkpoints to")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load("spider_train")
    n_unans = int(len(rows) * args.unanswerable)
    rows += make_unanswerable(rows, n_unans, seed=args.seed)
    random.shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    data = Dataset.from_list(make_examples(rows, tok, args.max_len))

    bf16 = has_bf16()
    dtype = torch.bfloat16 if bf16 else torch.float16  # T4 has no bf16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=args.out,
        seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        max_length=args.max_len,
        completion_only_loss=True,  # only learn from the SQL part
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=20,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        # also save checkpoints to the hub in case colab disconnects
        push_to_hub=bool(args.hub),
        hub_model_id=args.hub,
        hub_strategy="checkpoint",
        hub_private_repo=True,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=data,
                         processing_class=tok, peft_config=lora)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    result = trainer.train(resume_from_checkpoint=find_checkpoint(args))

    adapter = Path(args.out) / "adapter"
    trainer.save_model(adapter)
    tok.save_pretrained(adapter)

    info = {
        "model": args.model,
        "seed": args.seed,
        "examples": len(data),
        "steps": result.global_step,
        "train_loss": result.training_loss,
        "minutes": round((time.time() - start) / 60, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available() else None,
    }
    (Path(args.out) / "train_info.json").write_text(json.dumps(info, indent=2))
    print(info)

    if args.hub:
        HfApi().upload_folder(folder_path=args.out, repo_id=args.hub,
                              allow_patterns=["adapter/*", "train_info.json"])
        print(f"uploaded adapter to {args.hub}")


if __name__ == "__main__":
    main()