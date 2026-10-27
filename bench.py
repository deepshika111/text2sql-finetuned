"""Measure speed and cost of a model.

  python bench.py --model models/sft_s1/merged --gpu-price 0.35
  python bench.py --model models/sft_s1/gptq4  --gpu-price 0.35

Measures time for a single question and throughput at a few batch
sizes. --gpu-price is dollars per GPU hour, used to work out the cost
per 1,000 questions.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams

from data import load
from gpu import has_bf16
from prompts import build_messages

BATCHES = [1, 8, 32, 128]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--gpu-price", type=float, default=0.35)
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()

    rows = load("spider_dev")
    convs = [build_messages(r) for r in rows]
    bf16 = has_bf16()
    llm = LLM(model=args.model, dtype="bfloat16" if bf16 else "float16",
              max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=300)
    kwargs = {"use_tqdm": False,
              "chat_template_kwargs": {"enable_thinking": False}}

    llm.chat(convs[:8], params, **kwargs)  # warm up first

    single = []
    for conv in convs[:args.repeats]:
        start = time.time()
        llm.chat([conv], params, **kwargs)
        single.append(time.time() - start)

    results = {"model": args.model,
               "disk_gb": round(sum(f.stat().st_size for f in
                                    Path(args.model).rglob("*")) / 1e9, 2),
               "median_latency_s": round(statistics.median(single), 3),
               "throughput": {}}
    for size in BATCHES:
        batch = convs[:size]
        start = time.time()
        outs = llm.chat(batch, params, **kwargs)
        seconds = time.time() - start
        tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        results["throughput"][size] = {
            "questions_per_s": round(size / seconds, 2),
            "tokens_per_s": round(tokens / seconds, 1),
        }

    best = max(v["questions_per_s"] for v in results["throughput"].values())
    results["usd_per_1000"] = round(args.gpu_price * 1000 / best / 3600, 4)
    print(json.dumps(results, indent=2))
    name = Path(args.model).name
    Path("results").mkdir(exist_ok=True)
    Path(f"results/bench_{name}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
