"""Run the same questions through an API model, for comparison.

  export OPENAI_API_KEY=...      # or ANTHROPIC_API_KEY
  python api_baseline.py --split spider_dev --provider openai \\
      --model gpt-4o-mini --in-price 0.15 --out-price 0.60 \\
      --out results/api_gpt4omini

Prices are dollars per 1M tokens (check the provider's pricing page).
Output files match generate.py, so evaluate.py works on them as-is.
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data import load
from prompts import build_messages, extract_sql

MAX_TOKENS = 300


def call_openai(client, model, messages):
    r = client.chat.completions.create(
        model=model, messages=messages, temperature=0,
        max_completion_tokens=MAX_TOKENS)
    u = r.usage
    return r.choices[0].message.content, u.prompt_tokens, u.completion_tokens


def call_anthropic(client, model, messages):
    system, chat = messages[0]["content"], messages[1:]
    r = client.messages.create(model=model, system=system, messages=chat,
                               temperature=0, max_tokens=MAX_TOKENS)
    u = r.usage
    return r.content[0].text, u.input_tokens, u.output_tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--provider", choices=["openai", "anthropic"],
                   required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--in-price", type=float, required=True)
    p.add_argument("--out-price", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    if args.provider == "openai":
        from openai import OpenAI
        client, call = OpenAI(), call_openai
    else:
        from anthropic import Anthropic
        client, call = Anthropic(), call_anthropic

    rows = load(args.split)[:args.limit]

    def run(row):
        messages = build_messages(row)
        for attempt in range(4):  # wait and retry on rate limits
            try:
                start = time.time()
                text, tin, tout = call(client, args.model, messages)
                return row["id"], text, tin, tout, time.time() - start
            except Exception as e:
                if attempt == 3:
                    print(f"{row['id']} failed: {e}")
                    return row["id"], "", 0, 0, None
                time.sleep(2 ** attempt * 5)

    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, rows))
    wall = time.time() - start

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{args.split}.jsonl").open("w") as f:
        for qid, text, tin, tout, _ in results:
            cand = {"sql": extract_sql(text), "raw": text, "tokens": tout}
            f.write(json.dumps({"id": qid, "candidates": [cand]}) + "\n")

    tin = sum(r[2] for r in results)
    tout = sum(r[3] for r in results)
    cost = (tin * args.in_price + tout * args.out_price) / 1e6
    latencies = [r[4] for r in results if r[4] is not None]
    meta = {
        **vars(args),
        "questions": len(rows),
        "gen_seconds": round(wall, 1),
        "median_latency_s": round(statistics.median(latencies), 3),
        "p95_latency_s": round(sorted(latencies)[int(0.95 * len(latencies))], 3),
        "usd_per_1000": round(1000 * cost / len(rows), 4),
        "failed_calls": len(rows) - len(latencies),
    }
    (out_dir / f"{args.split}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
