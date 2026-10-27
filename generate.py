"""Have the model write SQL for every question in a split.

Examples:
  # base model, one answer per question
  python generate.py --split spider_dev --out results/base

  # fine-tuned model, 8 tries per question, retry if the SQL fails
  python generate.py --split spider_dev --adapter runs/sft_s1/adapter \\
      --n 8 --retry --out results/sft_s1

  # add example questions to the prompt
  python generate.py --split spider_dev --rag skeleton --out results/rag_skel

  # add 100 questions the database can't answer
  python generate.py --split spider_dev --adapter ... --unanswerable 100

Saves every answer to <out>/<split>.jsonl, so scoring and picking
don't need the GPU.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from data import load
from execution import run_sql
from gpu import has_bf16
from prompts import (REFUSE, count_tokens, extract_sql, fit_messages,
                     retry_messages)
from retrieval import Retriever
from safety import is_safe, make_unanswerable

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_TOKENS = 300


def check(row, sql):
    """Return the error for a query, or None if it runs fine."""
    if sql == REFUSE:
        return None
    if not is_safe(sql):
        return "only a single read-only SELECT query is allowed"
    _, err = run_sql(row["db_path"], sql, timeout=10)
    return err


def chat(llm, convs, params, lora):
    return llm.chat(convs, params, lora_request=lora, use_tqdm=True,
                    chat_template_kwargs={"enable_thinking": False})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--adapter")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--retry", action="store_true")
    p.add_argument("--rag", default="none",
                   choices=["none", "random", "similar", "skeleton"])
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--unanswerable", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-model-len", type=int, default=8192)
    args = p.parse_args()

    rows = load(args.split)
    if args.unanswerable:
        rows += make_unanswerable(rows, args.unanswerable)
    if args.limit:
        rows = rows[:args.limit]

    bf16 = has_bf16()
    llm = LLM(
        model=args.model,
        dtype="bfloat16" if bf16 else "float16",
        enable_lora=bool(args.adapter),
        max_lora_rank=64,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    lora = LoRARequest("sft", 1, args.adapter) if args.adapter else None

    # leave room for the answer and a possible retry
    tok = llm.get_tokenizer()
    budget = args.max_model_len - 2 * MAX_TOKENS - 100
    retriever = Retriever(load("spider_train"), args.rag, args.k)
    examples, convs, fit = [], [], Counter()
    for row in rows:
        ex = retriever.examples(row)
        msgs, how = fit_messages(row, ex, lambda m: count_tokens(tok, m),
                                 budget)
        fit[how] += 1
        examples.append(ex if how in ("full", "no sample rows") else [])
        convs.append(msgs)
    print("prompt fit:", dict(fit))
    ready = [i for i, c in enumerate(convs) if c is not None]

    greedy = args.n == 1
    params = SamplingParams(
        n=args.n,
        temperature=0.0 if greedy else args.temperature,
        top_p=1.0 if greedy else 0.95,
        max_tokens=MAX_TOKENS,
        logprobs=0,  # needed for cumulative_logprob (used by the picker)
        seed=0,
    )

    start = time.time()
    outputs = chat(llm, [convs[i] for i in ready], params, lora)
    gen_seconds = time.time() - start
    by_row = dict(zip(ready, outputs))

    records = []
    for i, (row, ex) in enumerate(zip(rows, examples)):
        if i not in by_row:  # prompt was too long even after trimming
            cand = {"sql": "", "raw": "", "tokens": 0, "logprob": 0.0,
                    "skipped": "prompt too long"}
            records.append({"id": row["id"], "candidates": [cand],
                            "examples": []})
            continue
        cands = []
        for o in by_row[i].outputs:
            sql = extract_sql(o.text)
            n_tok = max(len(o.token_ids), 1)
            cands.append({
                "sql": sql,
                "raw": o.text,
                "tokens": n_tok,
                "logprob": (o.cumulative_logprob or 0.0) / n_tok,
            })
        records.append({"id": row["id"], "candidates": cands,
                        "examples": [e["gold"] for e in ex]})

    retry_seconds = 0.0
    if args.retry:
        retry_seconds = self_correct(llm, lora, rows, convs, records)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.split}.jsonl"
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    meta = {**vars(args), "questions": len(rows), "prompt_fit": dict(fit),
            "gen_seconds": round(gen_seconds, 1),
            "retry_seconds": round(retry_seconds, 1)}
    (out_dir / f"{args.split}.meta.json").write_text(
        json.dumps(meta, indent=2))
    print(f"wrote {path} ({gen_seconds:.0f}s)")


def self_correct(llm, lora, rows, convs, records):
    """Give every failed query one retry."""
    todo = []
    for i, (row, rec) in enumerate(zip(rows, records)):
        if convs[i] is None:
            continue
        for j, cand in enumerate(rec["candidates"]):
            err = check(row, cand["sql"])
            if err:
                todo.append((i, j, retry_messages(convs[i], cand["sql"], err)))
    if not todo:
        return 0.0

    params = SamplingParams(n=1, temperature=0.0, max_tokens=MAX_TOKENS,
                            logprobs=0, seed=0)
    start = time.time()
    outputs = chat(llm, [conv for _, _, conv in todo], params, lora)
    for (i, j, _), out in zip(todo, outputs):
        cand = records[i]["candidates"][j]
        cand["first_sql"] = cand["sql"]
        cand["sql"] = extract_sql(out.outputs[0].text)
        cand["retried"] = True
    print(f"self-correction retried {len(todo)} candidates")
    return time.time() - start


if __name__ == "__main__":
    main()
