# text2sql

Fine-tuning a 1.5B open model (Qwen2.5-Coder-1.5B-Instruct) to turn English
questions into SQL, and testing what actually makes it better: training,
picking the best of several answers, self-correction and retrieval.
Everything runs on one free Colab T4.

Every score is **execution accuracy**: the predicted SQL is run on the real
database and must return the same rows as the correct query.

## Results

*To be filled in from `results/` once the runs finish.*

| Model | Spider test | Test-suite | BIRD test | Latency | $ / 1k questions |
|---|---|---|---|---|---|
| Base 1.5B, greedy | | | | | |
| Fine-tuned (3 seeds), greedy | | | | | |
| + self-correction | | | | | |
| + 8 samples, vote | | | | | |
| + 8 samples, picker | | | | | |
| Fine-tuned, 4-bit GPTQ | | | | | |
| API model (same prompt) | | | | | |

## How the numbers are kept honest

- **Unseen databases only.** Dev and test use databases that never appear
  in training.
- **Locked test set.** All choices (prompts, picker, settings) are made on
  `spider_dev` and `bird_dev`. `spider_test` and `bird_test` need
  `--final`, and every final run is logged to `results/test_log.txt`.
- **Two benchmarks.** Spider is old and may be in the base model's training
  data. BIRD Mini-Dev is harder and newer.
- **Three seeds, 95% intervals.** `stats.py` compares runs question by
  question with a paired bootstrap.
- **Test-suite scoring.** Spider test answers are also checked on extra
  copies of each database with different data, which catches SQL that is
  right by luck.

## Files

| File | What it does |
|---|---|
| `data.py` | Downloads Spider, BIRD Mini-Dev and the test suite; makes the splits |
| `schema.py` | Database → `CREATE TABLE` text with 3 example rows |
| `prompts.py` | Prompt, self-correction prompt, SQL extraction |
| `train.py` | QLoRA fine-tuning, one seed per run, resumes after disconnects |
| `generate.py` | vLLM generation: greedy or N samples, self-correction, RAG |
| `evaluate.py` | Scores every candidate, picks one (first / vote / picker / oracle) |
| `picker.py` | Trains the answer picker on dev candidates |
| `stats.py` | Confidence intervals and paired comparisons across seeds |
| `errors.py` | Labels each mistake and shows what training fixed |
| `retrieval.py` | RAG variants, and how often answers copy names from examples |
| `safety.py` | Read-only guard and unanswerable questions |
| `api_baseline.py` | Same prompt and scoring for an API model, with cost |
| `quantize.py` | Merges the adapter, exports 4-bit GPTQ |
| `bench.py` | Latency, throughput and cost per 1,000 questions |
| `demo.py` | Gradio demo: pick a database, ask a question |

## Run order (Colab, T4 GPU)

```bash
pip install -r requirements.txt     # then restart the runtime
python data.py
pytest -q

# 1. baseline
python generate.py --split spider_dev --out results/base
python generate.py --split bird_dev   --out results/base
python evaluate.py results/base/spider_dev.jsonl

# 2. fine-tune, three seeds (~1.5-2 h each on a T4)
for s in 1 2 3; do python train.py --seed $s --out runs/sft_s$s; done

# 3. greedy + self-correction, per seed
for s in 1 2 3; do
  for split in spider_dev bird_dev; do
    python generate.py --split $split --adapter runs/sft_s$s/adapter \
        --retry --out results/sft_s$s
    python evaluate.py results/sft_s$s/$split.jsonl
  done
done
python stats.py results/base/spider_dev.first.jsonl \
    --vs results/sft_s*/spider_dev.first.jsonl

# 4. answer picker: 8 samples, then vote and picker
python generate.py --split spider_dev --adapter runs/sft_s1/adapter \
    --n 8 --retry --out results/sft_s1_n8
python generate.py --split bird_dev --adapter runs/sft_s1/adapter \
    --n 8 --retry --out results/sft_s1_n8
python evaluate.py results/sft_s1_n8/spider_dev.jsonl --method vote
python picker.py results/sft_s1_n8/*_dev.scored.jsonl \
    --out results/sft_s1_n8/picker.json

# 5. what training fixed
python errors.py results/base/spider_dev.first.jsonl \
    results/sft_s1/spider_dev.first.jsonl

# 6. why RAG hurts (base model, three retrieval modes)
for mode in random similar skeleton; do
  python generate.py --split spider_dev --rag $mode --out results/rag_$mode
  python evaluate.py results/rag_$mode/spider_dev.jsonl
done
python retrieval.py results/rag_*/spider_dev.jsonl

# 7. safety: 100 questions the schema can't answer
python generate.py --split spider_dev --adapter runs/sft_s1/adapter \
    --unanswerable 100 --out results/safety
python evaluate.py results/safety/spider_dev.jsonl

# 8. 4-bit and speed
pip install gptqmodel --no-build-isolation
python quantize.py --adapter runs/sft_s1/adapter --out models/sft_s1
python bench.py --model models/sft_s1/merged
python bench.py --model models/sft_s1/gptq4

# 9. API comparison (needs a key; check today's prices)
python api_baseline.py --split spider_dev --provider openai \
    --model gpt-4o-mini --in-price 0.15 --out-price 0.60 \
    --out results/api
python evaluate.py results/api/spider_dev.jsonl

# 10. newer base model, same pipeline
python generate.py --split spider_dev --model Qwen/Qwen3-1.7B \
    --out results/qwen3_base
python train.py --model Qwen/Qwen3-1.7B --seed 1 --out runs/qwen3_s1
python generate.py --split spider_dev --model Qwen/Qwen3-1.7B \
    --adapter runs/qwen3_s1/adapter --out results/qwen3_s1
python stats.py results/qwen3_base/spider_dev.first.jsonl \
    --vs results/qwen3_s1/spider_dev.first.jsonl

# 11. final test: once, with the settings chosen on dev
python generate.py --split spider_test --adapter runs/sft_s1/adapter \
    --n 8 --retry --out results/final
python evaluate.py results/final/spider_test.jsonl \
    --method picker --picker results/sft_s1_n8/picker.json --final

# demo
python demo.py --model models/sft_s1/gptq4
```

## Related work

Other public text-to-SQL fine-tuning projects on Spider include
[instaquery](https://github.com/tarubee19/instaquery),
[sqltune](https://github.com/EnesUguzz/sqltune) and
[spider-sql-lora-ladder](https://github.com/frangarciaproff/spider-sql-lora-ladder).
