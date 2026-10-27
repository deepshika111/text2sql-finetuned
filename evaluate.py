"""Score the saved answers and pick one per question.

  python evaluate.py results/sft_s1/spider_dev.jsonl --method vote

Ways to pick:
  first    the first answer (use this for single-answer runs)
  vote     the result most answers agree on
  picker   use a trained picker (needs --picker path/to/picker.json)
  oracle   counts as right if any answer is right (best possible)

Test splits only run with --final, and each final run is written to
results/test_log.txt, so it's clear how often the test set was used.
"""

import argparse
import json
import time
from pathlib import Path

from data import load
from execution import run_sql, score, signature
from picker import choose_picker, choose_vote, load_picker
from prompts import REFUSE
from safety import is_safe, make_unanswerable
from stats import bootstrap_ci

TEST_LOG = Path("results/test_log.txt")


def load_rows(preds_path):
    meta_path = preds_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    rows = load(meta["split"])
    if meta.get("unanswerable"):
        rows += make_unanswerable(rows, meta["unanswerable"])
    return meta, {r["id"]: r for r in rows}


def score_candidate(row, sql):
    if row.get("unanswerable"):
        ok = sql == REFUSE
        return {"correct": ok, "error": None if ok else "answered"}
    if not sql:
        return {"correct": False, "error": "no answer"}
    if sql == REFUSE:
        return {"correct": False, "error": "refused", "sig": REFUSE}
    if not is_safe(sql):
        return {"correct": False, "error": "unsafe"}
    rows, err = run_sql(row["db_path"], sql)
    result = score(row, sql, rows, err, suite=False)
    result["sig"] = signature(rows)
    result["rows"] = len(rows) if rows is not None else None
    return result


def score_file(preds_path, rows):
    """Score every answer once and save it to <split>.scored.jsonl."""
    out = preds_path.with_suffix(".scored.jsonl")
    if out.exists() and out.stat().st_mtime > preds_path.stat().st_mtime:
        return [json.loads(line) for line in out.open()]

    scored = []
    start = time.time()
    for line in preds_path.open():
        rec = json.loads(line)
        row = rows[rec["id"]]
        for cand in rec["candidates"]:
            cand.update(score_candidate(row, cand["sql"]))
        scored.append(rec)
    with out.open("w") as f:
        for rec in scored:
            f.write(json.dumps(rec) + "\n")
    print(f"scored {len(scored)} questions in {time.time() - start:.0f}s")
    return scored


def pick(cands, method, weights=None):
    if method == "first":
        return 0
    if method == "vote":
        return choose_vote(cands)
    if method == "picker":
        return choose_picker(cands, weights)
    raise ValueError(method)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("preds", type=Path)
    p.add_argument("--method", default="first",
                   choices=["first", "vote", "picker", "oracle"])
    p.add_argument("--picker", type=Path)
    p.add_argument("--final", action="store_true")
    args = p.parse_args()

    meta, rows = load_rows(args.preds)
    split = meta["split"]
    if split.endswith("_test"):
        if not args.final:
            raise SystemExit(f"{split} is the locked test set. Use dev "
                             "splits for decisions, then pass --final once.")
        TEST_LOG.parent.mkdir(exist_ok=True)
        with TEST_LOG.open("a") as f:
            f.write(f"{time.ctime()}\t{args.preds}\t{args.method}\n")

    scored = score_file(args.preds, rows)
    weights = load_picker(args.picker) if args.method == "picker" else None

    choices = []
    for rec in scored:
        row, cands = rows[rec["id"]], rec["candidates"]
        if args.method == "oracle":
            flags = [c["correct"] for c in cands]
            correct = None if None in flags else any(flags)
            choice = {"id": rec["id"], "correct": correct, "index": None}
        else:
            i = pick(cands, args.method, weights)
            cand = cands[i]
            choice = {"id": rec["id"], "index": i, "sql": cand["sql"],
                      "correct": cand["correct"], "error": cand.get("error")}
            if row.get("suite_dir") and cand["correct"]:
                choice["suite"] = score(row, cand["sql"])["suite"]
            elif row.get("suite_dir"):
                choice["suite"] = cand["correct"]
        choice["unanswerable"] = bool(row.get("unanswerable"))
        choice["difficulty"] = row.get("difficulty", "")
        choices.append(choice)

    out = args.preds.with_suffix(f".{args.method}.jsonl")
    with out.open("w") as f:
        for c in choices:
            f.write(json.dumps(c) + "\n")

    summary = summarize(choices, meta)
    summary_path = args.preds.with_suffix(f".{args.method}.summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def summarize(choices, meta):
    normal = [c for c in choices
              if not c["unanswerable"] and c["correct"] is not None]
    unans = [c for c in choices if c["unanswerable"]]

    acc = [float(c["correct"]) for c in normal]
    low, high = bootstrap_ci(acc)
    summary = {
        "split": meta["split"],
        "questions": len(normal),
        "skipped_bad_gold": sum(c["correct"] is None for c in choices),
        "accuracy": round(100 * sum(acc) / max(len(acc), 1), 1),
        "ci95": [round(100 * low, 1), round(100 * high, 1)],
    }
    if any("suite" in c for c in normal):
        suite = [float(c.get("suite", False)) for c in normal]
        summary["test_suite_accuracy"] = round(100 * sum(suite) / len(suite), 1)

    by_level = {}
    for c in normal:
        by_level.setdefault(c["difficulty"] or "all", []).append(c["correct"])
    if len(by_level) > 1:
        summary["by_difficulty"] = {
            k: round(100 * sum(v) / len(v), 1) for k, v in sorted(by_level.items())}

    if unans:
        refused = sum(c["correct"] for c in unans)
        wrongly = sum(c.get("error") == "refused" for c in normal)
        summary["refused_unanswerable"] = f"{refused}/{len(unans)}"
        summary["refused_answerable"] = f"{wrongly}/{len(normal)}"

    seconds = meta.get("gen_seconds", 0) + meta.get("retry_seconds", 0)
    summary["seconds_per_question"] = round(seconds / max(len(choices), 1), 3)
    return summary


if __name__ == "__main__":
    main()
