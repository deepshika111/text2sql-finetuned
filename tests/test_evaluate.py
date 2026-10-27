import json
import sys

import pytest

import evaluate


def write_run(name, split, records, unanswerable=0):
    from pathlib import Path
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{split}.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    meta = {"split": split, "unanswerable": unanswerable,
            "gen_seconds": 10.0, "retry_seconds": 0.0}
    (out / f"{split}.meta.json").write_text(json.dumps(meta))
    return out / f"{split}.jsonl"


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", *map(str, argv)])
    evaluate.main()


def sql_cands(*sqls):
    return [{"sql": s, "logprob": -0.1, "tokens": 5} for s in sqls]


def test_vote_beats_first_sample(workdir, row, monkeypatch):
    rows = [{**row, "id": f"q{i}"} for i in range(3)]
    workdir("spider_dev", rows)
    good, bad = "SELECT count(*) FROM singer", "SELECT 1"
    records = [{"id": f"q{i}", "candidates": sql_cands(bad, good, good)}
               for i in range(3)]
    path = write_run("sft", "spider_dev", records)

    run(monkeypatch, path, "--method", "first")
    first = json.loads(path.with_suffix(".first.summary.json").read_text())
    run(monkeypatch, path, "--method", "vote")
    vote = json.loads(path.with_suffix(".vote.summary.json").read_text())
    run(monkeypatch, path, "--method", "oracle")
    oracle = json.loads(path.with_suffix(".oracle.summary.json").read_text())

    assert first["accuracy"] == 0.0
    assert vote["accuracy"] == 100.0 and oracle["accuracy"] == 100.0


def test_test_split_is_locked(workdir, row, monkeypatch):
    workdir("spider_test", [row])
    path = write_run("sft", "spider_test",
                     [{"id": "q1", "candidates": sql_cands("SELECT 1")}])
    with pytest.raises(SystemExit):
        run(monkeypatch, path)
    run(monkeypatch, path, "--final")
    log = open("results/test_log.txt").read()
    assert "spider_test.jsonl" in log


def test_refusals_are_scored(workdir, row, monkeypatch):
    row = {**row, "gold": "SELECT name FROM singer"}
    workdir("spider_dev", [row])
    records = [{"id": "q1", "candidates": sql_cands("SELECT name FROM singer")},
               {"id": "q1_unans", "candidates": sql_cands("CANNOT_ANSWER")}]
    path = write_run("sft", "spider_dev", records, unanswerable=1)
    run(monkeypatch, path)
    summary = json.loads(path.with_suffix(".first.summary.json").read_text())
    assert summary["accuracy"] == 100.0
    assert summary["refused_unanswerable"] == "1/1"
    assert summary["refused_answerable"] == "0/1"


def test_unsafe_sql_never_runs(workdir, row, monkeypatch):
    workdir("spider_dev", [row])
    path = write_run("sft", "spider_dev",
                     [{"id": "q1", "candidates": sql_cands("DROP TABLE singer")}])
    run(monkeypatch, path)
    choice = json.loads(path.with_suffix(".first.jsonl").read_text())
    assert choice["error"] == "unsafe" and choice["correct"] is False


def test_empty_answer_is_wrong(workdir, row, monkeypatch):
    workdir("spider_dev", [row])
    path = write_run("sft", "spider_dev",
                     [{"id": "q1", "candidates": sql_cands("")}])
    run(monkeypatch, path)
    choice = json.loads(path.with_suffix(".first.jsonl").read_text())
    assert choice["error"] == "no answer" and choice["correct"] is False
