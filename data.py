"""Download Spider and BIRD and split them into train, dev and test.

Files written to data/ (one question per line):
  spider_train  for training
  spider_dev    for trying things out (databases not used in training)
  bird_dev      for trying things out (half of BIRD Mini-Dev)
  spider_test   final check only (official Spider dev set)
  bird_test     final check only (other half of BIRD Mini-Dev)

Dev and test databases never show up in training, so every score is
on databases the model hasn't seen.

Usage: python data.py
"""

import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

DATA = Path("data")
SEED = 0
DEV_DB_SHARE = 0.15

SPIDER_DBS = "prem-research/spider"
BIRD_ZIP_ID = "13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG"
SUITE_ZIP_ID = "1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w"


def save(rows, name):
    path = DATA / f"{name}.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    dbs = len({r["db_id"] for r in rows})
    print(f"{name}: {len(rows)} questions, {dbs} databases")


def load(name):
    with open(DATA / f"{name}.jsonl") as f:
        return [json.loads(line) for line in f]


def fetch_zip(file_id, name):
    zip_path = DATA / f"{name}.zip"
    if not zip_path.exists():
        import gdown
        gdown.download(id=file_id, output=str(zip_path))
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA / name)
    return DATA / name


def check_dbs(rows):
    missing = {r["db_id"] for r in rows if not Path(r["db_path"]).exists()}
    if missing:
        raise FileNotFoundError(f"missing databases: {sorted(missing)[:5]}")


def make_row(i, name, source, db, db_path, ex, gold, evidence=""):
    return {
        "id": f"{name}_{i}",
        "source": source,
        "db_id": db,
        "db_path": str(db_path),
        "question": ex["question"],
        "evidence": evidence,
        "gold": gold,
        "difficulty": ex.get("difficulty", ""),
    }


def spider():
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        SPIDER_DBS,
        repo_type="dataset",
        allow_patterns="database/*/*.sqlite",
        local_dir=DATA / "spider",
    )
    db_dir = Path(root) / "database"
    ds = load_dataset("xlangai/spider")

    def rows_for(split, name):
        rows = []
        for i, ex in enumerate(ds[split]):
            db = ex["db_id"]
            path = db_dir / db / f"{db}.sqlite"
            rows.append(make_row(i, name, "spider", db, path,
                                 ex, ex["query"]))
        check_dbs(rows)
        return rows

    train = rows_for("train", "spider")

    # keep some whole databases out of training for dev
    dbs = sorted({r["db_id"] for r in train})
    random.Random(SEED).shuffle(dbs)
    dev_dbs = set(dbs[:int(len(dbs) * DEV_DB_SHARE)])

    save([r for r in train if r["db_id"] not in dev_dbs], "spider_train")
    save([r for r in train if r["db_id"] in dev_dbs], "spider_dev")

    test = rows_for("validation", "spider_test")
    add_test_suite(test)
    save(test, "spider_test")


def add_test_suite(rows):
    """Add the test-suite databases: copies of each test database with
    different data. A query that is only right by luck usually fails
    on one of them.
    """
    try:
        suite = fetch_zip(SUITE_ZIP_ID, "testsuite")
    except Exception as e:  # not required, normal scoring still works
        print(f"test suite skipped: {e}")
        return
    folders = {p.parent.name: p.parent for p in suite.rglob("*.sqlite")}
    for r in rows:
        if r["db_id"] in folders:
            r["suite_dir"] = str(folders[r["db_id"]])
    found = sum("suite_dir" in r for r in rows)
    print(f"test suite: {found}/{len(rows)} questions covered")


def bird():
    bird_dir = fetch_zip(BIRD_ZIP_ID, "bird")
    db_paths = {p.stem: p for p in bird_dir.rglob("*.sqlite")}
    questions = next(bird_dir.rglob("mini_dev_sqlite.json"))
    ds = json.loads(questions.read_text())

    rows = []
    for ex in ds:
        db = ex["db_id"]
        path = db_paths.get(db, f"missing/{db}")
        rows.append(make_row(ex["question_id"], "bird", "bird", db, path,
                             ex, ex["SQL"], ex.get("evidence") or ""))
    check_dbs(rows)

    # split each difficulty level in half, so both halves match
    groups = defaultdict(list)
    for r in rows:
        groups[r["difficulty"]].append(r)
    dev, test = [], []
    rng = random.Random(SEED)
    for key in sorted(groups):
        group = groups[key]
        rng.shuffle(group)
        half = len(group) // 2
        dev += group[:half]
        test += group[half:]

    save(dev, "bird_dev")
    save(test, "bird_test")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    spider()
    bird()
