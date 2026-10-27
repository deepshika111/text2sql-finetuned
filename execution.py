"""Check a predicted query by running it.

A prediction counts as correct if it returns the same rows as the
correct query. Row order only matters when the correct query has an
ORDER BY.
"""

import hashlib
import re
import sqlite3
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

TIMEOUT_S = 30
MAX_ROWS = 100_000


def run_sql(db_path, sql, timeout=TIMEOUT_S):
    """Run a query read-only. Returns (rows, error)."""
    uri = f"file:{db_path}?mode=ro"
    deadline = time.monotonic() + timeout
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as e:
        return None, f"connect: {e}"

    # stop the query once it runs past the time limit
    conn.set_progress_handler(
        lambda: time.monotonic() > deadline, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(MAX_ROWS + 1)
        if len(rows) > MAX_ROWS:
            return None, "too many rows"
        return rows, None
    except sqlite3.OperationalError as e:
        if time.monotonic() > deadline:
            return None, "timeout"
        return None, str(e)
    except (sqlite3.Error, ValueError) as e:
        return None, str(e)
    finally:
        conn.close()


def clean(value):
    # treat 2.0 and 2 as equal, and ignore tiny float differences
    if isinstance(value, float):
        value = round(value, 6)
        return int(value) if value.is_integer() else value
    return value


def is_ordered(sql):
    # True if there is an ORDER BY outside any subquery
    depth = 0
    for token in re.findall(r"\(|\)|\border\s+by\b", sql, re.I):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth == 0:
            return True
    return False


def same_rows(pred, gold, ordered=False, as_set=False):
    pred = [tuple(clean(v) for v in r) for r in pred]
    gold = [tuple(clean(v) for v in r) for r in gold]
    if as_set:  # how BIRD scores
        return set(pred) == set(gold)
    if ordered:
        return pred == gold
    return Counter(pred) == Counter(gold)


@lru_cache(maxsize=50_000)
def run_gold(db_path, sql):
    # cache so the correct query only runs once per question
    return run_sql(db_path, sql)


def compare(db_path, gold_sql, pred, pred_err, bird=False):
    """Compare prediction rows (already run) with the correct query."""
    gold, gold_err = run_gold(db_path, gold_sql)
    if gold_err:
        return {"correct": None, "error": f"gold failed: {gold_err}"}
    if pred_err:
        return {"correct": False, "error": pred_err}
    ordered = is_ordered(gold_sql)
    ok = same_rows(pred, gold, ordered=ordered, as_set=bird)
    return {"correct": ok, "error": None}


def exec_match(db_path, pred_sql, gold_sql, bird=False):
    """Run a predicted query and compare it with the correct one."""
    pred, pred_err = run_sql(db_path, pred_sql)
    return compare(db_path, gold_sql, pred, pred_err, bird)


def signature(rows):
    """Short id for a result, ignoring row order. Used for voting."""
    if rows is None:
        return None
    cleaned = sorted(repr(tuple(clean(v) for v in r)) for r in rows)
    return hashlib.md5("\n".join(cleaned).encode()).hexdigest()[:12]


def score(row, pred_sql, pred=None, pred_err=None, suite=True):
    """Score a prediction for one question.

    Pass pred/pred_err if the query already ran. For Spider test
    questions it also adds "suite": right on every test-suite database.
    """
    bird = row["source"] == "bird"
    if pred is None and pred_err is None:
        pred, pred_err = run_sql(row["db_path"], pred_sql)
    result = compare(row["db_path"], row["gold"], pred, pred_err, bird)

    suite = suite and row.get("suite_dir")
    if not suite:
        return result
    result["suite"] = result["correct"]
    if result["correct"]:
        for db in sorted(Path(suite).glob("*.sqlite")):
            r = exec_match(str(db), pred_sql, row["gold"])
            if r["correct"] is False:
                result["suite"] = False
                break
    return result


def accuracy(results):
    """Share correct, skipping questions whose correct query failed."""
    scored = [r["correct"] for r in results if r["correct"] is not None]
    if not scored:
        return 0.0
    return sum(scored) / len(scored)
