"""Sort wrong answers into types and see what fine-tuning fixed.

  python errors.py results/base/spider_dev.first.jsonl \\
                   results/sft_s1/spider_dev.first.jsonl

Types come from comparing the parts of the predicted and correct SQL
(tables, columns, filters, ...), so the same answer always gets the
same label.
"""

import argparse
import json
from collections import Counter

import sqlglot
from sqlglot import exp

from data import load

ORDER = ["refused", "unsafe", "syntax error", "made-up table",
         "made-up column", "other runtime error", "missing table",
         "extra table", "wrong columns", "wrong filter",
         "wrong aggregation", "wrong grouping", "wrong order/limit",
         "wrong values"]


def parse(sql):
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:
        return None


def names(tree, kind):
    return {n.name.lower() for n in tree.find_all(kind)}


def part(tree, kind):
    node = tree.find(kind)
    return node.sql().lower() if node else ""


def select_cols(tree):
    sel = tree.find(exp.Select)
    if not sel:
        return set()
    return {c.name.lower() for e in sel.expressions
            for c in e.find_all(exp.Column)}


def aggs(tree):
    return sorted(type(n).__name__ for n in tree.find_all(exp.AggFunc))


def where_parts(tree):
    where = tree.find(exp.Where)
    if not where:
        return set(), set()
    cols = {c.name.lower() for c in where.find_all(exp.Column)}
    ops = {type(n).__name__ for n in where.find_all(exp.Binary)}
    return cols, ops


def classify(choice, gold):
    """Give a wrong answer one label: the first rule that matches."""
    err = (choice.get("error") or "").lower()
    if err in ("refused", "unsafe"):
        return err
    if "syntax" in err or "incomplete input" in err:
        return "syntax error"
    if "no such table" in err:
        return "made-up table"
    if "no such column" in err:
        return "made-up column"
    if err:
        return "other runtime error"

    pred_t, gold_t = parse(choice["sql"]), parse(gold)
    if pred_t is None or gold_t is None:
        return "syntax error" if pred_t is None else "wrong values"

    p_tabs, g_tabs = names(pred_t, exp.Table), names(gold_t, exp.Table)
    if g_tabs - p_tabs:
        return "missing table"
    if p_tabs - g_tabs:
        return "extra table"
    if select_cols(pred_t) != select_cols(gold_t):
        return "wrong columns"
    if where_parts(pred_t) != where_parts(gold_t):
        return "wrong filter"
    if aggs(pred_t) != aggs(gold_t):
        return "wrong aggregation"
    if part(pred_t, exp.Group) != part(gold_t, exp.Group):
        return "wrong grouping"
    if (part(pred_t, exp.Order) != part(gold_t, exp.Order)
            or part(pred_t, exp.Limit) != part(gold_t, exp.Limit)):
        return "wrong order/limit"
    return "wrong values"  # same structure, but a different value


def label_run(path, gold):
    labels = {}
    for line in open(path):
        c = json.loads(line)
        if c.get("unanswerable") or c["correct"] is not False:
            continue
        labels[c["id"]] = classify(c, gold[c["id"]])
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("before")
    p.add_argument("after")
    args = p.parse_args()

    split = args.before.split("/")[-1].split(".")[0]
    gold = {r["id"]: r["gold"] for r in load(split)}
    before, after = label_run(args.before, gold), label_run(args.after, gold)

    cb, ca = Counter(before.values()), Counter(after.values())
    print(f"{'mistake':22s} {'before':>7s} {'after':>7s} {'change':>7s}")
    for kind in ORDER:
        if cb[kind] or ca[kind]:
            print(f"{kind:22s} {cb[kind]:7d} {ca[kind]:7d} "
                  f"{ca[kind] - cb[kind]:+7d}")
    print(f"{'total':22s} {len(before):7d} {len(after):7d} "
          f"{len(after) - len(before):+7d}")

    fixed = Counter(before[q] for q in before if q not in after)
    broke = Counter(after[q] for q in after if q not in before)
    print("\nfixed by the second run:", dict(fixed.most_common()))
    print("newly broken:", dict(broke.most_common()))


if __name__ == "__main__":
    main()
