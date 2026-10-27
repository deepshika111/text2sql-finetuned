"""Add example questions to the prompt, and check when that hurts.

Modes:
  random    k random training examples
  similar   k training questions with the closest meaning
  skeleton  match on the shape of the question, and show the example
            SQL with table and column names replaced by placeholders

My guess for why examples can hurt: they come from other databases,
and the model copies their table and column names. copied_names()
counts how often that happens.
"""

import random
import re

import numpy as np
import sqlglot
from sqlglot import exp

from schema import all_columns

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def mask_question(question, db_path):
    """Replace table/column words, numbers and quoted text with _."""
    names = set()
    for table, cols in all_columns(db_path).items():
        names |= set(re.split(r"[_\s]+", table)) | {
            part for col in cols for part in re.split(r"[_\s]+", col)}
    names = {n for n in names if len(n) > 2}
    text = re.sub(r"\"[^\"]*\"|'[^']*'", "_", question)
    text = re.sub(r"\d+(\.\d+)?", "_", text)
    words = [("_" if w.lower().strip("?,.") in names else w)
             for w in text.split()]
    return " ".join(words)


def mask_sql(sql):
    """Replace names and values, e.g. SELECT name FROM singer becomes
    SELECT col1 FROM table1."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:
        return sql
    tables, cols = {}, {}
    # use list() because the tree changes while we loop
    for node in list(tree.find_all(exp.Table)):
        key = node.name.lower()
        tables.setdefault(key, f"table{len(tables) + 1}")
        node.set("this", exp.to_identifier(tables[key]))
    for node in list(tree.find_all(exp.Column)):
        key = node.name.lower()
        cols.setdefault(key, f"col{len(cols) + 1}")
        node.set("this", exp.to_identifier(cols[key]))
        if node.table and node.table.lower() in tables:
            node.set("table", exp.to_identifier(tables[node.table.lower()]))
    for node in list(tree.find_all(exp.Literal)):
        node.replace(exp.Literal.string("value") if node.is_string
                     else exp.Literal.number(1))
    return tree.sql(dialect="sqlite")


class Retriever:
    def __init__(self, train_rows, mode, k=3, seed=0):
        self.rows, self.mode, self.k = train_rows, mode, k
        self.rng = random.Random(seed)
        if mode in ("similar", "skeleton"):
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBED_MODEL)
            texts = [self.key(r) for r in train_rows]
            self.vecs = self.embed(texts)

    def key(self, row):
        if self.mode == "skeleton":
            return mask_question(row["question"], row["db_path"])
        return row["question"]

    def embed(self, texts):
        return self.model.encode(texts, batch_size=256,
                                 normalize_embeddings=True)

    def examples(self, row):
        if self.mode == "none":
            return []
        if self.mode == "random":
            picks = self.rng.sample(self.rows, self.k)
        else:
            query = self.embed([self.key(row)])[0]
            best = np.argsort(-(self.vecs @ query))[:self.k]
            picks = [self.rows[i] for i in best]
        if self.mode == "skeleton":
            picks = [{**p, "question": mask_question(p["question"],
                                                     p["db_path"]),
                      "gold": mask_sql(p["gold"])} for p in picks]
        return picks


def identifiers(sql):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:
        return set()
    names = {t.name.lower() for t in tree.find_all(exp.Table)}
    return names | {c.name.lower() for c in tree.find_all(exp.Column)}


def copied_names(pred_sql, example_sqls, db_path):
    """Names in the prediction that aren't in this database but are in
    one of the examples, so they were copied from another database."""
    real = set(all_columns(db_path))
    for cols in all_columns(db_path).values():
        real |= cols
    from_examples = set().union(*(identifiers(s) for s in example_sqls))
    return sorted((identifiers(pred_sql) - real) & from_examples)


def main():
    """Count how often each run copied names from the examples.

    python retrieval.py results/rag_similar/spider_dev.jsonl ...
    (run evaluate.py on each file first)
    """
    import argparse
    import json
    from pathlib import Path

    from data import load

    p = argparse.ArgumentParser()
    p.add_argument("preds", nargs="+", type=Path)
    args = p.parse_args()

    for path in args.preds:
        split = path.name.split(".")[0]
        rows = {r["id"]: r for r in load(split)}
        preds = {json.loads(l)["id"]: json.loads(l) for l in path.open()}
        choices = path.with_suffix(".first.jsonl")
        copied, wrong_when_copied, total = 0, 0, 0
        for line in choices.open():
            c = json.loads(line)
            if c["id"] not in rows or c["correct"] is None:
                continue
            total += 1
            names = copied_names(c["sql"], preds[c["id"]]["examples"],
                                 rows[c["id"]]["db_path"])
            if names:
                copied += 1
                wrong_when_copied += not c["correct"]
        print(f"{path.parent.name}: copied names in {copied}/{total} "
              f"answers ({100 * copied / max(total, 1):.1f}%), "
              f"{wrong_when_copied} of those wrong")


if __name__ == "__main__":
    main()
