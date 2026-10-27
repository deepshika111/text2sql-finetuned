"""Block anything that isn't a plain SELECT, and make questions the
database can't answer (to check the model says so instead of guessing).
"""

import random

import sqlglot
from sqlglot import exp

from schema import all_columns

READ_ONLY = (exp.Select, exp.Union, exp.Intersect, exp.Except)
BANNED_WORDS = ("attach", "detach", "pragma", "vacuum", "load_extension")


def is_safe(sql):
    """True only for a single read-only query."""
    lowered = sql.lower()
    if any(word in lowered for word in BANNED_WORDS):
        return False
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:
        return False
    if len(statements) != 1 or statements[0] is None:
        return False
    tree = statements[0]
    if not isinstance(tree, READ_ONLY):
        return False
    # also catch writes hidden inside, like a WITH ... DELETE
    writes = (exp.Insert, exp.Update, exp.Delete, exp.Drop,
              exp.Create, exp.Alter)
    return not any(tree.find_all(*writes))


def gold_columns(row):
    """(table, column) pairs used by the correct query."""
    try:
        tree = sqlglot.parse_one(row["gold"], read="sqlite")
    except sqlglot.errors.SqlglotError:
        return []
    schema = all_columns(row["db_path"])
    used_tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    pairs = []
    # put SELECT columns first, hiding one of those is the clearest case
    selects = tree.find(exp.Select)
    first = list(selects.expressions) if selects else []
    for node in first + [tree]:
        for col in node.find_all(exp.Column):
            name = col.name.lower()
            for table in used_tables:
                if name in schema.get(table, set()):
                    pairs.append((table, name))
    return list(dict.fromkeys(pairs))  # remove duplicates, keep order


def make_unanswerable(rows, n, seed=0):
    """Copy n questions and hide a column each one needs.

    Without that column the question can't be answered, so the right
    reply is CANNOT_ANSWER. The column is hidden from every table that
    has it, so the model can't get it another way.
    """
    rng = random.Random(seed)
    pool = rows[:]
    rng.shuffle(pool)
    out = []
    for row in pool:
        pairs = [p for p in gold_columns(row) if p[1] != "id"]
        if not pairs:
            continue
        name = pairs[0][1]
        schema = all_columns(row["db_path"])
        drop = [(t, name) for t, cols in schema.items() if name in cols]
        out.append({**row, "id": row["id"] + "_unans", "drop": drop,
                    "gold": None, "unanswerable": True})
        if len(out) == n:
            break
    return out
