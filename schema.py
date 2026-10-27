"""Describe a SQLite database as text the model can read."""

import sqlite3
from functools import lru_cache

SAMPLE_ROWS = 3
MAX_CHARS = 40


@lru_cache(maxsize=1000)
def read_schema(db_path):
    """Read tables, columns, foreign keys and a few rows from a database."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode(errors="replace")
    tables = {}
    names = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    for (table,) in names:
        q = f'"{table}"'
        cols = [(c[1], c[2], c[5] > 0)
                for c in conn.execute(f"PRAGMA table_info({q})")]
        fks = [(f[3], f[2], f[4])  # column, other table, other column
               for f in conn.execute(f"PRAGMA foreign_key_list({q})")]
        try:
            rows = conn.execute(
                f"SELECT * FROM {q} LIMIT {SAMPLE_ROWS}").fetchall()
        except sqlite3.Error:
            rows = []
        tables[table] = {"columns": cols, "fks": fks, "rows": rows}
    conn.close()
    return tables


def short(value):
    text = str(value)
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "..."


def quote(name):
    return f"`{name}`" if not name.isidentifier() else name


def primary_key(tables, name):
    for table, info in tables.items():
        if table.lower() == name.lower():
            keys = [c[0] for c in info["columns"] if c[2]]
            return keys[0] if len(keys) == 1 else None
    return None


def schema_text(db_path, drop=None, sample_rows=True):
    """CREATE TABLE statements plus a few example rows per table.

    drop is a list of (table, column) to leave out (see safety.py).
    """
    hidden = {(t.lower(), c.lower()) for t, c in (drop or [])}
    tables = read_schema(db_path)
    parts = []
    for table, info in tables.items():
        keep = [i for i, (name, _, _) in enumerate(info["columns"])
                if (table.lower(), name.lower()) not in hidden]

        lines = []
        for i in keep:
            name, typ, pk = info["columns"][i]
            line = f"  {quote(name)} {typ or 'TEXT'}"
            lines.append(line + (" PRIMARY KEY" if pk else ""))
        for col, other, other_col in info["fks"]:
            if (table.lower(), col.lower()) in hidden:
                continue
            # no column given means the other table's primary key
            other_col = other_col or primary_key(tables, other)
            target = quote(other) + (f"({quote(other_col)})" if other_col
                                     else "")
            lines.append(f"  FOREIGN KEY ({quote(col)}) REFERENCES {target}")

        text = f"CREATE TABLE {quote(table)} (\n" + ",\n".join(lines) + "\n);"
        if sample_rows and info["rows"]:
            header = [info["columns"][i][0] for i in keep]
            rows = [[short(r[i]) for i in keep] for r in info["rows"]]
            body = "\n".join(" | ".join(r) for r in rows)
            text += f"\n/* example rows:\n{' | '.join(header)}\n{body}\n*/"
        parts.append(text)
    return "\n\n".join(parts)


def all_columns(db_path):
    """All table and column names, lowercased."""
    return {t.lower(): {c[0].lower() for c in info["columns"]}
            for t, info in read_schema(db_path).items()}
