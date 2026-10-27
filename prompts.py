"""Build the prompt for a question and pull the SQL out of the reply."""

import re

from schema import schema_text

REFUSE = "CANNOT_ANSWER"

SYSTEM = (
    "You write SQLite queries. Given a database schema and a question, "
    "reply with one SQL query that answers it, and nothing else. "
    f"If the schema has no data that can answer the question, reply "
    f"exactly {REFUSE}."
)


def user_text(row, schema, examples=()):
    parts = [f"Schema:\n{schema}"]
    for ex in examples:
        parts.append(f"Example question: {ex['question']}\n"
                     f"Example SQL: {ex['gold']}")
    if row.get("evidence"):
        parts.append(f"Hint: {row['evidence']}")
    parts.append(f"Question: {row['question']}")
    return "\n\n".join(parts)


def build_messages(row, examples=(), schema=None):
    """Chat messages for one question, with optional example questions."""
    if schema is None:
        schema = schema_text(row["db_path"], drop=row.get("drop"))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text(row, schema, examples)},
    ]


def count_tokens(tok, messages, add_generation_prompt=True):
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return len(tok.encode(text, add_special_tokens=False))


def fit_messages(row, examples, count, budget):
    """Make the prompt shorter until it fits in `budget` tokens.

    Some BIRD tables have over 100 columns, so the prompt can get too
    long. First drop the example rows, then the example questions.
    Returns (messages, how). messages is None if it still doesn't fit.
    """
    drop = row.get("drop")
    tries = [
        (examples, True, "full"),
        (examples, False, "no sample rows"),
        ((), False, "no sample rows or examples"),
    ]
    for exs, sample_rows, how in tries:
        if how.endswith("examples") and not examples:
            continue
        schema = schema_text(row["db_path"], drop=drop,
                             sample_rows=sample_rows)
        messages = build_messages(row, exs, schema)
        if count(messages) <= budget:
            return messages, how
    return None, "too long"


def retry_messages(messages, bad_sql, error):
    """Show the model its failed query and the error, and ask again."""
    return messages + [
        {"role": "assistant", "content": bad_sql},
        {"role": "user", "content":
            f"That query failed with: {error}\n"
            "Reply with a corrected SQL query only."},
    ]


def extract_sql(text):
    """Get the SQL out of the model's reply."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if REFUSE in text:
        return REFUSE
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1)
    text = text.strip()
    # keep the first statement only
    return text.split(";")[0].strip()
