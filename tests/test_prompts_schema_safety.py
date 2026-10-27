from prompts import (REFUSE, build_messages, extract_sql, fit_messages,
                     retry_messages)
from safety import is_safe, make_unanswerable
from schema import schema_text


def test_schema_has_tables_keys_and_sample_rows(concert_db):
    text = schema_text(concert_db)
    assert "CREATE TABLE singer" in text
    assert "singer_id INTEGER PRIMARY KEY" in text
    assert "REFERENCES singer(singer_id)" in text
    assert "Ana | 31 | France" in text


def test_schema_can_hide_a_column(concert_db):
    text = schema_text(concert_db, drop=[("singer", "country")])
    assert "country" not in text
    assert "Ana | 31" in text and "France" not in text


def test_prompt_includes_hint_and_question(row):
    row = {**row, "evidence": "adult means age >= 18"}
    user = build_messages(row)[1]["content"]
    assert "Hint: adult means age >= 18" in user
    assert user.endswith("Question: How many singers?")


def test_retry_message_shows_error(row):
    msgs = retry_messages(build_messages(row), "SELECT x", "no such column")
    assert msgs[-2] == {"role": "assistant", "content": "SELECT x"}
    assert "no such column" in msgs[-1]["content"]


def test_extract_sql_from_common_reply_shapes():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("<think>hmm</think>SELECT a FROM t") == "SELECT a FROM t"
    assert extract_sql("SELECT 1; DROP TABLE t") == "SELECT 1"
    assert extract_sql(f"Sorry. {REFUSE}") == REFUSE


def test_safety_allows_reads_only():
    assert is_safe("SELECT name FROM singer WHERE age > 30")
    assert is_safe("WITH a AS (SELECT 1) SELECT * FROM a")
    assert is_safe("SELECT 1 UNION SELECT 2")
    for bad in ["DELETE FROM singer", "DROP TABLE singer",
                "UPDATE singer SET age = 1", "PRAGMA table_info(singer)",
                "ATTACH 'x.db' AS x", "SELECT 1; DELETE FROM singer",
                "not sql at all ((("]:
        assert not is_safe(bad), bad


def test_unanswerable_hides_a_needed_column(row):
    row = {**row, "gold": "SELECT name FROM singer WHERE age > 30"}
    [unans] = make_unanswerable([row], 1)
    assert unans["unanswerable"] and unans["gold"] is None
    assert ("singer", "name") in unans["drop"]
    assert "  name TEXT" not in build_messages(unans)[1]["content"]


def test_fit_messages_shrinks_big_prompts(row):
    def chars(messages):
        return sum(len(m["content"]) for m in messages)

    full, how = fit_messages(row, [], chars, budget=10_000)
    assert how == "full" and "example rows" in full[1]["content"]

    small, how = fit_messages(row, [], chars, budget=chars(full) - 1)
    assert how == "no sample rows"
    assert "example rows" not in small[1]["content"]

    none, how = fit_messages(row, [], chars, budget=10)
    assert none is None and how == "too long"


def test_fit_messages_drops_examples_last(row):
    examples = [{"question": "q " * 200, "gold": "SELECT 1"}]

    def chars(messages):
        return sum(len(m["content"]) for m in messages)

    bare, _ = fit_messages(row, [], chars, budget=10_000)
    msgs, how = fit_messages(row, examples, chars, budget=chars(bare))
    assert how == "no sample rows or examples"
    assert "Example question" not in msgs[1]["content"]


def test_foreign_key_without_column_uses_primary_key(tmp_path):
    import sqlite3
    path = tmp_path / "league.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE team (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE player (id INTEGER, team_id INTEGER REFERENCES team);
        CREATE TABLE coach (team_id INTEGER REFERENCES missing);
    """)
    conn.close()
    text = schema_text(str(path))
    assert "FOREIGN KEY (team_id) REFERENCES team(id)" in text
    assert "FOREIGN KEY (team_id) REFERENCES missing" in text


def test_unparseable_sql_is_unsafe_not_a_crash():
    assert not is_safe("SELECT a FROM t WHERE b = 'it\\'s'")
