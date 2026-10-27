import sqlite3

import pytest

from execution import accuracy, exec_match, is_ordered, run_sql


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE singer (id INTEGER, name TEXT, age INTEGER);
        INSERT INTO singer VALUES
            (1, 'Ana', 31), (2, 'Bo', 25), (3, 'Cy', 40);
    """)
    conn.commit()
    conn.close()
    return str(path)


def test_same_query_written_differently_is_correct(db):
    gold = "SELECT count(*) FROM singer WHERE age > 30"
    pred = "select COUNT(id) from singer where not age <= 30"
    assert exec_match(db, pred, gold)["correct"] is True


def test_wrong_answer_is_wrong(db):
    gold = "SELECT count(*) FROM singer WHERE age > 30"
    pred = "SELECT count(*) FROM singer"
    assert exec_match(db, pred, gold)["correct"] is False


def test_row_order_ignored_without_order_by(db):
    gold = "SELECT name FROM singer"
    pred = "SELECT name FROM singer ORDER BY name DESC"
    assert exec_match(db, pred, gold)["correct"] is True


def test_row_order_matters_with_order_by(db):
    gold = "SELECT name FROM singer ORDER BY age"
    pred = "SELECT name FROM singer ORDER BY age DESC"
    assert exec_match(db, pred, gold)["correct"] is False


def test_duplicates_count_unless_bird(db):
    gold = "SELECT age > 30 FROM singer"
    pred = "SELECT DISTINCT age > 30 FROM singer"
    assert exec_match(db, pred, gold)["correct"] is False
    assert exec_match(db, pred, gold, bird=True)["correct"] is True


def test_int_and_float_match(db):
    gold = "SELECT avg(age) FROM singer WHERE id = 1"
    pred = "SELECT age FROM singer WHERE id = 1"
    assert exec_match(db, pred, gold)["correct"] is True


def test_broken_sql_is_wrong_with_error(db):
    result = exec_match(db, "SELECT nope FROM singer", "SELECT 1")
    assert result["correct"] is False
    assert "no such column" in result["error"]


def test_bad_gold_is_skipped(db):
    result = exec_match(db, "SELECT 1", "SELECT * FROM missing")
    assert result["correct"] is None
    assert accuracy([result, {"correct": True}]) == 1.0


def test_sql_cannot_modify_db(db):
    rows, err = run_sql(db, "DELETE FROM singer")
    assert err is not None
    rows, _ = run_sql(db, "SELECT count(*) FROM singer")
    assert rows == [(3,)]


def test_slow_query_times_out(db):
    slow = """
        WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n)
        SELECT count(*) FROM n
    """
    rows, err = run_sql(db, slow, timeout=0.5)
    assert rows is None and err == "timeout"


def test_order_by_inside_subquery_does_not_count():
    sql = "SELECT * FROM (SELECT a FROM t ORDER BY a) LIMIT 5"
    assert not is_ordered(sql)
    assert is_ordered("SELECT a FROM t ORDER BY a")
