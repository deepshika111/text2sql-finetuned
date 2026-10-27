import json
import sqlite3

import pytest


@pytest.fixture
def concert_db(tmp_path):
    path = tmp_path / "concert.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE singer (
            singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER,
            country TEXT);
        CREATE TABLE concert (
            concert_id INTEGER PRIMARY KEY, singer_id INTEGER, year INTEGER,
            FOREIGN KEY (singer_id) REFERENCES singer(singer_id));
        INSERT INTO singer VALUES
            (1, 'Ana', 31, 'France'), (2, 'Bo', 25, 'Japan'),
            (3, 'Cy', 40, 'France');
        INSERT INTO concert VALUES (1, 1, 2014), (2, 3, 2015), (3, 1, 2015);
    """)
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture
def row(concert_db):
    return {"id": "q1", "source": "spider", "db_id": "concert",
            "db_path": concert_db, "question": "How many singers?",
            "evidence": "", "gold": "SELECT count(*) FROM singer",
            "difficulty": ""}


@pytest.fixture
def workdir(tmp_path, monkeypatch, row):
    """A fake project folder with data/ and one generated run."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    def write_split(name, rows):
        with open(f"data/{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    return write_split
