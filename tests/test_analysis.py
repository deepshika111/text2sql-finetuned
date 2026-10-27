import random

from errors import classify
from picker import choose_picker, choose_vote, cross_validate, features, fit
from retrieval import copied_names, mask_question, mask_sql
from stats import bootstrap_ci, paired_diff


def cand(sig, correct, logprob=-0.2, error=None, rows=1):
    return {"sql": "SELECT 1", "sig": sig, "correct": correct,
            "logprob": logprob, "error": error, "rows": rows, "tokens": 10}


def test_vote_picks_the_most_common_result():
    cands = [cand("a", False), cand("b", True), cand("b", True),
             cand(None, False, error="no such column")]
    assert choose_vote(cands) in (1, 2)


def test_vote_breaks_ties_by_confidence():
    cands = [cand("a", False, logprob=-1.0), cand("b", True, logprob=-0.1)]
    assert choose_vote(cands) == 1


def test_vote_when_everything_errors():
    cands = [cand(None, False, error="x"), cand(None, False, error="y")]
    assert choose_vote(cands) == 0


def test_features_one_row_per_candidate():
    cands = [cand("a", True), cand("a", True), cand("b", False)]
    x = features(cands)
    assert x.shape == (3, 9)
    assert x[0][0] == 2 / 3 and x[2][1] == 0.0  # vote share, top vote


def fake_records(n, seed=0):
    """Right answer is the confident one, but often in the minority,
    so a picker that learns logprob can beat plain voting."""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        right = cand(f"r{i}", True, logprob=-0.1)
        wrong = [cand(f"w{i}", False, logprob=-1.5) for _ in range(3)]
        cands = [right] + wrong if rng.random() < 0.5 else wrong + [right]
        records.append({"id": str(i), "candidates": cands})
    return records


def test_picker_learns_what_voting_misses():
    records = fake_records(200)
    weights = fit(records)
    picked = [r["candidates"][choose_picker(r["candidates"], weights)]
              for r in records]
    voted = [r["candidates"][choose_vote(r["candidates"])] for r in records]
    assert sum(c["correct"] for c in picked) > sum(c["correct"] for c in voted)
    assert cross_validate(records) > 90


def test_bootstrap_interval_contains_the_mean():
    values = [1] * 70 + [0] * 30
    low, high = bootstrap_ci(values)
    assert low < 0.7 < high and high - low < 0.25


def test_paired_gain_is_detected():
    base = [0] * 50 + [1] * 50
    new = [0] * 30 + [1] * 70  # right on 20 more questions
    d, low, high, p = paired_diff(base, new)
    assert round(d, 2) == 0.2
    assert d > 0 and low > 0 and p < 0.01


def test_error_types():
    gold = "SELECT name FROM singer WHERE age > 30"
    def label(sql, error=None):
        return classify({"sql": sql, "error": error}, gold)
    assert label("SELECT x", "no such column: x") == "made-up column"
    assert label("SELECT x FROM t", "no such table: t") == "made-up table"
    assert label("SELECT name FROM concert WHERE age > 30") == "missing table"
    assert label("SELECT age FROM singer WHERE age > 30") == "wrong columns"
    assert label("SELECT name FROM singer WHERE age < 30") == "wrong filter"
    assert label("SELECT name FROM singer WHERE age > 40") == "wrong values"
    assert label("SELECT name FROM singer WHERE age > 30 LIMIT 1") \
        == "wrong order/limit"


def test_mask_sql_hides_names_and_values():
    masked = mask_sql("SELECT name FROM singer WHERE age > 30 "
                      "AND country = 'France'")
    assert "singer" not in masked and "France" not in masked
    assert "table1" in masked and "col1" in masked


def test_mask_question_hides_db_words(concert_db):
    masked = mask_question("How many singers from France are over 30?",
                           concert_db)
    assert "30" not in masked and masked.startswith("How many")


def test_copied_names_finds_names_from_other_databases(concert_db):
    examples = ["SELECT count(*) FROM employee WHERE salary > 10"]
    pred = "SELECT count(*) FROM singer WHERE salary > 10"
    assert copied_names(pred, examples, concert_db) == ["salary"]
    assert copied_names("SELECT name FROM singer", examples,
                        concert_db) == []
