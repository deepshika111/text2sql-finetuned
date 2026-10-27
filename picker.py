"""Choose the best SQL out of several tries.

Two ways:
  vote     go with the result most tries agree on
  picker   a small logistic regression trained on dev answers

The picker is checked with 5-fold cross-validation, so it is never
scored on a question it was trained on.

  python picker.py results/sft_s1/spider_dev.scored.jsonl \\
                   results/sft_s1/bird_dev.scored.jsonl \\
                   --out results/sft_s1/picker.json
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from prompts import REFUSE

FEATURES = ["vote_share", "is_top_vote", "error", "empty", "refuse",
            "logprob", "log_tokens", "log_rows", "retried"]


def votes(cands):
    return Counter(c.get("sig") for c in cands if c.get("sig"))


def features(cands):
    """Features for each answer, using only things known without the
    correct answer: votes, errors, model confidence, result size."""
    counts = votes(cands)
    top = max(counts.values()) if counts else 0
    rows = []
    for c in cands:
        n = counts.get(c.get("sig"), 0)
        error = bool(c.get("error")) and c.get("error") != "refused"
        rows.append([
            n / len(cands),
            float(n == top and n > 0),
            float(error),
            float(c.get("rows") == 0),
            float(c["sql"] == REFUSE),
            c.get("logprob", 0.0),
            math.log1p(c.get("tokens", 0)),
            math.log1p(c.get("rows") or 0),
            float(c.get("retried", False)),
        ])
    return np.array(rows)


def choose_vote(cands):
    """Most common result. On a tie, the more confident answer wins."""
    counts = votes(cands)
    if not counts:
        return 0  # every answer failed
    def rank(i):
        c = cands[i]
        return (counts.get(c.get("sig"), 0), c.get("logprob", 0.0))
    return max(range(len(cands)), key=rank)


def load_picker(path):
    return json.loads(Path(path).read_text())


def choose_picker(cands, weights):
    x = (features(cands) - weights["mean"]) / weights["std"]
    scores = x @ np.array(weights["coef"]) + weights["intercept"]
    return int(np.argmax(scores))


def fit(records, c=1.0):
    from sklearn.linear_model import LogisticRegression

    x, y = [], []
    for rec in records:
        cands = rec["candidates"]
        if any(cd["correct"] is None for cd in cands):
            continue
        x.append(features(cands))
        y += [float(cd["correct"]) for cd in cands]
    x = np.vstack(x)
    mean, std = x.mean(0), x.std(0) + 1e-6
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit((x - mean) / std, y)
    return {"features": FEATURES, "mean": mean.tolist(),
            "std": std.tolist(), "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0])}


def accuracy(records, choose):
    hits = [rec["candidates"][choose(rec["candidates"])]["correct"]
            for rec in records]
    hits = [h for h in hits if h is not None]
    return 100 * sum(hits) / max(len(hits), 1)


def cross_validate(records, folds=5, seed=0):
    """Picker accuracy, always on questions it wasn't trained on."""
    order = np.random.default_rng(seed).permutation(len(records))
    hits = 0
    for k in range(folds):
        held = set(order[k::folds].tolist())
        test = [records[i] for i in sorted(held)]
        train = [r for i, r in enumerate(records) if i not in held]
        weights = fit(train)
        hits += accuracy(test, lambda c: choose_picker(c, weights)) * len(test)
    return hits / len(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scored", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    records = []
    for path in args.scored:
        if "_test" in path.name:
            raise SystemExit("never train the picker on a test split")
        records += [json.loads(line) for line in path.open()]

    oracle = accuracy(records, lambda c: max(
        range(len(c)), key=lambda i: bool(c[i]["correct"])))
    print(f"questions: {len(records)}")
    print(f"first sample: {accuracy(records, lambda c: 0):.1f}")
    print(f"vote:         {accuracy(records, choose_vote):.1f}")
    print(f"picker (cv):  {cross_validate(records):.1f}")
    print(f"oracle:       {oracle:.1f}")

    weights = fit(records)
    args.out.write_text(json.dumps(weights, indent=2))
    for name, w in sorted(zip(FEATURES, weights["coef"]),
                          key=lambda t: -abs(t[1])):
        print(f"  {name:12s} {w:+.2f}")


if __name__ == "__main__":
    main()
