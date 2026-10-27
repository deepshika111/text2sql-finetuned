"""Is the difference between two runs real, or just luck?

  # compare two runs on the same questions
  python stats.py results/base/spider_dev.first.jsonl \\
      --vs results/sft_s1/spider_dev.first.jsonl

  # compare against all three seeds at once
  python stats.py results/base/spider_dev.first.jsonl \\
      --vs results/sft_s*/spider_dev.first.jsonl

With several seeds, each question's score is averaged over the seeds.
"""

import argparse
import json

import numpy as np

N_BOOT = 10_000


def bootstrap_ci(values, n=N_BOOT, seed=0):
    """95% confidence interval for the mean (bootstrap over questions)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (n, len(values)))
    means = values[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff(a, b, n=N_BOOT, seed=0):
    """Average gain of b over a, its 95% interval, and a p-value."""
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    means = diff[idx].mean(1)
    low, high = np.percentile(means, [2.5, 97.5])
    p = float((means <= 0).mean())  # how often there's no gain
    return float(diff.mean()), float(low), float(high), p


def read_scores(paths, key="correct"):
    """Score per question id, averaged over the given runs."""
    per_q = {}
    for path in paths:
        for line in open(path):
            c = json.loads(line)
            if c.get("unanswerable") or c.get("correct") is None:
                continue
            per_q.setdefault(c["id"], []).append(float(c.get(key, False)))
    return {q: sum(v) / len(v) for q, v in per_q.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base", nargs="+")
    p.add_argument("--vs", nargs="+", required=True)
    p.add_argument("--key", default="correct",
                   help="'suite' for test-suite accuracy")
    args = p.parse_args()

    a, b = read_scores(args.base, args.key), read_scores(args.vs, args.key)
    shared = sorted(a.keys() & b.keys())
    if len(shared) < len(a):
        print(f"note: comparing the {len(shared)} shared questions")
    av, bv = [a[q] for q in shared], [b[q] for q in shared]

    for name, vals in [("base", av), ("new ", bv)]:
        low, high = bootstrap_ci(vals)
        print(f"{name}: {100 * np.mean(vals):.1f}  "
              f"[{100 * low:.1f}, {100 * high:.1f}]")
    d, low, high, p_val = paired_diff(av, bv)
    print(f"gain: {100 * d:+.1f} points, 95% CI [{100 * low:+.1f}, "
          f"{100 * high:+.1f}], p = {p_val:.4f}")

    if len(args.vs) > 1:
        print("per seed:")
        for path in args.vs:
            s = read_scores([path], args.key)
            vals = [s[q] for q in shared if q in s]
            print(f"  {path}: {100 * np.mean(vals):.1f}")


if __name__ == "__main__":
    main()
