"""
verification/sample_pairs.py - Randomly sample 10 True and 10 False pairs.

Reads TruePairs.csv and FalsePairs.csv, samples 10 pairs from each,
and saves results to sampled_true_pairs.csv and sampled_false_pairs.csv.
"""

import random
import csv
from pathlib import Path

RANDOM_SEED = 42

SRC_DIR  = Path(__file__).parent.parent / "data"
TRUE_CSV  = Path(__file__).parent.parent / "data/TruePairs.csv"
FALSE_CSV = Path(__file__).parent.parent / "data/FalsePairs.csv"
OUT_DIR   = Path(__file__).parent.parent / "data"

N_SAMPLES = 10


def sample_csv(src_path: Path, out_path: Path, n: int):
    with open(src_path, newline="") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)

    sampled = random.sample(rows, n)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["videoA", "videoB"])
        writer.writeheader()
        writer.writerows(sampled)

    print(f"  {src_path.name} ({len(rows)} pairs) -> sampled {n} -> {out_path.name}")
    for row in sampled:
        print(f"    {row['videoA']}  vs  {row['videoB']}")


def run():
    random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {N_SAMPLES} pairs from each CSV (seed={RANDOM_SEED})\n")
    sample_csv(TRUE_CSV,  OUT_DIR / "sampled_true_pairs.csv",  N_SAMPLES)
    print()
    sample_csv(FALSE_CSV, OUT_DIR / "sampled_false_pairs.csv", N_SAMPLES)
    print("\nDone.")


if __name__ == "__main__":
    run()