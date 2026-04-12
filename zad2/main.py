import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path("videos-K-O")


def load_npz(path: Path) -> dict:
    """Load a single .npz file and return its arrays."""
    npz = np.load(path, allow_pickle=True)
    data = {key: npz[key] for key in npz.files}
    return data


def inspect(path: Path, data: dict):
    """Print a summary of what's inside an .npz file."""
    print(f"\n{'=' * 60}")
    print(f"File: {path.name}")
    print(f"{'=' * 60}")
    for key, arr in data.items():
        print(f"  {key:20s}  shape={arr.shape}  dtype={arr.dtype}")


def main():
    if not DATA_DIR.exists():
        print(f"[ERROR] Data directory '{DATA_DIR}' not found.")
        sys.exit(1)

    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{DATA_DIR}'.")
        sys.exit(1)

    print(f"Found {len(npz_files)} file(s) in '{DATA_DIR}'.\n")

    for path in npz_files:
        data = load_npz(path)
        inspect(path, data)


if __name__ == "__main__":
    main()