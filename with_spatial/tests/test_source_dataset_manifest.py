from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/source_station_dataset"
MANIFEST = ROOT / "data/source_station_dataset_manifest.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    table = pd.read_csv(MANIFEST)
    assert len(table) == 56, len(table)
    for row in table.itertuples(index=False):
        path = DATA_ROOT / row.relative_path
        assert path.is_file(), path
        assert path.stat().st_size == int(row.size_bytes), path
        assert sha256(path) == row.sha256, path
    print("PASS: source station dataset manifest")


if __name__ == "__main__":
    main()
