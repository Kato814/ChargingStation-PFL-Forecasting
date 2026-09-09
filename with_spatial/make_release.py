from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED = {"MANIFEST.csv", "SHA256SUMS.txt", "PACKAGE_SUMMARY.json"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def payload_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.name in EXCLUDED:
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def main() -> None:
    files = payload_files()
    rows = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        rows.append((rel, path.stat().st_size, sha256(path)))

    manifest = ROOT / "MANIFEST.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "size_bytes", "sha256"])
        writer.writerows(rows)

    sums = ROOT / "SHA256SUMS.txt"
    sums.write_text("".join(f"{digest}  {rel}\n" for rel, _, digest in rows), encoding="utf-8")

    verification_path = ROOT / "VERIFICATION.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else {}
    source_sha = None
    source_path = ROOT / "SOURCE_ARCHIVE.txt"
    if source_path.exists():
        for line in source_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("SHA256:") or line.startswith("SHA-256:"):
                source_sha = line.split(":", 1)[1].strip()
                break
    version_text = (ROOT / "VERSION.txt").read_text(encoding="utf-8")
    version = next((line.split(":", 1)[1].strip() for line in version_text.splitlines() if line.startswith("Version:")), "unknown")
    summary = {
        "package": "ChargingStation_PFL_Paper_Code_WITH_SPATIAL",
        "version": version,
        "variant": "WITH_SPATIAL_539D",
        "file_count_excluding_manifest_files": len(rows),
        "payload_bytes_excluding_manifest_files": int(sum(size for _, size, _ in rows)),
        "verification_status": verification.get("status", "UNKNOWN"),
        "source_archive_sha256": source_sha,
    }
    (ROOT / "PACKAGE_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
