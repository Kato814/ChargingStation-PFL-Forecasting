from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_LITERAL = "-0.007" + "060"


def main() -> None:
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".json", ".txt", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if FORBIDDEN_LITERAL in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"forbidden retired daily metric literal found in: {offenders}"
    print("PASS: retired daily metric is absent from paper code and documentation")


if __name__ == "__main__":
    main()
