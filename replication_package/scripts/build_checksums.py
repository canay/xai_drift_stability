"""Regenerate the deterministic SHA-256 manifest for public package files."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "provenance" / "SHA256SUMS"


def main() -> int:
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and "__pycache__" not in path.parts
        and ".venv" not in path.parts
    )
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(ROOT).as_posix()
        lines.append(f"{digest}  {relative}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(lines)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

