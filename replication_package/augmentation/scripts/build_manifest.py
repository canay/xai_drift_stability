"""Build the public scientific-augmentation SHA-256 manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "augmentation_manifest.json"
EXCLUDED = {
    "augmentation_manifest.json",
    "gate_a/validation/gate_a_independent_validation.json",
    "gate_b/decision/gate_b_full_validation.json",
    "gate_c/decision/gate_c_full_validation.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXCLUDED or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        files.append({"path": rel, "bytes": path.stat().st_size,
                      "sha256": digest(path)})
    payload = {
        "schema": "f02-public-augmentation-manifest-v1",
        "operation_id": "f02-scientific-augmentation-20260826",
        "excluded_mutable_validator_outputs": sorted(EXCLUDED - {"augmentation_manifest.json"}),
        "files": files,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT} with {len(files)} immutable files")


if __name__ == "__main__":
    main()
