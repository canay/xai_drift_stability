"""Verify the public scientific-augmentation package from saved evidence."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_validator(relative: str) -> None:
    path = ROOT / relative
    result = subprocess.run([sys.executable, str(path)], cwd=path.parent,
                            text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    require(result.returncode == 0, f"validator failed: {relative}")


def main() -> None:
    manifest = json.loads((ROOT / "augmentation_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == "f02-public-augmentation-manifest-v1",
            "unexpected manifest schema")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        require(path.is_file(), f"missing manifest file: {item['path']}")
        require(path.stat().st_size == int(item["bytes"]),
                f"size mismatch: {item['path']}")
        require(digest(path) == item["sha256"],
                f"hash mismatch: {item['path']}")

    run_validator("gate_a/tests/validate_gate_a_outputs.py")
    run_validator("gate_b/tests/validate_gate_b_outputs.py")
    run_validator("gate_c/tests/validate_gate_c_outputs.py")

    gate_a = json.loads((ROOT / "gate_a/decision/gate_a_decision.json").read_text(encoding="utf-8"))
    gate_b = json.loads((ROOT / "gate_b/decision/gate_b_full_decision.json").read_text(encoding="utf-8"))
    gate_c = json.loads((ROOT / "gate_c/decision/gate_c_full_decision.json").read_text(encoding="utf-8"))
    require(gate_a.get("status") == "PASS", "Gate A is not PASS")
    require(gate_b.get("decision") == "PASS", "Gate B is not PASS")
    require(gate_c.get("decision") == "KILL_PIVOT", "Gate C is not KILL_PIVOT")
    c1 = next(item for item in gate_c["criteria"]
              if item["criterion_id"] == "C1_HELDOUT_FAR_REDUCTION")
    require(c1.get("pass") is False, "Gate C false-alarm criterion was not preserved")
    require(gate_c.get("gate_d_authorized") is False, "Gate D must remain unauthorized")
    print("PASS: augmentation manifest and Gate A/B/C decisions are internally consistent")


if __name__ == "__main__":
    main()
