"""Mechanics, identity, atomic interruption, and deterministic resume tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parents[1]
SRC = RUN_ROOT / "src"
sys.path.insert(0, str(SRC))

from gate_a_core import construct_means, simulate_unit  # noqa: E402


class GateATests(unittest.TestCase):
    def smoke_config(self) -> dict:
        return {
            "schema_version": "f02-gate-a-config-v1",
            "base_seed": 20260826,
            "replications": 300,
            "dimensions": [8],
            "budgets": [16],
            "rhos": [-0.25, 0.0, 0.25],
            "signals": ["null"],
            "margins": {"near_tie": 0.05},
            "top_k": 3,
            "criteria": {},
        }

    def test_constructed_topk_shift(self) -> None:
        mean0, null = construct_means(8, 3, 0.05, "null")
        _, weak = construct_means(8, 3, 0.05, "weak")
        self.assertEqual(mean0.tolist(), null.tolist())
        self.assertGreater(weak[3], weak[2])

    def test_covariance_identity_direction(self) -> None:
        config = self.smoke_config()
        for rho, expected_sign in [(-0.75, -1), (0.75, 1)]:
            unit = {
                "unit_id": f"identity-{rho}",
                "dimension": 8,
                "budget": 32,
                "rho": rho,
                "signal": "moderate",
                "margin_name": "separated",
                "margin": 0.50,
            }
            config["replications"] = 4000
            result = simulate_unit(config, unit, "TEST")
            difference = result["empirical"][
                "independent_minus_paired_vector_mse"
            ]
            self.assertEqual(1 if difference > 0 else -1, expected_sign)
            theoretical = result["theory"][
                "independent_minus_paired_vector_mse"
            ]
            self.assertLess(abs(difference - theoretical) / abs(theoretical), 0.12)

    def test_interruption_resume_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(self.smoke_config(), indent=2), encoding="utf-8"
            )
            interrupted = root / "interrupted"
            clean = root / "clean"
            command = [
                sys.executable,
                str(SRC / "run_gate_a.py"),
                "--config",
                str(config_path),
                "--run-root",
                str(interrupted),
                "--inject-interrupt-after",
                "2",
            ]
            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 75, first.stderr)
            self.assertEqual(len(list((interrupted / "raw_outputs/units").glob("*.json"))), 2)
            resume = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "run_gate_a.py"),
                    "--config",
                    str(config_path),
                    "--run-root",
                    str(interrupted),
                    "--resume",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(resume.returncode, 0, resume.stderr)
            clean_run = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "run_gate_a.py"),
                    "--config",
                    str(config_path),
                    "--run-root",
                    str(clean),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(clean_run.returncode, 0, clean_run.stderr)
            interrupted_units = sorted((interrupted / "raw_outputs/units").glob("*.json"))
            clean_units = sorted((clean / "raw_outputs/units").glob("*.json"))
            self.assertEqual([path.name for path in interrupted_units], [path.name for path in clean_units])
            for left, right in zip(interrupted_units, clean_units):
                self.assertEqual(left.read_bytes(), right.read_bytes())
            heartbeat = json.loads((interrupted / "progress/heartbeat.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(heartbeat["sequence"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
