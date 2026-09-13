"""Run resumable Gate A units with atomic per-cell checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from gate_a_core import (
    atomic_json,
    expected_units,
    load_config,
    read_unit,
    sha256_file,
    simulate_unit,
)


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append_event(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--inject-interrupt-after", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    run_root = args.run_root.resolve()
    config = load_config(config_path)
    config_sha256 = sha256_file(config_path)
    all_units = expected_units(config)
    if args.max_units is not None:
        all_units = all_units[: args.max_units]
    unit_dir = run_root / "raw_outputs" / "units"
    progress_dir = run_root / "progress"
    events_path = run_root / "logs" / "runner_events.jsonl"
    unit_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    started_at = timestamp()
    started_perf = time.perf_counter()
    completed_this_invocation = 0
    skipped = 0
    heartbeat_sequence = 0
    append_event(
        events_path,
        {
            "event": "STARTED",
            "at": started_at,
            "pid": os.getpid(),
            "host": platform.node(),
            "config_sha256": config_sha256,
            "units_expected": len(all_units),
            "resume": args.resume,
        },
    )
    try:
        for index, unit in enumerate(all_units, start=1):
            unit_path = unit_dir / f"{unit['unit_id']}.json"
            if unit_path.exists():
                if not args.resume:
                    raise RuntimeError(
                        f"existing unit requires --resume: {unit_path.name}"
                    )
                prior = read_unit(unit_path, config_sha256)
                if prior["unit"] != unit:
                    raise RuntimeError(f"unit identity mismatch: {unit_path.name}")
                skipped += 1
                continue
            unit_started = time.perf_counter()
            payload = simulate_unit(config, unit, config_sha256)
            atomic_json(unit_path, payload)
            completed_this_invocation += 1
            heartbeat_sequence += 1
            elapsed = time.perf_counter() - started_perf
            heartbeat = {
                "schema_version": 1,
                "sequence": heartbeat_sequence,
                "at": timestamp(),
                "pid": os.getpid(),
                "unit_id": unit["unit_id"],
                "unit_index": index,
                "units_expected": len(all_units),
                "completed_this_invocation": completed_this_invocation,
                "skipped_existing": skipped,
                "elapsed_seconds": elapsed,
                "last_unit_seconds": time.perf_counter() - unit_started,
            }
            atomic_json(progress_dir / "heartbeat.json", heartbeat)
            atomic_json(
                progress_dir / "progress.json",
                {
                    **heartbeat,
                    "completed_total": len(list(unit_dir.glob("*.json"))),
                    "status": "RUNNING",
                },
            )
            append_event(events_path, {"event": "UNIT_COMPLETED", **heartbeat})
            print(
                f"UNIT {index}/{len(all_units)} {unit['unit_id']} "
                f"elapsed={elapsed:.3f}s",
                flush=True,
            )
            if (
                args.inject_interrupt_after is not None
                and completed_this_invocation >= args.inject_interrupt_after
            ):
                terminal = {
                    "schema_version": 1,
                    "status": "INTERRUPTED_TEST",
                    "exit_code": 75,
                    "at": timestamp(),
                    "completed_this_invocation": completed_this_invocation,
                    "skipped_existing": skipped,
                    "units_expected": len(all_units),
                }
                atomic_json(run_root / "terminal_status.json", terminal)
                append_event(events_path, {"event": "INTERRUPTED_TEST", **terminal})
                print("Injected interruption after atomic checkpoint", flush=True)
                return 75
        completed_total = len(list(unit_dir.glob("*.json")))
        terminal = {
            "schema_version": 1,
            "status": "COMPLETED",
            "exit_code": 0,
            "at": timestamp(),
            "completed_this_invocation": completed_this_invocation,
            "skipped_existing": skipped,
            "completed_total": completed_total,
            "units_expected": len(all_units),
            "wall_seconds": time.perf_counter() - started_perf,
        }
        atomic_json(run_root / "terminal_status.json", terminal)
        atomic_json(
            progress_dir / "progress.json",
            {**terminal, "status": "COMPLETED"},
        )
        append_event(events_path, {"event": "COMPLETED", **terminal})
        print(json.dumps(terminal, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        terminal = {
            "schema_version": 1,
            "status": "FAILED",
            "exit_code": 1,
            "at": timestamp(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_this_invocation": completed_this_invocation,
            "skipped_existing": skipped,
            "units_expected": len(all_units),
        }
        atomic_json(run_root / "terminal_status.json", terminal)
        append_event(events_path, {"event": "FAILED", **terminal})
        print(json.dumps(terminal, sort_keys=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
