# Commands

Date/time: 2026-08-26 15:44 +03:00  
Tool: Codex  
Model, if known: GPT-5  
Operation ID: `f02-scientific-augmentation-20260826`

Working directory: this run folder.

```powershell
python tests/test_gate_a.py
python src/run_gate_a.py --config config/gate_a_full.json --run-root .
python src/analyze_gate_a.py --config config/gate_a_full.json --run-root .
```

The test suite injects an exit-75 interruption after two atomic units, resumes,
and compares every resumed unit byte-for-byte with a clean run.
