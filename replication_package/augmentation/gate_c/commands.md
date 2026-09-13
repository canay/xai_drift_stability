# Gate C Commands

Date/time: 2026-08-26 18:10 +03:00  
Tool: Codex  
Model, if known: GPT-5  
Operation ID: `f02-scientific-augmentation-20260826`

Working directory: this run folder.

```powershell
python -m pytest tests/test_gate_c.py -q
python src/run_gate_c.py --config config/gate_c_full.json --splits calibration
python src/freeze_thresholds.py --config config/gate_c_full.json
python src/run_gate_c.py --config config/gate_c_full.json --splits validation
python src/validate_pretest.py --config config/gate_c_full.json
python src/run_gate_c.py --config config/gate_c_full.json --splits test
python src/analyze_gate_c.py --config config/gate_c_full.json
python tests/validate_gate_c_outputs.py
```

Calibration thresholds were frozen before validation/test access; the
validation pretest was frozen before test access. The test command opened only
the ten test units and completed with exit code 0.
