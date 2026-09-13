# Gate B Commands

Date/time: 2026-08-26 16:43 +03:00  
Tool: Codex  
Model, if known: GPT-5  
Operation ID: `f02-scientific-augmentation-20260826`

```powershell
python -m pytest tests/test_gate_b.py -q
python src/run_gate_b.py --config config/gate_b_smoke.json
python src/run_gate_b.py --config config/gate_b_resume_smoke.json --stop-after-units 1
python src/run_gate_b.py --config config/gate_b_resume_smoke.json
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
python src/run_gate_b.py --config config/gate_b_full.json
python src/analyze_gate_b.py --config config/gate_b_full.json
python tests/validate_gate_b_outputs.py
```

İlk resume-smoke süreç kaydı kontrollü olarak `exit_code: 75` yazdı; Windows
PTY dış dönüşü 1 olarak sundu. İkinci çağrı exit 0 ile tamamlandı ve ilk birim
SHA-256 `853882B439530A410CC96F511A9F8927717595AB72AC0F9D0CE51C68E63A07B6`
olarak byte-identical kaldı.
