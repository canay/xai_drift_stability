# Gate C Status — KILL/PIVOT

Date/time: 2026-08-26 18:10 +03:00  
Tool: Codex  
Model, if known: GPT-5  
Operation ID: `f02-scientific-augmentation-20260826`

- Run ID: `2026-08-26_codex_local_gate_c_heldout_detection`
- Host: `Windows workstation B (pseudonymized)` (local Windows)
- Execution status: `COMPLETE_VERIFIED`
- Scientific decision: `KILL_PIVOT`
- Atomic coverage: 20/20 units; 8,000 channel rows
- Once-opened test coverage: 4,000 rows from seeds 5--9
- Calibration/validation sealing: PASS; test rows read before opening: 0
- Independent output validation: 40/40 PASS
- C1 held-out FAR superiority: FAIL; paired 0.100 versus independent 0.090,
  difference 0.010, 95% CI [0.000, 0.025]
- C2 power non-inferiority: PASS; difference 0.0067 [0.0017, 0.0117]
- C3 AUROC non-inferiority: PASS; difference -0.0017 [-0.0073, 0.0032]
- C4 AUPRC non-inferiority: PASS; difference -0.0012 [-0.0030, 0.0005]
- C5--C7 transfer, comparator coverage, semantics, and provenance: PASS
- Gate D authorized: no
- Manuscript use: the negative detector boundary is reported prospectively;
  no detector-superiority claim is made
