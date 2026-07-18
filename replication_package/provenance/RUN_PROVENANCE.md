# Public run provenance

The archived evidence in this package is derived from the following canonical
experiment records. Host labels are descriptive; no credentials or private
machine paths are included.

| Run ID | Role | Execution profile | Status |
|---|---|---|---|
| `2026-07-17_codex_canayxps15_full_remediation_r01` | corrected 3,510-row main grid and temporal reruns | Windows 11, CPU-only, Python 3.12.12 | validated |
| `2026-07-17_codex_vps_crn_reference_revalidation_r01` | low/mid/high paired-estimator validation | Linux aarch64 VPS, CPU-only | 675/675 partitions completed; validation passed |

Corrected environment versions:

- NumPy 2.4.6
- pandas 2.2.3
- SciPy 1.16.3
- scikit-learn 1.8.0
- SHAP 0.49.1
- LIME 0.2.0.1
- Matplotlib 3.10.7

The main grid uses seeds 0-4. The paired-estimator low tier contains three
independently keyed paired draws per condition; its mid/high reference tiers
contain one draw per matched condition. The validation JSON records source,
configuration, dataset, shift, reference-index, and stream-seed hashes.

