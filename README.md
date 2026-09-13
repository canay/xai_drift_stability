# Explanation Drift under Distribution Shift: Coupling Strong Shapley Estimators and Separating Window Variability

This repository contains the replication package for the study by Özkan Canay,
Department of Information Systems and Technologies, Sakarya University.

The study examines how stochastic explainer error and window sampling affect
comparisons of feature attributions under distribution shift. It couples the
random streams of complementary KernelSHAP and fixed-budget LeverageSHAP while
preserving each estimator's marginal sampler and model-query budget. A separate
nested experiment measures the contribution of window sampling to signed
mean-Shapley contrasts.

The package retains null conditions, adverse coupling effects, rank diagnostics,
and the failed held-out detector gate. These results support a bounded claim
about estimator precision; they do not establish a universal detector advantage.

## Repository contents

The [replication package](replication_package/) contains experiment and analysis
code, configurations, seed policies, saved outputs, environment records, and
verification scripts. Its evidence has three complementary parts:

- The base package covers the controlled grid, paired-LIME validation, and
  chronological checks.
- The [augmentation](replication_package/augmentation/) preserves the synthetic
  estimator identity, cross-explainer comparison, and held-out detection tests.
- The [E1/E2 extension](replication_package/experiments/2026-09-05_codex_vps_estimator_window_extension/)
  contains all 720 completed units for strong-estimator coupling and window
  variability, with saved-run analysis and integrity verification.

Source datasets are not redistributed. The data manifests identify the public
OpenML and UCI records and the scripts that obtain them.

## Verify saved evidence

The E1/E2 integrity check needs only Python 3.12 or later:

```text
cd replication_package/experiments/2026-09-05_codex_vps_estimator_window_extension
python verify_saved.py
```

It checks all release hashes and all archived members without training a model
or rerunning the analysis. See the extension's README for safe extraction and
recomputation of its saved results.

For the base package, install its separate requirements in a virtual
environment and run `python scripts/verify_saved_results.py` from
`replication_package/`. Then run `python scripts/verify_augmentation.py` from
`replication_package/augmentation/`. Each component documents its own runtime;
do not combine their requirements into one environment by assumption.

Full experimental runs can take hours. Use the documented saved-result checks
first, and use a persistent workstation or server session for new runs.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite the software. Article publication metadata
will be added when available.

Project code and documentation use the [MIT License](LICENSE). Vendored
upstream code retains its accompanying license files. Dataset terms remain
those of the original providers.
