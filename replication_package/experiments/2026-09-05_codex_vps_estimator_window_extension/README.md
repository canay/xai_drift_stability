# Strong-estimator coupling and window-variability experiments

This extension contains the E1 and E2 evidence for *Explanation Drift under
Distribution Shift: Coupling Strong Shapley Estimators and Separating Window
Variability*. It retains all 720 completed experimental units, including null
conditions, adverse coupling effects, and rank diagnostics.

E1 compares independent and coupled random streams for IID KernelSHAP,
complementary KernelSHAP, fixed-budget LeverageSHAP, and a restricted cubic
PolySHAP implementation. E2 separates Monte Carlo variability from the variation
between sampled windows for signed mean-Shapley contrasts. These experiments
evaluate estimator precision; they do not establish detector superiority.

## Verify the saved run

Python 3.12 or later is sufficient for the integrity check; no packages or data
downloads are required. Run from this directory:

```text
python verify_saved.py
```

The expected result is `PASS`, with 720 saved units and 2,162 snapshot files.
This check verifies every release-file hash and every decompressed archive
member. It does not rerun an experiment or certify statistical conclusions.

## Recompute the saved-run analysis

The recorded runtime was Python 3.12.3 on Linux ARM64. Use a separate virtual
environment and install the versions in `requirements.txt`. Numerical libraries
and platform differences may affect results from a new experimental run.

```text
python -m pip install -r requirements.txt
python verify_saved.py --extract ../../e1_e2_saved_work
python src/analyze.py --work ../../e1_e2_saved_work --config config/full.json --output ../../e1_e2_recomputed
```

Both output directories must be new. Extraction verifies the saved arrays and
query records before and after writing. The analysis reads the extracted
snapshot and recreates the E1 cells, E2 windows and variance components, query
and rank receipts, and primary decision. Its formulas and resampling procedures
are unchanged from the recorded run. The portable version checks the released
configuration and original run fingerprint without requiring downloaded data.
This entry point is for the saved snapshot; it does not mix freshly computed
units into the archived run.

The corresponding saved results are in `results/analysis_v1/`. The query/rank
receipt CSV is gzip-compressed without changing its decompressed bytes.
`results/descriptive/` contains the final descriptive summaries. The two pooled
ratios in `primary_decision.json` round to 0.507 for complementary KernelSHAP
and 0.462 for LeverageSHAP. The E2 summary reports arithmetic means of five
split-level components, rather than a pooled population variance.

## Experimental source and inputs

The frozen configuration is `config/full.json`: Adult and Electricity, five
split seeds, two model families, the specified model-query budgets, and the
recorded replicate counts. The original `core.py` and `worker.py` are preserved
byte for byte. `run.py` retains the supervisor and experiment logic; its external
notification callback is disabled in this release. The source's historical gate
fields preserve the failed held-out detection gate and the decision against
scaling that detector experiment.

Source datasets are not redistributed. The parent replication package documents
the OpenML records (Adult 1590 and Electricity 151) and provides
`code/fetch_data.py`, which caches the inputs under its `data/` directory. A new
run can be launched from this directory after preparing those caches:

```text
python src/run.py --config config/full.json --work ../../e1_e2_new_run
```

A new run uses a new fingerprint because the portable supervisor differs from
the original notification-enabled source. It must use a fresh work directory;
it cannot resume the saved archival snapshot. Complete runs can take hours.
The published saved-run analysis command above is the direct reproduction path
for the article's numerical summaries.

## Provenance and licenses

`provenance.json` records the original scientific fingerprint, original and
portable code hashes, and the limited portability changes. Machine identifiers
are replaced consistently by `replication-host`; execution logs, local paths,
and private planning files are excluded. Numerical arrays and query records
retain their original bytes. `raw_metadata/SNAPSHOT_MANIFEST.json` binds the
portable snapshot, and `SHA256SUMS` binds the release files.

The pinned upstream sources are
[LeverageSHAP](https://github.com/rtealwitter/leverageshap), commit
`766107218179475915a2656e277ba7e0bd03db77`, and
[PolySHAP](https://github.com/FFmgll/PolySHAP), commit
`690a0eef1244b656981b963761d27d78a2a4fd16`. Their license files remain with the
vendored code. Project code uses the parent repository's MIT license; dataset
terms remain those of the original providers.

Eleven LaTeX table-rule string literals in the vendored LeverageSHAP reporting
utilities use equivalent adjacent raw/newline literals to distinguish their
source spelling from network paths. Their values and the complete parsed Python
syntax trees are unchanged; the original and portable file hashes are recorded
in `provenance.json`.
