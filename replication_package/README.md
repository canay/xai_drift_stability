# Replication package

## Scope

This package reproduces the controlled tabular-shift grid, paired-LIME
validation, and two chronological checks used in the study. The public payload
contains experiment evidence and analysis code only; it is not a manuscript
source archive.

## Requirements

- Python 3.12 (the corrected runs used Python 3.12.12)
- CPU execution; no GPU is required
- approximately 2 GB free disk space for downloaded data and run caches
- a persistent server or workstation session for full runs

Install the exact corrected-run environment:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify the archived evidence

The fast validation reads the saved metrics, checks the declared row coverage,
and verifies the package checksum manifest:

```bash
python scripts/verify_saved_results.py
```

Expected final line:

```text
PASS: saved main-grid and paired-estimator evidence is complete and internally consistent
```

Recompute the two statistical summaries from the archived raw outputs:

```bash
python code/analyze_main_grid_v2.py --run-dir outputs/main_grid
python code/analyze_crn_revalidation_v3.py --run-dir outputs/crn_validation
```

These commands use dataset-stratified seed-cluster resampling and may update the
derived CSV/JSON files inside the selected output directories.

## Scientific semantics tests

The test suite checks deterministic seed construction, model-independent shift
realizations, raw mixed-feature LIME neighborhoods, paired stream identity, and
the zero-vector cosine policy. It downloads the three OpenML datasets on the
first run unless `FH2_DATA_DIR` points to an existing cache.

```bash
set FH2_TEST_NEIGHBORHOODS=10000        # Windows cmd
set FH2_DATA_DIR=%CD%\data
python tests/test_scientific_semantics_v2.py
```

PowerShell:

```powershell
$env:FH2_TEST_NEIGHBORHOODS = '10000'
$env:FH2_DATA_DIR = (Join-Path $PWD 'data')
python tests/test_scientific_semantics_v2.py
```

## From-scratch experiment runs

Controlled grid (one dataset per process):

```bash
python code/run_experiment.py adult
python code/run_experiment.py bank-marketing
python code/run_experiment.py electricity
python code/analyze.py
```

Paired-estimator smoke run:

```bash
python code/run_crn_revalidation.py --config configs/crn_remote_smoke.json --partition adult:0
```

The locked low-, mid-, and high-budget configurations are in `configs/`.
Run dataset/seed partitions separately for resumability. The complete paired
grid is computationally expensive and should be executed in a persistent
server session.

Chronological checks:

```bash
python code/run_temporal.py
python code/run_second_temporal_stream.py --base-dir . --out-dir runs/bike_temporal
```

## Data

See [`data/README.md`](data/README.md). OpenML records are fetched by
`code/common.py`; the Bike Sharing archive is fetched by
`code/run_second_temporal_stream.py`. Cached raw data are ignored by Git.

## Output map

- `outputs/main_grid/`: 3,510-row corrected controlled grid and clustered summaries
- `outputs/crn_validation/`: 3,375 low-tier rows plus 225-row mid/high tiers
- `outputs/temporal/`: Electricity and Bike chronological outputs
- `outputs/figures/`: empirical plots generated from the archived evidence
- `logs/`: completion transcripts for the three corrected main-grid partitions
- `provenance/`: run registry snapshot, public manifest, and file checksums

## Determinism and interpretation

Seeds 0-4 control splitting and model fitting. Shift seeds are keyed by dataset,
seed, scenario, and severity; paired LIME streams are additionally keyed by the
reference instance and draw. The three explanation channels are distinct
estimands: local LIME and SHAP comparisons must not be treated as directly
rank-equivalent to global grouped permutation importance.

