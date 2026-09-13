# Scientific augmentation: estimator theory, strong baselines, and held-out boundary

This directory contains the version-specific evidence added for the article
*Common-Random-Number Estimation of Explanation Drift: Theory,
Cross-Explainer Validation, and a Held-Out Detection Boundary*.

The augmentation is organized as three prospectively ordered gates:

- `gate_a/`: 240 Gaussian ground-truth cells with 8000 replications per cell.
  It checks the covariance-conditional vector-MSE identity, the unbiased
  split-replicate squared-drift estimator, and the sufficient top-k condition.
- `gate_b/`: matched-budget LIME and sampling KernelSHAP experiments, plus
  GLIME-Binomial and official S-LIME comparisons. The archived full result has
  10 atomic units and 7440 method rows.
- `gate_c/`: separated calibration, validation, and once-opened test channels
  for paired/independent explanation scores, Explanation Shift, and input- and
  output-space comparators. The archived full result has 20 atomic units and
  8000 channel rows.

Gate A and Gate B pass their locked rules. Gate C returns `KILL_PIVOT` because
paired false-alarm superiority fails; this negative decision is part of the
paper's claim boundary. No Gate D scale-up was run.

## Fast verification

From `replication_package/augmentation`:

```bash
python scripts/verify_augmentation.py
```

The script checks the frozen file manifest, independently recomputes all three
gate decisions from the archived atomic/processed artifacts, and requires the
expected `PASS`, `PASS`, `KILL_PIVOT` sequence. Expected final output:

```text
PASS: augmentation manifest and Gate A/B/C decisions are internally consistent
```

## Environment

The archived augmentation used Python 3.12.7 on CPU-only Windows 11. Install the
version-specific environment with:

```bash
python -m pip install -r requirements.txt
```

Gate B pins official S-LIME semantics through source hashes recorded in
`gate_b/environment/slime_provenance.json`. Gate C records the official
`skshift` semantics boundary in
`gate_c/environment/skshift_provenance.json`. The portable copies of Gate B and
Gate C import the corrected raw-feature experiment substrate from
`../code/common.py`; no source dataset rows are redistributed.

## Rebuilding the estimator-stress figure

```bash
python figures/fig_7_estimand_stress_test.py
```

The script reads only the locked Gate A--C outputs in this directory and writes
PDF, SVG, and PNG renderings beside itself.

## Provenance and interpretation

The `criteria_locked.json`, resolved/full configurations, atomic unit files,
processed CSVs, decision JSONs, environment records, and independent validators
are retained together. Structural no-shift false-attribution rates in Gate B
are not deployment false-alarm rates. Gate C is the only held-out
detector-utility test, and its failed false-alarm criterion cannot be relabeled
as a detector success.

Date/time: 2026-08-26 18:49 +03:00  
Tool: Codex  
Model, if known: GPT-5  
Operation ID: `f02-scientific-augmentation-20260826`
