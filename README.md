# Paired-Sampling Explanation-Drift Estimation Under Tabular Data Shift

This repository contains the public replication package for the study by
Özkan Canay, Department of Information Systems and Technologies, Sakarya
University.

The study evaluates explanation drift under controlled tabular data shift and
introduces an instance-keyed common-random-number estimator for paired LIME
comparisons. In the matched-budget validation, the conventional independent
estimator has a 0.203 clean-clean top-5 instability floor and 0.022 mean drift
separation. Pairing gives a zero matched-stream structural floor and 0.142 mean
drift separation. The low-budget paired estimate has a mean absolute error of
0.0337 against the finite 150-instance/2000-sample reference.

## Repository contents

The [`replication_package`](replication_package/) directory contains:

- experiment and analysis code;
- locked configurations and seed policies;
- canonical controlled-grid, paired-estimator, and temporal outputs;
- scientific semantics tests and saved-result validation;
- environment and run-provenance records.

The source datasets are not redistributed. The runners fetch the public OpenML
and UCI records identified in the data manifest and cache them locally.

## Quick verification

```bash
cd replication_package
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/verify_saved_results.py
```

See [`replication_package/README.md`](replication_package/README.md) for the
analysis and from-scratch execution commands. Full experiment runs take hours;
use a persistent workstation or server session.

## Citation

Please cite the associated article using the metadata in
[`CITATION.cff`](CITATION.cff). The citation record will be updated when final
publication metadata is available.

## License

The code and repository documentation are released under the
[MIT License](LICENSE). Dataset terms remain those of their original providers.

