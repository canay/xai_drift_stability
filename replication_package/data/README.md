# Data acquisition manifest

Raw datasets are not redistributed in this repository. The experiment code
downloads the public records and caches them locally.

| Dataset | Provider record | Use in this package |
|---|---:|---|
| Adult | OpenML data ID 1590 | controlled tabular-shift grid |
| Bank Marketing | OpenML data ID 1461 | controlled tabular-shift grid |
| Electricity | OpenML data ID 151 | controlled grid and chronological check |
| Bike Sharing Dataset | UCI repository data ID 275 | external chronological check |

`code/common.py` downloads the OpenML records through
`sklearn.datasets.fetch_openml`. `code/run_second_temporal_stream.py` downloads
the UCI Bike Sharing archive. Dataset use and redistribution remain subject to
the terms published by the original providers.

