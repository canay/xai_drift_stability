# Secondary audit provenance

These files were copied byte-for-byte from the project's frozen `q1-audit/`
artifacts on 2026-08-12. No experiment was rerun during this package update.
`scripts/verify_saved_results.py` validates row coverage, the manuscript's 27
rounded baseline BalAcc/AUROC/AP values, the corrected LIME row contract, and
the package-wide checksum manifest.

Primary source hashes before the copy:

- `baseline_extended_by_seed.csv`: `A5C495D3D4BF358512D5B4048184DD9725FF900A4566321FE686E87AECBE2212`
- `baseline_extended_summary.csv`: `EED79D1D097EFA820EAC3E2C3BCED0D60C86B87BE75578091762CFDA0E1934C1`
- `standard_drift_monitor_cells.csv`: `89E74529383A3C7320D71753DA3F44918D0E292205AAD845B6E5C13A328B31EE`
- `explanation_vs_standard_monitor_leads.csv`: `3C523734B1309C26FAFC54DEE455C6B4BEFDC46D9685ECCFFB06BC5C5A65C28F`
- `explanation_vs_standard_monitor_summary.csv`: `9FFFA47CBFC221607F4E323EFEC5F3C22DC8AAF8CD0E6BE8CE81CE486766DD11`
- `lime_budget_raw_corrected.csv`: `28A74242FF5C3F310E4A0CDF321C9A67C1A4D61C6C881CE5A95642916302BF87`
- `lime_budget_corrected_summary.csv`: `BAB3B98E11B05C95870BD2D392351C02B1CB7DCFBD7809F3D6A346F1B5BA7655`
- `lime_budget_paired_bootstrap.csv`: `5130A07CCEB2A3262CC4FD77DC62AFCF6E6E8FBADF017852021646DC48D4AB96`
- `lime_clean_clean_floor_summary.csv`: `F24C3907C84EB06465F095ABA5BFEFE859097FA2ED6195FEAB0BDCCE4F48D4D8`
- `lime_budget_correction_summary.json`: `9EFC7F42488FAD0EAE1E83D23F2A5321529B8D13CD85D5025F8DFBCC03F2B076`

The controlled-grid multi-lens values remain sourced from
`outputs/main_grid/processed_outputs/main_grid_v2.csv`; this note does not create
a second copy of that evidence.

The documented corrected-versus-legacy analysis command also requires
`results/raw_all.csv`; the package copy is byte-identical to the project archive
(`BFAAE8E6552DD3471A3573CA88A158D38292308239616EB81772717714266F73`).

## KS alert-ordering table (Table 7), 2026-09-24

`explanation_vs_standard_monitor_leads.csv` and
`explanation_vs_standard_monitor_summary.csv` were computed from the legacy
independent-stream grid (`results/raw_all.csv`). The manuscript's KS
alert-ordering table (Table 7, `tab:ks-ordering`) is computed from the corrected
paired grid (`outputs/main_grid/processed_outputs/main_grid_v2.csv`) with the
same rule, the same 0.10 threshold, and the same saved monitor cells
(`standard_drift_monitor_cells.csv`, whose noise and missingness realizations are
seeded separately from the grid). `scripts/recompute_table7_paired.py` performs
this recomputation with the csv module and plain Python, first reproducing the
published legacy counts and all 270 legacy lead rows as a positive control, and
writes `outputs/secondary_audits/table7_paired/`:

- `table7_paired_counts.csv`: before/same/after counts per method and monitor (paired and legacy);
- `table7_paired_detail.csv`: first-alert indices for all 270 method-monitor-combination rows;
- `paired_seed_mean_instability.csv`: seed-mean top-5 instability for every drifted paired-grid cell;
- `verification.json`: rule, population, positive control, and input hashes.

No experiment was rerun. The legacy files are kept unchanged because the
positive control depends on them.
