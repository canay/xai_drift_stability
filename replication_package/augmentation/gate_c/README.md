# Gate C — Prospective Held-Out Detector Utility

Date/time: 2026-08-26 18:10 +03:00  
Tool: Codex  
Model, if known: GPT-5  
Operation ID: `f02-scientific-augmentation-20260826`

This run tests whether the Gate A--B reduction in explanation-drift estimation
error transfers to detector utility. Calibration seeds 0--2 freeze all
thresholds, validation seeds 3--4 test transfer without opening the test
channel, and seeds 5--9 form the once-opened test split. Every trial contains
paired and independent explanation scores, Explanation Shift, an input-space
discriminator, and an output-probability discriminator on identical disjoint
reference/candidate windows.

The locked primary criterion requires a strictly lower paired held-out
false-alarm rate with a nonpositive upper 95% dataset--seed cluster-bootstrap
limit. Power, AUROC, and AUPRC have separate non-inferiority rules. Gate C is
fail-closed: if false-alarm superiority fails, detector-superiority and Gate D
scale-up claims remain closed even when measurement-error results pass.

The final decision is `KILL_PIVOT`, because C1 fails while C2--C7 pass. This is
a prospective scientific boundary, not an execution failure. The supported
paper contribution remains the covariance-conditional estimator result and its
cross-explainer validation.
