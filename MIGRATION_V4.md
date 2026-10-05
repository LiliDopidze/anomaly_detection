# Version 4 migration and audit

The requested development branch is `development/industry-agnostic`, based on remote main `3333e789590ded14dd313ef42ffd32fd073b68a7`. Local main remains at `0ba9285e4a0c17c0488f4151fcdcc6d03745a5bd` with the pre-existing document edits intact. No push or merge is part of this delivery.

The recorded historical verification_v11 source is `ec89fb14d38b890c3a3d5040df853c7b6305b075`. All 17 saved source hashes reconcile. The retained reproduction has 6,912 native observations and six truth events; input/truth frames match exactly. Floating outputs agree at an absolute tolerance of 1e-10, with identical policies, event keys, masks and counts. The historical 40/15/20/25 split and Combined score remain isolated in the old package and checkout.

The separate 96-entity default historical attempt in temporary storage was interrupted after its historical final marker was written. It is not the reconciled verification_v11 benchmark and must not be reported as complete. The successful recorded small reproduction is sufficient for the source/input parity gate; its final period remains unopened.

Version 4 adds the generic `anomaly_detection` package while retaining the unchanged optical generator and adapter. Packaging version 13.0.0 refers to this repository release; model/schema version 4.0.0 refers to the methodology. Historical notebooks move under `notebooks/historical`; the four new notebooks are the supported path. Do not mix historical Combined thresholds or feature scales with version 4 policies.

## Audited evaluation corrections

The interrupted development result is retained intact. Before opening its final temporal period, the audit corrected recovery censoring at the evaluation horizon, early-event cohort handling when physical and observable onset cross a boundary, unknown-onset boundary workload, explicit schedule coverage when score rows are absent, and alarm duration during monitoring gaps. Group diagnostics now inspect all member confirmations for harmful merges, including faults after the anchor. Matched-fraction precision includes unverified investigations in its denominator; confirmed FP and unverified counts are separately identified.

A new run reuses only verified input files, fitted numerical artifacts and detector scores. Its cache lineage checks the previous freeze, configuration, dependency versions, every fitting/scoring module and the actual committed runner function bodies. It records hashes of copied scores, reruns all policy selection and diagnostics, and creates a new freeze. Original results are not overwritten. This is correction before final assessment, not test-driven tuning. The generator and candidate budget are unchanged.

The output `cache_lineage.json` identifies the prior run and source snapshot. `input_manifest.json`, `fit_manifest.json`, `FROZEN.json`, `FINAL_OPENED.json` and `FINAL_RESULT.json` together record the data, numerical source, selection source, access discipline and result. A clean development commit records the candidate implementation before freeze. Failed/interrupted runs remain explicitly labelled.

## Source discipline

The full numbered bibliography is retained in the supplied methodology. Implemented primitives were checked against STUMPY 1.14.1 and installed scikit-learn 1.5.1 source/API; the hosted 1.5 documentation currently identifies patch release 1.5.2. NIST supports the CUSUM recurrence and OTexts supports Fourier seasonality. AAAI evaluation work supports avoiding point adjustment. Anodot is vendor architecture material; MathWorks documents a modular workflow. None validates the complete project, specifies these four features or establishes Python/MATLAB parity. No extra MATLAB implementation is included. The browser could not retrieve three linked paper PDFs during this audit; the supplied citations are retained without claiming a fresh full-text verification.
