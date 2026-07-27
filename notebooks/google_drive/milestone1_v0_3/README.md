# Milestone 1 v0.3 — Google Drive notebooks

Run in this order:

1. `02_M1_V03_CONTRACT_AND_PACK.ipynb`
2. `03_M1_V03_TELECOM_TRANSLATOR.ipynb`
3. `04_M1_V03_LOCK_AND_ACCEPTANCE_TESTS.ipynb`
4. `05_M1_V03_PETROBRAS_3W_CONTRACT_CHALLENGE.ipynb`

Expected Drive data:

```text
MyDrive/anomaly_detection/
├── bootstrap/
│   └── M1_v0_3_runtime_bundle.zip
├── reference_dataset.parquet
├── topology.csv
├── entity_service_windows.csv
├── engineering_events.csv
├── tickets.csv
├── evaluation/                     # the supplied name may contain trailing space
│   └── ...
└── sources/
    └── petrobras_3w/
        └── 2.0.0/
            └── raw/
                └── 3w_dataset_2.0.0/
                    ├── 0/ ... 9/
                    ├── folds/
                    ├── dataset.ini
                    ├── LICENSE-CC-BY
                    └── README.md
```

Outputs are immutable and contract-versioned:

```text
MyDrive/anomaly_detection/
├── contracts/v0.3/
├── outputs/milestone_1/v0.3/
│   ├── telecom/<run_id>/{SPEC-CORE,SPEC-EVAL,validation}/
│   └── petrobras_3w/<run_id>/{SPEC-CORE,SPEC-EVAL}/
└── fixtures/petrobras_3w/2.0.0/v0.3/<run_id>/
```

Choose a new run ID to rerun. Existing artifacts are never overwritten.

Before running Notebook 02, copy the included `M1_v0_3_runtime_bundle.zip` file to
`MyDrive/anomaly_detection/bootstrap/`. It is kept separate so the notebook remains
readable and the implementation remains independently hash-verifiable.
