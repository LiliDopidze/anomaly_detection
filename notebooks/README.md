# Branch 2 synthetic workflow

Run the five numbered notebooks in order with a Python kernel that has this
project installed. Notebook 1 creates the data; no download or Drive mount is
needed. Notebook 5 stays sealed unless explicitly enabled.

1. `01_generate_and_validate.ipynb`: generate, verify, save audit reports.
2. `02_eda_and_seasonality.ipynb`: distributions, gaps, traces, dependencies.
3. `03_features.ipynb`: causal features and availability.
4. `04_develop_baselines.ipynb`: fit, calibrate, compare, select or STOP.
5. `05_final_evaluation.ipynb`: frozen final evaluation, disabled by default.

`legacy/` preserves the earlier notebooks. They use the earlier configuration
and data contracts and are not prerequisites for this workflow. The original
submitted generator is under `reference/`. See the root README and
SYNTHETIC_REVIEW for setup, scope and scientific limitations.
