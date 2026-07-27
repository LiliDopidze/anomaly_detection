"""Build the three thin, reviewable Milestone 1 Google Drive notebooks."""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "google_drive" / "milestone1_v0_3"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbformat.v4.new_code_cell(textwrap.dedent(source).strip())


RUNTIME_SETUP = code(
    """
    # Shared implementation: GitHub in Colab, local source when testing this repository.
    import os
    import subprocess
    import sys
    from pathlib import Path

    REPOSITORY = "https://github.com/LiliDopidze/anomaly_detection.git"
    RUNTIME_REF = os.getenv("ANOMALY_RUNTIME_REF", "main")
    LOCAL_SOURCE = os.getenv("ANOMALY_SOURCE_ROOT")

    if LOCAL_SOURCE:
        sys.path.insert(0, str(Path(LOCAL_SOURCE).resolve()))
        RUNTIME_COMMIT = "local-working-tree"
    else:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet",
            f"git+{REPOSITORY}@{RUNTIME_REF}",
        ])
        RUNTIME_COMMIT = subprocess.check_output(
            ["git", "ls-remote", REPOSITORY, RUNTIME_REF], text=True
        ).split()[0]

    os.environ["ANOMALY_RUNTIME_REF"] = RUNTIME_REF
    os.environ["ANOMALY_RUNTIME_COMMIT"] = RUNTIME_COMMIT
    print(f"Runtime: {RUNTIME_REF} ({RUNTIME_COMMIT[:12]})")
    """
)


DRIVE_SETUP = code(
    """
    # Mount Google Drive in Colab. Local validation skips this block.
    if "google.colab" in sys.modules:
        from google.colab import drive
        drive.mount("/content/drive")

    DRIVE_ROOT = Path(os.getenv(
        "ANOMALY_DRIVE_ROOT",
        "/content/drive/MyDrive/anomaly_detection",
    ))
    print(f"Drive root: {DRIVE_ROOT}")
    """
)


def notebook(title: str, purpose: str, cells: list) -> nbformat.NotebookNode:
    document = nbformat.v4.new_notebook()
    document["metadata"] = {
        "colab": {"name": title, "provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    document["cells"] = [
        markdown(
            f"""
            # {title}

            {purpose}

            This is a **thin orchestration notebook**. The tested implementation lives
            in the GitHub package; this notebook only sets paths, calls one workflow,
            and displays its evidence. Run cells from top to bottom.
            """
        ),
        RUNTIME_SETUP,
        DRIVE_SETUP,
        *cells,
    ]
    return document


def build_telecom() -> nbformat.NotebookNode:
    return notebook(
        "02 — Telecom materialisation",
        """
        Translate the native synthetic telecom fixture into physically separate
        `SPEC-CORE` and `SPEC-EVAL` outputs. It also records memory and representation
        reports. It never overwrites an existing run.
        """,
        [
            markdown(
                """
                ## Configuration

                For a first check, set a short `SAMPLE_START` / `SAMPLE_END` or list a
                few `ENTITY_IDS`. For the signed-off run, leave them empty to process
                the complete fixture. Change `RUN_ID` whenever you want a new run.
                """
            ),
            code(
                """
                TELECOM_SOURCE = Path(os.getenv(
                    "ANOMALY_TELECOM_SOURCE",
                    str(DRIVE_ROOT),
                ))
                OUTPUT_ROOT = Path(os.getenv(
                    "ANOMALY_OUTPUT_ROOT",
                    str(DRIVE_ROOT / "outputs" / "milestone_1" / "v0.3" / "telecom"),
                ))
                RUN_ID = os.getenv("ANOMALY_RUN_ID", "telecom_full_v1")
                RUN_ROOT = OUTPUT_ROOT / RUN_ID

                SAMPLE_START = os.getenv("ANOMALY_SAMPLE_START") or None
                SAMPLE_END = os.getenv("ANOMALY_SAMPLE_END") or None
                ENTITY_IDS = tuple(filter(None, os.getenv(
                    "ANOMALY_ENTITY_IDS", ""
                ).split(",")))
                BATCH_NATIVE_ROWS = int(os.getenv(
                    "ANOMALY_BATCH_NATIVE_ROWS", "250000"
                ))
                MEMORY_BUDGET_GIB = float(os.getenv(
                    "ANOMALY_MEMORY_BUDGET_GIB", "8"
                ))

                print(f"Source: {TELECOM_SOURCE}")
                print(f"New immutable run: {RUN_ROOT}")
                """
            ),
            code(
                """
                from anomaly_detection.workflows import contract_summary

                summary = contract_summary()
                print("SPEC-CORE:", summary["spec_core_tables"])
                print("SPEC-EVAL:", summary["spec_eval_tables"])
                print("Telecom metrics:", summary["packs"]["telecom"]["metric_count"])
                """
            ),
            code(
                """
                from anomaly_detection.workflows import materialise_telecom

                report = materialise_telecom(
                    TELECOM_SOURCE,
                    RUN_ROOT,
                    sample_start=SAMPLE_START,
                    sample_end=SAMPLE_END,
                    entity_ids=ENTITY_IDS,
                    batch_native_rows=BATCH_NATIVE_ROWS,
                    memory_budget_gib=MEMORY_BUDGET_GIB,
                )
                """
            ),
            code(
                """
                from pprint import pprint

                pprint({
                    "row_counts": report["materialisation"]["row_counts"],
                    "peak_rss_gib": report["memory"]["peak_rss_gib"],
                    "memory_budget_pass": report["memory"]["budget_pass"],
                    "canonical_rows": report["representation"]["canonical_rows"],
                    "run_root": report["run_root"],
                })
                print("Next: run Notebook 03 with the same RUN_ID.")
                """
            ),
        ],
    )


def build_acceptance() -> nbformat.NotebookNode:
    return notebook(
        "03 — Week 1 acceptance and truth lock",
        """
        Prove that translator output is identical after evaluation inputs—including
        `tickets.csv`—are removed. A timestamp canary and a deliberately leaky negative
        control prove the harness can detect value-level leakage. Runtime mount tests
        are secondary deployment evidence, not the primary proof.
        """,
        [
            markdown(
                """
                ## Configuration

                Use the same telecom source and `RUN_ID` that completed in Notebook 02.
                This notebook reads that immutable run and adds `acceptance_report.json`.
                """
            ),
            code(
                """
                TELECOM_SOURCE = Path(os.getenv(
                    "ANOMALY_TELECOM_SOURCE",
                    str(DRIVE_ROOT),
                ))
                OUTPUT_ROOT = Path(os.getenv(
                    "ANOMALY_OUTPUT_ROOT",
                    str(DRIVE_ROOT / "outputs" / "milestone_1" / "v0.3" / "telecom"),
                ))
                RUN_ID = os.getenv("ANOMALY_RUN_ID", "telecom_full_v1")
                RUN_ROOT = OUTPUT_ROOT / RUN_ID

                print(f"Source: {TELECOM_SOURCE}")
                print(f"Existing Notebook 02 run: {RUN_ROOT}")
                """
            ),
            code(
                """
                from anomaly_detection.workflows import run_week1_acceptance

                acceptance = run_week1_acceptance(TELECOM_SOURCE, RUN_ROOT)
                """
            ),
            code(
                """
                from pprint import pprint

                pprint({
                    "translator_invariance": acceptance["translator_invariance"]["passed"],
                    "value_leakage_test": acceptance["value_level_leakage"]["passed"],
                    "negative_control_detected":
                        acceptance["value_level_leakage"]["negative_control_detected"],
                    "runtime_isolation": acceptance["runtime_isolation"]["passed"],
                    "delayed_known_at_guard": acceptance["as_of_guard"]["passed"],
                    "quality_counts":
                        acceptance["materialised_telemetry"]["quality_counts"],
                    "exposure_ranges":
                        acceptance["materialised_telemetry"]["exposure_ranges"],
                })
                print(f"Evidence: {RUN_ROOT / 'acceptance_report.json'}")
                """
            ),
        ],
    )


def build_threew() -> nbformat.NotebookNode:
    return notebook(
        "04 — Petrobras 3W contract challenge",
        """
        Challenge the supposedly neutral contract with a real, non-telecom source.
        The adapter chooses the smallest deterministic subset that covers normal,
        transient, persistent, transition, and missing/frozen cases, then records
        what fits, what cannot be expressed, and what must not be invented.
        """,
        [
            markdown(
                """
                ## Configuration

                `THREEW_SOURCE` must point at the folder containing `dataset.ini` and
                directories `0` through `9`. The selected public fixture is copied to a
                separate immutable fixture folder for repeatable tests.
                """
            ),
            code(
                """
                THREEW_SOURCE = Path(os.getenv(
                    "ANOMALY_THREEW_SOURCE",
                    str(
                        DRIVE_ROOT
                        / "sources"
                        / "petrobras_3w"
                        / "2.0.0"
                        / "raw"
                        / "3w_dataset_2.0.0"
                    ),
                ))
                OUTPUT_ROOT = Path(os.getenv(
                    "ANOMALY_OUTPUT_ROOT",
                    str(DRIVE_ROOT / "outputs" / "milestone_1" / "v0.3" / "petrobras_3w"),
                ))
                RUN_ID = os.getenv("ANOMALY_RUN_ID", "threew_contract_v1")
                RUN_ROOT = OUTPUT_ROOT / RUN_ID
                FIXTURE_DESTINATION = Path(os.getenv(
                    "ANOMALY_FIXTURE_DESTINATION",
                    str(
                        DRIVE_ROOT
                        / "fixtures"
                        / "petrobras_3w"
                        / "2.0.0"
                        / "v0.3"
                        / RUN_ID
                    ),
                ))

                print(f"Source: {THREEW_SOURCE}")
                print(f"New immutable run: {RUN_ROOT}")
                print(f"New immutable fixture: {FIXTURE_DESTINATION}")
                """
            ),
            code(
                """
                from anomaly_detection.workflows import challenge_threew

                challenge = challenge_threew(
                    THREEW_SOURCE,
                    RUN_ROOT,
                    fixture_destination=FIXTURE_DESTINATION,
                )
                """
            ),
            code(
                """
                from pprint import pprint

                pprint({
                    "all_criteria_covered":
                        challenge["all_required_criteria_covered"],
                    "selected_instances": challenge["selected"],
                    "row_counts": challenge["row_counts"],
                    "generic_contract_changes":
                        challenge["contract_fit_report"][
                            "generic_contract_changes_required"
                        ],
                    "not_representable_without_invention":
                        challenge["contract_fit_report"][
                            "not_representable_without_invention"
                        ],
                })
                print(f"Evidence: {RUN_ROOT / 'workflow_report.json'}")
                """
            ),
        ],
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    targets = {
        "02_M1_V03_TELECOM_MATERIALISATION.ipynb": build_telecom(),
        "03_M1_V03_WEEK1_ACCEPTANCE.ipynb": build_acceptance(),
        "04_M1_V03_PETROBRAS_3W_CHALLENGE.ipynb": build_threew(),
    }
    for filename, document in targets.items():
        nbformat.write(document, OUTPUT / filename)
        print(OUTPUT / filename)


if __name__ == "__main__":
    main()
