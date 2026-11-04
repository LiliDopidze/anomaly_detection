"""Execute a project notebook using this interpreter; save only after success."""

import json
import argparse
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", default="01_data_assessment.ipynb",
                        choices=["01_data_assessment.ipynb", "02_expected_behaviour.ipynb",
                                 "03_feature_engineering.ipynb", "04_detection_and_validation.ipynb"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / args.notebook
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    with tempfile.TemporaryDirectory(prefix="assessment-kernel-") as directory:
        kernel = Path(directory) / "kernels/python3"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Time-series assessment", "language": "python",
            "env": {"MPLCONFIGDIR": str(root / "outputs/.matplotlib"),
                    "MPLBACKEND": "module://matplotlib_inline.backend_inline"},
        }))
        manager = KernelManager(
            kernel_name="python3",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]),
        )
        client = NotebookClient(
            notebook, km=manager, timeout=600, record_timing=False,
            resources={"metadata": {"path": str(root)}},
        )
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    nbformat.write(notebook, path)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    figures = sum("image/png" in output.get("data", {})
                  for cell in cells for output in cell.outputs)
    print(f"Executed {len(cells)} code cells and saved {figures} figures to {path.name}")


if __name__ == "__main__":
    main()
