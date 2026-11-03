"""Execute 00 and 01 with this Python environment; save successful outputs in place.

From the repository root: python tools/execute_foundation.py
Install the package with its [dev] extras first. The temporary kernel specification
avoids silently using a globally installed Python or an unrelated checkout.
"""

from pathlib import Path
import json
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    names = ["00_generate_and_diagnose.ipynb", "01_canonical_eda.ipynb"]
    with tempfile.TemporaryDirectory(prefix="foundation-kernel-") as directory:
        kernel = Path(directory) / "kernels" / "python3"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Foundation validation",
                    "language": "python",
                    "env": {
                        "MPLCONFIGDIR": str(Path(directory) / "mpl"),
                        "MPLBACKEND": "module://matplotlib_inline.backend_inline",
                    },
                }
            )
        )
        for name in names:
            print(f"Executing {name}", flush=True)
            path = root / "notebooks" / name
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            manager = KernelManager(
                kernel_name="python3",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]),
            )
            client = NotebookClient(
                notebook,
                km=manager,
                timeout=240,
                record_timing=False,
                resources={"metadata": {"path": str(root)}},
            )
            try:
                client.execute()
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            nbformat.write(notebook, path)
            cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
            plots = sum(
                "image/png" in output.get("data", {})
                for cell in cells
                for output in cell.get("outputs", [])
            )
            print(f"Passed: {len(cells)} code cells, {plots} figures", flush=True)


if __name__ == "__main__":
    main()
