"""Run the full practicum evaluation insights workflow in order.

Runs:
    1. pipeline.py
    2. create_visualizations.py
    3. analyze_program_alignment.py
    4. build_descriptive_tables.py

Does not run app.py. Launch the dashboard separately:
    streamlit run app.py
"""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent

steps = [
    "pipeline.py",
    "create_visualizations.py",
    "analyze_program_alignment.py",
    "build_descriptive_tables.py",
]


def run_step(script_name: str) -> None:
    """Run one script and stop immediately if it fails."""
    print(f"\nrunning {script_name}")

    result = subprocess.run(
        [sys.executable, script_name],
        cwd=project_root,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} stopped with exit code {result.returncode}"
        )


def main() -> None:
    """Run the full workflow in order."""
    print(f"project root: {project_root}")

    for script_name in steps:
        run_step(script_name)

    print("\ndone. launch the dashboard with: streamlit run app.py")


if __name__ == "__main__":
    main()
