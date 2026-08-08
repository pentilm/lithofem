"""M5-4: PML parameter scan (thickness x sigma-order); prints a summary table."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PML = Path(__file__).resolve().parent.parent / "solver" / "bin" / "pml_test"


def main() -> None:
    print("| thickness (wl) | sigma order | |r_PML| (theta=0, p=3, nz=12) |")
    print("|---|---|---|")
    for thick in (0.5, 1.0, 2.0):
        for order in (1, 2, 3):
            res = subprocess.run(
                [str(PML), "-t", "0", "-pol", "0", "-p", "3", "-nz", "12",
                 "-pt", str(thick), "-po", str(order)],
                capture_output=True, text=True, timeout=1800,
            )
            m = re.search(r"pml_reflection ([\d.eE+-]+)", res.stdout)
            val = m.group(1) if m else "FAIL"
            print(f"| {thick} | {order} | {val} |")


if __name__ == "__main__":
    main()
