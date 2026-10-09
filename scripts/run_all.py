"""Run the whole pipeline: tests, then Steps 1-4 (figures + tables into output/)."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = ["01_pricing_validation.py", "02_convergence.py", "03_implied_vol_smile.py", "04_delta_hedging.py"]


def run(cmd):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))}")
    print(f"  ({time.time() - t0:.0f}s)\n")


if __name__ == "__main__":
    (ROOT / "output" / "tables" / "key_results.json").unlink(missing_ok=True)
    if "--skip-tests" not in sys.argv:
        print("== tests =="); run([sys.executable, "-m", "pytest", "tests", "-q"])
    for s in STEPS:
        print(f"== {s} =="); run([sys.executable, f"scripts/{s}"])
    print("Done. Figures: output/figures   Tables: output/tables")
