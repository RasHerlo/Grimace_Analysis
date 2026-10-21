"""Re-run all analysis scripts (in order) and rebuild the PDF report.

Run:  python report/run_all.py            (everything)
      python report/run_all.py --build    (only rebuild the PDF from the saved blocks)

To include a new analysis: add its script to SCRIPTS (order = order of execution only; the position in the
report is decided by the question IDs in the script's Block).
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "00_data_overview/overview.py",
    "01_state_structure/01_patterns.py",
    "01_state_structure/02_temporal.py",
    "01_state_structure/03_missingness.py",
]

if "--build" not in sys.argv:
    for s in SCRIPTS:
        t = time.time()
        print(f"== {s}", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / s)], cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-3000:])
            sys.exit(f"FAILED: {s}")
        print(f"   ok ({time.time() - t:.0f} s)", flush=True)
print("== build report", flush=True)
subprocess.run([sys.executable, str(ROOT / "report" / "build_report.py")], cwd=ROOT, check=True)
