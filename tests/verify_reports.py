"""Validate downloaded reports with Perfetto's actual trace processor.

Usage: python tests/verify_reports.py [reports-directory]
Requires: pip install perfetto==0.59.1
"""
import csv
import json
from pathlib import Path
import sqlite3
import sys

from perfetto.trace_processor import TraceProcessor

root = Path(sys.argv[1] if len(sys.argv) > 1 else "reports")
summary = {}
expected = {
    "power_iteration": ("cupy_sqrt", "DeviceReduceSingleTileKernel"),
    "memcopy": ("copy_blocked", "copy_optimized"),
    "histogram": ("histogram_global", "histogram_localized"),
}
for example, fragments in expected.items():
    nsys = root / "nsys" / example / example
    ncu = root / "ncu" / example / example
    assert nsys.with_suffix(".nsys-rep").stat().st_size > 0
    assert ncu.with_suffix(".ncu-rep").stat().st_size > 0
    with sqlite3.connect(nsys.with_suffix(".sqlite")) as connection:
        names = [r[0] for r in connection.execute(
            "SELECT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id"
        )]
    with ncu.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    measured = [row for row in rows if row.get("Kernel Name") and row.get("gpu__time_duration.sum", "") not in ("", "0", "n/a")]
    with TraceProcessor(trace=str(nsys.with_suffix(".perfetto.json"))) as processor:
        slices = list(processor.query("SELECT name FROM slice WHERE dur > 0"))
        slice_names = [row.name for row in slices]
        errors = [dict(name=row.name, value=row.value) for row in processor.query(
            "SELECT name, value FROM stats WHERE severity='error' AND value > 0"
        )]
    for fragment in fragments:
        assert any(fragment.lower() in name.lower() for name in names), (example, "native nsys", fragment)
        assert any(fragment.lower() in name.lower() for name in slice_names), (example, "Perfetto", fragment)
        assert any(fragment.lower() in row["Kernel Name"].lower() for row in measured), (example, "ncu", fragment)
    assert not errors, (example, errors)
    summary[example] = {"nsys_kernels": len(names), "ncu_measured_kernels": len(measured), "perfetto_slices": len(slices), "perfetto_errors": errors}
print(json.dumps(summary, indent=2))
