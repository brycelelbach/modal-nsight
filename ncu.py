"""Run ACH examples under Nsight Compute; download native reports and metrics."""

import modal

from common import image, save_reports, select_examples

app = modal.App("modal-nsight-compute")


@app.function(image=image, gpu="H100", cpu=4, memory=16384, timeout=900)
def profile(example: str, launch_count: int = 20, metric_set: str = "basic") -> dict[str, bytes]:
    import csv
    import io
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tempfile

    select_examples(example)
    if launch_count < 1:
        raise ValueError("launch_count must be positive")
    if metric_set not in ("basic", "detailed", "full"):
        raise ValueError("metric_set must be basic, detailed, or full")
    ncu = os.environ["ACH_NCU_PATH"]
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory) / example
        subprocess.run([
            ncu, "--force-overwrite", "--set", metric_set, "--clock-control", "none",
            "--profile-from-start", "off", "--launch-count", str(launch_count),
            "-o", str(base), sys.executable, f"examples/{example}.py",
        ], check=True, timeout=600)
        native = base.with_suffix(".ncu-rep")
        exported = subprocess.run([
            ncu, "--import", str(native), "--csv", "--page", "raw",
        ], check=True, capture_output=True, text=True, timeout=120)
        rows = list(csv.DictReader(io.StringIO(exported.stdout)))
        measured = [row for row in rows if row.get("Kernel Name") and row.get("gpu__time_duration.sum", "") not in ("", "0", "n/a")]
        if not measured:
            raise RuntimeError("The report contains no measured CUDA kernels")
        print(f"Verified metrics for {len(measured)} CUDA kernels")
        return {native.name: native.read_bytes(), f"{example}.csv": exported.stdout.encode()}


@app.local_entrypoint()
def main(example: str = "all", output_dir: str = "reports/ncu", launch_count: int = 20, metric_set: str = "basic"):
    if launch_count < 1 or metric_set not in ("basic", "detailed", "full"):
        raise ValueError("Use a positive --launch-count and --metric-set basic, detailed, or full")
    for name in select_examples(example):
        save_reports(output_dir, name, profile.remote(name, launch_count, metric_set))
