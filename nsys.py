"""Run ACH examples under Nsight Systems; download native and Perfetto traces."""

import modal

from common import image, save_reports, select_examples

app = modal.App("modal-nsight-systems")


@app.function(image=image, gpu="H100", cpu=4, memory=16384, timeout=900)
def profile(example: str) -> dict[str, bytes]:
    import json
    import os
    from pathlib import Path
    import sqlite3
    import subprocess
    import sys
    import tempfile

    from nsightful.nsys import convert_nsys_sqlite_to_json

    select_examples(example)
    nsys = os.environ["ACH_NSYS_PATH"]  # Real binary, not the tutorial's Docker wrapper.
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory) / example
        command = [
            nsys, "profile", "--force-overwrite=true", "--sample=none", "--cpuctxsw=none",
            "--trace=cuda,nvtx", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
            "-o", str(base), sys.executable, f"examples/{example}.py",
        ]
        subprocess.run(command, check=True, timeout=600)
        native = base.with_suffix(".nsys-rep")
        sqlite = base.with_suffix(".sqlite")
        subprocess.run([
            nsys, "export", "--type=sqlite", "--force-overwrite=true",
            "-o", str(sqlite), str(native),
        ], check=True, timeout=120)
        with sqlite3.connect(sqlite) as connection:
            connection.row_factory = sqlite3.Row
            events = convert_nsys_sqlite_to_json(connection)
            kernels = connection.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
        if not kernels or not any(event.get("cat") == "cuda" and event.get("dur", 0) > 0 for event in events):
            raise RuntimeError("The report contains no CUDA kernel activity")
        print(f"Verified {kernels} CUDA kernels; converted {len(events)} events with nsightful")
        return {
            native.name: native.read_bytes(),
            f"{example}.perfetto.json": json.dumps({"traceEvents": events}).encode(),
            sqlite.name: sqlite.read_bytes(),
        }


@app.local_entrypoint()
def main(example: str = "all", output_dir: str = "reports/nsys"):
    for name in select_examples(example):
        save_reports(output_dir, name, profile.remote(name))
