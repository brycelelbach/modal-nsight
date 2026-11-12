"""Container and workload helpers shared by the two Modal entrypoints."""

from pathlib import Path
import re
import shlex

import modal

EXAMPLES = ("power_iteration", "memcopy", "histogram")
CUDA_IMAGE = "nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu22.04"
NSIGHTFUL_REVISION = "cbe425a4d8de4fee91d656ce6446da94f7f0f741"
NSYS_DEB_URL = (
    "https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/"
    "2026_1/NsightSystems-linux-cli-public-2026.1.1.204-3717666.deb"
)
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install("git", "curl", "nsight-compute-2026.1.1=2026.1.1.2-1")
    .run_commands(
        f"curl -fL {NSYS_DEB_URL} -o /tmp/nsight-systems.deb",
        "apt-get update && apt-get install -y --no-install-recommends /tmp/nsight-systems.deb",
        "ln -sf /opt/nvidia/nsight-systems-cli/2026.1.1/bin/nsys /usr/local/bin/nsys",
        "ln -sf /opt/nvidia/nsight-compute/2026.1.1/ncu /usr/local/bin/ncu",
        "nsys --version && ncu --version",
    )
    .pip_install(
        "numpy==2.3.5", "cupy-cuda13x==14.2.0", "numba==0.63.1",
        "numba-cuda==0.30.4", "cuda-cccl==1.1.1", "nvtx==0.2.16",
        "cuda-bindings==13.2.0", "cuda-core==1.2.0",
        f"nsightful @ git+https://github.com/brycelelbach/nsightful.git@{NSIGHTFUL_REVISION}",
    )
    .workdir("/opt/modal-nsight")
    .add_local_dir(Path(__file__).parent / "examples", "/opt/modal-nsight/examples", copy=True)
    .add_local_python_source("common")
)


def workloads(example: str, script: str, script_args: str):
    """Read a standalone local script, or select the bundled examples."""
    args = shlex.split(script_args)
    if script:
        if example != "all":
            raise ValueError("Choose either --script or --example")
        path = Path(script)
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", path.stem)
        return [(name, path.read_text(encoding="utf-8-sig"), args)]
    if args:
        raise ValueError("--script-args requires --script")
    if example != "all" and example not in EXAMPLES:
        raise ValueError(f"Choose all or one of: {', '.join(EXAMPLES)}")
    return [(name, None, []) for name in (EXAMPLES if example == "all" else (example,))]


def workload_path(directory: str, name: str, source: str | None) -> Path:
    if source is None:
        if name not in EXAMPLES:
            raise ValueError(f"Unknown example: {name}")
        return Path("/opt/modal-nsight/examples") / f"{name}.py"
    path = Path(directory) / "workload.py"
    path.write_text(source, encoding="utf-8")
    return path


def save_reports(output_dir: str, name: str, files: dict[str, bytes]) -> None:
    directory = Path(output_dir) / name
    directory.mkdir(parents=True, exist_ok=True)
    for filename, data in files.items():
        path = directory / filename
        path.write_bytes(data)
        print(f"Saved {path} ({len(data):,} bytes)")
