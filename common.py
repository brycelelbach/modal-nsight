"""Container and local output helpers shared by the two Modal entrypoints."""

from pathlib import Path

import modal

EXAMPLES = ("power_iteration", "memcopy", "histogram")
TUTORIAL_IMAGE = "ghcr.io/nvidia/accelerated-python-tutorial@sha256:2a5f12d784384a676a600dfc9a0254d0bc253c4a4ad622dcfdbb5e673d014fd0"
NSIGHTFUL_REVISION = "cbe425a4d8de4fee91d656ce6446da94f7f0f741"
image = (
    modal.Image.from_registry(TUTORIAL_IMAGE)
    .entrypoint([])
    .workdir("/opt/modal-nsight")
    .pip_install(
        f"nsightful @ git+https://github.com/brycelelbach/nsightful.git@{NSIGHTFUL_REVISION}",
        extra_options="--force-reinstall --no-deps",
    )
    .add_local_dir(Path(__file__).parent / "examples", "/opt/modal-nsight/examples", copy=True)
    .add_local_python_source("common")
)


def select_examples(example: str) -> tuple[str, ...]:
    if example == "all":
        return EXAMPLES
    if example not in EXAMPLES:
        raise ValueError(f"Choose all or one of: {', '.join(EXAMPLES)}")
    return (example,)


def save_reports(output_dir: str, example: str, files: dict[str, bytes]) -> None:
    directory = Path(output_dir) / example
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = directory / name
        path.write_bytes(data)
        print(f"Saved {path} ({len(data):,} bytes)")
