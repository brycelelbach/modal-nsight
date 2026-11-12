# Modal Nsight

Two small Modal scripts for profiling Python GPU workloads on an H100:

- `nsys.py` collects a native Nsight Systems report and uses **nsightful** to produce a Perfetto-compatible Chrome Trace JSON file. It also saves the intermediate SQLite export.
- `ncu.py` collects a native Nsight Compute report and exports kernel metrics to CSV.

Both scripts accept a standalone Python script or run the three bundled examples. A run starts a GPU function, downloads its reports to your machine, and exits.

## Run

Python 3.10+ is required locally; CUDA, CuPy, Numba, and the profilers run inside Modal.

```sh
git clone https://github.com/brycelelbach/modal-nsight.git
cd modal-nsight
python -m pip install -r requirements.txt
python -m modal setup  # Once, if not already authenticated.
```

Run all three examples with each profiler:

```sh
python -m modal run nsys.py
python -m modal run ncu.py
```

If you have multiple Modal profiles, select the workspace with `MODAL_PROFILE`: `export MODAL_PROFILE=modal-labs` in bash, or `$env:MODAL_PROFILE='modal-labs'` in PowerShell. Otherwise, the commands use your active profile. Compute is billed to the selected workspace.

Select one example or change the output location:

```sh
python -m modal run nsys.py --example power_iteration --output-dir reports/nsys
python -m modal run ncu.py --example memcopy
python -m modal run ncu.py --example histogram --metric-set full --launch-count 20
```

`--example` accepts `all` (the default), `power_iteration`, `memcopy`, or `histogram`. Runs are sequential. Re-running overwrites reports with the same names in the selected output directory.

## Profile your own script

```sh
python -m modal run nsys.py --script my_workload.py
python -m modal run ncu.py --script my_workload.py --script-args="--size 1048576"
```

The selected Python file is uploaded and run in the GPU container. Only that file is uploaded; package any additional modules or data in the image defined in `common.py`. Add workload dependencies to that image's `pip_install` list. Script arguments are split using shell-style quoting and passed directly to Python without a shell.

Custom scripts are profiled from process start by default. To capture only a marked region, call `cupy.cuda.profiler.start()` / `stop()` around the desired GPU work and add `--capture-range` to either command. Synchronize before starting and before stopping. The bundled examples use this mode automatically. Nsight Compute still limits collection to `--launch-count` kernels (20 by default).

Reports use your script's filename stem as their name. The script must execute CUDA kernels; runs without measured kernel activity fail validation.

## Reports

| Script | Local outputs for each example |
| --- | --- |
| `nsys.py` | `reports/nsys/<example>/<example>.nsys-rep`, `.perfetto.json`, and `.sqlite` |
| `ncu.py` | `reports/ncu/<example>/<example>.ncu-rep` and `.csv` |

Open **https://ui.perfetto.dev**, choose **Open trace file**, and select a `.perfetto.json` file. This is Chrome Trace JSON accepted by Perfetto, not Perfetto protobuf. The native `.nsys-rep` remains available for the complete Nsight Systems view. Open `.ncu-rep` in Nsight Compute, or inspect the exported CSV without installing a GUI.

The nsightful conversion includes CUDA kernels, CUDA API calls, and NVTX ranges. Its output is not a lossless conversion of every Nsight Systems track; keep the native report for analysis beyond the converted activities.

## Examples

| Example | Captured workload | Correctness check |
| --- | --- | --- |
| `power_iteration` | CuPy matrix-vector products, normalization, Rayleigh quotient, convergence checks, and checkpoint transfers | Dominant eigenvalue is approximately 1 |
| `memcopy` | Numba `copy_blocked` and coalesced `copy_optimized` kernels | Both destinations exactly match the source |
| `histogram` | Numba `histogram_global` and shared-memory `histogram_localized`, with CCCL striped block loads | All 256 bins from both kernels match NumPy |

Each example warms up before calling `cudaProfilerStart`, synchronizes before `cudaProfilerStop`, and validates its result afterward. NVTX labels identify the algorithms and variants. Nsight Systems captures the whole marked region. Nsight Compute defaults to the **basic** metric set and at most **20 kernel launches** in that region; it does not profile the whole power-iteration loop by default. Increase `--launch-count` as needed, or choose `--metric-set detailed` / `full` for more counters and additional replay overhead. The copy and histogram defaults include both variants.

The copy input is 128 MiB (`2**24` int64 values). Power iteration uses a 1024-by-1024 matrix with a 0.2 eigenvalue gap. The histogram downloads a book dataset from its Google Drive URL before capture. Internet access to that URL is required. Data generation, JIT compilation, and correctness checks are outside the captured region. Power iteration includes device-to-host checkpoint transfers.

## Container and profiler settings

`common.py` builds from NVIDIA's CUDA 13.2.1 development image with Python 3.12. It installs Nsight Systems 2026.1.1 and Nsight Compute 2026.1.1 directly and places `nsys` and `ncu` on `PATH`. The image also installs CuPy, Numba, CCCL, NVTX, and a pinned nsightful revision. The profiler scripts use standard executable names and have no project-specific environment variables. Versions and dependencies are explicit in `common.py`.

Functions request one H100, four CPUs, and 16 GiB of host memory. Nsight Systems CPU sampling and context-switch collection are disabled because they require host capabilities unavailable in this setup. Nsight Compute clock control is disabled for the managed GPU. GPU timings collected under a profiler include instrumentation/replay effects and should not be treated as uninstrumented benchmarks.

Each run fails if the workload's correctness assertions fail, a profiler or converter fails, or the resulting report has no measured CUDA kernels. Reports are returned directly to the local entrypoint; no Modal Volume, secret, or web endpoint is required.

## Verification

Validated on 2026-11-12 using H100 GPUs in `modal-labs`: all three bundled examples passed under both profilers, and Perfetto trace processor v59.1 loaded all three converted traces without import errors. Standalone-script runs also passed under both profilers with command-line arguments, using both full-process capture and explicit CUDA capture ranges.

To repeat the report validation after running both scripts:

```sh
python -m pip install perfetto==0.59.1
python tests/verify_reports.py reports
```

The verifier checks native Systems kernel records, Compute metrics, expected kernel names in Perfetto, and Perfetto import errors. It downloads Perfetto's official trace processor binary on first use.

## Sources and license

The extracted code is from [NVIDIA's Accelerated Computing Hub](https://github.com/brycelelbach/accelerated-computing-hub/tree/231ede9f433ee6513dd79ca88642cb842a9c04c3), revision `231ede9f433ee6513dd79ca88642cb842a9c04c3`:

- [Power iteration, memory spaces solution](https://github.com/brycelelbach/accelerated-computing-hub/blob/231ede9f433ee6513dd79ca88642cb842a9c04c3/tutorials/accelerated-python/notebooks/fundamentals/solutions/05__memory_spaces__power_iteration__SOLUTION.ipynb)
- [Kernel copy solution](https://github.com/brycelelbach/accelerated-computing-hub/blob/231ede9f433ee6513dd79ca88642cb842a9c04c3/tutorials/accelerated-python/notebooks/kernels/solutions/40__kernel_authoring__copy__SOLUTION.ipynb)
- [Book histogram solution](https://github.com/brycelelbach/accelerated-computing-hub/blob/231ede9f433ee6513dd79ca88642cb842a9c04c3/tutorials/accelerated-python/notebooks/kernels/solutions/41__kernel_authoring__book_histogram__SOLUTION.ipynb)

The kernel bodies and power-iteration algorithm are adapted into standalone programs with profiling controls and correctness checks. Code is under [Apache-2.0](LICENSE), matching ACH's code license. Original code remains attributable to NVIDIA and the ACH contributors. The external book dataset is downloaded separately and is not distributed by this repository. [Nsightful](https://github.com/brycelelbach/nsightful) and the container's dependencies retain their own licenses.
