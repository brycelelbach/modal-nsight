# Modal Nsight

Two small Modal scripts for profiling Accelerated Computing Hub (ACH) examples on an H100:

- `nsys.py` collects a native Nsight Systems report and uses **nsightful** to produce a Perfetto-compatible Chrome Trace JSON file. It also saves the intermediate SQLite export.
- `ncu.py` collects a native Nsight Compute report and exports kernel metrics to CSV.

Both scripts run the same three examples. There is no notebook or desktop server to deploy. A run starts a GPU function, downloads its reports to your machine, and exits.

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

The copy input is 128 MiB (`2**24` int64 values). Power iteration uses a 1024-by-1024 matrix and a larger eigenvalue gap than the notebook to keep the run short. The histogram downloads the original notebook's book dataset from its Google Drive URL before capture. Internet access to that URL is required. Data generation, JIT compilation, and correctness checks are outside the captured region. The power-iteration checkpoint transfer is retained, but checkpoint text-file writes, plots, notebook magics, and timing sweeps are omitted.

## Container and profiler settings

`common.py` uses the ACH tutorial container pinned by SHA-256, retaining its CUDA/CuPy/Numba/CCCL and Nsight installations. It installs nsightful from a pinned Git revision. Container and converter revisions are explicit constants in that file. The original tutorial image's GUI startup is bypassed; the scripts call the real profiler binaries through `ACH_NSYS_PATH` and `ACH_NCU_PATH`.

Functions request one H100, four CPUs, and 16 GiB of host memory. Nsight Systems CPU sampling and context-switch collection are disabled because they require host capabilities unavailable in this setup. Nsight Compute clock control is disabled for the managed GPU. GPU timings collected under a profiler include instrumentation/replay effects and should not be treated as uninstrumented benchmarks.

Each run fails if the workload's correctness assertions fail, a profiler or converter fails, or the resulting report has no measured CUDA kernels. Reports are returned directly to the local entrypoint; no Modal Volume, secret, or web endpoint is required.

## Verification

Validated on 2026-11-12 in the `modal-labs` workspace using H100 GPUs. All three examples passed their correctness checks under both profilers. Perfetto trace processor v59.1 loaded all three JSON traces with zero import errors and found the expected kernel activity:

| Example | Native Systems kernels | Compute measured kernels (basic) | Perfetto slices |
| --- | ---: | ---: | ---: |
| Power iteration | 717 | 20 | 2169 |
| Memcopy | 2 | 2 | 12 |
| Histogram | 2 | 2 | 12 |

Counts and timings can vary between runs. The full-metrics histogram run also succeeded; six `ctc__rx_bytes_data_user` / `ctc__tx_bytes_data_user` metrics were unavailable and produced warnings. The default basic set avoids those unavailable counters.

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
