# Jakal-Core

Graph-first heterogeneous compute runtime skeleton for C++20, with a shared C API, a diagnostic CLI, and install/package tooling.

This repository builds the `jakal_core` library target, the `jakal_runtime` shared library, and a set of build-tree tools around hardware discovery, workload planning, execution tuning, and packaged runtime diagnostics.

Jakal-Core is not a production runtime yet. It is a runtime skeleton that:

- discovers host and accelerator hardware
- turns that hardware into structural graphs
- builds placement and execution plans from those graphs
- runs a compact direct kernel set to check whether those plans make sense on a real machine
- exposes the same runtime through C++, C, CLI, and small Python helpers

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [What is this repository?](#what-is-this-repository)
- [Why graph-first planning?](#why-graph-first-planning)
- [Current scope](#current-scope)
- [API](#api)
  - [C++ entry points](#c-entry-points)
  - [Runtime options and install paths](#runtime-options-and-install-paths)
  - [Workload helpers and imported sources](#workload-helpers-and-imported-sources)
  - [C API and shared runtime](#c-api-and-shared-runtime)
  - [CLI](#cli)
  - [Python helpers](#python-helpers)
  - [Cache files and installed layout](#cache-files-and-installed-layout)
- [Examples](#examples)
- [Further reading](#further-reading)
- [License](#license)

## Installation

Jakal-Core is still source-first, but the tree now supports local installs, exported CMake package files, and CPack package generation for downstream use.

### Requirements

- CMake 3.20 or newer
- A C++20 compiler
- Internet access on the first configure unless `FetchContent` already has `Vulkan-Headers` cached. The top-level `CMakeLists.txt` downloads `v1.4.321` during configure.
- Vendor runtimes or drivers for the backends you want to exercise:
  - OpenCL for OpenCL discovery and direct OpenCL execution
  - Intel Level Zero for Level Zero discovery and native Intel GPU paths
  - Vulkan loader plus shader compiler tools if you want the Vulkan direct backend to become fully ready
  - CUDA or ROCm runtime libraries if you want those probes and modeled/native paths to activate
- Windows plus DirectML libraries if you want to build `jakal_directml_manifest_bench`

The project loads accelerator runtimes dynamically. If a given runtime library is missing, the corresponding probe stays inactive or modeled-only.

### Build the project

From the repository root:

```powershell
cmake -S . -B build
cmake --build build
```

The default build produces:

- libraries: `jakal_core`, `jakal_runtime`
- install and package tools: `jakal_core_cli`, `jakal_bootstrap`
- examples: `jakal_inspect`, `jakal_profile_workloads`, `jakal_explore_cpu_dl`, `jakal_partition_roles`, `jakal_profile_manifest`
- Windows-only example: `jakal_directml_manifest_bench`
- standalone test binaries: `jakal_smoke`, `jakal_optimization`, `jakal_planner_learning`, `jakal_partition_strategies`, `jakal_runtime_product`, `jakal_workload_import_adapters`, `jakal_backend_contracts`, `jakal_live_backend_smoke`, `jakal_preset_execution_diag`, `jakal_runtime_install_smoke`, `jakal_workload_bench`
- `ctest` entries: `jakal_bootstrap_status` when product tools are enabled, every standalone test above except `jakal_preset_execution_diag`, and `jakal_core_cli_doctor_json`

Useful CMake switches:

```powershell
cmake -S . -B build `
  -DJAKAL_CORE_BUILD_EXAMPLES=OFF `
  -DJAKAL_CORE_BUILD_TESTS=OFF `
  -DJAKAL_CORE_BUILD_PRODUCT_TOOLS=OFF
```

### Install locally

To create a local install tree:

```powershell
cmake --install build --config Release --prefix .\out\jakal-core
```

The install tree is smaller than the full development build. It installs:

- `jakal_core`
- `jakal_runtime`
- `jakal_core_cli`
- `jakal_bootstrap` when product tools are enabled
- `jakal_inspect` when examples are enabled
- public headers, exported CMake package files, docs, install/update/remove helpers, and Python helper scripts

Typical installed layout:

- `bin/`: `jakal_core_cli`, `jakal_bootstrap`, `jakal_inspect`, `jakal_runtime` shared library, `launch-jakal-hardware-setup.cmd`
- `lib/`: `jakal_core`, `jakal_runtime` import/static libraries, exported CMake package files
- `include/`: public headers under `jakal/`
- `share/jakal-core/`: runtime config, maintenance scripts, bundled prerequisite locations, branding assets, Python helpers
- `share/doc/JakalCore/`: `README.md`, `LICENSE`, `distribution.md`

### Consume the installed package

Jakal-Core installs exported CMake package files, so an installed copy can be consumed with `find_package`:

```cmake
find_package(JakalCore CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE jakal::core)
```

If you want the shared C runtime surface instead of the C++ library target, link `jakal::runtime`.

### Generate packages

For a ZIP or TGZ package:

```powershell
cmake -S . -B build -DJAKAL_CORE_BUILD_TESTS=OFF
cmake --build build --config Release --target package
```

On Windows the default package generator is ZIP. NSIS is added automatically when `makensis` is available.

For a Windows installer with optional signing and checksum generation:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\build-nsis-package.ps1 `
  -CodeSignCertSha1 "<thumbprint>" `
  -SignToolPath "C:\Program Files (x86)\Windows Kits\10\App Certification Kit\signtool.exe"
```

If you want build-time signing for installable Windows executables such as `jakal_core_cli.exe` and `jakal_bootstrap.exe`, configure:

- `JAKAL_CORE_ENABLE_CODE_SIGNING=ON`
- `JAKAL_CORE_SIGNTOOL_PATH=...`
- `JAKAL_CORE_CODESIGN_CERT_SHA1=...`
- `JAKAL_CORE_CODESIGN_TIMESTAMP_URL=...`

For the packaged runtime flow, including bundled prerequisite directories and the NSIS launcher entry points, see [`docs/distribution.md`](./docs/distribution.md).

### Run the tests

For multi-config generators such as Visual Studio:

```powershell
ctest --test-dir build -C Debug --output-on-failure
```

For single-config generators such as Ninja or Unix Makefiles:

```powershell
ctest --test-dir build --output-on-failure
```

## Quick start

If you want to consume the library from another CMake project, add this repository as a subdirectory and link `jakal::core`.

```cmake
add_subdirectory(path/to/Jakal-Core)
target_link_libraries(my_app PRIVATE jakal::core)
```

Minimal C++ example:

```cpp
#include "jakal/runtime.hpp"

#include <iostream>

int main() {
    jakal::Runtime runtime;

    for (const auto& graph : runtime.devices()) {
        std::cout << graph.presentation_name << '\n';
    }

    return 0;
}
```

`jakal::Runtime` refreshes hardware during construction by default, so `devices()` is ready immediately unless you explicitly disable every probe.

If you want to inspect the runtime without writing code, start with the build-tree tools:

```powershell
.\build\Debug\jakal_core_cli.exe doctor --json --host-only
.\build\Debug\jakal_inspect.exe
.\build\Debug\jakal_bootstrap.exe --status --self-check
```

On single-config generators, the executables are typically under `.\build\` instead of `.\build\Debug\`.

## What is this repository?

Jakal-Core treats hardware as a graph instead of a label. A discovered CPU, OpenCL device, Level Zero device, Vulkan device, CUDA device, or ROCm device becomes a set of compute, storage, control, and transfer nodes with weighted edges between them. The planner then works from that structure when it decides where a workload should land.

There are four main pieces in the tree today:

- hardware discovery and graph summarization
- workload graph generation and placement planning
- execution-graph construction and optimization
- direct execution plus lightweight validation for a small operation set

The built-in direct operation set covers:

- elementwise map
- reduction
- blocked matmul
- 3x3 convolution
- bilinear resample

The repository also ships:

- a shared C runtime surface in `jakal_runtime`
- a packaged-runtime CLI in `jakal_core_cli`
- install/update/remove scripts and a hardware-setup launcher
- Python ctypes helpers for simple diagnostics and manifest execution

There are built-in workload presets for:

- gaming-style upscaling and post-processing
- vision-style inference
- compact training-step surrogates
- CPU-heavy deep-learning exploration cases such as token decode, KV-cache updates, and dequant staging

## Why graph-first planning?

Flat device labels hide the details that actually drive placement. "GPU" does not tell you whether the device has unified memory, what its host link looks like, how much dispatch latency it carries, or what kind of execution structure it exposes for mapping work.

The planner in this repository scores things like:

- execution width and resident contexts
- matrix units and numeric capability flags
- directly attached memory and shared host-visible memory
- host-link bandwidth
- dispatch, synchronization, and transfer costs
- graph shape, not just vendor or backend name

That lets the runtime ask more specific questions:

- should a latency-sensitive decode stage stay on the host?
- is unified memory worth preferring for this workload?
- is sharding worth the transfer cost?
- which structural nodes should be mapped for a given operation?

That is the point of this codebase right now. It is closer to an executable model of the runtime architecture than a finished execution stack.

## Current scope

This is where the repository stands today:

- Discovery can use host, OpenCL, Level Zero, Vulkan, CUDA, and ROCm probes when the matching runtime libraries are present.
- Planning and optimization work across those discovered graphs.
- The direct executor can run host kernels, OpenCL kernels, Vulkan direct kernels, and some native Level Zero, CUDA, and ROCm kernels.
- The managed execution layer can run native workload manifests, do memory preflight checks, materialize residency and tensor-allocation traces, summarize backend buffer ownership and persistent resource reuse, correlate executed residency movement with direct-execution transfers, apply persisted-regression safety gates for explicit strategies, emit asset-prefetch plans, and write schema-versioned TSV telemetry.
- `load_workload_source(...)` can build workload graphs from native manifests and imported model descriptions such as ONNX and GGUF.
- `jakal_runtime` exposes the runtime through a smaller C ABI, including ABI/schema version queries plus managed-execution buffer-binding and residency-movement summaries after manifest execution, and `jakal_core_cli` exposes install-path and backend-health diagnostics for packaged runtime scenarios.
- The Python helpers under [`python/`](./python/) wrap the C ABI for lightweight `doctor`, `paths`, `optimize-smoke`, and `run-manifest` flows.

Known gaps in the current tree:

- The repository contains packaging and installer flows, but they are still documented and tested as source-first/developer flows rather than a finished production runtime release train.

What is still missing:

- backend-owned tensor allocators and residency movement beyond the current runtime-local allocation, spill-artifact, and backend buffer-binding diagnostics layer
- framework bridges for PyTorch, TensorFlow, or similar runtimes beyond the small export helper in `scripts/`
- fully validated production packaging, update, and silent-install flows across clean machines
- a stable production execution stack with mature native backend coverage across every supported accelerator path

## API

### C++ entry points

The main public headers are:

- [`include/jakal/runtime.hpp`](./include/jakal/runtime.hpp)
- [`include/jakal/planner.hpp`](./include/jakal/planner.hpp)
- [`include/jakal/execution.hpp`](./include/jakal/execution.hpp)
- [`include/jakal/workloads.hpp`](./include/jakal/workloads.hpp)
- [`include/jakal/c_api.h`](./include/jakal/c_api.h)

`jakal::Runtime` is the main entry point.

| Method | What it does |
| --- | --- |
| `refresh_hardware()` | Re-runs device discovery and rebuilds runtime backend status |
| `options()` | Returns the normalized runtime options in use |
| `install_paths()` | Returns the resolved install, cache, log, telemetry, and Python helper paths |
| `backend_statuses()` | Returns per-backend readiness such as `ready_direct` or `ready_modeled` |
| `devices()` | Returns discovered hardware graphs |
| `jakal_toolkit_index()` | Returns ranked backend variants per discovered device |
| `plan(workload)` | Builds or loads a cached placement plan |
| `optimize(workload)` | Builds workload and execution graphs, then chooses execution settings |
| `optimize(workload, workload_graph)` | Optimizes a caller-supplied workload graph instead of generating the default one |
| `execute(workload)` | Runs the selected execution path and feeds the results back into the optimizer |
| `execute_managed(workload)` | Runs placement, optimization, safety gates, and direct execution with extra diagnostics |
| `execute_managed(workload, workload_graph)` | Managed execution for a caller-supplied workload graph |
| `execute_manifest(path)` | Loads a manifest or imported workload source, then runs the managed path |

`jakal::RuntimeOptions` lets you:

- enable or disable host, OpenCL, Level Zero, Vulkan, CUDA, and ROCm probes
- prefer Level Zero over OpenCL when both match the same hardware
- override the install root, planner cache path, and execution cache path
- tune memory, safety, observability, and execution optimization policy

`jakal::WorkloadSpec` is the main planning input.

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | `std::string` | Human-readable workload name |
| `kind` | `jakal::WorkloadKind` | Broad workload class such as `inference` or `training` |
| `dataset_tag` | `std::string` | Optional preset or dataset identifier |
| `working_set_bytes` | `std::uint64_t` | Estimated active working set |
| `host_exchange_bytes` | `std::uint64_t` | Estimated host-device exchange volume |
| `estimated_flops` | `double` | Approximate compute demand |
| `batch_size` | `std::uint32_t` | Batch size hint |
| `latency_sensitive` | `bool` | Whether latency should be favored over throughput |
| `prefer_unified_memory` | `bool` | Whether unified memory should get extra weight |
| `matrix_friendly` | `bool` | Whether the workload looks friendly to GEMM-style hardware |
| `partition_strategy` | `jakal::PartitionStrategy` | Optional explicit strategy override such as `role_split` or `projection_sharded` |
| `phase` | `jakal::WorkloadPhase` | Phase hint such as `prefill`, `decode`, or `training_step` |
| `shape_bucket` | `std::string` | Optional bucket used for cache reuse and planning families |

### Runtime options and install paths

`runtime.hpp` also exposes install-aware helpers:

- `resolve_runtime_install_paths(...)`
- `make_runtime_options_for_install(...)`

Those helpers are what the CLI, the install smoke tests, and the packaged-runtime scripts use. They resolve the writable and config locations first, then point planner cache, execution cache, and telemetry output at those paths.

`RuntimeInstallPaths` contains:

| Field | Meaning |
| --- | --- |
| `install_root` | Explicit runtime install root or `JAKAL_INSTALL_ROOT` |
| `writable_root` | Writable runtime root chosen for caches and logs |
| `config_dir` | Directory that holds `jakal-runtime-config.ini` |
| `cache_dir` | Directory for planner and execution caches |
| `logs_dir` | Directory for runtime logs and telemetry |
| `telemetry_path` | Path to `runtime-telemetry.tsv` |
| `planner_cache_path` | Path to the planner cache TSV |
| `execution_cache_path` | Path to the execution cache TSV |
| `python_dir` | Runtime install-root Python helper directory, or a writable fallback when no install root is set |

### Workload helpers and imported sources

If you do not want to invent workloads by hand, [`include/jakal/workloads.hpp`](./include/jakal/workloads.hpp) exposes:

- `canonical_workload_presets()` for gaming, vision inference, and compact training-step surrogates
- `cpu_deep_learning_exploration_presets()` for host-heavy inference cases such as decode, KV-cache maintenance, and dequant pipelines
- `load_workload_source(path)` to load a native manifest or imported source file
- `load_workload_manifest(path)` as the explicit manifest entry point
- `compile_workload_graph(graph)` to index tensors and operations for downstream execution logic
- `normalize_workload_graph(graph)` to rebuild tensor lifetimes and dependency bookkeeping after manual edits

`default_workload_graph(workload)` in [`include/jakal/execution.hpp`](./include/jakal/execution.hpp) expands a `WorkloadSpec` into a `WorkloadGraph` with tensors, lifetimes, dependencies, and operation metadata.

The adapter and import coverage in the tree includes:

- native `.workload` manifests with optional `[asset]`, `[tensor]`, `[operation]`, and `[dependency]` sections
- `.onnx` graphs, including external data blobs
- `.gguf` weights, including shard discovery
- text-based imported descriptions used for PyTorch-export and GGML-style inputs

If you want to call `execute_manifest(...)` directly, a minimal graph-backed `.workload` file looks like this:

```ini
[workload]
name=manifest-exec
kind=inference
dataset_tag=manifest-exec-lite
phase=decode
working_set_bytes=8388608
host_exchange_bytes=1048576
estimated_flops=12000000
batch_size=1
latency_sensitive=true
prefer_unified_memory=true

[tensor]
id=input
bytes=16384
consumers=normalize
host_visible=true

[tensor]
id=hidden
bytes=16384
producer=normalize

[operation]
name=normalize
class=elementwise_map
extents=4096
input_bytes=16384
output_bytes=16384
estimated_flops=8192
parallelizable=true
streaming_friendly=true
inputs=input
outputs=hidden
```

If a manifest references required assets that are missing on disk, the managed path reports the failure, marks the run as not executed, and still writes telemetry.

There is also a small PyTorch bridge in [`scripts/export_torch_workload.py`](./scripts/export_torch_workload.py). It emits `pytorch_export` `.workload` files for built-in tensor-model presets such as:

- `finance-factor-risk-lite`
- `signal-filterbank-lite`
- `graph-ranking-lite`
- `scientific-solver-step-lite`

Example:

```powershell
python .\scripts\export_torch_workload.py --preset finance-factor-risk-lite --output .\finance-factor-risk-lite.workload
.\build\Debug\jakal_profile_manifest.exe .\finance-factor-risk-lite.workload --passes 3 --level-zero-only
.\build\Debug\jakal_directml_manifest_bench.exe .\finance-factor-risk-lite.workload --passes 3
```

### C API and shared runtime

The shared runtime target is `jakal_runtime`. Its C API lives in [`include/jakal/c_api.h`](./include/jakal/c_api.h) and mirrors the same runtime through a smaller, install-friendly ABI.

Core lifecycle and inspection:

- `jakal_core_runtime_create`
- `jakal_core_runtime_create_with_options`
- `jakal_core_runtime_destroy`
- `jakal_core_runtime_refresh`
- `jakal_core_runtime_get_install_paths`
- `jakal_core_runtime_backend_status_count`
- `jakal_core_runtime_get_backend_status`
- `jakal_core_runtime_device_count`
- `jakal_core_runtime_get_device`
- `jakal_core_runtime_graph_node_count`
- `jakal_core_runtime_get_graph_node`
- `jakal_core_runtime_graph_edge_count`
- `jakal_core_runtime_get_graph_edge`

Planning, optimization, and execution:

- `jakal_core_runtime_plan`
- `jakal_core_runtime_optimize`
- `jakal_core_runtime_execute`
- `jakal_core_runtime_execute_manifest`

The C API exposes install paths and backend readiness in addition to devices and execution reports. That is what `jakal_runtime_install_smoke` validates in the test suite.

### CLI

`src/jakal_core_cli.cpp` builds the packaged-runtime CLI.

| Command | What it does |
| --- | --- |
| `doctor [--host-only] [--json] [--runtime-root PATH]` | Prints or emits JSON for backend readiness, install paths, device summary, and recommended runtime setup |
| `devices [--host-only] [--runtime-root PATH]` | Prints discovered devices with a compact graph summary |
| `paths [--runtime-root PATH]` | Prints resolved install, config, cache, log, telemetry, and Python helper paths |
| `smoke [--host-only] [--runtime-root PATH]` | Runs a small built-in workload and reports success or failure |
| `run-manifest <path> [--host-only] [--runtime-root PATH]` | Loads a manifest or imported workload source and runs the managed execution path |

Examples:

```powershell
.\build\Debug\jakal_core_cli.exe doctor --json
.\build\Debug\jakal_core_cli.exe devices --host-only
.\build\Debug\jakal_core_cli.exe run-manifest .\finance-factor-risk-lite.workload --host-only
```

### Python helpers

The repository also ships lightweight Python wrappers in [`python/jakal_runtime.py`](./python/jakal_runtime.py) and [`python/jakal_app_adapter.py`](./python/jakal_app_adapter.py).

`jakal_runtime.py` loads `jakal_runtime.dll` or `libjakal_runtime.so` with `ctypes` and exposes:

- runtime creation with optional host-only defaults
- install-path queries
- backend status queries
- `optimize(...)`
- `execute_manifest(...)`

`jakal_app_adapter.py` is a small CLI on top of that wrapper:

```powershell
python .\python\jakal_app_adapter.py doctor --host-only
python .\python\jakal_app_adapter.py paths
python .\python\jakal_app_adapter.py optimize-smoke --host-only
python .\python\jakal_app_adapter.py run-manifest .\finance-factor-risk-lite.workload
```

### Cache files and installed layout

When you use `make_runtime_options_for_install(...)`, the runtime points its writable files at the resolved runtime directories instead of the old ad hoc temp-file naming scheme.

The important defaults are:

- `cache/planner-cache.tsv`
- `cache/execution-cache.tsv`
- `logs/runtime-telemetry.tsv`

The telemetry file now self-describes its schema in the header row, and the execution feedback caches (`.perf`, `.perf.family`) persist schema markers plus regression metadata so warm-state tuning and persisted-regression safety gates can survive process restarts.

If a runtime install root is available, `python_dir` resolves to `<install_root>/python`. Otherwise it falls back to `<writable_root>/python`.

The packaged runtime layout described in [`docs/distribution.md`](./docs/distribution.md) adds:

- `share/jakal-core/config/jakal-runtime-config.ini`
- `share/jakal-core/install/` and `share/jakal-core/install/prereqs/`
- `share/jakal-core/update/`
- `share/jakal-core/remove/`

## Examples

### Inspect discovered hardware and one sample workload

```powershell
.\build\Debug\jakal_inspect.exe
```

This prints:

- discovered hardware graphs
- graph nodes and edges
- ranked toolkit variants
- a sample placement plan
- optimization summaries
- direct execution results

### Inspect packaged-runtime health

```powershell
.\build\Debug\jakal_core_cli.exe doctor --json
```

This is the easiest way to inspect:

- resolved runtime paths
- backend readiness such as `ready-direct` or `ready-modeled`
- recommended setup presets such as `cpu-only`, `intel-level-zero`, `vulkan-runtime`, `opencl-fallback`, or `auto`

### Bootstrap install and self-check

```powershell
.\build\Debug\jakal_bootstrap.exe --status --self-check
```

This prints:

- install, update, and remove paths
- supported runtime backends detected on the machine
- active backends discovered by `jakal::Runtime`
- toolkit variants and the latest first-run self-check marker

### Profile canonical workload presets

```powershell
.\build\Debug\jakal_profile_workloads.exe
```

This runs the built-in gaming, inference, and training-style presets twice so you can compare cold and warm behavior and see what the learning cache changes.

### Profile imported manifests and model files

```powershell
.\build\Debug\jakal_profile_manifest.exe .\finance-factor-risk-lite.workload --passes 3 --host-only
```

You can also point it at imported model sources:

```powershell
.\build\Debug\jakal_profile_manifest.exe .\qwen2.5-0.5b.ollama.gguf --passes 3
.\build\Debug\jakal_profile_manifest.exe --ollama-model qwen2.5:0.5b --passes 3
```

Useful options include:

- `--host-only`
- `--level-zero-only`
- `--opencl-only`
- `--show-ops`
- `--partition-strategy auto_balanced|blind_sharded|role_split|reduce_on_gpu|projection_sharded|tpu_like`
- `--tuning-profile default|host-latency|hybrid-balanced|accelerator-throughput|cooperative-split`
- `--graph-rewrite-level N`
- `--graph-passes N`
- `--state key=value`

For repeated comparisons across backends, [`scripts/run_manifest_benchmarks.py`](./scripts/run_manifest_benchmarks.py) wraps:

- `jakal_profile_manifest` in `host` and `level-zero` modes
- optional `opencl` mode
- `jakal_directml_manifest_bench` when the executable is available

`jakal_directml_manifest_bench` is a standalone DirectML operator baseline. It is useful for backend-to-backend kernel checks, but it is not the same thing as the managed runtime total reported by `jakal_profile_manifest`.

Examples:

```powershell
python .\scripts\run_manifest_benchmarks.py --all-torch-presets --passes 3
python .\scripts\run_manifest_benchmarks.py --manifest .\qwen2.5-0.5b.ollama.gguf --passes 3
```

### Explore CPU-heavy deep-learning placement

By default, this example optimizes one preset without executing it:

```powershell
.\build\Debug\jakal_explore_cpu_dl.exe
```

Run the executor as well:

```powershell
.\build\Debug\jakal_explore_cpu_dl.exe --execute
```

Run every preset:

```powershell
.\build\Debug\jakal_explore_cpu_dl.exe --all --execute
```

You can also pass a preset name or dataset tag:

```powershell
.\build\Debug\jakal_explore_cpu_dl.exe llm-decode-token-lite --execute
```

The output includes:

- selected devices and ratios from the plan
- predicted transfer volume
- per-operation strategy and partition counts
- backend counts when execution is enabled

### Compare role-aware partition strategies

This example needs both a host device and at least one accelerator:

```powershell
.\build\Debug\jakal_partition_roles.exe
```

You can also pass a CPU-deep-learning preset name or dataset tag:

```powershell
.\build\Debug\jakal_partition_roles.exe llm-decode-token-lite
```

It replays the same workload through several hand-picked strategies such as `role-split`, `reduce-on-gpu`, and `projection-sharded-4`, then prints per-operation device placement, partition counts, backend choices, and runtime comparisons.

### Run the standalone tests directly

You can run the binaries directly when you want something narrower than full `ctest`:

```powershell
.\build\Debug\jakal_smoke.exe
.\build\Debug\jakal_optimization.exe --fast
.\build\Debug\jakal_runtime_install_smoke.exe
.\build\Debug\jakal_workload_bench.exe --smoke --host-only
```

Other useful standalone test binaries include:

- `jakal_planner_learning` for strategy-learning cache behavior
- `jakal_partition_strategies` for explicit partition strategy coverage
- `jakal_runtime_product` for managed execution, memory gates, manifests, and telemetry behavior
- `jakal_workload_import_adapters` for ONNX, GGUF, GGML-style, and PyTorch-export import coverage
- `jakal_backend_contracts` and `jakal_live_backend_smoke` for backend probing and execution checks

`jakal_preset_execution_diag` is built with tests enabled, but it is not currently registered with `ctest`.

## Further reading

- [`docs/distribution.md`](./docs/distribution.md) for install tree and package layout details
- [`docs/release.md`](./docs/release.md) for release automation and tag-based publishing
- [`examples/inspect_runtime.cpp`](./examples/inspect_runtime.cpp) for a fuller C++ walkthrough
- [`examples/profile_workloads.cpp`](./examples/profile_workloads.cpp) for preset profiling
- [`examples/profile_manifest.cpp`](./examples/profile_manifest.cpp) for manifest and imported-model profiling
- [`examples/explore_cpu_dl.cpp`](./examples/explore_cpu_dl.cpp) for host-versus-accelerator experiments
- [`examples/partition_roles.cpp`](./examples/partition_roles.cpp) for role-aware host/GPU partition experiments
- [`tests/runtime_product.cpp`](./tests/runtime_product.cpp) for managed execution, manifest parsing, asset prefetch, and telemetry behavior
- [`tests/runtime_install_smoke.cpp`](./tests/runtime_install_smoke.cpp) for the shared-runtime install-path and backend-status smoke test
- [`tests/workload_import_adapters.cpp`](./tests/workload_import_adapters.cpp) for imported workload-source coverage
- [`python/jakal_runtime.py`](./python/jakal_runtime.py) and [`python/jakal_app_adapter.py`](./python/jakal_app_adapter.py) for the lightweight Python surface

## License

MIT. See [`LICENSE`](./LICENSE).
