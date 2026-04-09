# Jakal-Core

Graph-first heterogeneous compute runtime skeleton for C++20.

This repository builds the `jakal_core` library target, publishes headers under `include/jakal`, and includes runnable examples and tests around the runtime skeleton.

Jakal-Core is not a production runtime yet. It is a small CMake library with examples and tests that:

- discovers host and accelerator hardware
- turns that hardware into structural graphs
- builds placement and execution plans from those graphs
- runs a compact set of direct kernels to check whether those plans make sense on a real machine

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [What is this repository?](#what-is-this-repository)
- [Why graph-first planning?](#why-graph-first-planning)
- [Current scope](#current-scope)
- [API](#api)
  - [C++ entry points](#c-entry-points)
  - [Workload helpers](#workload-helpers)
  - [Managed execution and manifests](#managed-execution-and-manifests)
  - [C API](#c-api)
  - [Cache files](#cache-files)
- [Examples](#examples)
- [Further reading](#further-reading)
- [License](#license)

## Installation

Jakal-Core is still source-first, but the tree now also exposes local install rules, exported CMake package files, and CPack package generation for downstream use.

### Requirements

- CMake 3.20 or newer
- A C++20 compiler
- An OpenCL runtime or driver if you want OpenCL discovery and direct OpenCL execution
- Optional runtime libraries for Level Zero, CUDA, or ROCm if you want those probes and native backends to activate automatically

The project loads accelerator runtimes dynamically. If a given runtime library is not present, the corresponding probe simply stays inactive.

### Build the project

From the repository root:

```powershell
cmake -S . -B build
cmake --build build
cmake --install build --config Release
```

The default build produces the `jakal_core` library plus these executables when product tools, examples, and tests are enabled:

- tools: `jakal_bootstrap`
- examples: `jakal_inspect`, `jakal_profile_workloads`, `jakal_explore_cpu_dl`, `jakal_partition_roles`
- test binaries: `jakal_smoke`, `jakal_optimization`, `jakal_planner_learning`, `jakal_partition_strategies`, `jakal_runtime_product`, `jakal_workload_import_adapters`, `jakal_backend_contracts`, `jakal_live_backend_smoke`, `jakal_preset_execution_diag`

Useful CMake switches:

```powershell
cmake -S . -B build -DJAKAL_CORE_BUILD_EXAMPLES=OFF -DJAKAL_CORE_BUILD_TESTS=OFF
```

To generate a distributable package:

```powershell
cmake -S . -B build -DJAKAL_CORE_BUILD_TESTS=OFF
cmake --build build --config Release --target package
```

If you are using a multi-config generator such as Visual Studio, CTest also needs a configuration name:

```powershell
ctest --test-dir build -C Debug --output-on-failure
```

On single-config generators such as Ninja or Unix Makefiles, the `-C Debug` part is not needed.

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

`jakal::Runtime` refreshes hardware during construction, so `devices()` is ready right away unless you explicitly disable every probe.

If you just want to see what the repository does without writing code, build the tree and run:

```powershell
.\build\Debug\jakal_inspect.exe
```

On single-config generators, the executable is typically `./build/jakal_inspect`.

For install and first-run diagnostics, the bootstrap tool is:

```powershell
.\build\Debug\jakal_bootstrap.exe --status --self-check
```

## What is this repository?

Jakal-Core treats hardware as a graph instead of a label. A discovered CPU, OpenCL device, CUDA device, or Level Zero device becomes a set of compute, storage, control, and transfer nodes with weighted edges between them. The planner then works from that structure when it decides where a workload should land.

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

There are also built-in workload presets for:

- gaming-style upscaling and post-processing
- vision-style inference
- compact training-step surrogates
- CPU-heavy deep-learning exploration cases such as token decode, KV-cache updates, and dequant staging

## Why graph-first planning?

Flat device labels hide the details that actually drive placement. "GPU" does not tell you whether the device has unified memory, what its host link looks like, how much dispatch latency it carries, or how much structure it exposes for mapping work.

The planner in this repository scores things like:

- execution width and resident contexts
- matrix units and numeric capability flags
- directly attached memory and shared host-visible memory
- host-link bandwidth
- dispatch, synchronization, and transfer costs
- graph shape, not just vendor or backend name

That lets the runtime ask more useful questions:

- should a latency-sensitive decode stage stay on the host?
- is unified memory worth preferring for this workload?
- is sharding worth the transfer cost?
- which structural nodes should be mapped for a given operation?

That is the point of this codebase right now. It is closer to an executable model of the runtime architecture than a finished execution stack.

## Current scope

This is where the repository actually stands today:

- Discovery can use host, OpenCL, Level Zero, CUDA, and ROCm probes when the matching runtime libraries are present.
- Planning and optimization work across those discovered graphs.
- The direct executor can run host kernels, OpenCL kernels, and some native Level Zero, CUDA, and ROCm kernels.
- The managed execution layer can run native workload manifests, do memory preflight checks, emit residency and asset-prefetch plans, and write TSV telemetry.
- `load_workload_source(...)` can build workload graphs from native manifests and imported model descriptions such as ONNX and GGUF.
- Native backend coverage is uneven. Missing native kernels fall back to host execution rather than pretending the backend is complete.
- The toolkit-ranking layer and planner can reason about more backend variants than the direct executor can run end to end.

What is still missing:

- real tensor allocators and residency movement beyond the current planning and diagnostics layer
- framework bridges for PyTorch, TensorFlow, or similar runtimes
- polished production packaging beyond the current local install and CPack flow
- a stable production execution stack with mature backend coverage

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
| `refresh_hardware()` | Re-runs device discovery and rebuilds the toolkit index |
| `devices()` | Returns discovered hardware graphs |
| `jakal_toolkit_index()` | Returns ranked backend variants per discovered device |
| `plan(workload)` | Builds or loads a cached placement plan |
| `optimize(workload)` | Builds workload and execution graphs, then picks execution settings |
| `optimize(workload, workload_graph)` | Optimizes an explicit workload graph instead of generating the default one |
| `execute(workload)` | Runs the selected execution path and feeds the results back into the optimizer |
| `execute_managed(workload)` | Runs placement, optimization, safety gates, and direct execution with extra diagnostics |
| `execute_managed(workload, workload_graph)` | Managed execution for a caller-supplied workload graph |
| `execute_manifest(path)` | Loads a manifest or imported workload source, then runs the managed path |

`jakal::RuntimeOptions` lets you:

- enable or disable host, OpenCL, Level Zero, CUDA, and ROCm probes
- prefer Level Zero over OpenCL when both match the same hardware
- override the plan cache path
- override the execution cache path
- tune memory, safety, and observability policy through `RuntimeProductPolicy`

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

### Workload helpers

If you do not want to invent workloads by hand, [`include/jakal/workloads.hpp`](./include/jakal/workloads.hpp) exposes two helper sets:

- `canonical_workload_presets()` for gaming, vision inference, and compact training-step surrogates
- `cpu_deep_learning_exploration_presets()` for host-heavy inference cases such as decode, KV-cache maintenance, and dequant pipelines

`default_workload_graph(workload)` expands a `WorkloadSpec` into a `WorkloadGraph` with tensors, lifetimes, dependencies, and operation metadata.

The same header also exposes:

- `load_workload_source(path)` to load a native manifest or an imported source file
- `load_workload_manifest(path)` as the explicit manifest entry point
- `normalize_workload_graph(graph)` to rebuild tensor lifetimes and dependency bookkeeping after manual edits

The adapter tests currently cover:

- native `.workload` manifests with optional `[asset]`, `[tensor]`, `[operation]`, and `[dependency]` sections
- `.onnx` graphs, including external data blobs
- `.gguf` weights, including shard discovery
- text-based imported descriptions used for PyTorch-export and GGML-style inputs

There is also a small PyTorch bridge in [`scripts/export_torch_workload.py`](./scripts/export_torch_workload.py).
It emits `pytorch_export` `.workload` files for built-in tensor-model presets such as:

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

### Managed execution and manifests

`jakal::RuntimeProductPolicy` is the part of `RuntimeOptions` that controls:

- memory reserve ratios and preflight blocking
- planner confidence and rollback gates
- telemetry persistence

`ManagedExecutionReport` adds these diagnostics on top of the direct execution report:

- resolved placement plan and planner confidence
- memory preflight and spill predictions
- kernel coverage checks
- asset prefetch entries
- residency sequence actions
- telemetry output path

If you want to run a manifest directly, `execute_manifest(...)` accepts both a spec-only manifest and a graph-backed manifest. A minimal graph-backed example looks like this:

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

[asset]
id=weights-shard
path=weights.bin
tensor_ids=weights
preload_required=true
persistent=true
host_visible=true

[tensor]
id=input
bytes=16384
consumers=normalize
host_visible=true

[tensor]
id=weights
bytes=16384
consumers=normalize
persistent=true
host_visible=true

[tensor]
id=hidden
bytes=16384
producer=normalize
consumers=score

[operation]
name=normalize
class=elementwise_map
extents=4096
input_bytes=32768
output_bytes=16384
estimated_flops=8192
parallelizable=true
streaming_friendly=true
inputs=input,weights
outputs=hidden
```

If `weights.bin` is required and missing, `execute_manifest(...)` returns a managed report with `asset_prefetch.missing_required_assets=true`, marks the run as not executed, and still writes telemetry.

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

### C API

The C API in [`include/jakal/c_api.h`](./include/jakal/c_api.h) mirrors the same runtime in a smaller surface.

Core lifecycle and inspection:

- `jakal_core_runtime_create`
- `jakal_core_runtime_destroy`
- `jakal_core_runtime_refresh`
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

The C API currently stops at the direct execution layer. Managed execution, manifest loading, asset-prefetch planning, and telemetry controls are C++-only for now.

Accepted `jakal_core_workload_spec.kind` strings are:

- `custom`
- `inference`
- `image`
- `tensor`
- `gaming`
- `training`

The array-returning functions follow the usual "capacity plus out-count" pattern. Pass a buffer and its capacity, and the function writes the number of required entries to `out_count`. If the buffer is missing or too small, the function returns an error code after telling you how many entries were needed.

### Cache files

By default the runtime writes lightweight TSV caches to the system temp directory:

- `jakal_core_plan_cache.tsv`
- `jakal_core_execution_cache.tsv`
- `jakal_core_execution_cache.tsv.perf`
- `jakal_core_runtime_telemetry.tsv`

You can redirect those files through `jakal::RuntimeOptions`.

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

### Run the tests

For multi-config generators such as Visual Studio:

```powershell
ctest --test-dir build -C Debug --output-on-failure
```

For single-config generators:

```powershell
ctest --test-dir build --output-on-failure
```

You can also run the binaries directly:

```powershell
.\build\Debug\jakal_smoke.exe
.\build\Debug\jakal_optimization.exe
```

Other useful standalone test binaries include:

- `jakal_planner_learning` for strategy-learning cache behavior
- `jakal_partition_strategies` for explicit partition strategy coverage
- `jakal_runtime_product` for managed execution, memory gates, manifests, and telemetry
- `jakal_workload_import_adapters` for ONNX, GGUF, GGML-style, and PyTorch-export import coverage
- `jakal_backend_contracts` and `jakal_live_backend_smoke` for backend probing and execution checks

`jakal_preset_execution_diag` is built with tests enabled, but it is not currently registered with `ctest`.

## Further reading

- [`examples/inspect_runtime.cpp`](./examples/inspect_runtime.cpp) for a fuller C++ walkthrough
- [`examples/profile_workloads.cpp`](./examples/profile_workloads.cpp) for preset profiling
- [`examples/explore_cpu_dl.cpp`](./examples/explore_cpu_dl.cpp) for host-versus-accelerator experiments
- [`examples/partition_roles.cpp`](./examples/partition_roles.cpp) for role-aware host/GPU partition experiments
- [`examples/compare_host_workloads.cpp`](./examples/compare_host_workloads.cpp) for an extra profiling utility that is present in the source tree but not wired into the default CMake targets
- [`tests/smoke.cpp`](./tests/smoke.cpp) for the smallest end-to-end path
- [`tests/optimization.cpp`](./tests/optimization.cpp) for graph, cache, and backend coverage checks
- [`tests/runtime_product.cpp`](./tests/runtime_product.cpp) for managed execution, manifest parsing, asset prefetch, and telemetry behavior
- [`tests/workload_import_adapters.cpp`](./tests/workload_import_adapters.cpp) for imported workload-source coverage

## License

MIT. See [`LICENSE`](./LICENSE).

