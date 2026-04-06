# Jakal-Core

Graph-first heterogeneous compute runtime skeleton for C++20.

This repository is `Jakal-Core`. The CMake project and library target are still named `GPU` and `gpu::runtime`.

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
  - [C API](#c-api)
  - [Cache files](#cache-files)
- [Examples](#examples)
- [Further reading](#further-reading)
- [License](#license)

## Installation

Jakal-Core is source-first right now. There are no install rules or published packages in this tree yet, so the normal way to use it is to build from source or add it to another CMake project with `add_subdirectory(...)`.

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
```

The default build produces:

- `gpu_runtime`
- `gpu_inspect`
- `gpu_profile_workloads`
- `gpu_explore_cpu_dl`
- `gpu_smoke`
- `gpu_optimization`

Useful CMake switches:

```powershell
cmake -S . -B build -DGPU_BUILD_EXAMPLES=OFF -DGPU_BUILD_TESTS=OFF
```

If you are using a multi-config generator such as Visual Studio, CTest also needs a configuration name:

```powershell
ctest --test-dir build -C Debug --output-on-failure
```

On single-config generators such as Ninja or Unix Makefiles, the `-C Debug` part is not needed.

## Quick start

If you want to consume the library from another CMake project, add this repository as a subdirectory and link `gpu::runtime`.

```cmake
add_subdirectory(path/to/Jakal-Core)
target_link_libraries(my_app PRIVATE gpu::runtime)
```

Minimal C++ example:

```cpp
#include "gpu/runtime.hpp"

#include <iostream>

int main() {
    gpu::Runtime runtime;

    for (const auto& graph : runtime.devices()) {
        std::cout << graph.presentation_name << '\n';
    }

    return 0;
}
```

`gpu::Runtime` refreshes hardware during construction, so `devices()` is ready right away unless you explicitly disable every probe.

If you just want to see what the repository does without writing code, build the tree and run:

```powershell
.\build\Debug\gpu_inspect.exe
```

On single-config generators, the executable is typically `./build/gpu_inspect`.

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
- Native backend coverage is uneven. Missing native kernels fall back to host execution rather than pretending the backend is complete.
- The toolkit-ranking layer and planner can reason about more backend variants than the direct executor can run end to end.

What is still missing:

- real tensor residency management and allocators
- framework bridges for PyTorch, TensorFlow, or similar runtimes
- packaging and install rules for downstream consumers
- a stable production execution stack with mature backend coverage

## API

### C++ entry points

The main public headers are:

- [`include/gpu/runtime.hpp`](./include/gpu/runtime.hpp)
- [`include/gpu/planner.hpp`](./include/gpu/planner.hpp)
- [`include/gpu/execution.hpp`](./include/gpu/execution.hpp)
- [`include/gpu/workloads.hpp`](./include/gpu/workloads.hpp)
- [`include/gpu/c_api.h`](./include/gpu/c_api.h)

`gpu::Runtime` is the main entry point.

| Method | What it does |
| --- | --- |
| `refresh_hardware()` | Re-runs device discovery and rebuilds the toolkit index |
| `devices()` | Returns discovered hardware graphs |
| `gpu_toolkit_index()` | Returns ranked backend variants per discovered device |
| `plan(workload)` | Builds or loads a cached placement plan |
| `optimize(workload)` | Builds workload and execution graphs, then picks execution settings |
| `execute(workload)` | Runs the selected execution path and feeds the results back into the optimizer |

`gpu::RuntimeOptions` lets you:

- enable or disable host, OpenCL, Level Zero, CUDA, and ROCm probes
- override the plan cache path
- override the execution cache path

`gpu::WorkloadSpec` is the main planning input.

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | `std::string` | Human-readable workload name |
| `kind` | `gpu::WorkloadKind` | Broad workload class such as `inference` or `training` |
| `dataset_tag` | `std::string` | Optional preset or dataset identifier |
| `working_set_bytes` | `std::uint64_t` | Estimated active working set |
| `host_exchange_bytes` | `std::uint64_t` | Estimated host-device exchange volume |
| `estimated_flops` | `double` | Approximate compute demand |
| `batch_size` | `std::uint32_t` | Batch size hint |
| `latency_sensitive` | `bool` | Whether latency should be favored over throughput |
| `prefer_unified_memory` | `bool` | Whether unified memory should get extra weight |
| `matrix_friendly` | `bool` | Whether the workload looks friendly to GEMM-style hardware |

### Workload helpers

If you do not want to invent workloads by hand, [`include/gpu/workloads.hpp`](./include/gpu/workloads.hpp) exposes two helper sets:

- `canonical_workload_presets()` for gaming, vision inference, and compact training-step surrogates
- `cpu_deep_learning_exploration_presets()` for host-heavy inference cases such as decode, KV-cache maintenance, and dequant pipelines

`default_workload_graph(workload)` expands a `WorkloadSpec` into a `WorkloadGraph` with tensors, lifetimes, dependencies, and operation metadata.

### C API

The C API in [`include/gpu/c_api.h`](./include/gpu/c_api.h) mirrors the same runtime in a smaller surface.

Core lifecycle and inspection:

- `gpu_runtime_create`
- `gpu_runtime_destroy`
- `gpu_runtime_refresh`
- `gpu_runtime_device_count`
- `gpu_runtime_get_device`
- `gpu_runtime_graph_node_count`
- `gpu_runtime_get_graph_node`
- `gpu_runtime_graph_edge_count`
- `gpu_runtime_get_graph_edge`

Planning, optimization, and execution:

- `gpu_runtime_plan`
- `gpu_runtime_optimize`
- `gpu_runtime_execute`

Accepted `gpu_workload_spec.kind` strings are:

- `custom`
- `inference`
- `image`
- `tensor`
- `gaming`
- `training`

The array-returning functions follow the usual "capacity plus out-count" pattern. Pass a buffer and its capacity, and the function writes the number of required entries to `out_count`. If the buffer is missing or too small, the function returns an error code after telling you how many entries were needed.

### Cache files

By default the runtime writes lightweight TSV caches to the system temp directory:

- `gpu_runtime_plan_cache.tsv`
- `gpu_runtime_execution_cache.tsv`
- `gpu_runtime_execution_cache.tsv.perf`

You can redirect those files through `gpu::RuntimeOptions`.

## Examples

### Inspect discovered hardware and one sample workload

```powershell
.\build\Debug\gpu_inspect.exe
```

This prints:

- discovered hardware graphs
- graph nodes and edges
- ranked toolkit variants
- a sample placement plan
- optimization summaries
- direct execution results

### Profile canonical workload presets

```powershell
.\build\Debug\gpu_profile_workloads.exe
```

This runs the built-in gaming, inference, and training-style presets twice so you can compare cold and warm behavior and see what the learning cache changes.

### Explore CPU-heavy deep-learning placement

By default, this example optimizes one preset without executing it:

```powershell
.\build\Debug\gpu_explore_cpu_dl.exe
```

Run the executor as well:

```powershell
.\build\Debug\gpu_explore_cpu_dl.exe --execute
```

Run every preset:

```powershell
.\build\Debug\gpu_explore_cpu_dl.exe --all --execute
```

You can also pass a preset name or dataset tag:

```powershell
.\build\Debug\gpu_explore_cpu_dl.exe llm-decode-token-lite --execute
```

The output includes:

- selected devices and ratios from the plan
- predicted transfer volume
- per-operation strategy and partition counts
- backend counts when execution is enabled

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
.\build\Debug\gpu_smoke.exe
.\build\Debug\gpu_optimization.exe
```

## Further reading

- [`examples/inspect_runtime.cpp`](./examples/inspect_runtime.cpp) for a fuller C++ walkthrough
- [`examples/profile_workloads.cpp`](./examples/profile_workloads.cpp) for preset profiling
- [`examples/explore_cpu_dl.cpp`](./examples/explore_cpu_dl.cpp) for host-versus-accelerator experiments
- [`examples/compare_host_workloads.cpp`](./examples/compare_host_workloads.cpp) for an extra profiling utility that is present in the source tree but not wired into the default CMake targets
- [`tests/smoke.cpp`](./tests/smoke.cpp) for the smallest end-to-end path
- [`tests/optimization.cpp`](./tests/optimization.cpp) for graph, cache, and backend coverage checks

## License

MIT. See [`LICENSE`](./LICENSE).
