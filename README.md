# GPU

Graph-first heterogeneous compute runtime skeleton for C++20.

This repository is a small CMake library with examples and tests. It discovers host and OpenCL devices, turns them into structural graphs, builds placement plans from those graphs, and runs a compact set of direct kernels to validate the planner.

It is still an early runtime skeleton, not a production execution stack.

## installation

### requirements

- CMake 3.20 or newer
- A C++20 compiler
- An OpenCL runtime or driver if you want OpenCL device discovery and direct OpenCL execution

The project does not need CUDA, Level Zero, or Vulkan SDKs to build. Those backends appear in the toolkit-ranking layer, but native execution for them is not wired up in this tree yet.

### build the project

```powershell
# if you have not cloned the repository yet
# git clone <repo-url>
# cd GPU

cmake -S . -B build
cmake --build build
```

The default build creates:

- `gpu_runtime` as a static library
- `gpu_inspect` and `gpu_profile_workloads` example executables
- `gpu_smoke` and `gpu_optimization` test executables

Optional CMake switches:

```powershell
cmake -S . -B build -DGPU_BUILD_EXAMPLES=OFF -DGPU_BUILD_TESTS=OFF
```

## quick start

If you want to use the library from another CMake project, add this repository as a subdirectory and link `gpu::runtime`.

```cmake
add_subdirectory(path/to/GPU)
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

`gpu::Runtime` refreshes hardware during construction, so `devices()` is ready immediately unless you disable all probes.

To inspect the discovered graph and a sample plan from this repository directly:

```powershell
.\build\gpu_inspect.exe
```

On Unix-like shells, run `./build/gpu_inspect` instead.

## what is this repository?

The runtime tries to reason about hardware from structure instead of from labels like "CPU", "GPU", or vendor names. Each discovered device becomes a graph of compute, storage, transfer, and control objects. The planner and execution layers work from that graph and from a workload description.

Right now the library covers four main pieces:

- hardware discovery for the host and OpenCL devices
- graph summarization and cost materialization
- workload planning and execution-graph generation
- direct execution and lightweight validation for a small operation set

The operation set in the current tree includes:

- elementwise map
- reduction
- blocked matmul
- 3x3 convolution
- bilinear resample

Canonical workload presets are also built in for:

- gaming upscaling and post-processing
- vision-style inference
- compact training-step surrogates

## why graph-first planning?

Flat device labels lose the details that actually matter for placement. The planner in this repository scores things like execution width, memory capacity, host-link bandwidth, dispatch cost, synchronization cost, and exposed hierarchy. That makes it possible to talk about:

- how much work should stay on the host
- when unified or coherent memory should matter
- whether sharding is worth the transfer cost
- which structural nodes should be mapped for a given workload

That is the idea being tested here. The code is more useful as an executable model of the architecture than as a finished runtime.

## current status

Implemented today:

- C++20 runtime core in [`include/gpu`](./include/gpu) and [`src`](./src)
- host probe and OpenCL probe
- structural hardware graph with weighted edges
- planner cache and execution cache persistence
- workload graph generation with tensors, dependencies, and lifetimes
- execution-graph construction with residency and transfer schedules
- system-profile-aware optimization policies
- direct host execution
- direct OpenCL execution for the current operation set
- C ABI for device, node, edge, and planning inspection
- examples and tests

Not finished yet:

- real tensor residency and allocator management
- native Level Zero, CUDA, and Vulkan execution backends
- framework bridges for Torch, TensorFlow, or similar runtimes
- full inter-device transfer scheduling backed by native transports
- packaging and install rules for downstream consumption

One detail worth calling out: the toolkit-ranking layer can score OpenCL, Level Zero, CUDA, and Vulkan variants. The runtime probes in this repository still come from the host and OpenCL, and the non-OpenCL GPU backends are currently model-driven rather than native integrations.

## api

### C++ entry points

The main public headers are:

- [`include/gpu/runtime.hpp`](./include/gpu/runtime.hpp)
- [`include/gpu/planner.hpp`](./include/gpu/planner.hpp)
- [`include/gpu/execution.hpp`](./include/gpu/execution.hpp)
- [`include/gpu/workloads.hpp`](./include/gpu/workloads.hpp)
- [`include/gpu/c_api.h`](./include/gpu/c_api.h)

`gpu::Runtime` is the main entry point:

| Method | What it does |
| --- | --- |
| `devices()` | Returns the discovered hardware graphs |
| `gpu_toolkit_index()` | Returns ranked backend variants per discovered device |
| `plan(workload)` | Builds or loads a cached placement plan |
| `optimize(workload)` | Builds workload and execution graphs, then picks execution settings |
| `execute(workload)` | Runs the optimized configuration and records validation feedback |

`gpu::RuntimeOptions` lets you:

- disable the host probe or OpenCL probe
- override the plan cache path
- override the execution cache path

`gpu::WorkloadSpec` is the main planning input. It includes:

- workload kind
- working-set size
- host exchange size
- estimated flops
- batch size
- latency sensitivity
- unified-memory preference
- matrix-friendly hint

If you do not want to invent workloads by hand, [`include/gpu/workloads.hpp`](./include/gpu/workloads.hpp) exposes `canonical_workload_presets()`.

### C API

The C API in [`include/gpu/c_api.h`](./include/gpu/c_api.h) exposes a smaller surface:

- `gpu_runtime_create` and `gpu_runtime_destroy`
- `gpu_runtime_refresh`
- `gpu_runtime_device_count` and `gpu_runtime_get_device`
- `gpu_runtime_graph_node_count` and `gpu_runtime_get_graph_node`
- `gpu_runtime_graph_edge_count` and `gpu_runtime_get_graph_edge`
- `gpu_runtime_plan`

`gpu_runtime_plan` is currently a planning API, not a full execution API.

For `gpu_workload_spec.kind`, the C layer currently recognizes these strings:

- `custom`
- `inference`
- `image`
- `tensor`

Other values fall back to `custom`.

### cache files

By default the runtime writes lightweight TSV caches to the system temp directory:

- `gpu_runtime_plan_cache.tsv`
- `gpu_runtime_execution_cache.tsv`
- `gpu_runtime_execution_cache.tsv.perf`

You can redirect those files through `gpu::RuntimeOptions`.

## examples

### inspect discovered hardware and a sample workload

```powershell
.\build\gpu_inspect.exe
```

This prints:

- discovered hardware graphs
- graph nodes and edges
- ranked toolkit variants
- a sample plan
- optimization and direct-execution summaries

The output can get large on machines with detailed OpenCL graphs.

### run canonical workload presets

```powershell
.\build\gpu_profile_workloads.exe
```

This example runs the built-in gaming, inference, and training-style presets and prints warm-versus-cold execution summaries. It is the slowest example in the repository because it executes each preset more than once to exercise the learning cache.

### run the tests

If `ctest` is available in your shell:

```powershell
ctest --test-dir build --output-on-failure
```

Or run the test executables directly:

```powershell
.\build\gpu_smoke.exe
.\build\gpu_optimization.exe
```

A recent `gpu_smoke` run ended with:

```text
Graphs=1 allocations=1 operations=5 executed=5 cache=hit
```

A recent `gpu_optimization` run ended with:

```text
operations=5 cached=yes graphs=1 graph_passes=6
```

Treat those numbers as sanity checks, not fixed golden values across every machine.

## further reading

- [`examples/inspect_runtime.cpp`](./examples/inspect_runtime.cpp) for a fuller C++ walkthrough
- [`examples/profile_workloads.cpp`](./examples/profile_workloads.cpp) for preset profiling
- [`tests/smoke.cpp`](./tests/smoke.cpp) for the smallest end-to-end path
- [`tests/optimization.cpp`](./tests/optimization.cpp) for cache, optimization, and execution assertions

## license

MIT. See [`LICENSE`](./LICENSE).
