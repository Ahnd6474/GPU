#include "gpu/runtime.hpp"

#include <iostream>

int main() {
    gpu::RuntimeOptions options;
    options.enable_opencl_probe = false;
    gpu::Runtime runtime(options);

    if (runtime.devices().empty()) {
        std::cerr << "No hardware graphs discovered.\n";
        return 1;
    }

    for (const auto& graph : runtime.devices()) {
        if (graph.nodes.empty()) {
            std::cerr << "Discovered graph without nodes.\n";
            return 1;
        }
        if (graph.edges.empty()) {
            std::cerr << "Discovered graph without edges.\n";
            return 1;
        }
    }

    const gpu::WorkloadSpec workload{
        "smoke",
        gpu::WorkloadKind::tensor,
        "",
        128ull * 1024ull * 1024ull,
        64ull * 1024ull * 1024ull,
        2.0e11,
        4,
        false,
        false,
        true};

    const auto plan = runtime.plan(workload);
    if (plan.allocations.empty()) {
        std::cerr << "No plan allocations created.\n";
        return 1;
    }

    const auto report = runtime.optimize(workload);
    if (report.operations.empty()) {
        std::cerr << "No execution optimizations created.\n";
        return 1;
    }
    if (report.workload_graph.operations.empty() ||
        report.workload_graph.tensors.empty() ||
        report.workload_graph.lifetimes.empty()) {
        std::cerr << "Workload graph metadata missing.\n";
        return 1;
    }
    for (const auto& operation : report.operations) {
        if (operation.graph.residency_plan.empty() || operation.graph.peak_resident_bytes == 0) {
            std::cerr << "Execution residency metadata missing.\n";
            return 1;
        }
    }

    const auto executed = runtime.execute(workload);
    if (executed.operations.empty() || !executed.all_succeeded) {
        std::cerr << "Direct execution failed.\n";
        return 1;
    }

    std::cout << "Graphs=" << runtime.devices().size()
              << " allocations=" << plan.allocations.size()
              << " operations=" << report.operations.size()
              << " executed=" << executed.operations.size()
              << " cache=" << (plan.loaded_from_cache ? "hit" : "miss")
              << '\n';

    return 0;
}
