#include "gpu/device.hpp"
#include "gpu/runtime.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>

namespace {

std::filesystem::path unique_temp_file(const std::string& stem) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (stem + "-" + std::to_string(nonce) + ".tsv");
}

bool verify_cost_propagation() {
    gpu::HardwareGraph graph;
    graph.uid = "manual";
    graph.probe = "test";
    graph.presentation_name = "manual";

    graph.nodes.push_back({"root", "root", "", gpu::HardwareObjectDomain::control, gpu::HardwareObjectRole::root});
    graph.nodes.push_back({"queue", "queue", "root", gpu::HardwareObjectDomain::control, gpu::HardwareObjectRole::queue});
    graph.nodes.push_back({"tile", "tile", "root", gpu::HardwareObjectDomain::compute, gpu::HardwareObjectRole::tile});
    graph.nodes.push_back({"lane", "lane", "tile", gpu::HardwareObjectDomain::compute, gpu::HardwareObjectRole::lane});
    graph.nodes.back().compute.execution_width = 32;
    graph.nodes.push_back({"mem", "mem", "root", gpu::HardwareObjectDomain::storage, gpu::HardwareObjectRole::global_memory});

    graph.edges.push_back({"root", "tile", gpu::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "queue", gpu::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"tile", "lane", gpu::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"queue", "tile", gpu::GraphEdgeSemantics::dispatches, true, 0.0, 0.0, 3.0});
    graph.edges.push_back({"mem", "tile", gpu::GraphEdgeSemantics::feeds, true, 0.0, 128.0, 1.0});
    graph.edges.push_back({"lane", "mem", gpu::GraphEdgeSemantics::feeds, true, 0.0, 512.0, 0.5});

    gpu::materialize_graph_costs(graph);

    double contains_weight = 0.0;
    double dispatch_weight = 0.0;
    for (const auto& edge : graph.edges) {
        if (edge.source_id == "root" && edge.target_id == "tile" && edge.semantics == gpu::GraphEdgeSemantics::contains) {
            contains_weight = edge.weight;
        }
        if (edge.source_id == "queue" && edge.target_id == "tile" && edge.semantics == gpu::GraphEdgeSemantics::dispatches) {
            dispatch_weight = edge.weight;
        }
    }

    return contains_weight > 0.0 && dispatch_weight > 3.0;
}

}  // namespace

int main() {
    const auto plan_cache = unique_temp_file("gpu-runtime-plan-test");
    const auto exec_cache = unique_temp_file("gpu-runtime-exec-test");

    gpu::RuntimeOptions options;
    options.cache_path = plan_cache;
    options.execution_cache_path = exec_cache;

    gpu::Runtime runtime(options);

    if (runtime.devices().empty()) {
        std::cerr << "No hardware graphs discovered.\n";
        return 1;
    }

    const gpu::WorkloadSpec workload{
        "optimization-suite",
        gpu::WorkloadKind::tensor,
        512ull * 1024ull * 1024ull,
        128ull * 1024ull * 1024ull,
        4.0e12,
        8,
        false,
        false,
        true};

    const auto first = runtime.optimize(workload);
    if (first.operations.empty()) {
        std::cerr << "No operation optimizations created.\n";
        return 1;
    }

    for (const auto& result : first.operations) {
        if (result.graph.nodes.empty() || result.graph.edges.empty()) {
            std::cerr << "Execution graph missing structure for " << result.operation.name << ".\n";
            return 1;
        }
        if (result.config.signature.empty() || result.config.participating_devices.empty()) {
            std::cerr << "Execution config missing identifiers for " << result.operation.name << ".\n";
            return 1;
        }
        if (result.config.mapped_structural_nodes.empty()) {
            std::cerr << "Execution config missing structural mappings for " << result.operation.name << ".\n";
            return 1;
        }
        if (result.benchmark.predicted_latency_us <= 0.0 || result.benchmark.effective_latency_us <= 0.0) {
            std::cerr << "Benchmark missing latency values for " << result.operation.name << ".\n";
            return 1;
        }
        if (result.benchmark.surrogate_latency_us <= 0.0 || result.benchmark.shape_bucket.empty()) {
            std::cerr << "Benchmark missing surrogate metadata for " << result.operation.name << ".\n";
            return 1;
        }
        if (!result.benchmark.accuracy_within_tolerance) {
            std::cerr << "Accuracy drift exceeded tolerance for " << result.operation.name << ".\n";
            return 1;
        }
    }

    if (first.system_profile.sustained_slowdown < 1.0) {
        std::cerr << "System profile sustained slowdown must be >= 1.\n";
        return 1;
    }

    const auto second = runtime.optimize(workload);
    if (!second.loaded_from_cache) {
        std::cerr << "Expected cached execution settings on second optimize call.\n";
        return 1;
    }
    if (second.system_profile.cold_start) {
        std::cerr << "Expected warm execution path on second optimize call.\n";
        return 1;
    }

    if (!verify_cost_propagation()) {
        std::cerr << "Aggressive-to-hierarchy cost propagation check failed.\n";
        return 1;
    }

    const auto executed = runtime.execute(workload);
    if (!executed.all_succeeded || executed.operations.empty()) {
        std::cerr << "Direct executor did not complete successfully.\n";
        return 1;
    }

    bool saw_backend = false;
    for (const auto& operation : executed.operations) {
        if (!operation.verified || operation.runtime_us <= 0.0) {
            std::cerr << "Direct execution verification failed for " << operation.operation_name << ".\n";
            return 1;
        }
        saw_backend = saw_backend || operation.used_host || operation.used_opencl;
    }

    if (!saw_backend) {
        std::cerr << "Expected direct execution to use at least one backend.\n";
        return 1;
    }

    std::cout << "operations=" << first.operations.size()
              << " cached=" << (second.loaded_from_cache ? "yes" : "no")
              << " graphs=" << runtime.devices().size()
              << " executed=" << executed.operations.size()
              << '\n';

    std::error_code ec;
    std::filesystem::remove(plan_cache, ec);
    std::filesystem::remove(exec_cache, ec);
    std::filesystem::remove(exec_cache.string() + ".perf", ec);
    return 0;
}
