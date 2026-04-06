#include "gpu/runtime.hpp"

#include <iomanip>
#include <iostream>

int main() {
    gpu::Runtime runtime;

    std::cout << "Discovered hardware graphs\n";
    for (const auto& graph : runtime.devices()) {
        const auto summary = gpu::summarize_graph(graph);
        std::cout
            << "\n[" << graph.ordinal << "] " << graph.presentation_name
            << " | probe=" << graph.probe
            << " | execution_objects=" << summary.execution_objects
            << " | lanes/object=" << summary.lanes_per_object
            << " | memory=" << (summary.addressable_bytes / (1024ull * 1024ull)) << " MiB"
            << " | nodes=" << graph.nodes.size()
            << " | edges=" << graph.edges.size()
            << '\n';

        std::cout << "  Nodes\n";
        for (const auto& node : graph.nodes) {
            std::cout
                << "    " << std::setw(24) << std::left << node.label
                << " domain=" << gpu::to_string(node.domain)
                << " role=" << gpu::to_string(node.role)
                << " resolution=" << gpu::to_string(node.resolution)
                << '\n';
        }

        std::cout << "  Edges\n";
        for (const auto& edge : graph.edges) {
            std::cout
                << "    " << edge.source_id
                << " -> " << edge.target_id
                << " semantics=" << gpu::to_string(edge.semantics)
                << " weight=" << edge.weight
                << " bw=" << edge.bandwidth_gbps
                << "GB/s latency=" << edge.latency_us
                << "us\n";
        }
    }

    const gpu::WorkloadSpec workload{
        "sample-inference",
        gpu::WorkloadKind::inference,
        4ull * 1024ull * 1024ull * 1024ull,
        512ull * 1024ull * 1024ull,
        30.0e12,
        16,
        false,
        false,
        true};

    const auto plan = runtime.plan(workload);
    std::cout << "\nPlan signature: " << plan.signature << '\n';
    std::cout << "Loaded from cache: " << (plan.loaded_from_cache ? "yes" : "no") << '\n';

    for (const auto& allocation : plan.allocations) {
        std::cout
            << "  " << std::setw(24) << std::left << allocation.device.presentation_name
            << " ratio=" << std::fixed << std::setprecision(3) << allocation.ratio
            << " score=" << allocation.score
            << '\n';
    }

    const auto report = runtime.optimize(workload);
    std::cout << "\nExecution report: " << report.signature << '\n';
    std::cout << "Execution settings cache: " << (report.loaded_from_cache ? "hit" : "miss") << '\n';
    std::cout << "System profile: low_spec=" << (report.system_profile.low_spec_mode ? "yes" : "no")
              << " cold=" << (report.system_profile.cold_start ? "yes" : "no")
              << " battery=" << (report.system_profile.on_battery ? "yes" : "no")
              << " free_mem_ratio=" << std::fixed << std::setprecision(3) << report.system_profile.free_memory_ratio
              << " paging_risk=" << report.system_profile.paging_risk
              << " sustained=" << report.system_profile.sustained_slowdown
              << '\n';

    for (const auto& result : report.operations) {
        std::cout
            << "  op=" << std::setw(20) << std::left << result.operation.name
            << " strategy=" << std::setw(12) << gpu::to_string(result.config.strategy)
            << " devices=" << result.config.participating_devices.size()
            << " predicted=" << std::fixed << std::setprecision(3) << result.graph.predicted_latency_us << "us"
            << " surrogate=" << result.benchmark.surrogate_latency_us << "us"
            << " effective=" << result.benchmark.effective_latency_us << "us"
            << " error=" << result.benchmark.relative_error
            << '\n';
    }

    const auto executed = runtime.execute(workload);
    std::cout << "\nDirect execution: " << (executed.all_succeeded ? "ok" : "failed") << '\n';
    for (const auto& operation : executed.operations) {
        std::cout
            << "  op=" << std::setw(20) << std::left << operation.operation_name
            << " backend=" << std::setw(14) << operation.backend_name
            << " runtime=" << std::fixed << std::setprecision(3) << operation.runtime_us << "us"
            << " ref=" << operation.reference_runtime_us << "us"
            << " error=" << operation.relative_error
            << '\n';
    }

    return 0;
}
