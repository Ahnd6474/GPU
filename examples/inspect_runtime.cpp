#include "jakal/runtime.hpp"

#include <iomanip>
#include <iostream>

int main() {
    jakal::Runtime runtime;

    std::cout << "Discovered hardware graphs\n";
    for (const auto& graph : runtime.devices()) {
        const auto summary = jakal::summarize_graph(graph);
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
                << " domain=" << jakal::to_string(node.domain)
                << " role=" << jakal::to_string(node.role)
                << " resolution=" << jakal::to_string(node.resolution)
                << '\n';
        }

        std::cout << "  Edges\n";
        for (const auto& edge : graph.edges) {
            std::cout
                << "    " << edge.source_id
                << " -> " << edge.target_id
                << " semantics=" << jakal::to_string(edge.semantics)
                << " weight=" << edge.weight
                << " bw=" << edge.bandwidth_gbps
                << "GB/s latency=" << edge.latency_us
                << "us\n";
        }
    }

    std::cout << "\nGPU toolkit index\n";
    for (const auto& entry : runtime.jakal_toolkit_index()) {
        std::cout << "  device=" << entry.device_uid << " variants=" << entry.variants.size() << '\n';
        for (const auto& variant : entry.variants) {
            std::cout
                << "    vendor=" << std::setw(10) << std::left << jakal::to_string(variant.binding.vendor)
                << " backend=" << std::setw(18) << jakal::to_string(variant.binding.backend)
                << " adapter=" << std::setw(20) << variant.binding.adapter_id
                << " executable=" << (variant.executable ? "yes" : "no")
                << " score=" << std::fixed << std::setprecision(3) << variant.toolkit_score
                << " wg=(" << variant.tuning.workgroup_x << "," << variant.tuning.workgroup_y << "," << variant.tuning.workgroup_z << ")"
                << " batch=" << variant.tuning.queue_batch
                << " rationale=" << variant.rationale
                << '\n';
        }
    }

    const jakal::WorkloadSpec workload{
        "sample-inference",
        jakal::WorkloadKind::inference,
        "",
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
              << " battery=" << (report.system_profile.on_battery ? "yes" : "no")
              << " readiness=" << std::fixed << std::setprecision(3) << report.system_profile.readiness_score
              << " stability=" << report.system_profile.stability_score
              << " free_mem_ratio=" << std::fixed << std::setprecision(3) << report.system_profile.free_memory_ratio
              << " paging_risk=" << report.system_profile.paging_risk
              << " sustained=" << report.system_profile.sustained_slowdown
              << '\n';
    std::cout << "Graph optimizer: " << report.graph_optimization.optimizer_name
              << " passes=" << report.graph_optimization.passes.size()
              << " initial=" << report.graph_optimization.initial_objective_us
              << "us final=" << report.graph_optimization.final_objective_us
              << "us logical_partitions=" << report.graph_optimization.total_logical_partitions
              << '\n';

    for (const auto& result : report.operations) {
        std::cout
            << "  op=" << std::setw(20) << std::left << result.operation.name
            << " strategy=" << std::setw(12) << jakal::to_string(result.config.strategy)
            << " optimizer=" << std::setw(15) << result.benchmark.optimizer_name
            << " partitions=" << result.config.logical_partitions
            << " devices=" << result.config.participating_devices.size()
            << " predicted=" << std::fixed << std::setprecision(3) << result.graph.predicted_latency_us << "us"
            << " surrogate=" << result.benchmark.surrogate_latency_us << "us"
            << " effective=" << result.benchmark.effective_latency_us << "us"
            << " objective=" << result.benchmark.objective_score
            << " error=" << result.benchmark.relative_error
            << '\n';
    }

    const auto executed = runtime.execute(workload);
    std::cout << "\nDirect execution: " << (executed.all_succeeded ? "ok" : "failed") << '\n';
    std::cout << "Total runtime=" << std::fixed << std::setprecision(3) << executed.total_runtime_us
              << "us reference=" << executed.total_reference_runtime_us
              << "us improvement=" << executed.speedup_vs_reference << "x\n";
    for (const auto& operation : executed.operations) {
        std::cout
            << "  op=" << std::setw(20) << std::left << operation.operation_name
            << " backend=" << std::setw(14) << operation.backend_name
            << " requested=";
        if (!operation.requested_gpu_backend.empty()) {
            std::cout << operation.requested_gpu_vendor << ':' << operation.requested_gpu_backend;
        } else {
            std::cout << "host";
        }
        std::cout
            << " partitions=" << operation.logical_partitions_used
            << " runtime=" << std::fixed << std::setprecision(3) << operation.runtime_us << "us"
            << " ref=" << operation.reference_runtime_us << "us"
            << " speedup=" << operation.speedup_vs_reference << "x"
            << " error=" << operation.relative_error
            << '\n';
    }

    return 0;
}

