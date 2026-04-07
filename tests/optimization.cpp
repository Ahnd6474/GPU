#include "jakal/device.hpp"
#include "jakal/executor.hpp"
#include "jakal/jakal_toolkit.hpp"
#include "jakal/runtime.hpp"
#include "jakal/workloads.hpp"

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
    jakal::HardwareGraph graph;
    graph.uid = "manual";
    graph.probe = "test";
    graph.presentation_name = "manual";

    graph.nodes.push_back({"root", "root", "", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::root});
    graph.nodes.push_back({"queue", "queue", "root", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::queue});
    graph.nodes.push_back({"tile", "tile", "root", jakal::HardwareObjectDomain::compute, jakal::HardwareObjectRole::tile});
    graph.nodes.push_back({"lane", "lane", "tile", jakal::HardwareObjectDomain::compute, jakal::HardwareObjectRole::lane});
    graph.nodes.back().compute.execution_width = 32;
    graph.nodes.push_back({"mem", "mem", "root", jakal::HardwareObjectDomain::storage, jakal::HardwareObjectRole::global_memory});

    graph.edges.push_back({"root", "tile", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "queue", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"tile", "lane", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"queue", "tile", jakal::GraphEdgeSemantics::dispatches, true, 0.0, 0.0, 3.0});
    graph.edges.push_back({"mem", "tile", jakal::GraphEdgeSemantics::feeds, true, 0.0, 128.0, 1.0});
    graph.edges.push_back({"lane", "mem", jakal::GraphEdgeSemantics::feeds, true, 0.0, 512.0, 0.5});

    jakal::materialize_graph_costs(graph);

    double contains_weight = 0.0;
    double dispatch_weight = 0.0;
    for (const auto& edge : graph.edges) {
        if (edge.source_id == "root" && edge.target_id == "tile" && edge.semantics == jakal::GraphEdgeSemantics::contains) {
            contains_weight = edge.weight;
        }
        if (edge.source_id == "queue" && edge.target_id == "tile" && edge.semantics == jakal::GraphEdgeSemantics::dispatches) {
            dispatch_weight = edge.weight;
        }
    }

    return contains_weight > 0.0 && dispatch_weight > 3.0;
}

bool verify_jakal_toolkit_index() {
    jakal::HardwareGraph graph;
    graph.uid = "opencl:test:0";
    graph.probe = "opencl";
    graph.presentation_name = "Intel Iris Xe";

    graph.nodes.push_back({"root", "root", "", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::root});
    graph.nodes.push_back({"queue", "queue", "root", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::queue});
    graph.nodes.back().control.supports_asynchronous_dispatch = true;
    graph.nodes.push_back({"cluster", "cluster", "root", jakal::HardwareObjectDomain::compute, jakal::HardwareObjectRole::cluster});
    graph.nodes.back().compute.execution_width = 96;
    graph.nodes.back().compute.clock_mhz = 1300;
    graph.nodes.back().compute.supports_fp16 = true;
    graph.nodes.push_back({"memory", "memory", "root", jakal::HardwareObjectDomain::storage, jakal::HardwareObjectRole::global_memory});
    graph.nodes.back().storage.capacity_bytes = 6ull * 1024ull * 1024ull * 1024ull;
    graph.nodes.back().storage.unified_address_space = true;
    graph.nodes.back().storage.coherent_with_host = true;
    graph.nodes.back().storage.shared_host_bytes = graph.nodes.back().storage.capacity_bytes;
    graph.nodes.push_back({"host-link", "host-link", "root", jakal::HardwareObjectDomain::transfer, jakal::HardwareObjectRole::transfer_link});
    graph.nodes.back().transfer.read_bandwidth_gbps = 96.0;
    graph.nodes.back().transfer.write_bandwidth_gbps = 96.0;
    graph.nodes.back().transfer.dispatch_latency_us = 8.0;
    graph.nodes.back().transfer.synchronization_latency_us = 6.0;

    graph.edges.push_back({"root", "queue", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "cluster", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "memory", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "host-link", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"queue", "cluster", jakal::GraphEdgeSemantics::dispatches, true, 1.0, 0.0, 8.0});
    graph.edges.push_back({"host-link", "memory", jakal::GraphEdgeSemantics::transfers_to, true, 1.0, 96.0, 6.0});
    jakal::materialize_graph_costs(graph);

    jakal::JakalToolkit toolkit;
    const jakal::JakalL0WorkloadTraits traits{
        jakal::OperationClass::matmul,
        {1024, 1024, 1024},
        256ull * 1024ull * 1024ull,
        2.0e12,
        false,
        true,
        false};
    const auto variants = toolkit.rank_variants(graph, traits);
    if (variants.empty()) {
        return false;
    }
    const auto best = toolkit.select_best(graph, traits);
    if (!best.has_value()) {
        return false;
    }
    return best->binding.vendor == jakal::JakalVendorFamily::intel &&
           (best->binding.backend == jakal::JakalBackendKind::level_zero ||
            best->binding.backend == jakal::JakalBackendKind::opencl);
}

jakal::HardwareGraph make_manual_gpu_graph(
    const std::string& uid,
    const std::string& presentation_name,
    const bool fp16,
    const bool int8,
    const bool unified_memory) {
    jakal::HardwareGraph graph;
    graph.uid = uid;
    graph.probe = "opencl";
    graph.presentation_name = presentation_name;

    graph.nodes.push_back({"root", "root", "", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::root});
    graph.nodes.push_back({"queue", "queue", "root", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::queue});
    graph.nodes.back().control.supports_asynchronous_dispatch = true;
    graph.nodes.push_back({"cluster", "cluster", "root", jakal::HardwareObjectDomain::compute, jakal::HardwareObjectRole::cluster});
    graph.nodes.back().compute.execution_width = 128;
    graph.nodes.back().compute.clock_mhz = 1800;
    graph.nodes.back().compute.matrix_engines = 16;
    graph.nodes.back().compute.supports_fp16 = fp16;
    graph.nodes.back().compute.supports_int8 = int8;
    graph.nodes.push_back({"memory", "memory", "root", jakal::HardwareObjectDomain::storage, jakal::HardwareObjectRole::global_memory});
    graph.nodes.back().storage.capacity_bytes = 8ull * 1024ull * 1024ull * 1024ull;
    graph.nodes.back().storage.unified_address_space = unified_memory;
    graph.nodes.back().storage.coherent_with_host = unified_memory;
    graph.nodes.back().storage.shared_host_bytes = unified_memory ? graph.nodes.back().storage.capacity_bytes : 0ull;
    graph.nodes.push_back({"host-link", "host-link", "root", jakal::HardwareObjectDomain::transfer, jakal::HardwareObjectRole::transfer_link});
    graph.nodes.back().transfer.read_bandwidth_gbps = 128.0;
    graph.nodes.back().transfer.write_bandwidth_gbps = 128.0;
    graph.nodes.back().transfer.dispatch_latency_us = 6.0;
    graph.nodes.back().transfer.synchronization_latency_us = 5.0;

    graph.edges.push_back({"root", "queue", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "cluster", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "memory", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "host-link", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"queue", "cluster", jakal::GraphEdgeSemantics::dispatches, true, 1.0, 0.0, 6.0});
    graph.edges.push_back({"host-link", "memory", jakal::GraphEdgeSemantics::transfers_to, true, 1.0, 128.0, 5.0});
    jakal::materialize_graph_costs(graph);
    return graph;
}

bool verify_gpu_direct_variants() {
    const struct VariantCase {
        std::string name;
        std::string presentation_name;
        jakal::JakalVendorFamily vendor;
        jakal::JakalBackendKind backend;
        jakal::OperationClass op_class;
        std::vector<std::uint64_t> extents;
        bool fp16;
        bool int8;
        bool unified_memory;
    } cases[] = {
        {"level-zero", "Intel Arc A770", jakal::JakalVendorFamily::intel, jakal::JakalBackendKind::level_zero, jakal::OperationClass::matmul, {64, 64, 64}, true, false, true},
        {"cuda", "NVIDIA RTX 4090", jakal::JakalVendorFamily::nvidia, jakal::JakalBackendKind::cuda, jakal::OperationClass::matmul, {64, 64, 64}, true, true, false},
        {"vulkan", "AMD Radeon RX 7900", jakal::JakalVendorFamily::amd, jakal::JakalBackendKind::vulkan_compute, jakal::OperationClass::resample_2d, {128, 128, 256, 256}, true, false, false},
    };

    jakal::DirectExecutor executor;
    for (const auto& test_case : cases) {
        auto graph = make_manual_gpu_graph(
            "graph:" + test_case.name,
            test_case.presentation_name,
            test_case.fp16,
            test_case.int8,
            test_case.unified_memory);

        jakal::OperationSpec operation;
        operation.name = "op-" + test_case.name;
        operation.op_class = test_case.op_class;
        operation.extents = test_case.extents;
        operation.input_bytes = 1ull << 20;
        operation.output_bytes = 1ull << 20;
        operation.temporary_bytes = 1ull << 18;
        operation.estimated_flops = 1.0e9;
        operation.max_relative_error = 1.0e-4;
        operation.parallelizable = true;
        operation.reduction_like = test_case.op_class == jakal::OperationClass::reduction;
        operation.streaming_friendly = test_case.op_class == jakal::OperationClass::resample_2d;
        operation.matrix_friendly = test_case.op_class == jakal::OperationClass::matmul;

        jakal::ExecutionConfig config;
        config.signature = "cfg-" + test_case.name;
        config.operation_name = operation.name;
        config.primary_device_uid = graph.uid;
        config.participating_devices = {graph.uid};
        config.mapped_structural_nodes = {"cluster"};
        config.logical_partitions = 1;
        config.target_error_tolerance = operation.max_relative_error;

        jakal::ExecutionGraph execution_graph;
        execution_graph.signature = "exec-" + test_case.name;
        execution_graph.workload_signature = "manual";
        execution_graph.operation = operation;
        execution_graph.participating_devices = {graph.uid};

        jakal::OperationOptimizationResult optimized;
        optimized.operation = operation;
        optimized.config = config;
        optimized.graph = execution_graph;

        jakal::OptimizationReport report;
        report.signature = "report-" + test_case.name;
        report.placement.signature = "plan-" + test_case.name;
        report.placement.allocations.push_back({graph, 1.0, 1.0});
        report.operations.push_back(optimized);

        jakal::JakalToolkitVariant variant;
        variant.binding.device_uid = graph.uid;
        variant.binding.graph_fingerprint = jakal::structural_fingerprint(graph);
        variant.binding.adapter_id = "adapter-" + test_case.name;
        variant.binding.presentation_name = graph.presentation_name;
        variant.binding.vendor = test_case.vendor;
        variant.binding.backend = test_case.backend;
        variant.binding.capabilities.adapter_available = true;
        variant.binding.capabilities.kernel_specialization = true;
        variant.binding.capabilities.asynchronous_dispatch = true;
        variant.executable = true;
        variant.toolkit_score = 2.0;

        jakal::JakalToolkitIndexEntry index_entry;
        index_entry.device_uid = graph.uid;
        index_entry.graph_fingerprint = jakal::structural_fingerprint(graph);
        index_entry.variants.push_back(variant);

        const auto execution = executor.execute(report, {graph}, {index_entry});
        if (execution.operations.size() != 1) {
            std::cerr << "Expected one execution record for " << test_case.name
                      << ", got " << execution.operations.size() << ".\n";
            return false;
        }

        const auto& record = execution.operations.front();
        if (record.requested_gpu_backend != jakal::to_string(test_case.backend)) {
            std::cerr << "Requested backend mismatch for " << test_case.name
                      << ": expected " << jakal::to_string(test_case.backend)
                      << ", got " << record.requested_gpu_backend << ".\n";
            return false;
        }
        if (record.backend_name.find(jakal::to_string(test_case.backend)) == std::string::npos) {
            std::cerr << "Backend name mismatch for " << test_case.name
                      << ": expected substring " << jakal::to_string(test_case.backend)
                      << ", got " << record.backend_name << ".\n";
            return false;
        }
        if (execution.all_succeeded && !record.verified) {
            std::cerr << "Successful direct execution was not verified for " << test_case.name << ".\n";
            return false;
        }
    }

    return true;
}

}  // namespace

int main() {
    const auto plan_cache = unique_temp_file("gpu-runtime-plan-test");
    const auto exec_cache = unique_temp_file("gpu-runtime-exec-test");

    jakal::RuntimeOptions options;
    options.cache_path = plan_cache;
    options.execution_cache_path = exec_cache;
    options.enable_opencl_probe = false;

    jakal::Runtime runtime(options);

    if (runtime.devices().empty()) {
        std::cerr << "No hardware graphs discovered.\n";
        return 1;
    }

    const jakal::WorkloadSpec workload{
        "optimization-suite",
        jakal::WorkloadKind::tensor,
        "",
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
    if (first.workload_graph.operations.empty() ||
        first.workload_graph.tensors.empty() ||
        first.workload_graph.lifetimes.empty() ||
        first.workload_graph.dependencies.empty()) {
        std::cerr << "Expected workload DAG, tensors, and lifetimes.\n";
        return 1;
    }
    if (first.graph_optimization.optimizer_name.empty() || first.graph_optimization.passes.empty()) {
        std::cerr << "Expected graph-level optimization passes.\n";
        return 1;
    }
    if (first.graph_optimization.final_objective_us <= 0.0) {
        std::cerr << "Expected graph-level objective.\n";
        return 1;
    }

    bool saw_transfer_schedule = false;
    for (const auto& result : first.operations) {
        if (result.graph.nodes.empty() || result.graph.edges.empty()) {
            std::cerr << "Execution graph missing structure for " << result.operation.name << ".\n";
            return 1;
        }
        if (result.graph.residency_plan.empty()) {
            std::cerr << "Execution graph missing residency plan for " << result.operation.name << ".\n";
            return 1;
        }
        if (result.graph.peak_resident_bytes == 0) {
            std::cerr << "Execution graph missing peak residency estimate for " << result.operation.name << ".\n";
            return 1;
        }
        if (result.config.signature.empty() || result.config.participating_devices.empty()) {
            std::cerr << "Execution config missing identifiers for " << result.operation.name << ".\n";
            return 1;
        }
        if (result.config.logical_partitions == 0) {
            std::cerr << "Execution config missing logical partitions for " << result.operation.name << ".\n";
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
        if (result.benchmark.calibrated_prediction_us <= 0.0 ||
            result.benchmark.calibration_ratio <= 0.0 ||
            result.benchmark.validation_samples < 1u) {
            std::cerr << "Benchmark missing calibration metadata for " << result.operation.name << ".\n";
            return 1;
        }
        if (result.benchmark.optimizer_name.empty() || result.benchmark.objective_score <= 0.0) {
            std::cerr << "Benchmark missing optimizer metadata for " << result.operation.name << ".\n";
            return 1;
        }
        if (!result.benchmark.accuracy_within_tolerance) {
            std::cerr << "Accuracy drift exceeded tolerance for " << result.operation.name << ".\n";
            return 1;
        }
        saw_transfer_schedule = saw_transfer_schedule || !result.graph.transfer_schedule.empty();
    }
    if (!saw_transfer_schedule) {
        std::cerr << "Expected at least one scheduled transfer in optimized workload.\n";
        return 1;
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
    if (second.system_profile.readiness_score <= 0.0) {
        std::cerr << "Expected non-zero readiness score on second optimize call.\n";
        return 1;
    }

    if (!verify_cost_propagation()) {
        std::cerr << "Aggressive-to-hierarchy cost propagation check failed.\n";
        return 1;
    }
    if (!verify_jakal_toolkit_index()) {
        std::cerr << "GPU L0 toolkit ranking check failed.\n";
        return 1;
    }
    if (!verify_gpu_direct_variants()) {
        std::cerr << "GPU direct variant execution check failed.\n";
        return 1;
    }

    const auto presets = jakal::canonical_workload_presets();
    const auto preset_it = std::find_if(presets.begin(), presets.end(), [](const jakal::CanonicalWorkloadPreset& preset) {
        return preset.workload.kind == jakal::WorkloadKind::inference;
    });
    if (preset_it == presets.end()) {
        std::cerr << "Missing canonical inference preset.\n";
        return 1;
    }

    const auto macro_report = runtime.optimize(preset_it->workload);
    if (macro_report.operations.empty()) {
        std::cerr << "Canonical workload optimization failed: " << preset_it->workload.name << ".\n";
        return 1;
    }
    if (macro_report.workload_graph.dependencies.empty() || macro_report.workload_graph.lifetimes.empty()) {
        std::cerr << "Canonical workload DAG metadata missing: " << preset_it->workload.name << ".\n";
        return 1;
    }
    if (macro_report.system_profile.readiness_score < 0.0 ||
        macro_report.system_profile.readiness_score > 1.0 ||
        macro_report.system_profile.stability_score < 0.0 ||
        macro_report.system_profile.stability_score > 1.0) {
        std::cerr << "Canonical workload system metrics out of range: " << preset_it->workload.name << ".\n";
        return 1;
    }

    std::cout << "operations=" << first.operations.size()
              << " cached=" << (second.loaded_from_cache ? "yes" : "no")
              << " graphs=" << runtime.devices().size()
              << " graph_passes=" << first.graph_optimization.passes.size()
              << '\n';

    std::error_code ec;
    std::filesystem::remove(plan_cache, ec);
    std::filesystem::remove(exec_cache, ec);
    std::filesystem::remove(exec_cache.string() + ".perf", ec);
    return 0;
}

