#include "jakal/device.hpp"
#include "jakal/executor.hpp"
#include "jakal/jakal_toolkit.hpp"
#include "jakal/operation_variant_registry.hpp"
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

jakal::HardwareGraph make_manual_host_graph() {
    jakal::HardwareGraph graph;
    graph.uid = "host:test:0";
    graph.probe = "host";
    graph.presentation_name = "Test CPU";

    graph.nodes.push_back({"root", "root", "", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::root});
    graph.nodes.push_back({"queue", "queue", "root", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::queue});
    graph.nodes.back().control.supports_asynchronous_dispatch = false;
    graph.nodes.push_back({"core-cluster", "core-cluster", "root", jakal::HardwareObjectDomain::compute, jakal::HardwareObjectRole::cluster});
    graph.nodes.back().compute.execution_width = 16;
    graph.nodes.back().compute.clock_mhz = 4200;
    graph.nodes.back().compute.supports_fp16 = true;
    graph.nodes.push_back({"host-memory", "host-memory", "root", jakal::HardwareObjectDomain::storage, jakal::HardwareObjectRole::host_memory});
    graph.nodes.back().storage.capacity_bytes = 32ull * 1024ull * 1024ull * 1024ull;
    graph.nodes.back().storage.unified_address_space = true;
    graph.nodes.back().storage.coherent_with_host = true;
    graph.nodes.back().storage.shared_host_bytes = graph.nodes.back().storage.capacity_bytes;

    graph.edges.push_back({"root", "queue", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "core-cluster", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "host-memory", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"queue", "core-cluster", jakal::GraphEdgeSemantics::dispatches, true, 1.0, 0.0, 12.0});
    graph.edges.push_back({"host-memory", "core-cluster", jakal::GraphEdgeSemantics::feeds, true, 1.0, 64.0, 8.0});
    jakal::materialize_graph_costs(graph);
    return graph;
}

bool verify_partition_strategies() {
    const auto plan_cache = unique_temp_file("partition-plan-test");
    const auto exec_cache = unique_temp_file("partition-exec-test");

    jakal::Planner planner(plan_cache);
    jakal::ExecutionOptimizer optimizer(exec_cache);
    const std::vector<jakal::HardwareGraph> graphs{
        make_manual_host_graph(),
        make_manual_gpu_graph("gpu:test:0", "Intel Arc A770", true, true, true)};

    auto make_workload = [](const jakal::PartitionStrategy strategy) {
        return jakal::WorkloadSpec{
            "llm-decode-token-lite",
            jakal::WorkloadKind::inference,
            "llm-decode-token-lite",
            640ull * 1024ull * 1024ull,
            12ull * 1024ull * 1024ull,
            3.8e10,
            1,
            true,
            true,
            true,
            strategy};
    };
    const auto find_operation = [](const jakal::OptimizationReport& report, const std::string& name)
        -> const jakal::OperationOptimizationResult* {
        const auto it = std::find_if(report.operations.begin(), report.operations.end(), [&](const auto& result) {
            return result.operation.name == name;
        });
        return it == report.operations.end() ? nullptr : &*it;
    };

    const auto role_split = optimizer.optimize(
        make_workload(jakal::PartitionStrategy::role_split),
        planner.build_plan(make_workload(jakal::PartitionStrategy::role_split), graphs),
        graphs);
    const auto* role_qkv = find_operation(role_split, "decode-qkv");
    const auto* role_kv = find_operation(role_split, "kv-append");
    if (role_qkv == nullptr || role_kv == nullptr) {
        return false;
    }
    if (role_qkv->config.primary_device_uid != "gpu:test:0" ||
        role_qkv->config.participating_devices.size() != 1u ||
        role_kv->config.primary_device_uid != "host:test:0") {
        return false;
    }

    const auto projection_sharded = optimizer.optimize(
        make_workload(jakal::PartitionStrategy::projection_sharded),
        planner.build_plan(make_workload(jakal::PartitionStrategy::projection_sharded), graphs),
        graphs);
    const auto* sharded_qkv = find_operation(projection_sharded, "decode-qkv");
    if (sharded_qkv == nullptr) {
        return false;
    }
    if (sharded_qkv->config.strategy != jakal::ExecutionStrategy::sharded ||
        sharded_qkv->config.participating_devices.size() != 2u ||
        sharded_qkv->config.logical_partitions != 4u) {
        return false;
    }

    const auto tpu_like_plan = planner.build_plan(make_workload(jakal::PartitionStrategy::tpu_like), graphs);
    if (tpu_like_plan.allocations.size() != 2u ||
        tpu_like_plan.allocations.front().device.uid != "host:test:0" ||
        tpu_like_plan.allocations.back().device.uid != "gpu:test:0") {
        return false;
    }
    if (tpu_like_plan.allocations.front().ratio >= tpu_like_plan.allocations.back().ratio) {
        return false;
    }

    const auto tpu_like = optimizer.optimize(make_workload(jakal::PartitionStrategy::tpu_like), tpu_like_plan, graphs);
    const auto* tpu_qkv = find_operation(tpu_like, "decode-qkv");
    const auto* tpu_reduce = find_operation(tpu_like, "decode-score-reduce");
    const auto* tpu_kv = find_operation(tpu_like, "kv-append");
    if (tpu_qkv == nullptr || tpu_reduce == nullptr || tpu_kv == nullptr) {
        return false;
    }
    if (tpu_qkv->config.primary_device_uid != "gpu:test:0" ||
        tpu_qkv->config.strategy != jakal::ExecutionStrategy::overlapped ||
        tpu_qkv->config.logical_partitions != 4u ||
        tpu_qkv->config.queue_depth < 3u ||
        tpu_qkv->config.stages < 3u) {
        return false;
    }
    if (tpu_reduce->config.primary_device_uid != "gpu:test:0" ||
        tpu_reduce->config.strategy != jakal::ExecutionStrategy::overlapped) {
        return false;
    }
    if (tpu_kv->config.primary_device_uid != "host:test:0" ||
        tpu_kv->config.participating_devices.size() != 1u) {
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(plan_cache, ec);
    std::filesystem::remove(exec_cache, ec);
    std::filesystem::remove(exec_cache.string() + ".perf", ec);
    return true;
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

bool verify_operation_variant_registry() {
    const auto has_variant = [](const std::vector<jakal::OperationVariantSpec>& variants, const std::string& id) {
        return std::any_of(variants.begin(), variants.end(), [&](const jakal::OperationVariantSpec& spec) {
            return spec.id == id;
        });
    };

    jakal::WorkloadSpec workload{
        "registry-check",
        jakal::WorkloadKind::tensor,
        "",
        256ull * 1024ull * 1024ull,
        64ull * 1024ull * 1024ull,
        8.0e11,
        8,
        false,
        false,
        true};
    auto latency_workload = workload;
    latency_workload.name = "registry-check-latency";
    latency_workload.latency_sensitive = true;

    jakal::OperationSpec matmul;
    matmul.name = "registry-matmul";
    matmul.op_class = jakal::OperationClass::matmul;
    matmul.extents = {1024, 1024, 1024};
    matmul.parallelizable = true;
    matmul.streaming_friendly = false;
    matmul.matrix_friendly = true;
    matmul.max_relative_error = 1.0e-3;

    const jakal::OperationVariantRequest matmul_request{
        workload,
        matmul,
        3u,
        2u,
        false,
        false,
        true,
        true,
        true};
    const auto matmul_variants = jakal::OperationVariantRegistry::builtin().resolve(matmul_request);

    jakal::OperationSpec conv = matmul;
    conv.name = "registry-conv";
    conv.op_class = jakal::OperationClass::convolution_2d;
    conv.extents = {128, 128};
    conv.matrix_friendly = false;
    const jakal::OperationVariantRequest conv_request{
        workload,
        conv,
        3u,
        2u,
        false,
        false,
        true,
        true,
        true};
    const auto conv_variants = jakal::OperationVariantRegistry::builtin().resolve(conv_request);

    jakal::OperationSpec reduction = matmul;
    reduction.name = "registry-reduction";
    reduction.op_class = jakal::OperationClass::reduction;
    reduction.extents = {4096};
    reduction.streaming_friendly = false;
    reduction.matrix_friendly = false;
    const jakal::OperationVariantRequest reduction_request{
        workload,
        reduction,
        3u,
        2u,
        false,
        false,
        true,
        true,
        true};
    const auto reduction_variants = jakal::OperationVariantRegistry::builtin().resolve(reduction_request);
    const jakal::OperationVariantRequest reduction_latency_request{
        latency_workload,
        reduction,
        3u,
        2u,
        false,
        false,
        true,
        true,
        true};
    const auto reduction_latency_variants = jakal::OperationVariantRegistry::builtin().resolve(reduction_latency_request);

    return has_variant(matmul_variants, "single-device-balanced") &&
           has_variant(matmul_variants, "throughput-overlap") &&
           has_variant(matmul_variants, "low-precision-overlap") &&
           has_variant(matmul_variants, "placement-sharded") &&
           has_variant(matmul_variants, "accelerator-ddp") &&
           has_variant(conv_variants, "single-device-balanced") &&
           has_variant(conv_variants, "throughput-overlap") &&
           has_variant(conv_variants, "low-precision-overlap") &&
           has_variant(conv_variants, "accelerator-ddp") &&
           has_variant(reduction_variants, "single-device-balanced") &&
           has_variant(reduction_variants, "throughput-overlap") &&
           has_variant(reduction_variants, "placement-sharded") &&
           has_variant(reduction_latency_variants, "single-device-latency");
}

}  // namespace

int main() {
    std::cerr << "stage: init\n";
    const auto plan_cache = unique_temp_file("gpu-runtime-plan-test");
    const auto exec_cache = unique_temp_file("gpu-runtime-exec-test");

    jakal::RuntimeOptions options;
    options.cache_path = plan_cache;
    options.execution_cache_path = exec_cache;
    options.enable_opencl_probe = false;

    jakal::Runtime runtime(options);
    std::cerr << "stage: runtime\n";

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
    std::cerr << "stage: first-optimize\n";
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
    if (first.graph_optimization.optimizer_name.find("bootstrap_general_optimizer:") != 0) {
        std::cerr << "Expected bootstrap optimizer route on first optimize.\n";
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
        if (result.config.variant_id.empty()) {
            std::cerr << "Execution config missing variant id for " << result.operation.name << ".\n";
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
    std::cerr << "stage: second-optimize\n";
    if (!second.loaded_from_cache) {
        std::cerr << "Expected cached execution settings on second optimize call.\n";
        return 1;
    }
    if (second.graph_optimization.optimizer_name.find("runtime_sensitive_optimizer:") != 0) {
        std::cerr << "Expected runtime-sensitive optimizer route on cached optimize.\n";
        return 1;
    }
    if (!second.graph_optimization.passes.empty()) {
        std::cerr << "Expected lightweight runtime path to skip heavy graph passes.\n";
        return 1;
    }
    if (second.system_profile.readiness_score <= 0.0) {
        std::cerr << "Expected non-zero readiness score on second optimize call.\n";
        return 1;
    }

    std::cerr << "stage: verify-cost\n";
    if (!verify_cost_propagation()) {
        std::cerr << "Aggressive-to-hierarchy cost propagation check failed.\n";
        return 1;
    }
    std::cerr << "stage: verify-toolkit\n";
    if (!verify_jakal_toolkit_index()) {
        std::cerr << "GPU L0 toolkit ranking check failed.\n";
        return 1;
    }
    std::cerr << "stage: verify-direct\n";
    if (!verify_gpu_direct_variants()) {
        std::cerr << "GPU direct variant execution check failed.\n";
        return 1;
    }
    std::cerr << "stage: verify-registry\n";
    if (!verify_operation_variant_registry()) {
        std::cerr << "Operation variant registry check failed.\n";
        return 1;
    }
    std::cerr << "stage: registry\n";
    if (!verify_partition_strategies()) {
        std::cerr << "CPU/GPU partition strategy check failed.\n";
        return 1;
    }
    std::cerr << "stage: partition\n";

    const auto presets = jakal::canonical_workload_presets();
    const auto preset_it = std::find_if(presets.begin(), presets.end(), [](const jakal::CanonicalWorkloadPreset& preset) {
        return preset.workload.kind == jakal::WorkloadKind::inference;
    });
    if (preset_it == presets.end()) {
        std::cerr << "Missing canonical inference preset.\n";
        return 1;
    }

    const auto macro_report = runtime.optimize(preset_it->workload);
    std::cerr << "stage: macro\n";
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

