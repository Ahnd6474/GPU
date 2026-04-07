#include "jakal/c_api.h"
#include "jakal/device.hpp"
#include "jakal/runtime.hpp"
#include "jakal/workloads.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iterator>

namespace {

std::filesystem::path unique_temp_file(const std::string& stem) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (stem + "-" + std::to_string(nonce) + ".tsv");
}

jakal::HardwareGraph make_manual_gpu_graph(
    const std::string& uid,
    const std::string& presentation_name,
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
    graph.nodes.back().compute.supports_fp16 = true;
    graph.nodes.back().compute.supports_int8 = true;
    graph.nodes.push_back({"memory", "memory", "root", jakal::HardwareObjectDomain::storage, jakal::HardwareObjectRole::global_memory});
    graph.nodes.back().storage.capacity_bytes = 8ull * 1024ull * 1024ull * 1024ull;
    graph.nodes.back().storage.unified_address_space = unified_memory;
    graph.nodes.back().storage.coherent_with_host = unified_memory;
    graph.nodes.back().storage.shared_host_bytes = unified_memory ? graph.nodes.back().storage.capacity_bytes : 0ull;
    graph.nodes.push_back({"host-link", "host-link", "root", jakal::HardwareObjectDomain::transfer, jakal::HardwareObjectRole::transfer_link});
    graph.nodes.back().transfer.read_bandwidth_gbps = 96.0;
    graph.nodes.back().transfer.write_bandwidth_gbps = 96.0;
    graph.nodes.back().transfer.dispatch_latency_us = 6.0;
    graph.nodes.back().transfer.synchronization_latency_us = 5.0;

    graph.edges.push_back({"root", "queue", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "cluster", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "memory", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "host-link", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"queue", "cluster", jakal::GraphEdgeSemantics::dispatches, true, 1.0, 0.0, 6.0});
    graph.edges.push_back({"host-link", "memory", jakal::GraphEdgeSemantics::transfers_to, true, 1.0, 96.0, 5.0});
    jakal::materialize_graph_costs(graph);
    return graph;
}

bool verify_auto_accelerator_data_parallel() {
    const auto intel = make_manual_gpu_graph("opencl:intel:0", "Intel Arc A770", true);
    const auto amd = make_manual_gpu_graph("opencl:amd:0", "AMD Radeon RX 7900", false);
    const auto nvidia = make_manual_gpu_graph("opencl:nvidia:0", "NVIDIA RTX 4090", false);

    jakal::ExecutionPlan placement;
    placement.signature = "ddp-smoke";
    placement.allocations.push_back({intel, 0.34, 2.0});
    placement.allocations.push_back({amd, 0.33, 1.9});
    placement.allocations.push_back({nvidia, 0.33, 2.1});

    jakal::WorkloadSpec workload{
        "ddp-smoke",
        jakal::WorkloadKind::custom,
        "",
        64ull * 1024ull * 1024ull,
        16ull * 1024ull * 1024ull,
        3.0e10,
        16,
        false,
        false,
        true};

    jakal::ExecutionOptimizer optimizer(unique_temp_file("gpu-ddp-smoke"));
    const auto report = optimizer.optimize(workload, placement, {intel, amd, nvidia});
    const bool success = std::any_of(
        report.operations.begin(),
        report.operations.end(),
        [](const jakal::OperationOptimizationResult& result) {
            return result.operation.op_class == jakal::OperationClass::matmul &&
                   result.config.strategy == jakal::ExecutionStrategy::sharded &&
                   result.config.participating_devices.size() == 3 &&
                   std::find(result.config.participating_devices.begin(), result.config.participating_devices.end(), "host:0") ==
                       result.config.participating_devices.end();
        });
    if (!success) {
        for (const auto& result : report.operations) {
            std::cerr << "DDP debug op=" << result.operation.name
                      << " class=" << jakal::to_string(result.operation.op_class)
                      << " strategy=" << jakal::to_string(result.config.strategy)
                      << " primary=" << result.config.primary_device_uid
                      << " devices=" << result.config.participating_devices.size()
                      << '\n';
        }
    }
    return success;
}

}  // namespace

int main() {
    jakal::RuntimeOptions options;
    options.enable_opencl_probe = false;
    jakal::Runtime runtime(options);

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

    const jakal::WorkloadSpec workload{
        "smoke",
        jakal::WorkloadKind::tensor,
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

    const auto cpu_dl_presets = jakal::cpu_deep_learning_exploration_presets();
    if (cpu_dl_presets.size() < 3) {
        std::cerr << "CPU deep-learning presets missing.\n";
        return 1;
    }
    for (const auto& preset : cpu_dl_presets) {
        const auto graph = jakal::default_workload_graph(preset.workload);
        if (graph.operations.empty() || graph.tensors.empty() || graph.signature.empty()) {
            std::cerr << "CPU deep-learning workload graph missing metadata.\n";
            return 1;
        }
    }
    const auto decode_it = std::find_if(
        cpu_dl_presets.begin(),
        cpu_dl_presets.end(),
        [](const jakal::CpuDeepLearningExplorationPreset& preset) {
            return preset.workload.dataset_tag == "llm-decode-token-lite";
        });
    if (decode_it == cpu_dl_presets.end()) {
        std::cerr << "Decode preset missing.\n";
        return 1;
    }
    const auto decode_report = runtime.optimize(decode_it->workload);
    const auto host_primary_count = std::count_if(
        decode_report.operations.begin(),
        decode_report.operations.end(),
        [](const jakal::OperationOptimizationResult& result) {
            return result.config.primary_device_uid == "host:0";
        });
    if (host_primary_count == 0) {
        std::cerr << "CPU policy did not assign decode operations to host.\n";
        return 1;
    }
    if (!verify_auto_accelerator_data_parallel()) {
        std::cerr << "Auto accelerator data parallel policy did not produce sharded multi-GPU configs.\n";
        return 1;
    }

    const auto executed = runtime.execute(workload);
    if (executed.operations.empty() || !executed.all_succeeded) {
        std::cerr << "Direct execution failed.\n";
        return 1;
    }

    jakal_core_runtime_t* c_runtime = jakal_core_runtime_create();
    if (c_runtime == nullptr) {
        std::cerr << "Failed to create C runtime.\n";
        return 1;
    }

    const jakal_core_workload_spec c_workload{
        "smoke",
        "tensor",
        128ull * 1024ull * 1024ull,
        64ull * 1024ull * 1024ull,
        2.0e11,
        4,
        0,
        0,
        1};

    jakal_core_optimization_info optimization_info{};
    jakal_core_operation_optimization_info optimization_ops[8]{};
    size_t optimization_count = 0;
    if (jakal_core_runtime_optimize(
            c_runtime,
            &c_workload,
            &optimization_info,
            optimization_ops,
            std::size(optimization_ops),
            &optimization_count) != 0 ||
        optimization_count == 0) {
        std::cerr << "C API optimize failed.\n";
        jakal_core_runtime_destroy(c_runtime);
        return 1;
    }

    jakal_core_execution_info execution_info{};
    jakal_core_execution_operation_info execution_ops[8]{};
    size_t execution_count = 0;
    if (jakal_core_runtime_execute(
            c_runtime,
            &c_workload,
            &execution_info,
            execution_ops,
            std::size(execution_ops),
            &execution_count) != 0 ||
        execution_count == 0 ||
        execution_info.all_succeeded == 0) {
        std::cerr << "C API execute failed.\n";
        jakal_core_runtime_destroy(c_runtime);
        return 1;
    }

    jakal_core_runtime_destroy(c_runtime);

    std::cout << "Graphs=" << runtime.devices().size()
              << " allocations=" << plan.allocations.size()
              << " operations=" << report.operations.size()
              << " executed=" << executed.operations.size()
              << " cache=" << (plan.loaded_from_cache ? "hit" : "miss")
              << '\n';

    return 0;
}

