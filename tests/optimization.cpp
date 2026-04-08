#include "jakal/device.hpp"
#include "jakal/executor.hpp"
#include "jakal/executors/scheduler.hpp"
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

void attach_cache_and_scratch(
    jakal::HardwareGraph& graph,
    const std::uint64_t cache_bytes,
    const std::uint64_t scratch_bytes) {
    graph.nodes.push_back({"l2-cache", "l2-cache", "root", jakal::HardwareObjectDomain::storage, jakal::HardwareObjectRole::cache});
    graph.nodes.back().storage.capacity_bytes = cache_bytes;
    graph.nodes.push_back({"scratch", "scratch", "root", jakal::HardwareObjectDomain::storage, jakal::HardwareObjectRole::scratchpad});
    graph.nodes.back().storage.capacity_bytes = scratch_bytes;
    graph.edges.push_back({"root", "l2-cache", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "scratch", jakal::GraphEdgeSemantics::contains, true});
    jakal::materialize_graph_costs(graph);
}

const jakal::OperationOptimizationResult* find_operation_by_name(
    const jakal::OptimizationReport& report,
    const std::string& name) {
    const auto it = std::find_if(report.operations.begin(), report.operations.end(), [&](const auto& result) {
        return result.operation.name == name;
    });
    return it == report.operations.end() ? nullptr : &*it;
}

const jakal::OperationOptimizationResult* find_first_operation_by_class(
    const jakal::OptimizationReport& report,
    const jakal::OperationClass op_class) {
    const auto it = std::find_if(report.operations.begin(), report.operations.end(), [&](const auto& result) {
        return result.operation.op_class == op_class;
    });
    return it == report.operations.end() ? nullptr : &*it;
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
    const auto role_split = optimizer.optimize(
        make_workload(jakal::PartitionStrategy::role_split),
        planner.build_plan(make_workload(jakal::PartitionStrategy::role_split), graphs),
        graphs);
    const auto* role_qkv = find_operation_by_name(role_split, "decode-qkv");
    const auto* role_kv = find_operation_by_name(role_split, "kv-append");
    if (role_qkv == nullptr || role_kv == nullptr) {
        std::cerr << "partition: missing role_split ops\n";
        return false;
    }
    if (role_qkv->config.primary_device_uid != "gpu:test:0" ||
        role_qkv->config.participating_devices.size() != 1u ||
        role_kv->config.primary_device_uid != "host:test:0") {
        std::cerr << "partition: role_split placement mismatch qkv=" << role_qkv->config.primary_device_uid
                  << " kv=" << role_kv->config.primary_device_uid << '\n';
        return false;
    }

    const auto projection_sharded = optimizer.optimize(
        make_workload(jakal::PartitionStrategy::projection_sharded),
        planner.build_plan(make_workload(jakal::PartitionStrategy::projection_sharded), graphs),
        graphs);
    const auto* sharded_qkv = find_operation_by_name(projection_sharded, "decode-qkv");
    if (sharded_qkv == nullptr) {
        std::cerr << "partition: missing sharded qkv\n";
        return false;
    }
    if (sharded_qkv->config.strategy != jakal::ExecutionStrategy::sharded ||
        sharded_qkv->config.participating_devices.size() != 2u ||
        sharded_qkv->config.logical_partitions != 4u) {
        std::cerr << "partition: sharded config mismatch strategy="
                  << jakal::to_string(sharded_qkv->config.strategy)
                  << " devices=" << sharded_qkv->config.participating_devices.size()
                  << " parts=" << sharded_qkv->config.logical_partitions << '\n';
        return false;
    }

    const auto tpu_like_plan = planner.build_plan(make_workload(jakal::PartitionStrategy::tpu_like), graphs);
    if (tpu_like_plan.allocations.size() != 2u ||
        tpu_like_plan.allocations.front().device.uid != "host:test:0" ||
        tpu_like_plan.allocations.back().device.uid != "gpu:test:0") {
        std::cerr << "partition: tpu_like allocation order mismatch\n";
        return false;
    }
    if (tpu_like_plan.allocations.front().ratio >= tpu_like_plan.allocations.back().ratio) {
        std::cerr << "partition: tpu_like ratios not gpu-biased host=" << tpu_like_plan.allocations.front().ratio
                  << " gpu=" << tpu_like_plan.allocations.back().ratio << '\n';
        return false;
    }

    const auto tpu_like = optimizer.optimize(make_workload(jakal::PartitionStrategy::tpu_like), tpu_like_plan, graphs);
    const auto* tpu_qkv = find_operation_by_name(tpu_like, "decode-qkv");
    const auto* tpu_reduce = find_operation_by_name(tpu_like, "decode-score-reduce");
    const auto* tpu_kv = find_operation_by_name(tpu_like, "kv-append");
    if (tpu_qkv == nullptr || tpu_reduce == nullptr || tpu_kv == nullptr) {
        std::cerr << "partition: missing tpu_like ops\n";
        return false;
    }
    if (tpu_qkv->config.primary_device_uid != "gpu:test:0" ||
        tpu_qkv->config.strategy != jakal::ExecutionStrategy::overlapped ||
        tpu_qkv->config.logical_partitions != 4u ||
        tpu_qkv->config.queue_depth < 3u ||
        tpu_qkv->config.stages < 3u) {
        std::cerr << "partition: tpu qkv mismatch primary=" << tpu_qkv->config.primary_device_uid
                  << " strategy=" << jakal::to_string(tpu_qkv->config.strategy)
                  << " parts=" << tpu_qkv->config.logical_partitions
                  << " queue=" << tpu_qkv->config.queue_depth
                  << " stages=" << tpu_qkv->config.stages << '\n';
        return false;
    }
    if (tpu_reduce->config.primary_device_uid != "gpu:test:0" ||
        tpu_reduce->config.strategy != jakal::ExecutionStrategy::overlapped) {
        std::cerr << "partition: tpu reduce mismatch primary=" << tpu_reduce->config.primary_device_uid
                  << " strategy=" << jakal::to_string(tpu_reduce->config.strategy) << '\n';
        return false;
    }
    if (tpu_kv->config.primary_device_uid != "host:test:0" ||
        tpu_kv->config.participating_devices.size() != 1u) {
        std::cerr << "partition: tpu kv mismatch primary=" << tpu_kv->config.primary_device_uid
                  << " devices=" << tpu_kv->config.participating_devices.size() << '\n';
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(plan_cache, ec);
    std::filesystem::remove(exec_cache, ec);
    std::filesystem::remove(exec_cache.string() + ".perf", ec);
    return true;
}

bool verify_device_agnostic_planning_and_execution() {
    const auto plan_cache = unique_temp_file("device-agnostic-plan");
    const auto exec_cache = unique_temp_file("device-agnostic-exec");

    jakal::Planner planner(plan_cache);
    jakal::ExecutionOptimizer optimizer(exec_cache);
    const auto host = make_manual_host_graph();
    auto gpu = make_manual_gpu_graph("gpu:device-agnostic:0", "Balanced GPU", true, true, true);
    gpu.nodes[2].compute.execution_width = 192;
    gpu.nodes[2].compute.matrix_engines = 24;
    gpu.nodes[2].compute.clock_mhz = 1900;
    attach_cache_and_scratch(gpu, 2ull * 1024ull * 1024ull, 256ull * 1024ull);
    const std::vector<jakal::HardwareGraph> graphs{host, gpu};

    const jakal::WorkloadSpec host_exchange_workload{
        "device-agnostic-risk-lite",
        jakal::WorkloadKind::tensor,
        "device-agnostic-risk-lite",
        384ull * 1024ull * 1024ull,
        160ull * 1024ull * 1024ull,
        8.0e10,
        1,
        true,
        true,
        false,
        jakal::PartitionStrategy::auto_balanced};
    const auto host_exchange_plan = planner.build_plan(host_exchange_workload, graphs);
    if (host_exchange_plan.allocations.size() < 2u) {
        std::cerr << "device-agnostic: expected mixed plan, got "
                  << host_exchange_plan.allocations.size() << " allocations\n";
        return false;
    }

    const auto host_allocation = std::find_if(
        host_exchange_plan.allocations.begin(),
        host_exchange_plan.allocations.end(),
        [&](const jakal::PlanAllocation& allocation) {
            return allocation.device.uid == host.uid;
        });
    const auto accelerator_allocation = std::find_if(
        host_exchange_plan.allocations.begin(),
        host_exchange_plan.allocations.end(),
        [&](const jakal::PlanAllocation& allocation) {
            return allocation.device.uid == gpu.uid;
        });
    if (host_allocation == host_exchange_plan.allocations.end() ||
        accelerator_allocation == host_exchange_plan.allocations.end()) {
        std::cerr << "device-agnostic: missing host or accelerator allocation\n";
        return false;
    }
    if (host_allocation->ratio < 0.18 || accelerator_allocation->ratio < 0.18) {
        std::cerr << "device-agnostic: expected meaningful host/accelerator split host="
                  << host_allocation->ratio << " accel=" << accelerator_allocation->ratio << '\n';
        return false;
    }

    const jakal::WorkloadSpec matrix_workload{
        "device-agnostic-matmul-lite",
        jakal::WorkloadKind::inference,
        "device-agnostic-matmul-lite",
        384ull * 1024ull * 1024ull,
        8ull * 1024ull * 1024ull,
        3.5e12,
        8,
        false,
        false,
        true,
        jakal::PartitionStrategy::auto_balanced};
    const auto matrix_plan = planner.build_plan(matrix_workload, graphs);
    const auto matrix_host = std::find_if(
        matrix_plan.allocations.begin(),
        matrix_plan.allocations.end(),
        [&](const jakal::PlanAllocation& allocation) {
            return allocation.device.uid == host.uid;
        });
    const auto matrix_accel = std::find_if(
        matrix_plan.allocations.begin(),
        matrix_plan.allocations.end(),
        [&](const jakal::PlanAllocation& allocation) {
            return allocation.device.uid == gpu.uid;
        });
    if (matrix_host == matrix_plan.allocations.end() ||
        matrix_accel == matrix_plan.allocations.end() ||
        matrix_accel->ratio <= matrix_host->ratio) {
        std::cerr << "device-agnostic: matrix workload lost accelerator preference host="
                  << (matrix_host == matrix_plan.allocations.end() ? -1.0 : matrix_host->ratio)
                  << " accel="
                  << (matrix_accel == matrix_plan.allocations.end() ? -1.0 : matrix_accel->ratio) << '\n';
        return false;
    }

    jakal::WorkloadGraph workload_graph;
    workload_graph.signature = "device-agnostic-manual";
    workload_graph.tensors = {
        {"risk-input", "risk-input", "", {"feature-normalize"}, 8ull * 1024ull * 1024ull, true, false, true},
        {"risk-normalized", "risk-normalized", "feature-normalize", {"risk-reduce", "projection-update"}, 8ull * 1024ull * 1024ull, false, true, true},
        {"risk-summary", "risk-summary", "risk-reduce", {}, 1ull * 1024ull * 1024ull, false, true, true},
        {"projection-weights", "projection-weights", "", {"projection-update"}, 24ull * 1024ull * 1024ull, true, false, false},
        {"projection-output", "projection-output", "projection-update", {}, 8ull * 1024ull * 1024ull, false, false, false}};
    workload_graph.operations = {
        {"feature-normalize",
         jakal::OperationClass::elementwise_map,
         {2'097'152},
         8ull * 1024ull * 1024ull,
         8ull * 1024ull * 1024ull,
         1ull * 1024ull * 1024ull,
         2.0e8,
         1.0e-4,
         true,
         false,
         false,
         false,
         {"risk-input"},
         {"risk-normalized"}},
        {"risk-reduce",
         jakal::OperationClass::reduction,
         {2'097'152},
         8ull * 1024ull * 1024ull,
         1ull * 1024ull * 1024ull,
         1ull * 1024ull * 1024ull,
         1.5e8,
         1.0e-4,
         true,
         true,
         false,
         false,
         {"risk-normalized"},
         {"risk-summary"}},
        {"projection-update",
         jakal::OperationClass::matmul,
         {1024, 1024, 512},
         32ull * 1024ull * 1024ull,
         8ull * 1024ull * 1024ull,
         4ull * 1024ull * 1024ull,
         4.0e10,
         1.0e-3,
         true,
         false,
         false,
         true,
         {"risk-normalized", "projection-weights"},
         {"projection-output"}}};

    const auto report =
        optimizer.optimize(host_exchange_workload, host_exchange_plan, graphs, &workload_graph);
    const auto* normalize = find_operation_by_name(report, "feature-normalize");
    const auto* reduce = find_operation_by_name(report, "risk-reduce");
    const auto* projection = find_operation_by_name(report, "projection-update");
    if (normalize == nullptr || reduce == nullptr || projection == nullptr) {
        std::cerr << "device-agnostic: missing optimized operations\n";
        return false;
    }
    if (normalize->config.primary_device_uid != host.uid ||
        reduce->config.primary_device_uid != host.uid) {
        std::cerr << "device-agnostic: expected host placement for small ops normalize="
                  << normalize->config.primary_device_uid
                  << " reduce=" << reduce->config.primary_device_uid << '\n';
        return false;
    }
    if (projection->config.primary_device_uid != gpu.uid) {
        std::cerr << "device-agnostic: expected accelerator placement for projection, got "
                  << projection->config.primary_device_uid << '\n';
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(plan_cache, ec);
    std::filesystem::remove(exec_cache, ec);
    std::filesystem::remove(exec_cache.string() + ".perf", ec);
    return true;
}

bool verify_aggressive_graph_rewrites() {
    const auto exec_cache = unique_temp_file("rewrite-exec");
    jakal::ExecutionOptimizer optimizer(exec_cache);

    auto host = make_manual_host_graph();
    auto gpu = make_manual_gpu_graph("gpu:rewrite:0", "Rewrite GPU", true, true, true);
    attach_cache_and_scratch(host, 2ull * 1024ull * 1024ull, 64ull * 1024ull);
    attach_cache_and_scratch(gpu, 2ull * 1024ull * 1024ull, 128ull * 1024ull);

    jakal::ExecutionPlan plan;
    plan.signature = "manual-rewrite";
    plan.allocations.push_back({host, 0.4, 1.0});
    plan.allocations.push_back({gpu, 0.6, 2.0});

    const jakal::WorkloadSpec workload{
        "rewrite-risk-lite",
        jakal::WorkloadKind::tensor,
        "rewrite-risk-lite",
        256ull * 1024ull * 1024ull,
        96ull * 1024ull * 1024ull,
        6.0e10,
        1,
        true,
        true,
        false};

    jakal::WorkloadGraph graph;
    graph.signature = "rewrite-manual";
    graph.tensors = {
        {"input", "input", "", {"ew-pre"}, 8ull * 1024ull * 1024ull, true, false, true},
        {"norm-1", "norm-1", "ew-pre", {"ew-post"}, 8ull * 1024ull * 1024ull, false, true, true},
        {"norm-2", "norm-2", "ew-post", {"projection"}, 8ull * 1024ull * 1024ull, false, true, true},
        {"weights", "weights", "", {"projection"}, 16ull * 1024ull * 1024ull, true, false, false},
        {"output", "output", "projection", {}, 8ull * 1024ull * 1024ull, false, false, false}};
    graph.operations = {
        {"ew-pre",
         jakal::OperationClass::elementwise_map,
         {2'097'152},
         8ull * 1024ull * 1024ull,
         8ull * 1024ull * 1024ull,
         1ull * 1024ull * 1024ull,
         2.0e8,
         1.0e-4,
         true,
         false,
         false,
         false,
         {"input"},
         {"norm-1"}},
        {"ew-post",
         jakal::OperationClass::elementwise_map,
         {2'097'152},
         8ull * 1024ull * 1024ull,
         8ull * 1024ull * 1024ull,
         1ull * 1024ull * 1024ull,
         2.5e8,
         1.0e-4,
         true,
         false,
         false,
         false,
         {"norm-1"},
         {"norm-2"}},
        {"projection",
         jakal::OperationClass::matmul,
         {1024, 1024, 512},
         24ull * 1024ull * 1024ull,
         8ull * 1024ull * 1024ull,
         4ull * 1024ull * 1024ull,
         5.0e10,
         1.0e-3,
         true,
         false,
         false,
         true,
         {"norm-2", "weights"},
         {"output"}}};

    jakal::ExecutionTuningOverrides tuning;
    tuning.graph_rewrite_level = 2u;
    tuning.graph_optimization_passes_override = 1u;
    const auto report = optimizer.optimize(workload, plan, {host, gpu}, &graph, &tuning);
    const auto* fused = find_operation_by_name(report, "ew-pre");
    const auto* removed = find_operation_by_name(report, "ew-post");
    if (fused == nullptr || removed != nullptr) {
        std::cerr << "rewrite: expected ew-post to fuse into ew-pre\n";
        return false;
    }
    if (std::find(
            fused->operation.fused_operation_names.begin(),
            fused->operation.fused_operation_names.end(),
            "ew-post") == fused->operation.fused_operation_names.end()) {
        std::cerr << "rewrite: ew-pre missing fused ew-post tag\n";
        return false;
    }
    if (report.workload_graph.operations.size() != 2u) {
        std::cerr << "rewrite: expected 2 ops after aggressive rewrite, got "
                  << report.workload_graph.operations.size() << '\n';
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(exec_cache, ec);
    std::filesystem::remove(exec_cache.string() + ".perf", ec);
    return true;
}

jakal::WorkloadGraph make_runtime_signal_meta_graph() {
    jakal::WorkloadGraph graph;
    graph.signature = "runtime-meta-signal";
    graph.tensors = {
        {"signal-in", "signal-in", "", {"analysis", "gate"}, 32ull * 1024ull * sizeof(float), false, false, true},
        {"analysis-w", "analysis-w", "", {"analysis"}, 1024ull * 512ull * sizeof(float), true, false, false},
        {"spectrum", "spectrum", "analysis", {"relu"}, 32ull * 512ull * sizeof(float), false, true, false},
        {"activated", "activated", "relu", {"synthesis"}, 32ull * 512ull * sizeof(float), false, true, false},
        {"synthesis-w", "synthesis-w", "", {"synthesis"}, 512ull * 256ull * sizeof(float), true, false, false},
        {"bank", "bank", "synthesis", {"mix"}, 32ull * 256ull * sizeof(float), false, true, false},
        {"gate-w", "gate-w", "", {"gate"}, 1024ull * 256ull * sizeof(float), true, false, false},
        {"gate-raw", "gate-raw", "gate", {"sigmoid"}, 32ull * 256ull * sizeof(float), false, true, false},
        {"gains", "gains", "sigmoid", {"mix"}, 32ull * 256ull * sizeof(float), false, true, false},
        {"mixed", "mixed", "mix", {"energy"}, 32ull * 256ull * sizeof(float), false, true, false},
        {"energy-out", "energy-out", "energy", {}, 32ull * sizeof(float), false, false, true}};

    jakal::OperationSpec analysis;
    analysis.name = "analysis";
    analysis.op_class = jakal::OperationClass::matmul;
    analysis.extents = {32, 512, 1024};
    analysis.input_bytes = (32ull * 1024ull + 1024ull * 512ull) * sizeof(float);
    analysis.output_bytes = 32ull * 512ull * sizeof(float);
    analysis.estimated_flops = 2.0 * 32.0 * 512.0 * 1024.0;
    analysis.matrix_friendly = true;
    analysis.input_tensor_ids = {"signal-in", "analysis-w"};
    analysis.output_tensor_ids = {"spectrum"};

    jakal::OperationSpec relu;
    relu.name = "relu";
    relu.op_class = jakal::OperationClass::elementwise_map;
    relu.extents = {32ull * 512ull};
    relu.input_bytes = 32ull * 512ull * sizeof(float);
    relu.output_bytes = 32ull * 512ull * sizeof(float);
    relu.estimated_flops = 3.0 * 32.0 * 512.0;
    relu.input_tensor_ids = {"spectrum"};
    relu.output_tensor_ids = {"activated"};

    jakal::OperationSpec synthesis;
    synthesis.name = "synthesis";
    synthesis.op_class = jakal::OperationClass::matmul;
    synthesis.extents = {32, 256, 512};
    synthesis.input_bytes = (32ull * 512ull + 512ull * 256ull) * sizeof(float);
    synthesis.output_bytes = 32ull * 256ull * sizeof(float);
    synthesis.estimated_flops = 2.0 * 32.0 * 256.0 * 512.0;
    synthesis.matrix_friendly = true;
    synthesis.input_tensor_ids = {"activated", "synthesis-w"};
    synthesis.output_tensor_ids = {"bank"};

    jakal::OperationSpec gate;
    gate.name = "gate";
    gate.op_class = jakal::OperationClass::matmul;
    gate.extents = {32, 256, 1024};
    gate.input_bytes = (32ull * 1024ull + 1024ull * 256ull) * sizeof(float);
    gate.output_bytes = 32ull * 256ull * sizeof(float);
    gate.estimated_flops = 2.0 * 32.0 * 256.0 * 1024.0;
    gate.matrix_friendly = true;
    gate.input_tensor_ids = {"signal-in", "gate-w"};
    gate.output_tensor_ids = {"gate-raw"};

    jakal::OperationSpec sigmoid;
    sigmoid.name = "sigmoid";
    sigmoid.op_class = jakal::OperationClass::elementwise_map;
    sigmoid.extents = {32ull * 256ull};
    sigmoid.input_bytes = 32ull * 256ull * sizeof(float);
    sigmoid.output_bytes = 32ull * 256ull * sizeof(float);
    sigmoid.estimated_flops = 4.0 * 32.0 * 256.0;
    sigmoid.input_tensor_ids = {"gate-raw"};
    sigmoid.output_tensor_ids = {"gains"};

    jakal::OperationSpec mix;
    mix.name = "mix";
    mix.op_class = jakal::OperationClass::elementwise_map;
    mix.extents = {32ull * 256ull};
    mix.input_bytes = 2ull * 32ull * 256ull * sizeof(float);
    mix.output_bytes = 32ull * 256ull * sizeof(float);
    mix.estimated_flops = 3.0 * 32.0 * 256.0;
    mix.input_tensor_ids = {"bank", "gains"};
    mix.output_tensor_ids = {"mixed"};

    jakal::OperationSpec energy;
    energy.name = "energy";
    energy.op_class = jakal::OperationClass::reduction;
    energy.extents = {32ull * 256ull};
    energy.input_bytes = 32ull * 256ull * sizeof(float);
    energy.output_bytes = 32ull * sizeof(float);
    energy.estimated_flops = 32.0 * 256.0;
    energy.reduction_like = true;
    energy.input_tensor_ids = {"mixed"};
    energy.output_tensor_ids = {"energy-out"};

    graph.operations = {analysis, relu, synthesis, gate, sigmoid, mix, energy};
    jakal::normalize_workload_graph(graph);
    return graph;
}

jakal::WorkloadGraph make_runtime_reduce_meta_graph() {
    jakal::WorkloadGraph graph;
    graph.signature = "runtime-meta-reduce";
    graph.tensors = {
        {"features", "features", "", {"norm"}, 64ull * 256ull * sizeof(float), false, false, true},
        {"normalized", "normalized", "norm", {"projection"}, 64ull * 256ull * sizeof(float), false, true, false},
        {"proj-w", "proj-w", "", {"projection"}, 256ull * 512ull * sizeof(float), true, false, false},
        {"projected", "projected", "projection", {"gelu"}, 64ull * 512ull * sizeof(float), false, true, false},
        {"activated", "activated", "gelu", {"scenario"}, 64ull * 512ull * sizeof(float), false, true, false},
        {"scenario-w", "scenario-w", "", {"scenario"}, 512ull * 128ull * sizeof(float), true, false, false},
        {"scores", "scores", "scenario", {"softmax"}, 64ull * 128ull * sizeof(float), false, true, false},
        {"probabilities", "probabilities", "softmax", {"portfolio-sum"}, 64ull * 128ull * sizeof(float), false, true, false},
        {"portfolio-out", "portfolio-out", "portfolio-sum", {}, 64ull * sizeof(float), false, false, true}};

    jakal::OperationSpec norm;
    norm.name = "norm";
    norm.op_class = jakal::OperationClass::elementwise_map;
    norm.extents = {64ull * 256ull};
    norm.input_bytes = 64ull * 256ull * sizeof(float);
    norm.output_bytes = 64ull * 256ull * sizeof(float);
    norm.estimated_flops = 5.0 * 64.0 * 256.0;
    norm.input_tensor_ids = {"features"};
    norm.output_tensor_ids = {"normalized"};

    jakal::OperationSpec projection;
    projection.name = "projection";
    projection.op_class = jakal::OperationClass::matmul;
    projection.extents = {64, 512, 256};
    projection.input_bytes = (64ull * 256ull + 256ull * 512ull) * sizeof(float);
    projection.output_bytes = 64ull * 512ull * sizeof(float);
    projection.estimated_flops = 2.0 * 64.0 * 512.0 * 256.0;
    projection.matrix_friendly = true;
    projection.input_tensor_ids = {"normalized", "proj-w"};
    projection.output_tensor_ids = {"projected"};

    jakal::OperationSpec gelu;
    gelu.name = "gelu";
    gelu.op_class = jakal::OperationClass::elementwise_map;
    gelu.extents = {64ull * 512ull};
    gelu.input_bytes = 64ull * 512ull * sizeof(float);
    gelu.output_bytes = 64ull * 512ull * sizeof(float);
    gelu.estimated_flops = 6.0 * 64.0 * 512.0;
    gelu.input_tensor_ids = {"projected"};
    gelu.output_tensor_ids = {"activated"};

    jakal::OperationSpec scenario;
    scenario.name = "scenario";
    scenario.op_class = jakal::OperationClass::matmul;
    scenario.extents = {64, 128, 512};
    scenario.input_bytes = (64ull * 512ull + 512ull * 128ull) * sizeof(float);
    scenario.output_bytes = 64ull * 128ull * sizeof(float);
    scenario.estimated_flops = 2.0 * 64.0 * 128.0 * 512.0;
    scenario.matrix_friendly = true;
    scenario.input_tensor_ids = {"activated", "scenario-w"};
    scenario.output_tensor_ids = {"scores"};

    jakal::OperationSpec softmax;
    softmax.name = "softmax";
    softmax.op_class = jakal::OperationClass::reduction;
    softmax.extents = {64ull * 128ull};
    softmax.input_bytes = 64ull * 128ull * sizeof(float);
    softmax.output_bytes = 64ull * 128ull * sizeof(float);
    softmax.estimated_flops = 3.0 * 64.0 * 128.0;
    softmax.reduction_like = true;
    softmax.input_tensor_ids = {"scores"};
    softmax.output_tensor_ids = {"probabilities"};

    jakal::OperationSpec portfolio_sum;
    portfolio_sum.name = "portfolio-sum";
    portfolio_sum.op_class = jakal::OperationClass::reduction;
    portfolio_sum.extents = {64ull * 128ull};
    portfolio_sum.input_bytes = 64ull * 128ull * sizeof(float);
    portfolio_sum.output_bytes = 64ull * sizeof(float);
    portfolio_sum.estimated_flops = 64.0 * 128.0;
    portfolio_sum.reduction_like = true;
    portfolio_sum.input_tensor_ids = {"probabilities"};
    portfolio_sum.output_tensor_ids = {"portfolio-out"};

    graph.operations = {norm, projection, gelu, scenario, softmax, portfolio_sum};
    jakal::normalize_workload_graph(graph);
    return graph;
}

bool verify_runtime_graph_aware_meta_policy() {
    const auto plan_cache = unique_temp_file("runtime-meta-plan");
    const auto exec_cache = unique_temp_file("runtime-meta-exec");

    jakal::RuntimeOptions options;
    options.cache_path = plan_cache;
    options.execution_cache_path = exec_cache;
    options.enable_opencl_probe = false;
    options.product.observability.persist_telemetry = false;
    jakal::Runtime runtime(options);

    bool saw_host = false;
    bool saw_accelerator = false;
    for (const auto& device : runtime.devices()) {
        saw_host = saw_host || device.probe == "host";
        saw_accelerator = saw_accelerator || device.probe != "host";
    }
    if (!(saw_host && saw_accelerator)) {
        std::error_code ec;
        std::filesystem::remove(plan_cache, ec);
        std::filesystem::remove(plan_cache.string() + ".strategy", ec);
        std::filesystem::remove(plan_cache.string() + ".strategy_family", ec);
        std::filesystem::remove(plan_cache.string() + ".confidence", ec);
        std::filesystem::remove(exec_cache, ec);
        std::filesystem::remove(exec_cache.string() + ".perf", ec);
        return true;
    }

    const jakal::WorkloadSpec projection_workload{
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
        jakal::PartitionStrategy::auto_balanced,
        jakal::WorkloadPhase::decode,
        "decode-small"};
    const auto projection_graph = jakal::default_workload_graph(projection_workload);
    const auto projection_report = runtime.optimize(projection_workload, projection_graph);
    if (projection_report.partition_strategy != jakal::PartitionStrategy::projection_sharded) {
        std::cerr << "runtime-meta: expected projection_sharded for decode projection graph, got "
                  << jakal::to_string(projection_report.partition_strategy) << '\n';
        return false;
    }
    if (projection_report.graph_optimization.passes.size() != 4u) {
        std::cerr << "runtime-meta: expected 4 graph passes for projection graph, got "
                  << projection_report.graph_optimization.passes.size() << '\n';
        return false;
    }

    const jakal::WorkloadSpec signal_workload{
        "signal-filterbank-meta",
        jakal::WorkloadKind::tensor,
        "signal-filterbank-meta",
        96ull * 1024ull * 1024ull,
        20ull * 1024ull * 1024ull,
        4.2e7,
        32,
        true,
        true,
        true,
        jakal::PartitionStrategy::auto_balanced,
        jakal::WorkloadPhase::unknown,
        "b32-f1024-c256"};
    const auto signal_report = runtime.optimize(signal_workload, make_runtime_signal_meta_graph());
    if (signal_report.partition_strategy != jakal::PartitionStrategy::auto_balanced) {
        std::cerr << "runtime-meta: expected cooperative auto_balanced for signal graph, got "
                  << jakal::to_string(signal_report.partition_strategy) << '\n';
        return false;
    }
    if (signal_report.graph_optimization.passes.size() != 2u) {
        std::cerr << "runtime-meta: expected 2 graph passes for signal graph, got "
                  << signal_report.graph_optimization.passes.size() << '\n';
        return false;
    }

    const jakal::WorkloadSpec reduce_workload{
        "finance-factor-meta",
        jakal::WorkloadKind::tensor,
        "finance-factor-meta",
        256ull * 1024ull * 1024ull,
        16ull * 1024ull * 1024ull,
        6.0e8,
        64,
        false,
        false,
        true,
        jakal::PartitionStrategy::auto_balanced,
        jakal::WorkloadPhase::unknown,
        "b64-f256-s128"};
    const auto reduce_report = runtime.optimize(reduce_workload, make_runtime_reduce_meta_graph());
    if (reduce_report.partition_strategy != jakal::PartitionStrategy::reduce_on_gpu) {
        std::cerr << "runtime-meta: expected reduce_on_gpu for reduction-tail graph, got "
                  << jakal::to_string(reduce_report.partition_strategy) << '\n';
        return false;
    }
    if (reduce_report.graph_optimization.passes.size() != 4u) {
        std::cerr << "runtime-meta: expected 4 graph passes for reduction-tail graph, got "
                  << reduce_report.graph_optimization.passes.size() << '\n';
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(plan_cache, ec);
    std::filesystem::remove(plan_cache.string() + ".strategy", ec);
    std::filesystem::remove(plan_cache.string() + ".strategy_family", ec);
    std::filesystem::remove(plan_cache.string() + ".confidence", ec);
    std::filesystem::remove(exec_cache, ec);
    std::filesystem::remove(exec_cache.string() + ".perf", ec);
    return true;
}

bool verify_cache_aware_tiling_for_inference() {
    const auto weak_cache = unique_temp_file("tiling-weak-exec");
    const auto strong_cache = unique_temp_file("tiling-strong-exec");

    auto weak_gpu = make_manual_gpu_graph("gpu:weak:0", "Weak GPU", false, false, false);
    weak_gpu.nodes[2].compute.execution_width = 32;
    weak_gpu.nodes[2].compute.clock_mhz = 900;
    weak_gpu.nodes[2].compute.matrix_engines = 0;
    attach_cache_and_scratch(weak_gpu, 128ull * 1024ull, 16ull * 1024ull);

    auto strong_gpu = make_manual_gpu_graph("gpu:strong:0", "Strong GPU", true, true, true);
    strong_gpu.nodes[2].compute.execution_width = 256;
    strong_gpu.nodes[2].compute.clock_mhz = 2200;
    strong_gpu.nodes[2].compute.matrix_engines = 64;
    strong_gpu.nodes[4].transfer.read_bandwidth_gbps = 256.0;
    strong_gpu.nodes[4].transfer.write_bandwidth_gbps = 256.0;
    attach_cache_and_scratch(strong_gpu, 2ull * 1024ull * 1024ull, 256ull * 1024ull);

    const jakal::WorkloadSpec workload{
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
        jakal::PartitionStrategy::auto_balanced,
        jakal::WorkloadPhase::decode,
        "decode-small"};

    const auto make_plan = [](const jakal::HardwareGraph& graph) {
        jakal::ExecutionPlan plan;
        plan.signature = "manual-" + graph.uid;
        plan.allocations.push_back({graph, 1.0, 1.0});
        return plan;
    };

    jakal::ExecutionOptimizer weak_optimizer(weak_cache);
    jakal::ExecutionOptimizer strong_optimizer(strong_cache);
    const auto weak_report = weak_optimizer.optimize(workload, make_plan(weak_gpu), {weak_gpu});
    const auto strong_report = strong_optimizer.optimize(workload, make_plan(strong_gpu), {strong_gpu});

    const auto* weak_matmul = find_first_operation_by_class(weak_report, jakal::OperationClass::matmul);
    const auto* strong_matmul = find_first_operation_by_class(strong_report, jakal::OperationClass::matmul);
    if (weak_matmul == nullptr || strong_matmul == nullptr) {
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(weak_cache, ec);
    std::filesystem::remove(strong_cache, ec);
    std::filesystem::remove(weak_cache.string() + ".perf", ec);
    std::filesystem::remove(strong_cache.string() + ".perf", ec);

    return strong_matmul->config.tile_x > weak_matmul->config.tile_x &&
           strong_matmul->config.tile_y > weak_matmul->config.tile_y &&
           strong_matmul->config.tile_k >= weak_matmul->config.tile_k;
}

bool verify_topology_aware_scheduler_for_inference() {
    auto edge_gpu = make_manual_gpu_graph("gpu:edge:0", "Edge GPU", false, false, false);
    edge_gpu.nodes[2].compute.execution_width = 64;
    edge_gpu.nodes[2].compute.clock_mhz = 1200;
    edge_gpu.nodes[2].compute.matrix_engines = 4;
    edge_gpu.nodes[4].transfer.read_bandwidth_gbps = 64.0;
    edge_gpu.nodes[4].transfer.write_bandwidth_gbps = 64.0;
    attach_cache_and_scratch(edge_gpu, 256ull * 1024ull, 32ull * 1024ull);

    auto datacenter_gpu = make_manual_gpu_graph("gpu:dc:0", "Datacenter GPU", true, true, true);
    datacenter_gpu.nodes[2].compute.execution_width = 256;
    datacenter_gpu.nodes[2].compute.clock_mhz = 2400;
    datacenter_gpu.nodes[2].compute.matrix_engines = 64;
    datacenter_gpu.nodes[4].transfer.read_bandwidth_gbps = 320.0;
    datacenter_gpu.nodes[4].transfer.write_bandwidth_gbps = 320.0;
    attach_cache_and_scratch(datacenter_gpu, 4ull * 1024ull * 1024ull, 512ull * 1024ull);

    jakal::OperationSpec operation;
    operation.name = "decode-qkv";
    operation.op_class = jakal::OperationClass::matmul;
    operation.extents = {1024, 1024, 1024};
    operation.input_bytes = 32ull * 1024ull * 1024ull;
    operation.output_bytes = 16ull * 1024ull * 1024ull;
    operation.temporary_bytes = 4ull * 1024ull * 1024ull;
    operation.estimated_flops = 4.0e11;
    operation.parallelizable = true;
    operation.matrix_friendly = true;

    jakal::OperationOptimizationResult optimized;
    optimized.operation = operation;
    optimized.config.primary_device_uid = datacenter_gpu.uid;
    optimized.config.participating_devices = {edge_gpu.uid, datacenter_gpu.uid};
    optimized.config.logical_partitions = 4;
    optimized.config.overlap_transfers = true;
    optimized.config.use_low_precision = true;

    jakal::OptimizationReport report;
    report.placement.allocations.push_back({edge_gpu, 0.5, 1.0});
    report.placement.allocations.push_back({datacenter_gpu, 0.5, 1.0});

    const std::vector<jakal::HardwareGraph> graphs{edge_gpu, datacenter_gpu};
    jakal::executors::DefaultIntraDeviceScheduler scheduler;
    const auto assignments = scheduler.make_assignments(report, optimized, graphs, 1024u);
    if (assignments.size() != 8u) {
        return false;
    }

    std::size_t edge_count = 0;
    std::size_t datacenter_count = 0;
    for (const auto& assignment : assignments) {
        if (assignment.graph == nullptr) {
            return false;
        }
        if (assignment.graph->uid == edge_gpu.uid) {
            edge_count += assignment.shard.count;
        } else if (assignment.graph->uid == datacenter_gpu.uid) {
            datacenter_count += assignment.shard.count;
        }
    }

    return datacenter_count > edge_count;
}

bool verify_operation_lowering_for_cpu_and_gpu_inference() {
    const auto exec_cache = unique_temp_file("lowering-exec");
    jakal::ExecutionOptimizer optimizer(exec_cache);

    auto host = make_manual_host_graph();
    host.nodes[2].compute.native_vector_bits = 512;
    host.nodes[2].compute.execution_width = 32;
    attach_cache_and_scratch(host, 2ull * 1024ull * 1024ull, 64ull * 1024ull);

    auto gpu = make_manual_gpu_graph("gpu:infer:0", "Inference GPU", true, true, true);
    gpu.nodes[2].compute.execution_width = 256;
    gpu.nodes[2].compute.matrix_engines = 64;
    gpu.nodes[2].compute.clock_mhz = 2200;
    attach_cache_and_scratch(gpu, 4ull * 1024ull * 1024ull, 256ull * 1024ull);

    jakal::ExecutionPlan plan;
    plan.signature = "manual-lowering";
    plan.allocations.push_back({host, 0.2, 1.0});
    plan.allocations.push_back({gpu, 0.8, 4.0});

    const jakal::WorkloadSpec workload{
        "ai-vision-inference-lite",
        jakal::WorkloadKind::inference,
        "ai-vision-inference-224",
        512ull * 1024ull * 1024ull,
        64ull * 1024ull * 1024ull,
        4.0e12,
        8,
        true,
        false,
        true,
        jakal::PartitionStrategy::auto_balanced,
        jakal::WorkloadPhase::prefill,
        "vision-mid"};

    const auto report = optimizer.optimize(workload, plan, {host, gpu});
    const auto* mlp_up = find_operation_by_name(report, "mlp-up");
    const auto* patch_proj = find_operation_by_name(report, "patch-proj");
    const auto* mlp_activation = find_operation_by_name(report, "mlp-activation");
    if (mlp_up == nullptr || patch_proj == nullptr) {
        std::cerr << "lowering: missing mlp-up or patch-proj\n";
        return false;
    }
    if (mlp_activation != nullptr) {
        std::cerr << "lowering: ops";
        for (const auto& operation : report.workload_graph.operations) {
            std::cerr << ' ' << operation.name;
        }
        std::cerr << '\n';
        const auto tensor_it = std::find_if(
            report.workload_graph.tensors.begin(),
            report.workload_graph.tensors.end(),
            [](const jakal::WorkloadTensor& tensor) { return tensor.id == "mlp-up-out"; });
        if (tensor_it != report.workload_graph.tensors.end()) {
            std::cerr << "lowering: mlp-up-out producer=" << tensor_it->producer_operation
                      << " consumers=" << tensor_it->consumer_operations.size();
            for (const auto& consumer : tensor_it->consumer_operations) {
                std::cerr << " " << consumer;
            }
            std::cerr << '\n';
        }
        std::cerr << "lowering: mlp-activation still present\n";
        return false;
    }
    if (std::find(mlp_up->operation.fused_operation_names.begin(),
                  mlp_up->operation.fused_operation_names.end(),
                  "mlp-activation") == mlp_up->operation.fused_operation_names.end()) {
        std::cerr << "lowering: mlp-up fused ops missing mlp-activation\n";
        return false;
    }
    if (!patch_proj->operation.cpu_pack_weights ||
        !patch_proj->operation.gpu_pack_weights ||
        !patch_proj->operation.cpu_pretranspose_rhs ||
        !patch_proj->operation.gpu_pretranspose_rhs ||
        !patch_proj->operation.cpu_vectorized ||
        !patch_proj->operation.gpu_tensorized ||
        patch_proj->operation.cpu_weight_layout == "native" ||
        patch_proj->operation.gpu_weight_layout == "native" ||
        patch_proj->operation.cpu_output_layout == "native" ||
        patch_proj->operation.gpu_output_layout == "native" ||
        patch_proj->operation.cpu_micro_kernel_unroll < 2u ||
        patch_proj->operation.gpu_micro_kernel_unroll < 2u) {
        std::cerr << "lowering: patch-proj hints cpu_pack=" << patch_proj->operation.cpu_pack_weights
                  << " gpu_pack=" << patch_proj->operation.gpu_pack_weights
                  << " cpu_preT=" << patch_proj->operation.cpu_pretranspose_rhs
                  << " gpu_preT=" << patch_proj->operation.gpu_pretranspose_rhs
                  << " cpu_vec=" << patch_proj->operation.cpu_vectorized
                  << " gpu_tensor=" << patch_proj->operation.gpu_tensorized
                  << " cpu_w=" << patch_proj->operation.cpu_weight_layout
                  << " gpu_w=" << patch_proj->operation.gpu_weight_layout
                  << " cpu_out=" << patch_proj->operation.cpu_output_layout
                  << " gpu_out=" << patch_proj->operation.gpu_output_layout
                  << " cpu_u=" << patch_proj->operation.cpu_micro_kernel_unroll
                  << " gpu_u=" << patch_proj->operation.gpu_micro_kernel_unroll << '\n';
        return false;
    }
    if (patch_proj->config.variant_id.find("gpu-lowered") == std::string::npos) {
        std::cerr << "lowering: patch-proj variant missing gpu-lowered tag: "
                  << patch_proj->config.variant_id << '\n';
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(exec_cache, ec);
    std::filesystem::remove(exec_cache.string() + ".perf", ec);
    if (report.workload_graph.operations.size() >= 8u) {
        std::cerr << "lowering: expected fewer than 8 ops, got " << report.workload_graph.operations.size() << '\n';
        return false;
    }
    return true;
}

bool verify_operation_lowering_for_non_dl_graphs() {
    const auto exec_cache = unique_temp_file("lowering-gaming-exec");
    jakal::ExecutionOptimizer optimizer(exec_cache);

    auto host = make_manual_host_graph();
    host.nodes[2].compute.native_vector_bits = 256;
    attach_cache_and_scratch(host, 1024ull * 1024ull, 64ull * 1024ull);

    auto gpu = make_manual_gpu_graph("gpu:gaming:0", "Gaming GPU", true, false, false);
    gpu.nodes[2].compute.execution_width = 192;
    gpu.nodes[2].compute.matrix_engines = 16;
    attach_cache_and_scratch(gpu, 2ull * 1024ull * 1024ull, 128ull * 1024ull);

    jakal::ExecutionPlan plan;
    plan.signature = "manual-gaming-lowering";
    plan.allocations.push_back({host, 0.35, 1.0});
    plan.allocations.push_back({gpu, 0.65, 2.0});

    const jakal::WorkloadSpec workload{
        "gaming-upscale-1080p",
        jakal::WorkloadKind::gaming,
        "gaming-fsr-like-720p-to-1080p",
        512ull * 1024ull * 1024ull,
        96ull * 1024ull * 1024ull,
        2.0e12,
        1,
        true,
        false,
        false};

    const auto report = optimizer.optimize(workload, plan, {host, gpu});
    const auto* upscale = find_operation_by_name(report, "upscale-resolve");
    const auto* post = find_operation_by_name(report, "post-tonemap");
    const auto* history = find_operation_by_name(report, "history-reconstruction");
    if (upscale == nullptr || history == nullptr) {
        return false;
    }
    if (post != nullptr) {
        return false;
    }
    if (std::find(upscale->operation.fused_operation_names.begin(),
                  upscale->operation.fused_operation_names.end(),
                  "post-tonemap") == upscale->operation.fused_operation_names.end()) {
        return false;
    }
    if (!history->operation.cpu_vectorized ||
        !history->operation.gpu_tensorized ||
        history->operation.cpu_input_layout == "native" ||
        history->operation.gpu_input_layout == "native") {
        return false;
    }
    if (upscale->config.variant_id.find("gpu-lowered") == std::string::npos &&
        upscale->config.variant_id.find("cpu-lowered") == std::string::npos) {
        std::cerr << "gaming lowering: upscale-resolve variant missing lowered tag: "
                  << upscale->config.variant_id << '\n';
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(exec_cache, ec);
    std::filesystem::remove(exec_cache.string() + ".perf", ec);
    return report.workload_graph.operations.size() < 7u;
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
        {"level-zero-conv", "Intel Arc A770", jakal::JakalVendorFamily::intel, jakal::JakalBackendKind::level_zero, jakal::OperationClass::convolution_2d, {64, 64}, true, false, true},
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
        switch (test_case.backend) {
        case jakal::JakalBackendKind::level_zero:
            graph.probe = "level-zero";
            break;
        case jakal::JakalBackendKind::cuda:
            graph.probe = "cuda";
            break;
        case jakal::JakalBackendKind::rocm:
            graph.probe = "rocm";
            break;
        case jakal::JakalBackendKind::vulkan_compute:
            graph.probe = "vulkan";
            break;
        case jakal::JakalBackendKind::opencl:
            graph.probe = "opencl";
            break;
        default:
            break;
        }

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
        if (test_case.op_class == jakal::OperationClass::matmul) {
            operation.gpu_pack_weights = true;
            operation.gpu_pretranspose_rhs = true;
            operation.gpu_tensorized = true;
            operation.gpu_weight_layout = "gpu-tensorcore-tiled";
            operation.gpu_output_layout = "gpu-tile-accumulator";
            operation.gpu_micro_kernel_unroll = 4;
        } else if (test_case.op_class == jakal::OperationClass::convolution_2d) {
            operation.gpu_input_layout = "gpu-conv-patch9";
            operation.gpu_output_layout = "gpu-conv-accumulator";
            operation.gpu_tensorized = true;
        } else if (test_case.op_class == jakal::OperationClass::resample_2d) {
            operation.gpu_input_layout = "gpu-resample-packed6";
            operation.gpu_output_layout = "gpu-resample-linear";
            operation.gpu_tensorized = true;
        }

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
        if ((test_case.op_class == jakal::OperationClass::matmul ||
             test_case.op_class == jakal::OperationClass::convolution_2d ||
             test_case.op_class == jakal::OperationClass::resample_2d) &&
            record.backend_name.find("gpu-lowered") == std::string::npos) {
            std::cerr << "Expected lowered GPU backend tag for " << test_case.name
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
           has_variant(reduction_variants, "host-critical-path") &&
           has_variant(matmul_variants, "throughput-overlap") &&
           has_variant(matmul_variants, "cooperative-split") &&
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
    if (!verify_device_agnostic_planning_and_execution()) {
        std::cerr << "Device-agnostic planning check failed.\n";
        return 1;
    }
    std::cerr << "stage: device-agnostic\n";
    if (!verify_aggressive_graph_rewrites()) {
        std::cerr << "Aggressive graph rewrite check failed.\n";
        return 1;
    }
    std::cerr << "stage: rewrites\n";
    if (!verify_runtime_graph_aware_meta_policy()) {
        std::cerr << "Runtime graph-aware meta-policy check failed.\n";
        return 1;
    }
    std::cerr << "stage: runtime-meta\n";
    if (!verify_cache_aware_tiling_for_inference()) {
        std::cerr << "Inference cache-aware tiling check failed.\n";
        return 1;
    }
    std::cerr << "stage: tiling\n";
    if (!verify_topology_aware_scheduler_for_inference()) {
        std::cerr << "Inference topology-aware scheduler check failed.\n";
        return 1;
    }
    std::cerr << "stage: scheduler\n";
    if (!verify_operation_lowering_for_cpu_and_gpu_inference()) {
        std::cerr << "Inference operation lowering check failed.\n";
        return 1;
    }
    std::cerr << "stage: lowering\n";
    if (!verify_operation_lowering_for_non_dl_graphs()) {
        std::cerr << "Non-DL operation lowering check failed.\n";
        return 1;
    }
    std::cerr << "stage: lowering-non-dl\n";

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

