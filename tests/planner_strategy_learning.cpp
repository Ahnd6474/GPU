#include "jakal/device.hpp"
#include "jakal/planner.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::filesystem::path unique_temp_file(const std::string& stem) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (stem + "-" + std::to_string(nonce) + ".tsv");
}

jakal::HardwareGraph make_host_graph() {
    jakal::HardwareGraph graph;
    graph.uid = "host:test:0";
    graph.probe = "host";
    graph.presentation_name = "Test CPU";

    graph.nodes.push_back({"root", "root", "", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::root});
    graph.nodes.push_back({"queue", "queue", "root", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::queue});
    graph.nodes.push_back({"cluster", "cluster", "root", jakal::HardwareObjectDomain::compute, jakal::HardwareObjectRole::cluster});
    graph.nodes.back().compute.execution_width = 16;
    graph.nodes.back().compute.clock_mhz = 4000;
    graph.nodes.push_back({"memory", "memory", "root", jakal::HardwareObjectDomain::storage, jakal::HardwareObjectRole::host_memory});
    graph.nodes.back().storage.capacity_bytes = 32ull * 1024ull * 1024ull * 1024ull;
    graph.nodes.back().storage.unified_address_space = true;
    graph.nodes.back().storage.coherent_with_host = true;
    graph.nodes.back().storage.shared_host_bytes = graph.nodes.back().storage.capacity_bytes;

    graph.edges.push_back({"root", "queue", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "cluster", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "memory", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"queue", "cluster", jakal::GraphEdgeSemantics::dispatches, true, 1.0, 0.0, 12.0});
    graph.edges.push_back({"memory", "cluster", jakal::GraphEdgeSemantics::feeds, true, 1.0, 64.0, 8.0});
    jakal::materialize_graph_costs(graph);
    return graph;
}

jakal::HardwareGraph make_gpu_graph() {
    jakal::HardwareGraph graph;
    graph.uid = "gpu:test:0";
    graph.probe = "opencl";
    graph.presentation_name = "Test GPU";

    graph.nodes.push_back({"root", "root", "", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::root});
    graph.nodes.push_back({"queue", "queue", "root", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::queue});
    graph.nodes.back().control.supports_asynchronous_dispatch = true;
    graph.nodes.push_back({"cluster", "cluster", "root", jakal::HardwareObjectDomain::compute, jakal::HardwareObjectRole::cluster});
    graph.nodes.back().compute.execution_width = 128;
    graph.nodes.back().compute.clock_mhz = 1800;
    graph.nodes.back().compute.matrix_engines = 16;
    graph.nodes.push_back({"memory", "memory", "root", jakal::HardwareObjectDomain::storage, jakal::HardwareObjectRole::global_memory});
    graph.nodes.back().storage.capacity_bytes = 8ull * 1024ull * 1024ull * 1024ull;
    graph.nodes.push_back({"link", "link", "root", jakal::HardwareObjectDomain::transfer, jakal::HardwareObjectRole::transfer_link});
    graph.nodes.back().transfer.read_bandwidth_gbps = 128.0;
    graph.nodes.back().transfer.write_bandwidth_gbps = 128.0;
    graph.nodes.back().transfer.dispatch_latency_us = 5.0;
    graph.nodes.back().transfer.synchronization_latency_us = 4.0;

    graph.edges.push_back({"root", "queue", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "cluster", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "memory", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "link", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"queue", "cluster", jakal::GraphEdgeSemantics::dispatches, true, 1.0, 0.0, 5.0});
    graph.edges.push_back({"link", "memory", jakal::GraphEdgeSemantics::transfers_to, true, 1.0, 128.0, 4.0});
    jakal::materialize_graph_costs(graph);
    return graph;
}

jakal::HardwareGraph make_gpu_graph_family_variant() {
    auto graph = make_gpu_graph();
    graph.uid = "gpu:test:1";
    graph.presentation_name = "Test GPU Variant";
    graph.nodes[2].compute.execution_width = 96;
    graph.nodes[2].compute.clock_mhz = 1650;
    graph.nodes[2].compute.matrix_engines = 12;
    graph.nodes[3].storage.capacity_bytes = 6ull * 1024ull * 1024ull * 1024ull;
    graph.nodes[4].transfer.read_bandwidth_gbps = 96.0;
    graph.nodes[4].transfer.write_bandwidth_gbps = 96.0;
    graph.nodes[4].transfer.dispatch_latency_us = 6.0;
    graph.nodes[4].transfer.synchronization_latency_us = 5.0;
    jakal::materialize_graph_costs(graph);
    return graph;
}

}  // namespace

bool expect_strategy(
    const char* label,
    const jakal::ExecutionPlan& plan,
    const jakal::PartitionStrategy expected) {
    if (plan.resolved_partition_strategy == expected) {
        return true;
    }

    std::cerr << label << ": expected " << jakal::to_string(expected) << " but got "
              << jakal::to_string(plan.resolved_partition_strategy) << '\n';
    return false;
}

bool test_strategy_exploration() {
    const auto cache_path = unique_temp_file("planner-exploration");
    jakal::Planner planner(cache_path);
    const std::vector<jakal::HardwareGraph> graphs{make_host_graph(), make_gpu_graph()};

    const jakal::WorkloadSpec workload{
        "decode-lite",
        jakal::WorkloadKind::inference,
        "llm-decode-token-lite",
        640ull * 1024ull * 1024ull,
        12ull * 1024ull * 1024ull,
        3.8e10,
        1,
        true,
        true,
        true,
        jakal::PartitionStrategy::auto_balanced};

    const auto initial_plan = planner.build_plan(workload, graphs);
    if (!expect_strategy("initial exploration plan", initial_plan, jakal::PartitionStrategy::auto_balanced)) {
        return false;
    }

    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::auto_balanced, 1200.0, 420.0, 1.0, 1.0, true});

    const auto exploratory_plan = planner.build_plan(workload, graphs);
    if (!expect_strategy("exploratory plan", exploratory_plan, jakal::PartitionStrategy::role_split)) {
        return false;
    }

    const auto strategy_cache = cache_path.string() + ".strategy";
    const auto family_strategy_cache = cache_path.string() + ".strategy_family";
    std::error_code ec;
    std::filesystem::remove(cache_path, ec);
    std::filesystem::remove(strategy_cache, ec);
    std::filesystem::remove(family_strategy_cache, ec);
    return true;
}

bool test_strategy_learning() {
    const auto cache_path = unique_temp_file("planner-learning");
    jakal::Planner planner(cache_path);
    const std::vector<jakal::HardwareGraph> graphs{make_host_graph(), make_gpu_graph()};

    const jakal::WorkloadSpec workload{
        "decode-lite",
        jakal::WorkloadKind::inference,
        "llm-decode-token-lite",
        640ull * 1024ull * 1024ull,
        12ull * 1024ull * 1024ull,
        3.8e10,
        1,
        true,
        true,
        true,
        jakal::PartitionStrategy::auto_balanced};

    const auto initial_plan = planner.build_plan(workload, graphs);
    if (!expect_strategy("initial learning plan", initial_plan, jakal::PartitionStrategy::auto_balanced)) {
        return false;
    }

    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::auto_balanced, 1200.0, 420.0, 1.0, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::tpu_like, 960.0, 360.0, 1.18, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::role_split, 900.0, 290.0, 1.20, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::role_split, 880.0, 280.0, 1.24, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::reduce_on_gpu, 1100.0, 390.0, 1.05, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::projection_sharded, 1030.0, 325.0, 1.11, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::blind_sharded, 1300.0, 510.0, 0.92, 0.4, false});

    const auto learned_plan = planner.build_plan(workload, graphs);
    if (!expect_strategy("learned plan", learned_plan, jakal::PartitionStrategy::role_split)) {
        return false;
    }

    const auto strategy_cache = cache_path.string() + ".strategy";
    const auto family_strategy_cache = cache_path.string() + ".strategy_family";
    std::error_code ec;
    std::filesystem::remove(cache_path, ec);
    std::filesystem::remove(strategy_cache, ec);
    std::filesystem::remove(family_strategy_cache, ec);
    return true;
}

bool test_latency_weighted_learning() {
    const auto cache_path = unique_temp_file("planner-latency-weight");
    jakal::Planner planner(cache_path);
    const std::vector<jakal::HardwareGraph> graphs{make_host_graph(), make_gpu_graph()};

    const jakal::WorkloadSpec workload{
        "decode-latency",
        jakal::WorkloadKind::inference,
        "llm-decode-token-lite",
        640ull * 1024ull * 1024ull,
        12ull * 1024ull * 1024ull,
        3.8e10,
        1,
        true,
        true,
        true,
        jakal::PartitionStrategy::auto_balanced};

    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::auto_balanced, 1000.0, 360.0, 1.00, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::blind_sharded, 1040.0, 355.0, 0.97, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::role_split, 930.0, 340.0, 1.16, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::reduce_on_gpu, 980.0, 330.0, 1.03, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::projection_sharded, 970.0, 210.0, 1.01, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::tpu_like, 960.0, 240.0, 1.02, 1.0, true});

    const auto latency_plan = planner.build_plan(workload, graphs);
    if (!expect_strategy("latency-weighted plan", latency_plan, jakal::PartitionStrategy::projection_sharded)) {
        return false;
    }

    const auto strategy_cache = cache_path.string() + ".strategy";
    const auto family_strategy_cache = cache_path.string() + ".strategy_family";
    std::error_code ec;
    std::filesystem::remove(cache_path, ec);
    std::filesystem::remove(strategy_cache, ec);
    std::filesystem::remove(family_strategy_cache, ec);
    return true;
}

bool test_failure_penalty_learning() {
    const auto cache_path = unique_temp_file("planner-failure-penalty");
    jakal::Planner planner(cache_path);
    const std::vector<jakal::HardwareGraph> graphs{make_host_graph(), make_gpu_graph()};

    const jakal::WorkloadSpec workload{
        "throughput-batch",
        jakal::WorkloadKind::training,
        "tensor-batch",
        2ull * 1024ull * 1024ull * 1024ull,
        256ull * 1024ull * 1024ull,
        1.4e12,
        16,
        false,
        false,
        true,
        jakal::PartitionStrategy::auto_balanced};

    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::auto_balanced, 2100.0, 2100.0, 1.00, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::blind_sharded, 2000.0, 2000.0, 1.05, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::role_split, 1600.0, 1600.0, 1.32, 0.45, false});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::reduce_on_gpu, 1720.0, 1720.0, 1.22, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::projection_sharded, 1800.0, 1800.0, 1.15, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        graphs,
        {jakal::PartitionStrategy::tpu_like, 1760.0, 1760.0, 1.19, 0.92, true});

    const auto robust_plan = planner.build_plan(workload, graphs);
    if (!expect_strategy("failure-penalty plan", robust_plan, jakal::PartitionStrategy::reduce_on_gpu)) {
        return false;
    }

    const auto strategy_cache = cache_path.string() + ".strategy";
    const auto family_strategy_cache = cache_path.string() + ".strategy_family";
    std::error_code ec;
    std::filesystem::remove(cache_path, ec);
    std::filesystem::remove(strategy_cache, ec);
    std::filesystem::remove(family_strategy_cache, ec);
    return true;
}

bool test_phase_scoped_learning() {
    const auto cache_path = unique_temp_file("planner-phase-split");
    jakal::Planner planner(cache_path);
    const std::vector<jakal::HardwareGraph> graphs{make_host_graph(), make_gpu_graph()};

    const jakal::WorkloadSpec prefill_workload{
        "llm-shared-phase-lite",
        jakal::WorkloadKind::inference,
        "llm-shared-phase-lite",
        768ull * 1024ull * 1024ull,
        48ull * 1024ull * 1024ull,
        2.2e11,
        1,
        false,
        false,
        true,
        jakal::PartitionStrategy::auto_balanced,
        jakal::WorkloadPhase::prefill};

    const auto decode_workload = jakal::WorkloadSpec{
        "llm-shared-phase-lite",
        jakal::WorkloadKind::inference,
        "llm-shared-phase-lite",
        768ull * 1024ull * 1024ull,
        48ull * 1024ull * 1024ull,
        2.2e11,
        1,
        true,
        true,
        true,
        jakal::PartitionStrategy::auto_balanced,
        jakal::WorkloadPhase::decode};

    planner.ingest_strategy_feedback(
        prefill_workload,
        graphs,
        {jakal::PartitionStrategy::auto_balanced, 1400.0, 1400.0, 1.00, 1.0, true});
    planner.ingest_strategy_feedback(
        prefill_workload,
        graphs,
        {jakal::PartitionStrategy::blind_sharded, 1500.0, 1500.0, 0.93, 1.0, true});
    planner.ingest_strategy_feedback(
        prefill_workload,
        graphs,
        {jakal::PartitionStrategy::role_split, 1320.0, 1320.0, 1.06, 1.0, true});
    planner.ingest_strategy_feedback(
        prefill_workload,
        graphs,
        {jakal::PartitionStrategy::reduce_on_gpu, 1180.0, 1180.0, 1.19, 1.0, true});
    planner.ingest_strategy_feedback(
        prefill_workload,
        graphs,
        {jakal::PartitionStrategy::projection_sharded, 1240.0, 1240.0, 1.11, 1.0, true});
    planner.ingest_strategy_feedback(
        prefill_workload,
        graphs,
        {jakal::PartitionStrategy::tpu_like, 1210.0, 1210.0, 1.14, 1.0, true});

    planner.ingest_strategy_feedback(
        decode_workload,
        graphs,
        {jakal::PartitionStrategy::auto_balanced, 910.0, 320.0, 1.00, 1.0, true});
    planner.ingest_strategy_feedback(
        decode_workload,
        graphs,
        {jakal::PartitionStrategy::blind_sharded, 980.0, 360.0, 0.93, 1.0, true});
    planner.ingest_strategy_feedback(
        decode_workload,
        graphs,
        {jakal::PartitionStrategy::role_split, 860.0, 280.0, 1.05, 1.0, true});
    planner.ingest_strategy_feedback(
        decode_workload,
        graphs,
        {jakal::PartitionStrategy::reduce_on_gpu, 900.0, 290.0, 1.01, 1.0, true});
    planner.ingest_strategy_feedback(
        decode_workload,
        graphs,
        {jakal::PartitionStrategy::projection_sharded, 845.0, 210.0, 1.03, 1.0, true});
    planner.ingest_strategy_feedback(
        decode_workload,
        graphs,
        {jakal::PartitionStrategy::tpu_like, 870.0, 245.0, 1.02, 1.0, true});

    const auto prefill_plan = planner.build_plan(prefill_workload, graphs);
    if (!expect_strategy("prefill phase plan", prefill_plan, jakal::PartitionStrategy::reduce_on_gpu)) {
        return false;
    }

    const auto decode_plan = planner.build_plan(decode_workload, graphs);
    if (!expect_strategy("decode phase plan", decode_plan, jakal::PartitionStrategy::projection_sharded)) {
        return false;
    }

    const auto strategy_cache = cache_path.string() + ".strategy";
    const auto family_strategy_cache = cache_path.string() + ".strategy_family";
    std::error_code ec;
    std::filesystem::remove(cache_path, ec);
    std::filesystem::remove(strategy_cache, ec);
    std::filesystem::remove(family_strategy_cache, ec);
    return true;
}

bool test_shape_bucket_scoped_learning() {
    const auto cache_path = unique_temp_file("planner-shape-split");
    jakal::Planner planner(cache_path);
    const std::vector<jakal::HardwareGraph> graphs{make_host_graph(), make_gpu_graph()};

    const jakal::WorkloadSpec short_decode_workload{
        "llm-shape-shared-lite",
        jakal::WorkloadKind::inference,
        "llm-shape-shared-lite",
        640ull * 1024ull * 1024ull,
        12ull * 1024ull * 1024ull,
        3.8e10,
        1,
        true,
        true,
        true,
        jakal::PartitionStrategy::auto_balanced,
        jakal::WorkloadPhase::decode};

    const jakal::WorkloadSpec long_decode_workload{
        "llm-shape-shared-lite",
        jakal::WorkloadKind::inference,
        "llm-shape-shared-lite",
        1536ull * 1024ull * 1024ull,
        96ull * 1024ull * 1024ull,
        1.3e11,
        4,
        true,
        true,
        true,
        jakal::PartitionStrategy::auto_balanced,
        jakal::WorkloadPhase::decode};

    planner.ingest_strategy_feedback(
        short_decode_workload,
        graphs,
        {jakal::PartitionStrategy::auto_balanced, 920.0, 325.0, 1.00, 1.0, true});
    planner.ingest_strategy_feedback(
        short_decode_workload,
        graphs,
        {jakal::PartitionStrategy::blind_sharded, 980.0, 360.0, 0.94, 1.0, true});
    planner.ingest_strategy_feedback(
        short_decode_workload,
        graphs,
        {jakal::PartitionStrategy::role_split, 860.0, 275.0, 1.06, 1.0, true});
    planner.ingest_strategy_feedback(
        short_decode_workload,
        graphs,
        {jakal::PartitionStrategy::reduce_on_gpu, 900.0, 290.0, 1.02, 1.0, true});
    planner.ingest_strategy_feedback(
        short_decode_workload,
        graphs,
        {jakal::PartitionStrategy::projection_sharded, 845.0, 205.0, 1.04, 1.0, true});
    planner.ingest_strategy_feedback(
        short_decode_workload,
        graphs,
        {jakal::PartitionStrategy::tpu_like, 870.0, 245.0, 1.02, 1.0, true});

    planner.ingest_strategy_feedback(
        long_decode_workload,
        graphs,
        {jakal::PartitionStrategy::auto_balanced, 1680.0, 610.0, 1.00, 1.0, true});
    planner.ingest_strategy_feedback(
        long_decode_workload,
        graphs,
        {jakal::PartitionStrategy::blind_sharded, 1810.0, 700.0, 0.93, 1.0, true});
    planner.ingest_strategy_feedback(
        long_decode_workload,
        graphs,
        {jakal::PartitionStrategy::role_split, 1610.0, 580.0, 1.04, 1.0, true});
    planner.ingest_strategy_feedback(
        long_decode_workload,
        graphs,
        {jakal::PartitionStrategy::reduce_on_gpu, 1430.0, 520.0, 1.17, 1.0, true});
    planner.ingest_strategy_feedback(
        long_decode_workload,
        graphs,
        {jakal::PartitionStrategy::projection_sharded, 1510.0, 560.0, 1.10, 1.0, true});
    planner.ingest_strategy_feedback(
        long_decode_workload,
        graphs,
        {jakal::PartitionStrategy::tpu_like, 1475.0, 540.0, 1.12, 1.0, true});

    const auto short_plan = planner.build_plan(short_decode_workload, graphs);
    if (!expect_strategy("short decode shape-bucket plan", short_plan, jakal::PartitionStrategy::projection_sharded)) {
        return false;
    }

    const auto long_plan = planner.build_plan(long_decode_workload, graphs);
    if (!expect_strategy("long decode shape-bucket plan", long_plan, jakal::PartitionStrategy::reduce_on_gpu)) {
        return false;
    }

    const auto strategy_cache = cache_path.string() + ".strategy";
    const auto family_strategy_cache = cache_path.string() + ".strategy_family";
    std::error_code ec;
    std::filesystem::remove(cache_path, ec);
    std::filesystem::remove(strategy_cache, ec);
    std::filesystem::remove(family_strategy_cache, ec);
    return true;
}

bool test_hardware_family_transfer_learning() {
    const auto cache_path = unique_temp_file("planner-family-transfer");
    jakal::Planner planner(cache_path);
    const std::vector<jakal::HardwareGraph> source_graphs{make_host_graph(), make_gpu_graph()};
    const std::vector<jakal::HardwareGraph> family_graphs{make_host_graph(), make_gpu_graph_family_variant()};

    const jakal::WorkloadSpec workload{
        "decode-family-transfer",
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
        jakal::WorkloadPhase::decode};

    planner.ingest_strategy_feedback(
        workload,
        source_graphs,
        {jakal::PartitionStrategy::auto_balanced, 1180.0, 405.0, 1.00, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        source_graphs,
        {jakal::PartitionStrategy::auto_balanced, 1210.0, 415.0, 0.98, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        source_graphs,
        {jakal::PartitionStrategy::role_split, 910.0, 290.0, 1.22, 1.0, true});
    planner.ingest_strategy_feedback(
        workload,
        source_graphs,
        {jakal::PartitionStrategy::role_split, 890.0, 275.0, 1.25, 1.0, true});

    const auto family_plan = planner.build_plan(workload, family_graphs);
    if (!expect_strategy("hardware family transfer plan", family_plan, jakal::PartitionStrategy::role_split)) {
        return false;
    }

    const auto strategy_cache = cache_path.string() + ".strategy";
    const auto family_strategy_cache = cache_path.string() + ".strategy_family";
    std::error_code ec;
    std::filesystem::remove(cache_path, ec);
    std::filesystem::remove(strategy_cache, ec);
    std::filesystem::remove(family_strategy_cache, ec);
    return true;
}

int main() {
    if (!test_strategy_exploration()) {
        return 1;
    }
    if (!test_strategy_learning()) {
        return 1;
    }
    if (!test_latency_weighted_learning()) {
        return 1;
    }
    if (!test_failure_penalty_learning()) {
        return 1;
    }
    if (!test_phase_scoped_learning()) {
        return 1;
    }
    if (!test_shape_bucket_scoped_learning()) {
        return 1;
    }
    if (!test_hardware_family_transfer_learning()) {
        return 1;
    }
    std::cout << "planner strategy learning ok\n";
    return 0;
}
