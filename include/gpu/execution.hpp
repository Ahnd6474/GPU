#pragma once

#include "gpu/planner.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpu {

enum class OperationClass {
    elementwise_map,
    reduction,
    matmul,
    convolution_2d,
    resample_2d
};

enum class ExecutionNodeKind {
    source,
    dispatch,
    compute,
    aggregate,
    sink,
    synchronize
};

enum class ExecutionEdgeKind {
    dataflow,
    control,
    dependency,
    aggregation
};

enum class ExecutionStrategy {
    single_device,
    sharded,
    streaming,
    overlapped
};

struct OperationSpec {
    std::string name;
    OperationClass op_class = OperationClass::elementwise_map;
    std::vector<std::uint64_t> extents;
    std::uint64_t input_bytes = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t temporary_bytes = 0;
    double estimated_flops = 0.0;
    double max_relative_error = 1.0e-4;
    bool parallelizable = true;
    bool reduction_like = false;
    bool streaming_friendly = false;
    bool matrix_friendly = false;
};

struct ExecutionNode {
    std::string id;
    std::string label;
    std::string device_uid;
    std::string structural_node_id;
    ExecutionNodeKind kind = ExecutionNodeKind::compute;
    std::uint64_t bytes = 0;
    double work_units = 0.0;
    double predicted_latency_us = 0.0;
};

struct ExecutionEdge {
    std::string source_id;
    std::string target_id;
    ExecutionEdgeKind kind = ExecutionEdgeKind::dataflow;
    bool directed = true;
    std::uint64_t bytes = 0;
    double predicted_latency_us = 0.0;
    double overlap_ratio = 0.0;
};

struct ExecutionGraph {
    std::string signature;
    std::string workload_signature;
    OperationSpec operation;
    std::vector<std::string> participating_devices;
    std::vector<ExecutionNode> nodes;
    std::vector<ExecutionEdge> edges;
    double predicted_latency_us = 0.0;
    double predicted_speedup_vs_reference = 1.0;
    double expected_relative_error = 0.0;
};

struct ExecutionConfig {
    std::string signature;
    std::string operation_name;
    ExecutionStrategy strategy = ExecutionStrategy::single_device;
    std::string primary_device_uid;
    std::vector<std::string> participating_devices;
    std::vector<std::string> mapped_structural_nodes;
    std::uint32_t queue_depth = 1;
    std::uint32_t stages = 1;
    std::uint32_t tile_x = 0;
    std::uint32_t tile_y = 0;
    std::uint32_t tile_k = 0;
    bool overlap_transfers = false;
    bool use_low_precision = false;
    double target_error_tolerance = 1.0e-4;
};

struct SystemProfile {
    bool low_spec_mode = false;
    bool cold_start = true;
    bool on_battery = false;
    bool battery_saver = false;
    double battery_percent = 100.0;
    std::uint64_t available_memory_bytes = 0;
    double free_memory_ratio = 1.0;
    double paging_risk = 0.0;
    double sustained_slowdown = 1.0;
    double initialization_penalty_us = 0.0;
    double amortization_gain = 1.0;
};

struct BenchmarkRecord {
    std::string operation_name;
    std::string config_signature;
    std::string shape_bucket;
    double reference_latency_us = 0.0;
    double validation_latency_us = 0.0;
    double predicted_latency_us = 0.0;
    double surrogate_latency_us = 0.0;
    double effective_latency_us = 0.0;
    double system_penalty_us = 0.0;
    double speedup_vs_reference = 1.0;
    double relative_error = 0.0;
    bool accuracy_within_tolerance = true;
    bool simulated = false;
};

struct OperationOptimizationResult {
    OperationSpec operation;
    ExecutionConfig config;
    ExecutionGraph graph;
    BenchmarkRecord benchmark;
};

struct OptimizationReport {
    std::string signature;
    ExecutionPlan placement;
    std::vector<OperationOptimizationResult> operations;
    SystemProfile system_profile;
    bool loaded_from_cache = false;
};

struct ExecutionFeedbackRecord {
    std::string operation_name;
    double runtime_us = 0.0;
    double reference_runtime_us = 0.0;
    double relative_error = 0.0;
    bool verified = false;
};

[[nodiscard]] std::string to_string(OperationClass op_class);
[[nodiscard]] std::string to_string(ExecutionNodeKind kind);
[[nodiscard]] std::string to_string(ExecutionEdgeKind kind);
[[nodiscard]] std::string to_string(ExecutionStrategy strategy);

class ExecutionOptimizer {
public:
    struct PerformanceSummary {
        std::string shape_bucket;
        ExecutionConfig config;
        std::uint32_t observations = 0;
        double average_effective_latency_us = 0.0;
        double average_relative_error = 0.0;
        double average_prediction_scale = 1.0;
        double average_system_penalty_us = 0.0;
    };

    [[nodiscard]] static std::filesystem::path default_cache_path();

    explicit ExecutionOptimizer(std::filesystem::path cache_path = default_cache_path());

    [[nodiscard]] OptimizationReport optimize(
        const WorkloadSpec& workload,
        const ExecutionPlan& placement,
        const std::vector<HardwareGraph>& graphs);
    void ingest_execution_feedback(
        const OptimizationReport& report,
        const std::vector<ExecutionFeedbackRecord>& feedback,
        const std::vector<HardwareGraph>& graphs);

private:
    struct CachedConfig {
        std::string operation_name;
        ExecutionConfig config;
    };

    void load_cache();
    void persist_cache() const;

    std::filesystem::path cache_path_;
    std::filesystem::path performance_cache_path_;
    bool cache_loaded_ = false;
    std::unordered_map<std::string, std::vector<CachedConfig>> cache_;
    std::unordered_map<std::string, PerformanceSummary> performance_cache_;
    std::unordered_map<std::string, double> device_sustained_slowdown_;
    std::unordered_map<std::string, bool> warmed_devices_;
};

[[nodiscard]] std::vector<OperationSpec> default_operation_suite(const WorkloadSpec& workload);

}  // namespace gpu
