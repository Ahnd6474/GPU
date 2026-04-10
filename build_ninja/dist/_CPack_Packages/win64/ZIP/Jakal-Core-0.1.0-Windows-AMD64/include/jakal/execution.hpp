#pragma once

#include "jakal/planner.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace jakal {

constexpr std::uint32_t kInvalidExecutionIndex = 0xffffffffu;

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

enum class ValidationTier {
    full,
    adaptive,
    minimal,
    disabled
};

enum class OptimizationPolicy {
    heuristic_greedy,
    learned_greedy,
    trace_replay,
    ucb_explore,
    reinforce_softmax,
    spsa_local_search,
    adam_surrogate
};

struct ContinuousExecutionState {
    double queue_depth_raw = 0.0;
    double stage_raw = 0.0;
    double tile_raw = 0.0;
    double overlap_raw = 0.0;
    double partition_raw = 0.0;
    double precision_raw = 0.0;
    double single_device_logit = 1.2;
    double sharded_logit = -0.8;
    double streaming_logit = 0.4;
    double overlapped_logit = 0.6;
};

struct ExecutionTuningOverrides {
    std::optional<ContinuousExecutionState> initial_state_override;
    std::optional<std::uint32_t> graph_optimization_passes_override;
    std::optional<std::uint32_t> optimizer_wall_time_budget_ms;
    std::uint32_t graph_rewrite_level = 1u;
    ValidationTier validation_tier = ValidationTier::adaptive;
};

struct GraphOptimizationPass {
    std::uint32_t pass_index = 0;
    double objective_us = 0.0;
    double gradient_norm = 0.0;
    double learning_rate = 0.0;
    ContinuousExecutionState state;
};

struct GraphOptimizationSummary {
    std::string optimizer_name;
    ContinuousExecutionState initial_state;
    ContinuousExecutionState final_state;
    std::vector<GraphOptimizationPass> passes;
    double initial_objective_us = 0.0;
    double final_objective_us = 0.0;
    std::uint32_t total_logical_partitions = 0;
    std::uint32_t time_budget_ms = 0;
    bool budget_exhausted = false;
    bool converged = false;
};

struct CandidatePolicySummary {
    std::uint32_t max_ranked_strategies = 0;
    std::uint32_t max_devices = 0;
    std::uint32_t max_candidates = 0;
    std::uint32_t validation_shortlist = 0;
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
    std::vector<std::string> input_tensor_ids;
    std::vector<std::string> output_tensor_ids;
    std::vector<std::string> temporary_tensor_ids;
    std::vector<std::string> dependency_operation_names;
    std::vector<std::string> fused_operation_names;
    bool cpu_vectorized = false;
    bool gpu_tensorized = false;
    std::string cpu_input_layout = "native";
    std::string cpu_weight_layout = "native";
    std::string cpu_output_layout = "native";
    std::string gpu_input_layout = "native";
    std::string gpu_weight_layout = "native";
    std::string gpu_output_layout = "native";
    bool cpu_pack_weights = false;
    bool gpu_pack_weights = false;
    bool cpu_pretranspose_rhs = false;
    bool gpu_pretranspose_rhs = false;
    std::uint32_t cpu_micro_kernel_unroll = 1;
    std::uint32_t gpu_micro_kernel_unroll = 1;
    std::uint32_t cpu_tile_m = 0;
    std::uint32_t cpu_tile_n = 0;
    std::uint32_t cpu_tile_k = 0;
    std::uint32_t cpu_parallel_chunk = 0;
    bool cpu_use_avx512 = false;
    std::uint64_t cpu_pack_budget_bytes = 0;
    std::uint32_t cpu_single_thread_cutoff = 0;
    std::string cpu_kernel_family = "auto";
    std::string cpu_low_precision_kernel_family = "auto";
    std::uint32_t attention_head_count = 0;
    std::uint32_t attention_head_group_size = 0;
    std::uint32_t preferred_token_block = 0;
    std::string preferred_kv_residency = "auto";
    bool residency_sensitive_fusion = false;
};

struct WorkloadTensor {
    std::string id;
    std::string alias_group;
    std::string producer_operation;
    std::vector<std::string> consumer_operations;
    std::uint64_t bytes = 0;
    bool persistent = false;
    bool temporary = false;
    bool host_visible = false;
};

struct TensorLifetime {
    std::string tensor_id;
    std::uint32_t first_operation_index = 0;
    std::uint32_t last_operation_index = 0;
    std::uint64_t bytes = 0;
    bool persistent = false;
};

struct WorkloadDependency {
    std::string source_operation_name;
    std::string target_operation_name;
    std::string tensor_id;
    bool requires_residency = true;
};

struct WorkloadGraph {
    std::string signature;
    std::vector<WorkloadTensor> tensors;
    std::vector<TensorLifetime> lifetimes;
    std::vector<WorkloadDependency> dependencies;
    std::vector<OperationSpec> operations;
};

struct TensorResidencyPlanEntry {
    std::string tensor_id;
    std::string device_uid;
    std::string structural_node_id;
    std::uint32_t tensor_index = kInvalidExecutionIndex;
    std::uint32_t device_index = kInvalidExecutionIndex;
    std::uint32_t structural_node_index = kInvalidExecutionIndex;
    std::uint64_t bytes = 0;
    bool persistent = false;
    bool live_in = false;
    bool live_out = false;
    bool requires_transfer = false;
    double pressure_ratio = 0.0;
};

struct TransferScheduleEntry {
    std::string tensor_id;
    std::string source_device_uid;
    std::string target_device_uid;
    std::string source_operation_name;
    std::string target_operation_name;
    std::uint32_t tensor_index = kInvalidExecutionIndex;
    std::uint32_t source_device_index = kInvalidExecutionIndex;
    std::uint32_t target_device_index = kInvalidExecutionIndex;
    std::uint32_t source_operation_index = kInvalidExecutionIndex;
    std::uint32_t target_operation_index = kInvalidExecutionIndex;
    std::uint64_t bytes = 0;
    double predicted_latency_us = 0.0;
    double overlap_ratio = 0.0;
    bool cross_device = false;
};

struct ExecutionNode {
    std::string id;
    std::string label;
    std::string device_uid;
    std::string structural_node_id;
    std::uint32_t node_index = kInvalidExecutionIndex;
    std::uint32_t device_index = kInvalidExecutionIndex;
    std::uint32_t structural_node_index = kInvalidExecutionIndex;
    std::uint32_t operation_index = kInvalidExecutionIndex;
    std::uint32_t tensor_index = kInvalidExecutionIndex;
    ExecutionNodeKind kind = ExecutionNodeKind::compute;
    std::uint64_t bytes = 0;
    double work_units = 0.0;
    double predicted_latency_us = 0.0;
};

struct ExecutionEdge {
    std::string source_id;
    std::string target_id;
    std::uint32_t source_node_index = kInvalidExecutionIndex;
    std::uint32_t target_node_index = kInvalidExecutionIndex;
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
    std::vector<std::string> indexed_devices;
    std::vector<std::string> indexed_operations;
    std::vector<std::string> indexed_structural_nodes;
    std::vector<TensorResidencyPlanEntry> residency_plan;
    std::vector<TransferScheduleEntry> transfer_schedule;
    std::vector<ExecutionNode> nodes;
    std::vector<ExecutionEdge> edges;
    double predicted_latency_us = 0.0;
    double predicted_speedup_vs_reference = 1.0;
    double expected_relative_error = 0.0;
    double predicted_transfer_latency_us = 0.0;
    double predicted_overlap_gain_us = 0.0;
    double predicted_memory_pressure = 0.0;
    std::uint64_t peak_resident_bytes = 0;
};

struct ExecutionConfig {
    std::string signature;
    std::string operation_name;
    std::string variant_id;
    ExecutionStrategy strategy = ExecutionStrategy::single_device;
    std::string primary_device_uid;
    std::vector<std::string> participating_devices;
    std::vector<std::string> mapped_structural_nodes;
    std::uint32_t queue_depth = 1;
    std::uint32_t stages = 1;
    std::uint32_t tile_x = 0;
    std::uint32_t tile_y = 0;
    std::uint32_t tile_k = 0;
    std::uint32_t logical_partitions = 1;
    bool overlap_transfers = false;
    bool use_low_precision = false;
    double target_error_tolerance = 1.0e-4;
    double queue_depth_scale = 1.0;
    double stage_scale = 1.0;
    double tile_scale = 1.0;
    double overlap_ratio = 0.0;
    double partition_intensity = 0.0;
    double precision_mix = 0.0;
    double telemetry_staging_hit_rate = 0.0;
    double telemetry_cross_device_sync_cost_us = 0.0;
    double telemetry_residency_pressure = 0.0;
    bool semantic_head_partitioning = false;
    bool semantic_token_partitioning = false;
};

struct SystemProfile {
    bool low_spec_mode = false;
    bool on_battery = false;
    bool battery_saver = false;
    double battery_percent = 100.0;
    std::uint64_t available_memory_bytes = 0;
    double free_memory_ratio = 1.0;
    double paging_risk = 0.0;
    double sustained_slowdown = 1.0;
    double readiness_score = 0.0;
    double stability_score = 1.0;
    double initialization_penalty_us = 0.0;
    double amortization_gain = 1.0;
};

struct BenchmarkRecord {
    std::string operation_name;
    std::string config_signature;
    std::string shape_bucket;
    std::string optimizer_name;
    double reference_latency_us = 0.0;
    double validation_latency_us = 0.0;
    double predicted_latency_us = 0.0;
    double surrogate_latency_us = 0.0;
    double effective_latency_us = 0.0;
    double system_penalty_us = 0.0;
    double objective_score = 0.0;
    double trace_weight = 1.0;
    double speedup_vs_reference = 1.0;
    double relative_error = 0.0;
    double calibrated_prediction_us = 0.0;
    double calibration_ratio = 1.0;
    double calibration_confidence = 0.0;
    double reference_spread_us = 0.0;
    double candidate_spread_us = 0.0;
    std::uint32_t validation_samples = 0;
    bool accuracy_within_tolerance = true;
    bool simulated = false;
};

struct PerformanceSummary {
    std::string shape_bucket;
    std::string device_family_signature;
    ExecutionConfig config;
    std::uint32_t observations = 0;
    double average_effective_latency_us = 0.0;
    double average_relative_error = 0.0;
    double average_prediction_scale = 1.0;
    double average_calibration_ratio = 1.0;
    double average_system_penalty_us = 0.0;
    double average_validation_spread_us = 0.0;
    double average_reward = 0.0;
    double average_copy_share = 0.0;
    double average_transfer_overlap_ratio = 0.0;
    double average_budget_pressure = 0.0;
    double average_queue_separation_ratio = 0.0;
    double average_staging_hit_rate = 0.0;
    double average_cross_device_sync_cost_us = 0.0;
    double average_residency_pressure = 0.0;
    double average_kv_host_residency_ratio = 0.0;
};

struct CpuRuntimeHintSummary {
    std::string shape_bucket;
    std::uint32_t preferred_parallel_chunk = 0;
    std::uint32_t preferred_tile_m = 0;
    std::uint32_t preferred_tile_n = 0;
    std::uint32_t preferred_tile_k = 0;
    std::uint32_t preferred_single_thread_cutoff = 0;
    std::string preferred_kernel_family = "auto";
    bool preferred_use_avx512 = false;
    std::uint32_t observations = 0;
    double average_effective_latency_us = 0.0;
};

struct OperationOptimizationResult {
    OperationSpec operation;
    ExecutionConfig config;
    ExecutionGraph graph;
    BenchmarkRecord benchmark;
};

struct OptimizationReport {
    std::string signature;
    WorkloadKind workload_kind = WorkloadKind::custom;
    WorkloadPhase workload_phase = WorkloadPhase::unknown;
    std::string workload_shape_bucket;
    std::string dataset_tag;
    std::uint64_t workload_working_set_bytes = 0;
    std::uint64_t workload_host_exchange_bytes = 0;
    PartitionStrategy partition_strategy = PartitionStrategy::auto_balanced;
    WorkloadGraph workload_graph;
    ExecutionPlan placement;
    std::vector<OperationOptimizationResult> operations;
    SystemProfile system_profile;
    GraphOptimizationSummary graph_optimization;
    bool loaded_from_cache = false;
};

struct ExecutionFeedbackRecord {
    std::string operation_name;
    std::string backend_name;
    std::vector<std::string> participating_devices;
    double runtime_us = 0.0;
    double reference_runtime_us = 0.0;
    double relative_error = 0.0;
    bool verified = false;
    bool used_host = false;
    bool used_opencl = false;
    bool used_multiple_devices = false;
    std::uint32_t logical_partitions_used = 1;
    double copy_share = 0.0;
    double transfer_overlap_ratio = 0.0;
    double budget_pressure = 0.0;
    double queue_separation_ratio = 0.0;
    double staging_hit_rate = 0.0;
    double cross_device_sync_cost_us = 0.0;
    double residency_pressure = 0.0;
    double kv_host_residency_ratio = 0.0;
    std::uint32_t dispatch_count = 0;
    std::uint32_t event_wait_count = 0;
};

struct CachedExecutionConfig {
    std::string operation_name;
    ExecutionConfig config;
};

using CachedConfig = CachedExecutionConfig;

[[nodiscard]] std::string to_string(OperationClass op_class);
[[nodiscard]] std::string to_string(ExecutionNodeKind kind);
[[nodiscard]] std::string to_string(ExecutionEdgeKind kind);
[[nodiscard]] std::string to_string(ExecutionStrategy strategy);
[[nodiscard]] std::string to_string(OptimizationPolicy policy);
[[nodiscard]] std::string to_string(ValidationTier tier);

class BootstrapExecutionOptimizer {
public:
    explicit BootstrapExecutionOptimizer(std::filesystem::path cache_path);

    void load_cache();
    [[nodiscard]] bool has_full_cache(
        const std::string& report_signature,
        const std::vector<OperationSpec>& operations,
        const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
        std::unordered_map<std::string, ExecutionConfig>* cached_by_operation);
    void store_configs(const std::string& report_signature, std::vector<CachedExecutionConfig> configs);
    void persist_cache() const;

private:
    std::filesystem::path cache_path_;
    bool cache_loaded_ = false;
    std::unordered_map<std::string, std::vector<CachedExecutionConfig>> cache_;
};

class AdaptiveExecutionOptimizer {
public:
    explicit AdaptiveExecutionOptimizer(std::filesystem::path cache_path);

    void load_cache();
    void persist_cache() const;
    void apply_runtime_state(SystemProfile& profile, const std::vector<HardwareGraph>& graphs) const;
    [[nodiscard]] bool should_use_lightweight_path(const std::string& report_signature, bool fully_cached) const;
    [[nodiscard]] bool should_reoptimize(const std::string& report_signature) const;
    [[nodiscard]] bool should_reoptimize_operation(
        const std::string& report_signature,
        const std::string& operation_name) const;
    [[nodiscard]] const std::unordered_map<std::string, PerformanceSummary>& performance_cache() const;
    [[nodiscard]] const std::unordered_map<std::string, PerformanceSummary>& graph_family_performance_cache() const;
    [[nodiscard]] const std::unordered_map<std::string, CpuRuntimeHintSummary>& cpu_runtime_hint_cache() const;
    [[nodiscard]] const std::unordered_map<std::string, double>& backend_penalty_cache() const;
    [[nodiscard]] const std::unordered_map<std::string, bool>& warmed_devices() const;
    void apply_feedback_tuning_hints(WorkloadGraph& graph) const;
    void ingest_execution_feedback(
        const OptimizationReport& report,
        const std::vector<ExecutionFeedbackRecord>& feedback,
        const std::vector<HardwareGraph>& graphs);

private:
    std::filesystem::path performance_cache_path_;
    std::filesystem::path graph_family_cache_path_;
    std::filesystem::path cpu_runtime_hint_cache_path_;
    bool cache_loaded_ = false;
    std::unordered_map<std::string, PerformanceSummary> performance_cache_;
    std::unordered_map<std::string, PerformanceSummary> graph_family_performance_cache_;
    std::unordered_map<std::string, CpuRuntimeHintSummary> cpu_runtime_hint_cache_;
    std::unordered_map<std::string, double> device_sustained_slowdown_;
    std::unordered_map<std::string, bool> warmed_devices_;
    std::unordered_map<std::string, double> backend_penalty_cache_;
    std::unordered_map<std::string, std::uint32_t> reoptimization_pressure_;
    std::unordered_map<std::string, std::uint32_t> operation_reoptimization_pressure_;
};

class ExecutionOptimizer {
public:
    [[nodiscard]] static std::filesystem::path default_cache_path();

    explicit ExecutionOptimizer(std::filesystem::path cache_path = default_cache_path());

    [[nodiscard]] OptimizationReport optimize(
        const WorkloadSpec& workload,
        const ExecutionPlan& placement,
        const std::vector<HardwareGraph>& graphs,
        const WorkloadGraph* workload_graph_override = nullptr,
        const ExecutionTuningOverrides* tuning_overrides = nullptr);
    void ingest_execution_feedback(
        const OptimizationReport& report,
        const std::vector<ExecutionFeedbackRecord>& feedback,
        const std::vector<HardwareGraph>& graphs);

private:
    BootstrapExecutionOptimizer bootstrap_optimizer_;
    AdaptiveExecutionOptimizer adaptive_optimizer_;
};

[[nodiscard]] std::vector<OperationSpec> default_operation_suite(const WorkloadSpec& workload);
[[nodiscard]] WorkloadGraph default_workload_graph(const WorkloadSpec& workload);
[[nodiscard]] CandidatePolicySummary describe_candidate_policy(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const SystemProfile& system,
    bool continuous_state_active,
    double placement_affinity);

}  // namespace jakal

