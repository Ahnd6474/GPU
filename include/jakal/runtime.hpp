#pragma once

#include "jakal/backend.hpp"
#include "jakal/execution.hpp"
#include "jakal/executor.hpp"
#include "jakal/jakal_toolkit.hpp"
#include "jakal/planner.hpp"
#include "jakal/workloads.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace jakal {

struct RuntimeMemoryPolicy {
    double host_reserve_ratio = 0.10;
    double accelerator_reserve_ratio = 0.12;
    double max_pressure_ratio = 0.92;
    bool allow_host_spill = true;
    bool enforce_preflight = true;
};

struct RuntimeSafetyPolicy {
    bool enable_canary = true;
    bool enable_strategy_rollback = true;
    bool enable_planner_risk_gate = true;
    double minimum_planner_confidence = 0.45;
    double planner_risk_gate = 0.50;
    double max_runtime_regression_ratio = 1.10;
    std::uint32_t blacklist_after_failures = 2;
    std::uint64_t blacklist_cooldown_epochs = 16;
};

struct RuntimeObservabilityOptions {
    bool persist_telemetry = true;
    std::filesystem::path telemetry_path;
};

struct RuntimeProductPolicy {
    RuntimeMemoryPolicy memory;
    RuntimeSafetyPolicy safety;
    RuntimeObservabilityOptions observability;
};

struct RuntimeOptimizationPolicy {
    std::optional<PartitionStrategy> forced_partition_strategy;
    bool enable_graph_aware_strategy_hints = true;
    bool enable_graph_aware_execution_tuning = true;
    ExecutionTuningOverrides execution;
};

struct RuntimeOptions {
    bool enable_host_probe = true;
    bool enable_opencl_probe = true;
    bool enable_level_zero_probe = true;
    bool enable_cuda_probe = true;
    bool enable_rocm_probe = true;
    bool prefer_level_zero_over_opencl = true;
    std::filesystem::path cache_path;
    std::filesystem::path execution_cache_path;
    RuntimeProductPolicy product;
    RuntimeOptimizationPolicy optimization;
};

struct DeviceMemoryReservation {
    std::string device_uid;
    bool host = false;
    std::uint64_t effective_capacity_bytes = 0;
    std::uint64_t reserved_bytes = 0;
    std::uint64_t persistent_bytes = 0;
    std::uint64_t transient_bytes = 0;
    double pressure_ratio = 0.0;
};

struct MemoryPreflightReport {
    std::vector<DeviceMemoryReservation> devices;
    std::uint64_t pinned_host_visible_bytes = 0;
    std::uint64_t aggregate_persistent_bytes = 0;
    std::uint64_t aggregate_transient_bytes = 0;
    std::uint64_t predicted_spill_bytes = 0;
    std::uint64_t predicted_reload_bytes = 0;
    std::uint32_t forced_spill_count = 0;
    double peak_pressure_ratio = 0.0;
    bool requires_spill = false;
    bool safe_to_run = true;
    std::string summary;
};

struct StrategySafetyDecision {
    PartitionStrategy requested_strategy = PartitionStrategy::auto_balanced;
    PartitionStrategy selected_strategy = PartitionStrategy::auto_balanced;
    PartitionStrategy final_strategy = PartitionStrategy::auto_balanced;
    PlanStrategySource planner_strategy_source = PlanStrategySource::heuristic_auto;
    double planner_confidence = 0.0;
    double planner_risk_score = 0.0;
    bool blacklisted_before_run = false;
    bool memory_forced_auto = false;
    bool planner_forced_auto = false;
    bool rolled_back_to_auto = false;
    bool blocked_by_memory = false;
    bool canary_triggered = false;
    std::string summary;
};

struct KernelCoverageIssue {
    std::string operation_name;
    std::string device_uid;
    std::string backend_name;
    OperationClass op_class = OperationClass::elementwise_map;
    bool supported = true;
    std::string reason;
};

struct KernelCoverageReport {
    std::vector<KernelCoverageIssue> issues;
    bool all_supported = true;
    bool forced_auto = false;
    std::string summary;
};

struct AssetPrefetchEntry {
    std::string asset_id;
    std::string source_asset_id;
    std::filesystem::path path;
    std::string tensor_id;
    std::string device_uid;
    std::uint64_t file_offset = 0;
    std::uint64_t bytes = 0;
    std::string queue_hint = "host_io";
    std::string target_residency = "auto";
    std::string materialization_kind = "raw";
    std::string backend_hint = "any";
    std::string backend_cache_tag = "none";
    bool exists_on_disk = false;
    bool preload_required = true;
    bool persistent = true;
    bool host_visible = false;
    bool pin_host_staging = false;
    bool derived_cache = false;
};

struct AssetPrefetchReport {
    std::vector<AssetPrefetchEntry> entries;
    std::uint64_t total_prefetch_bytes = 0;
    std::uint64_t total_host_io_bytes = 0;
    std::uint64_t total_host_to_device_bytes = 0;
    std::uint64_t total_layout_cache_bytes = 0;
    bool missing_required_assets = false;
    std::string summary;
};

enum class ResidencyActionKind {
    prefetch,
    reload,
    spill,
    evict
};

struct ResidencyAction {
    ResidencyActionKind kind = ResidencyActionKind::prefetch;
    std::string tensor_id;
    std::string device_uid;
    std::string trigger_operation_name;
    std::uint32_t operation_index = 0;
    std::uint64_t bytes = 0;
    bool persistent = false;
};

struct ResidencySequenceReport {
    std::vector<ResidencyAction> actions;
    std::uint64_t peak_live_bytes = 0;
    std::uint64_t spill_bytes = 0;
    std::uint64_t reload_bytes = 0;
    std::uint32_t forced_spill_count = 0;
    bool viable_without_spill = true;
    std::string summary;
};

struct ManagedExecutionReport {
    ExecutionPlan planning;
    DirectExecutionReport execution;
    MemoryPreflightReport memory_preflight;
    StrategySafetyDecision safety;
    KernelCoverageReport kernel_coverage;
    AssetPrefetchReport asset_prefetch;
    ResidencySequenceReport residency_sequence;
    std::filesystem::path telemetry_path;
    bool executed = false;
};

class Runtime {
public:
    explicit Runtime(RuntimeOptions options = {});

    void refresh_hardware();

    [[nodiscard]] const std::vector<HardwareGraph>& devices() const;
    [[nodiscard]] const std::vector<JakalToolkitIndexEntry>& jakal_toolkit_index() const;
    [[nodiscard]] ExecutionPlan plan(const WorkloadSpec& workload);
    [[nodiscard]] OptimizationReport optimize(const WorkloadSpec& workload);
    [[nodiscard]] OptimizationReport optimize(const WorkloadSpec& workload, const WorkloadGraph& workload_graph);
    [[nodiscard]] DirectExecutionReport execute(const WorkloadSpec& workload);
    [[nodiscard]] ManagedExecutionReport execute_managed(const WorkloadSpec& workload);
    [[nodiscard]] ManagedExecutionReport execute_managed(const WorkloadSpec& workload, const WorkloadGraph& workload_graph);
    [[nodiscard]] ManagedExecutionReport execute_manifest(const std::filesystem::path& manifest_path);

private:
    [[nodiscard]] bool should_include_descriptor(const HardwareGraph& candidate) const;
    [[nodiscard]] std::filesystem::path telemetry_path() const;
    [[nodiscard]] std::string strategy_safety_key(
        const WorkloadSpec& workload,
        PartitionStrategy strategy) const;
    [[nodiscard]] bool is_strategy_blacklisted(
        const WorkloadSpec& workload,
        PartitionStrategy strategy) const;
    void record_strategy_failure(const WorkloadSpec& workload, PartitionStrategy strategy);
    void record_strategy_success(const WorkloadSpec& workload, PartitionStrategy strategy);
    [[nodiscard]] MemoryPreflightReport build_memory_preflight(const OptimizationReport& optimization) const;
    [[nodiscard]] KernelCoverageReport build_kernel_coverage(const OptimizationReport& optimization) const;
    [[nodiscard]] AssetPrefetchReport build_asset_prefetch(
        const WorkloadManifest& manifest,
        const OptimizationReport& optimization) const;
    [[nodiscard]] ResidencySequenceReport build_residency_sequence(const OptimizationReport& optimization) const;
    [[nodiscard]] DirectExecutionReport execute_with_feedback(
        const WorkloadSpec& workload,
        const OptimizationReport& optimization,
        const WorkloadGraph* workload_graph_override = nullptr);
    void persist_telemetry(
        const WorkloadSpec& workload,
        const ManagedExecutionReport& report) const;

    RuntimeOptions options_;
    Planner planner_;
    ExecutionOptimizer execution_optimizer_;
    DirectExecutor direct_executor_;
    JakalToolkit jakal_toolkit_;
    std::vector<std::unique_ptr<IDeviceProbe>> probes_;
    std::vector<HardwareGraph> devices_;
    std::vector<JakalToolkitIndexEntry> jakal_toolkit_index_;
    std::uint64_t execution_epoch_ = 0;
    std::unordered_map<std::string, std::uint32_t> strategy_failure_counts_;
    std::unordered_map<std::string, std::uint64_t> strategy_blacklist_until_epoch_;
};

[[nodiscard]] std::string runtime_backend_name_for_graph(const HardwareGraph& graph);
[[nodiscard]] std::string runtime_backend_cache_tag_for_graph(const HardwareGraph& graph);
[[nodiscard]] bool runtime_backend_supports_operation(
    const HardwareGraph& graph,
    OperationClass op_class,
    std::string* reason = nullptr);

}  // namespace jakal

