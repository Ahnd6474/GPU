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

constexpr std::uint32_t kRuntimeTelemetrySchemaVersion = 2u;
constexpr std::uint32_t kRuntimeTelemetryBudgetSchemaVersion = 1u;

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
    bool enable_persisted_regression_gate = true;
    double minimum_planner_confidence = 0.45;
    double planner_risk_gate = 0.50;
    double max_runtime_regression_ratio = 1.10;
    double persisted_regression_slowdown_ratio = 1.20;
    std::uint32_t blacklist_after_failures = 2;
    std::uint32_t persisted_regression_gate_after_events = 3;
    std::uint64_t blacklist_cooldown_epochs = 16;
};

struct RuntimeObservabilityOptions {
    bool persist_telemetry = true;
    bool async_telemetry_flush = true;
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
    bool enable_vulkan_probe = true;
    bool enable_vulkan_status = true;
    bool enable_cuda_probe = true;
    bool enable_rocm_probe = true;
    bool prefer_level_zero_over_opencl = true;
    bool eager_hardware_refresh = true;
    std::filesystem::path install_root;
    std::filesystem::path cache_path;
    std::filesystem::path execution_cache_path;
    RuntimeProductPolicy product;
    RuntimeOptimizationPolicy optimization;
};

struct RuntimeInstallPaths {
    std::filesystem::path install_root;
    std::filesystem::path writable_root;
    std::filesystem::path config_dir;
    std::filesystem::path cache_dir;
    std::filesystem::path logs_dir;
    std::filesystem::path telemetry_path;
    std::filesystem::path planner_cache_path;
    std::filesystem::path execution_cache_path;
    std::filesystem::path python_dir;
};

enum class RuntimeBackendStatusCode {
    disabled,
    unavailable,
    no_devices,
    ready_direct,
    ready_modeled
};

struct RuntimeBackendStatus {
    std::string backend_name;
    std::string device_uid;
    std::string detail;
    RuntimeBackendStatusCode code = RuntimeBackendStatusCode::unavailable;
    bool enabled = false;
    bool available = false;
    bool direct_execution = false;
    bool modeled_fallback = false;
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
    bool regression_gate_forced_auto = false;
    bool rolled_back_to_auto = false;
    bool blocked_by_memory = false;
    bool canary_triggered = false;
    std::uint32_t persisted_regression_events = 0;
    double persisted_worst_slowdown = 1.0;
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
    std::uint32_t tensor_index = kInvalidExecutionIndex;
    std::uint32_t device_index = kInvalidExecutionIndex;
    std::uint32_t operation_index = 0;
    std::uint64_t bytes = 0;
    bool persistent = false;
};

struct ResidencySequenceReport {
    std::vector<ResidencyAction> actions;
    std::vector<std::string> indexed_tensors;
    std::vector<std::string> indexed_devices;
    std::vector<std::string> indexed_operations;
    std::uint64_t peak_live_bytes = 0;
    std::uint64_t spill_bytes = 0;
    std::uint64_t reload_bytes = 0;
    std::uint32_t forced_spill_count = 0;
    bool viable_without_spill = true;
    std::string summary;
};

enum class TensorAllocationEventKind {
    allocate,
    reuse,
    release
};

struct TensorAllocationEvent {
    TensorAllocationEventKind kind = TensorAllocationEventKind::allocate;
    ResidencyActionKind residency_cause = ResidencyActionKind::prefetch;
    std::string tensor_id;
    std::string device_uid;
    std::string trigger_operation_name;
    std::uint32_t tensor_index = kInvalidExecutionIndex;
    std::uint32_t device_index = kInvalidExecutionIndex;
    std::uint32_t operation_index = 0;
    std::uint32_t block_id = 0;
    std::uint64_t offset_bytes = 0;
    std::uint64_t bytes = 0;
    bool persistent = false;
};

struct TensorAllocatorReport {
    std::vector<TensorAllocationEvent> events;
    std::vector<std::string> indexed_tensors;
    std::vector<std::string> indexed_devices;
    std::vector<std::string> indexed_operations;
    std::uint64_t peak_live_bytes = 0;
    std::uint64_t peak_reserved_bytes = 0;
    std::uint32_t allocation_count = 0;
    std::uint32_t reuse_count = 0;
    std::string summary;
};

struct SpillArtifactEntry {
    ResidencyActionKind kind = ResidencyActionKind::spill;
    std::string tensor_id;
    std::string device_uid;
    std::string trigger_operation_name;
    std::filesystem::path path;
    std::uint32_t tensor_index = kInvalidExecutionIndex;
    std::uint32_t device_index = kInvalidExecutionIndex;
    std::uint32_t operation_index = 0;
    std::uint64_t bytes = 0;
    bool exists_on_disk = false;
};

struct SpillArtifactReport {
    std::vector<SpillArtifactEntry> entries;
    std::uint64_t materialized_spill_bytes = 0;
    std::uint64_t materialized_reload_bytes = 0;
    std::string summary;
};

struct BackendBufferBindingEntry {
    std::string device_uid;
    std::string backend_name;
    std::string ownership_scope = "runtime-local";
    std::string pool_id;
    std::string resource_tag;
    std::uint64_t planned_peak_bytes = 0;
    std::uint64_t reserved_bytes = 0;
    std::uint64_t spill_bytes = 0;
    std::uint64_t reload_bytes = 0;
    std::uint64_t materialized_spill_bytes = 0;
    std::uint64_t materialized_reload_bytes = 0;
    std::uint32_t direct_execution_operation_count = 0;
    std::uint32_t persistent_resource_reuse_hits = 0;
    bool direct_execution_active = false;
    bool uses_runtime_spill_artifacts = false;
};

struct BackendBufferBindingReport {
    std::vector<BackendBufferBindingEntry> entries;
    std::uint64_t backend_owned_peak_bytes = 0;
    std::uint64_t runtime_local_peak_bytes = 0;
    std::uint32_t total_persistent_resource_reuse_hits = 0;
    std::string summary;
};

struct ExecutedResidencyMovementEntry {
    std::string kind;
    std::string tensor_id;
    std::string device_uid;
    std::string backend_name;
    std::string trigger_operation_name;
    std::string pool_id;
    std::filesystem::path spill_artifact_path;
    std::uint32_t operation_index = 0;
    std::uint64_t bytes = 0;
    double runtime_us = 0.0;
    bool from_direct_execution = false;
    bool from_spill_artifact = false;
};

struct ExecutedResidencyMovementReport {
    std::vector<ExecutedResidencyMovementEntry> entries;
    std::uint64_t executed_h2d_bytes = 0;
    std::uint64_t executed_d2h_bytes = 0;
    std::uint64_t executed_spill_bytes = 0;
    std::uint64_t executed_reload_bytes = 0;
    double total_transfer_runtime_us = 0.0;
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
    TensorAllocatorReport tensor_allocator;
    SpillArtifactReport spill_artifacts;
    BackendBufferBindingReport backend_buffer_bindings;
    ExecutedResidencyMovementReport executed_residency_movements;
    std::filesystem::path telemetry_path;
    bool executed = false;
};

class Runtime {
public:
    explicit Runtime(RuntimeOptions options = {});

    void refresh_hardware();

    [[nodiscard]] const RuntimeOptions& options() const;
    [[nodiscard]] const RuntimeInstallPaths& install_paths() const;
    [[nodiscard]] const std::vector<RuntimeBackendStatus>& backend_statuses() const;
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
    void ensure_hardware_refreshed() const;
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
    [[nodiscard]] TensorAllocatorReport build_tensor_allocator(
        const ResidencySequenceReport& residency_sequence) const;
    [[nodiscard]] SpillArtifactReport materialize_spill_artifacts(
        const ResidencySequenceReport& residency_sequence) const;
    [[nodiscard]] BackendBufferBindingReport build_backend_buffer_bindings(
        const DirectExecutionReport* execution,
        const TensorAllocatorReport& tensor_allocator,
        const ResidencySequenceReport& residency_sequence,
        const SpillArtifactReport& spill_artifacts) const;
    [[nodiscard]] ExecutedResidencyMovementReport build_executed_residency_movements(
        const DirectExecutionReport* execution,
        const ResidencySequenceReport& residency_sequence,
        const SpillArtifactReport& spill_artifacts) const;
    [[nodiscard]] DirectExecutionReport execute_with_feedback(
        const WorkloadSpec& workload,
        const OptimizationReport& optimization,
        const WorkloadGraph* workload_graph_override = nullptr);
    void persist_telemetry(
        const WorkloadSpec& workload,
        const ManagedExecutionReport& report) const;
    void rebuild_backend_statuses();

    RuntimeOptions options_;
    RuntimeInstallPaths install_paths_;
    Planner planner_;
    ExecutionOptimizer execution_optimizer_;
    DirectExecutor direct_executor_;
    JakalToolkit jakal_toolkit_;
    std::vector<std::unique_ptr<IDeviceProbe>> probes_;
    std::vector<HardwareGraph> devices_;
    std::vector<RuntimeBackendStatus> backend_statuses_;
    std::vector<JakalToolkitIndexEntry> jakal_toolkit_index_;
    bool hardware_refreshed_ = false;
    std::uint64_t execution_epoch_ = 0;
    std::unordered_map<std::string, std::uint32_t> strategy_failure_counts_;
    std::unordered_map<std::string, std::uint64_t> strategy_blacklist_until_epoch_;
};

[[nodiscard]] std::string runtime_backend_name_for_graph(const HardwareGraph& graph);
[[nodiscard]] std::string runtime_backend_cache_tag_for_graph(const HardwareGraph& graph);
[[nodiscard]] std::string to_string(RuntimeBackendStatusCode code);
[[nodiscard]] RuntimeInstallPaths resolve_runtime_install_paths(const std::filesystem::path& install_root = {});
[[nodiscard]] RuntimeOptions make_runtime_options_for_install(const std::filesystem::path& install_root = {});
[[nodiscard]] bool runtime_backend_supports_operation(
    const HardwareGraph& graph,
    OperationClass op_class,
    std::string* reason = nullptr);

}  // namespace jakal

