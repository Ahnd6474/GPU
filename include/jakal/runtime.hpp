#pragma once

#include "jakal/backend.hpp"
#include "jakal/execution.hpp"
#include "jakal/executor.hpp"
#include "jakal/jakal_toolkit.hpp"
#include "jakal/planner.hpp"
#include "jakal/workloads.hpp"

#include <filesystem>
#include <memory>
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

struct RuntimeOptions {
    bool enable_host_probe = true;
    bool enable_opencl_probe = true;
    bool enable_level_zero_probe = true;
    bool enable_cuda_probe = true;
    bool enable_rocm_probe = true;
    std::filesystem::path cache_path;
    std::filesystem::path execution_cache_path;
    RuntimeProductPolicy product;
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
    double peak_pressure_ratio = 0.0;
    bool requires_spill = false;
    bool safe_to_run = true;
    std::string summary;
};

struct StrategySafetyDecision {
    PartitionStrategy requested_strategy = PartitionStrategy::auto_balanced;
    PartitionStrategy selected_strategy = PartitionStrategy::auto_balanced;
    PartitionStrategy final_strategy = PartitionStrategy::auto_balanced;
    bool blacklisted_before_run = false;
    bool memory_forced_auto = false;
    bool rolled_back_to_auto = false;
    bool blocked_by_memory = false;
    bool canary_triggered = false;
    std::string summary;
};

struct ManagedExecutionReport {
    DirectExecutionReport execution;
    MemoryPreflightReport memory_preflight;
    StrategySafetyDecision safety;
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

}  // namespace jakal

