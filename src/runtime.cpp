#include "jakal/runtime.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace jakal {
namespace {

bool runtime_regressed(
    const DirectExecutionReport& report,
    const double max_runtime_regression_ratio);

std::vector<ExecutionFeedbackRecord> make_feedback_records(const DirectExecutionReport& report) {
    std::vector<ExecutionFeedbackRecord> feedback;
    feedback.reserve(report.operations.size());
    for (const auto& operation : report.operations) {
        feedback.push_back(ExecutionFeedbackRecord{
            operation.operation_name,
            operation.backend_name,
            operation.participating_devices,
            operation.runtime_us,
            operation.reference_runtime_us,
            operation.relative_error,
            operation.verified,
            operation.used_host,
            operation.used_opencl,
            operation.used_multiple_devices,
            operation.logical_partitions_used});
    }
    return feedback;
}

double total_runtime_us(const DirectExecutionReport& report) {
    return std::accumulate(
        report.operations.begin(),
        report.operations.end(),
        0.0,
        [](const double total, const OperationExecutionRecord& operation) {
            return total + operation.runtime_us;
        });
}

bool should_retry_execution(const DirectExecutionReport& report) {
    if (!report.all_succeeded) {
        return true;
    }

    return std::any_of(report.operations.begin(), report.operations.end(), [](const OperationExecutionRecord& operation) {
        return !operation.used_host &&
               operation.reference_runtime_us > 0.0 &&
               operation.runtime_us > (operation.reference_runtime_us * 1.10);
    });
}

bool selection_changed(const OptimizationReport& left, const OptimizationReport& right) {
    if (left.operations.size() != right.operations.size()) {
        return true;
    }

    std::unordered_map<std::string, std::string> left_by_operation;
    left_by_operation.reserve(left.operations.size());
    for (const auto& operation : left.operations) {
        left_by_operation.emplace(operation.operation.name, operation.config.signature);
    }

    for (const auto& operation : right.operations) {
        const auto it = left_by_operation.find(operation.operation.name);
        if (it == left_by_operation.end() || it->second != operation.config.signature) {
            return true;
        }
    }

    return false;
}

double head_runtime_us(const WorkloadSpec& workload, const DirectExecutionReport& report) {
    if (report.operations.empty()) {
        return std::max(report.total_runtime_us, 0.0);
    }
    if (!workload.latency_sensitive) {
        return std::max(report.total_runtime_us, 0.0);
    }

    const std::size_t lead_operations =
        std::min<std::size_t>(3u, std::max<std::size_t>(1u, (report.operations.size() + 2u) / 3u));
    return std::accumulate(
        report.operations.begin(),
        report.operations.begin() + static_cast<std::ptrdiff_t>(lead_operations),
        0.0,
        [](const double total, const OperationExecutionRecord& operation) {
            return total + std::max(operation.runtime_us, 0.0);
        });
}

double successful_operation_ratio(const DirectExecutionReport& report) {
    if (report.operations.empty()) {
        return report.all_succeeded ? 1.0 : 0.0;
    }

    const auto successful = static_cast<double>(std::count_if(
        report.operations.begin(),
        report.operations.end(),
        [](const OperationExecutionRecord& operation) {
            return operation.backend_error.empty() && operation.verified;
        }));
    return successful / static_cast<double>(report.operations.size());
}

StrategyFeedbackSample make_strategy_feedback_sample(
    const WorkloadSpec& requested_workload,
    const DirectExecutionReport& report,
    const PartitionStrategy strategy,
    const PlanStrategySource strategy_source,
    const double planned_confidence,
    const bool rolled_back_to_auto,
    const double max_runtime_regression_ratio) {
    const bool regressed = runtime_regressed(report, max_runtime_regression_ratio);
    return StrategyFeedbackSample{
        strategy,
        report.total_runtime_us,
        head_runtime_us(requested_workload, report),
        report.speedup_vs_reference,
        successful_operation_ratio(report),
        report.all_succeeded,
        strategy_source,
        planned_confidence,
        rolled_back_to_auto,
        regressed};
}

void record_partition_strategy_feedback(
    Planner& planner,
    const WorkloadSpec& requested_workload,
    const std::vector<HardwareGraph>& graphs,
    const StrategyFeedbackSample& feedback) {
    planner.ingest_strategy_feedback(
        requested_workload,
        graphs,
        feedback);
}

bool runtime_regressed(
    const DirectExecutionReport& report,
    const double max_runtime_regression_ratio) {
    return report.total_reference_runtime_us > 0.0 &&
           report.total_runtime_us > (report.total_reference_runtime_us * max_runtime_regression_ratio);
}

std::uint64_t effective_capacity_bytes(
    const HardwareGraph& graph,
    const RuntimeMemoryPolicy& policy) {
    const auto summary = summarize_graph(graph);
    std::uint64_t capacity = 0u;
    if (graph.probe == "host") {
        capacity = summary.addressable_bytes == 0u ? summary.shared_host_bytes : summary.addressable_bytes;
    } else {
        capacity = summary.directly_attached_bytes;
        if (policy.allow_host_spill && (summary.unified_address_space || summary.coherent_with_host)) {
            capacity += summary.shared_host_bytes;
        }
        if (capacity == 0u) {
            capacity = summary.addressable_bytes;
        }
    }
    const double reserve_ratio = graph.probe == "host" ? policy.host_reserve_ratio : policy.accelerator_reserve_ratio;
    return static_cast<std::uint64_t>(std::max(0.0, static_cast<double>(capacity) * std::max(0.0, 1.0 - reserve_ratio)));
}

const HardwareGraph* find_graph_by_uid(
    const std::vector<HardwareGraph>& graphs,
    const std::string& uid) {
    const auto it = std::find_if(graphs.begin(), graphs.end(), [&](const HardwareGraph& graph) {
        return graph.uid == uid;
    });
    return it == graphs.end() ? nullptr : &(*it);
}

std::string lowercase_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool looks_like_intel_opencl_graph(const HardwareGraph& graph) {
    if (graph.probe != "opencl") {
        return false;
    }
    const auto haystack = lowercase_copy(graph.presentation_name + " " + graph.uid);
    return haystack.find("intel") != std::string::npos;
}

bool has_level_zero_graph(const std::vector<HardwareGraph>& graphs) {
    return std::any_of(graphs.begin(), graphs.end(), [](const HardwareGraph& graph) {
        return graph.probe == "level-zero";
    });
}

std::string backend_name_for_graph(const HardwareGraph& graph) {
    if (graph.probe == "host") {
        return "host-direct";
    }
    if (graph.probe == "level-zero") {
        return "level-zero-native";
    }
    if (graph.probe == "cuda") {
        return "cuda-native";
    }
    if (graph.probe == "rocm") {
        return "rocm-native";
    }
    if (graph.probe == "vulkan") {
        return "vulkan-direct";
    }
    if (graph.probe == "opencl") {
        return "opencl-direct";
    }
    return graph.probe + "-direct";
}

bool backend_supports_operation(
    const HardwareGraph& graph,
    const OperationClass op_class,
    std::string* reason = nullptr) {
    (void)op_class;
    if (graph.probe == "host" || graph.probe == "opencl" || graph.probe == "vulkan" || graph.probe == "cuda" ||
        graph.probe == "rocm" || graph.probe == "level-zero") {
        return true;
    }
    if (reason != nullptr) {
        *reason = "unknown backend kernel coverage";
    }
    return false;
}

double planner_risk_score(
    const WorkloadSpec& workload,
    const ExecutionPlan& planning,
    const OptimizationReport& optimization,
    const MemoryPreflightReport& memory,
    const KernelCoverageReport& kernel_coverage,
    const ResidencySequenceReport& residency) {
    double score = 0.0;
    if (planning.strategy_source == PlanStrategySource::exploration) {
        score += 0.35;
    } else if (planning.strategy_source == PlanStrategySource::family_learning) {
        score += 0.18;
    }
    if (!planning.loaded_from_cache) {
        score += 0.05;
    }
    if (workload.latency_sensitive) {
        score += 0.05;
    }
    if (optimization.workload_host_exchange_bytes >= (64ull * 1024ull * 1024ull)) {
        score += 0.10;
    }
    if (memory.requires_spill) {
        score += 0.25;
    }
    if (memory.peak_pressure_ratio >= 0.90) {
        score += 0.10;
    }
    if (residency.spill_bytes > 0u) {
        score += 0.15;
    }
    if (residency.forced_spill_count > 0u) {
        score += 0.25;
    }
    if (!kernel_coverage.all_supported) {
        score += 0.30;
    }
    return std::clamp(score, 0.0, 1.0);
}

}  // namespace

std::string runtime_backend_name_for_graph(const HardwareGraph& graph) {
    return backend_name_for_graph(graph);
}

bool runtime_backend_supports_operation(
    const HardwareGraph& graph,
    const OperationClass op_class,
    std::string* reason) {
    return backend_supports_operation(graph, op_class, reason);
}

Runtime::Runtime(RuntimeOptions options)
    : options_(std::move(options)),
      planner_(options_.cache_path.empty() ? Planner::default_cache_path() : options_.cache_path),
      execution_optimizer_(
          options_.execution_cache_path.empty()
              ? ExecutionOptimizer::default_cache_path()
              : options_.execution_cache_path) {
    if (options_.enable_host_probe) {
        probes_.push_back(make_host_probe());
    }
    if (options_.enable_opencl_probe) {
        probes_.push_back(make_opencl_probe());
    }
    if (options_.enable_level_zero_probe) {
        probes_.push_back(make_level_zero_probe());
    }
    if (options_.enable_cuda_probe) {
        probes_.push_back(make_cuda_probe());
    }
    if (options_.enable_rocm_probe) {
        probes_.push_back(make_rocm_probe());
    }
    refresh_hardware();
}

void Runtime::refresh_hardware() {
    devices_.clear();
    jakal_toolkit_index_.clear();
    std::vector<HardwareGraph> discovered;

    for (auto& probe : probes_) {
        if (!probe->available()) {
            continue;
        }

        for (auto& graph : probe->discover_hardware()) {
            if (should_include_descriptor(graph)) {
                discovered.push_back(std::move(graph));
            }
        }
    }

    const bool shadow_intel_opencl =
        options_.prefer_level_zero_over_opencl && options_.enable_level_zero_probe && has_level_zero_graph(discovered);
    for (auto& graph : discovered) {
        if (shadow_intel_opencl && looks_like_intel_opencl_graph(graph)) {
            continue;
        }
        devices_.push_back(std::move(graph));
    }

    std::sort(devices_.begin(), devices_.end(), [](const HardwareGraph& left, const HardwareGraph& right) {
        const auto left_summary = summarize_graph(left);
        const auto right_summary = summarize_graph(right);

        if (left_summary.execution_objects != right_summary.execution_objects) {
            return left_summary.execution_objects > right_summary.execution_objects;
        }
        if (left_summary.addressable_bytes != right_summary.addressable_bytes) {
            return left_summary.addressable_bytes > right_summary.addressable_bytes;
        }
        if (left_summary.host_read_gbps != right_summary.host_read_gbps) {
            return left_summary.host_read_gbps > right_summary.host_read_gbps;
        }
        return structural_fingerprint(left) < structural_fingerprint(right);
    });

    jakal_toolkit_index_ = jakal_toolkit_.build_index(devices_);
}

const std::vector<HardwareGraph>& Runtime::devices() const {
    return devices_;
}

const std::vector<JakalToolkitIndexEntry>& Runtime::jakal_toolkit_index() const {
    return jakal_toolkit_index_;
}

ExecutionPlan Runtime::plan(const WorkloadSpec& workload) {
    if (devices_.empty()) {
        refresh_hardware();
    }

    return planner_.build_plan(workload, devices_);
}

OptimizationReport Runtime::optimize(const WorkloadSpec& workload) {
    if (devices_.empty()) {
        refresh_hardware();
    }

    const auto placement = planner_.build_plan(workload, devices_);
    return execution_optimizer_.optimize(workload, placement, devices_, nullptr);
}

OptimizationReport Runtime::optimize(const WorkloadSpec& workload, const WorkloadGraph& workload_graph) {
    if (devices_.empty()) {
        refresh_hardware();
    }

    const auto placement = planner_.build_plan(workload, devices_);
    return execution_optimizer_.optimize(workload, placement, devices_, &workload_graph);
}

DirectExecutionReport Runtime::execute_with_feedback(
    const WorkloadSpec& workload,
    const OptimizationReport& optimization,
    const WorkloadGraph* workload_graph_override) {
    auto initial_report = direct_executor_.execute(optimization, devices_, jakal_toolkit_index_);
    execution_optimizer_.ingest_execution_feedback(
        initial_report.optimization,
        make_feedback_records(initial_report),
        devices_);

    if (!should_retry_execution(initial_report)) {
        return initial_report;
    }

    const auto refined_optimization =
        workload_graph_override == nullptr ? optimize(workload) : optimize(workload, *workload_graph_override);
    if (!selection_changed(initial_report.optimization, refined_optimization)) {
        return initial_report;
    }

    auto refined_report = direct_executor_.execute(refined_optimization, devices_, jakal_toolkit_index_);
    execution_optimizer_.ingest_execution_feedback(
        refined_report.optimization,
        make_feedback_records(refined_report),
        devices_);

    if (!refined_report.all_succeeded) {
        return initial_report;
    }
    if (!initial_report.all_succeeded) {
        return refined_report;
    }
    if (total_runtime_us(refined_report) < (total_runtime_us(initial_report) * 0.95)) {
        return refined_report;
    }
    return initial_report;
}

DirectExecutionReport Runtime::execute(const WorkloadSpec& workload) {
    return execute_managed(workload).execution;
}

ManagedExecutionReport Runtime::execute_managed(const WorkloadSpec& workload) {
    return execute_managed(workload, default_workload_graph(workload));
}

ManagedExecutionReport Runtime::execute_managed(const WorkloadSpec& workload, const WorkloadGraph& workload_graph) {
    if (devices_.empty()) {
        refresh_hardware();
    }

    ++execution_epoch_;

    ManagedExecutionReport managed;
    managed.telemetry_path = telemetry_path();
    managed.safety.requested_strategy = workload.partition_strategy;

    std::ostringstream safety_summary;
    auto effective_workload = workload;
    if (workload.partition_strategy != PartitionStrategy::auto_balanced &&
        is_strategy_blacklisted(workload, workload.partition_strategy)) {
        effective_workload.partition_strategy = PartitionStrategy::auto_balanced;
        managed.safety.blacklisted_before_run = true;
        safety_summary << "strategy blacklisted -> auto";
    }

    auto update_planner_diagnostics = [&](const ExecutionPlan& plan) {
        managed.planning = plan;
        managed.safety.planner_strategy_source = plan.strategy_source;
        managed.safety.planner_confidence = plan.strategy_confidence;
    };

    auto planned = planner_.build_plan(effective_workload, devices_);
    update_planner_diagnostics(planned);
    auto optimization = execution_optimizer_.optimize(effective_workload, planned, devices_, &workload_graph);
    managed.safety.selected_strategy = optimization.partition_strategy;
    managed.memory_preflight = build_memory_preflight(optimization);
    managed.kernel_coverage = build_kernel_coverage(optimization);
    managed.residency_sequence = build_residency_sequence(optimization);
    managed.memory_preflight.predicted_spill_bytes = managed.residency_sequence.spill_bytes;
    managed.memory_preflight.predicted_reload_bytes = managed.residency_sequence.reload_bytes;
    managed.memory_preflight.forced_spill_count = managed.residency_sequence.forced_spill_count;
    managed.memory_preflight.requires_spill =
        managed.memory_preflight.requires_spill || managed.residency_sequence.spill_bytes > 0u;
    managed.safety.planner_risk_score = planner_risk_score(
        effective_workload,
        managed.planning,
        optimization,
        managed.memory_preflight,
        managed.kernel_coverage,
        managed.residency_sequence);

    auto force_auto_if_needed = [&](const char* reason) {
        if (optimization.partition_strategy == PartitionStrategy::auto_balanced) {
            return;
        }
        auto fallback_workload = workload;
        fallback_workload.partition_strategy = PartitionStrategy::auto_balanced;
        auto fallback_plan = planner_.build_plan(fallback_workload, devices_);
        auto fallback_optimization = execution_optimizer_.optimize(fallback_workload, fallback_plan, devices_, &workload_graph);
        auto fallback_memory = build_memory_preflight(fallback_optimization);
        if (fallback_memory.safe_to_run || !managed.memory_preflight.safe_to_run) {
            effective_workload = fallback_workload;
            planned = std::move(fallback_plan);
            update_planner_diagnostics(planned);
            optimization = std::move(fallback_optimization);
            managed.memory_preflight = std::move(fallback_memory);
            managed.kernel_coverage = build_kernel_coverage(optimization);
            managed.residency_sequence = build_residency_sequence(optimization);
            managed.memory_preflight.predicted_spill_bytes = managed.residency_sequence.spill_bytes;
            managed.memory_preflight.predicted_reload_bytes = managed.residency_sequence.reload_bytes;
            managed.memory_preflight.forced_spill_count = managed.residency_sequence.forced_spill_count;
            managed.memory_preflight.requires_spill =
                managed.memory_preflight.requires_spill || managed.residency_sequence.spill_bytes > 0u;
            managed.safety.planner_risk_score = planner_risk_score(
                effective_workload,
                managed.planning,
                optimization,
                managed.memory_preflight,
                managed.kernel_coverage,
                managed.residency_sequence);
            managed.safety.memory_forced_auto = true;
            if (safety_summary.tellp() > 0) {
                safety_summary << "; ";
            }
            safety_summary << reason;
        }
    };

    if (optimization.partition_strategy != PartitionStrategy::auto_balanced &&
        is_strategy_blacklisted(workload, optimization.partition_strategy)) {
        force_auto_if_needed("selected strategy blacklisted -> auto");
    }

    if (!managed.memory_preflight.safe_to_run) {
        force_auto_if_needed("memory preflight forced auto");
    }
    if (!managed.kernel_coverage.all_supported) {
        force_auto_if_needed("kernel coverage forced auto");
        managed.kernel_coverage.forced_auto = managed.safety.memory_forced_auto;
    }
    if (options_.product.safety.enable_planner_risk_gate &&
        managed.planning.resolved_partition_strategy != PartitionStrategy::auto_balanced &&
        managed.safety.planner_confidence < options_.product.safety.minimum_planner_confidence &&
        managed.safety.planner_risk_score >= options_.product.safety.planner_risk_gate) {
        force_auto_if_needed("planner risk gate forced auto");
        managed.safety.planner_forced_auto = true;
    }

    managed.safety.selected_strategy = optimization.partition_strategy;
    if (!managed.memory_preflight.safe_to_run && options_.product.memory.enforce_preflight) {
        managed.safety.blocked_by_memory = true;
        managed.safety.final_strategy = optimization.partition_strategy;
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << managed.memory_preflight.summary;
        managed.safety.summary = safety_summary.str();
        persist_telemetry(workload, managed);
        return managed;
    }

    managed.execution = execute_with_feedback(effective_workload, optimization, &workload_graph);
    managed.executed = true;
    managed.safety.final_strategy = managed.execution.optimization.partition_strategy;
    auto planner_feedback = make_strategy_feedback_sample(
        effective_workload,
        managed.execution,
        managed.safety.selected_strategy,
        managed.planning.strategy_source,
        managed.planning.strategy_confidence,
        false,
        options_.product.safety.max_runtime_regression_ratio);

    const bool explicit_strategy = managed.execution.optimization.partition_strategy != PartitionStrategy::auto_balanced;
    const bool canary_triggered =
        explicit_strategy &&
        options_.product.safety.enable_canary &&
        (!managed.execution.all_succeeded ||
         runtime_regressed(managed.execution, options_.product.safety.max_runtime_regression_ratio));

    if (canary_triggered) {
        managed.safety.canary_triggered = true;
        record_strategy_failure(workload, managed.execution.optimization.partition_strategy);
        planner_feedback.runtime_regressed =
            runtime_regressed(managed.execution, options_.product.safety.max_runtime_regression_ratio);
        if (options_.product.safety.enable_strategy_rollback) {
            auto fallback_workload = workload;
            fallback_workload.partition_strategy = PartitionStrategy::auto_balanced;
            auto fallback_optimization = optimize(fallback_workload, workload_graph);
            auto fallback_memory = build_memory_preflight(fallback_optimization);
            if (fallback_memory.safe_to_run || !options_.product.memory.enforce_preflight) {
                auto fallback_execution = execute_with_feedback(fallback_workload, fallback_optimization, &workload_graph);
                if (fallback_execution.all_succeeded &&
                    (!managed.execution.all_succeeded ||
                     total_runtime_us(fallback_execution) <= total_runtime_us(managed.execution))) {
                    managed.execution = std::move(fallback_execution);
                    managed.memory_preflight = std::move(fallback_memory);
                    managed.safety.rolled_back_to_auto = true;
                    planner_feedback.rolled_back_to_auto = true;
                    managed.safety.final_strategy = managed.execution.optimization.partition_strategy;
                    if (safety_summary.tellp() > 0) {
                        safety_summary << "; ";
                    }
                    safety_summary << "rolled back to auto";
                }
            }
        }
    } else if (explicit_strategy) {
        record_strategy_success(workload, managed.execution.optimization.partition_strategy);
    }

    if (managed.executed) {
        record_partition_strategy_feedback(planner_, workload, devices_, planner_feedback);
    }

    if (managed.memory_preflight.summary.size() > 0) {
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << managed.memory_preflight.summary;
    }
    if (!managed.planning.strategy_reason.empty()) {
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << "planner=" << to_string(managed.planning.strategy_source)
                       << " conf=" << managed.planning.strategy_confidence
                       << " reason=" << managed.planning.strategy_reason;
    }
    if (!managed.residency_sequence.summary.empty()) {
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << managed.residency_sequence.summary;
    }
    managed.safety.summary = safety_summary.str();
    persist_telemetry(workload, managed);
    return managed;
}

ManagedExecutionReport Runtime::execute_manifest(const std::filesystem::path& manifest_path) {
    const auto manifest = load_workload_manifest(manifest_path);
    ManagedExecutionReport report;
    report.telemetry_path = telemetry_path();
    if (!manifest.assets.empty()) {
        const auto missing_required = std::find_if(
            manifest.assets.begin(),
            manifest.assets.end(),
            [](const WorkloadAsset& asset) {
                return asset.preload_required && (!asset.path.empty()) && !std::filesystem::exists(asset.path);
            });
        if (missing_required != manifest.assets.end()) {
            report.asset_prefetch.missing_required_assets = true;
            report.safety.summary = "required asset missing: " + missing_required->id;
            persist_telemetry(manifest.workload, report);
            return report;
        }
    }

    report = manifest.has_graph ? execute_managed(manifest.workload, manifest.graph) : execute_managed(manifest.workload);
    if (manifest.has_graph) {
        report.asset_prefetch = build_asset_prefetch(manifest, report.execution.optimization);
        if (report.asset_prefetch.missing_required_assets && report.executed) {
            report.executed = false;
            report.safety.summary += report.safety.summary.empty() ? "" : "; ";
            report.safety.summary += report.asset_prefetch.summary;
        }
    }
    return report;
}

std::filesystem::path Runtime::telemetry_path() const {
    if (!options_.product.observability.telemetry_path.empty()) {
        return options_.product.observability.telemetry_path;
    }
    try {
        return std::filesystem::temp_directory_path() / "jakal_core_runtime_telemetry.tsv";
    } catch (const std::exception&) {
        return std::filesystem::path("jakal_core_runtime_telemetry.tsv");
    }
}

std::string Runtime::strategy_safety_key(
    const WorkloadSpec& workload,
    const PartitionStrategy strategy) const {
    std::ostringstream stream;
    stream << workload.name << '|'
           << to_string(workload.kind) << '|'
           << workload.dataset_tag << '|'
           << to_string(canonical_workload_phase(workload)) << '|'
           << canonical_workload_shape_bucket(workload) << '|'
           << to_string(strategy);
    return stream.str();
}

bool Runtime::is_strategy_blacklisted(
    const WorkloadSpec& workload,
    const PartitionStrategy strategy) const {
    if (strategy == PartitionStrategy::auto_balanced) {
        return false;
    }
    const auto it = strategy_blacklist_until_epoch_.find(strategy_safety_key(workload, strategy));
    return it != strategy_blacklist_until_epoch_.end() && it->second > execution_epoch_;
}

void Runtime::record_strategy_failure(const WorkloadSpec& workload, const PartitionStrategy strategy) {
    if (strategy == PartitionStrategy::auto_balanced) {
        return;
    }
    const auto key = strategy_safety_key(workload, strategy);
    const auto failures = ++strategy_failure_counts_[key];
    if (failures >= options_.product.safety.blacklist_after_failures) {
        strategy_blacklist_until_epoch_[key] = execution_epoch_ + options_.product.safety.blacklist_cooldown_epochs;
        strategy_failure_counts_[key] = 0u;
    }
}

void Runtime::record_strategy_success(const WorkloadSpec& workload, const PartitionStrategy strategy) {
    if (strategy == PartitionStrategy::auto_balanced) {
        return;
    }
    const auto key = strategy_safety_key(workload, strategy);
    strategy_failure_counts_.erase(key);
    strategy_blacklist_until_epoch_.erase(key);
}

MemoryPreflightReport Runtime::build_memory_preflight(const OptimizationReport& optimization) const {
    MemoryPreflightReport report;
    if (devices_.empty()) {
        report.safe_to_run = false;
        report.summary = "no devices available";
        return report;
    }

    std::unordered_map<std::string, DeviceMemoryReservation> reservations;
    reservations.reserve(devices_.size());
    for (const auto& graph : devices_) {
        DeviceMemoryReservation reservation;
        reservation.device_uid = graph.uid;
        reservation.host = graph.probe == "host";
        reservation.effective_capacity_bytes = effective_capacity_bytes(graph, options_.product.memory);
        reservations.emplace(graph.uid, reservation);
    }

    std::unordered_map<std::string, std::uint64_t> persistent_seen;
    std::unordered_map<std::string, std::uint64_t> transient_seen;
    for (const auto& result : optimization.operations) {
        for (const auto& entry : result.graph.residency_plan) {
            const auto key = entry.device_uid + "|" + entry.tensor_id;
            auto& target = entry.persistent ? persistent_seen[key] : transient_seen[key];
            target = std::max(target, entry.bytes);
        }
    }

    for (const auto& [key, bytes] : persistent_seen) {
        const auto delimiter = key.find('|');
        const auto device_uid = key.substr(0u, delimiter);
        if (const auto it = reservations.find(device_uid); it != reservations.end()) {
            it->second.persistent_bytes += bytes;
        }
        report.aggregate_persistent_bytes += bytes;
    }

    for (const auto& [key, bytes] : transient_seen) {
        const auto delimiter = key.find('|');
        const auto device_uid = key.substr(0u, delimiter);
        if (const auto it = reservations.find(device_uid); it != reservations.end()) {
            it->second.transient_bytes += bytes;
        }
        report.aggregate_transient_bytes += bytes;
    }

    if (!optimization.placement.allocations.empty()) {
        for (const auto& allocation : optimization.placement.allocations) {
            if (const auto it = reservations.find(allocation.device.uid); it != reservations.end()) {
                it->second.transient_bytes += static_cast<std::uint64_t>(
                    std::round(static_cast<double>(optimization.workload_working_set_bytes) * allocation.ratio));
                it->second.transient_bytes += static_cast<std::uint64_t>(
                    std::round(static_cast<double>(optimization.workload_host_exchange_bytes) * allocation.ratio * 0.5));
            }
        }
        report.aggregate_transient_bytes += optimization.workload_working_set_bytes;
        report.aggregate_transient_bytes += optimization.workload_host_exchange_bytes / 2u;
    }

    report.pinned_host_visible_bytes = std::accumulate(
        optimization.workload_graph.tensors.begin(),
        optimization.workload_graph.tensors.end(),
        std::uint64_t{0},
        [](const std::uint64_t total, const WorkloadTensor& tensor) {
            return total + (tensor.host_visible ? tensor.bytes : 0ull);
        });

    std::ostringstream summary;
    for (auto& [device_uid, reservation] : reservations) {
        reservation.reserved_bytes = reservation.persistent_bytes + reservation.transient_bytes;
        reservation.pressure_ratio =
            reservation.effective_capacity_bytes == 0u
                ? (reservation.reserved_bytes > 0u ? 1.0 : 0.0)
                : static_cast<double>(reservation.reserved_bytes) /
                      static_cast<double>(std::max<std::uint64_t>(1u, reservation.effective_capacity_bytes));
        report.peak_pressure_ratio = std::max(report.peak_pressure_ratio, reservation.pressure_ratio);
        if (reservation.pressure_ratio > options_.product.memory.max_pressure_ratio) {
            report.requires_spill = true;
            if (reservation.host || !options_.product.memory.allow_host_spill || reservation.pressure_ratio > 1.15) {
                report.safe_to_run = false;
            }
        }
        report.devices.push_back(reservation);
        if (reservation.pressure_ratio > options_.product.memory.max_pressure_ratio) {
            if (summary.tellp() > 0) {
                summary << "; ";
            }
            summary << device_uid << " pressure=" << reservation.pressure_ratio;
        }
    }

    if (!report.safe_to_run && report.devices.empty()) {
        report.summary = "no devices available for memory preflight";
    } else if (report.safe_to_run && report.requires_spill) {
        report.summary = "memory spill expected";
    } else if (!report.safe_to_run && summary.tellp() > 0) {
        report.summary = summary.str();
    }
    return report;
}

ResidencySequenceReport Runtime::build_residency_sequence(const OptimizationReport& optimization) const {
    struct SequencedTensor {
        std::string tensor_id;
        std::string device_uid;
        std::uint64_t bytes = 0;
        std::uint32_t first = 0;
        std::uint32_t last = 0;
        bool persistent = false;
    };

    ResidencySequenceReport report;
    if (optimization.workload_graph.operations.empty()) {
        return report;
    }

    std::unordered_map<std::string, TensorLifetime> lifetimes_by_id;
    lifetimes_by_id.reserve(optimization.workload_graph.lifetimes.size());
    for (const auto& lifetime : optimization.workload_graph.lifetimes) {
        lifetimes_by_id.emplace(lifetime.tensor_id, lifetime);
    }

    std::unordered_map<std::string, std::vector<SequencedTensor>> tensors_by_device;
    for (const auto& optimized : optimization.operations) {
        for (const auto& entry : optimized.graph.residency_plan) {
            auto& device_tensors = tensors_by_device[entry.device_uid];
            const auto duplicate = std::find_if(
                device_tensors.begin(),
                device_tensors.end(),
                [&](const SequencedTensor& tensor) {
                    return tensor.tensor_id == entry.tensor_id;
                });
            const auto lifetime_it = lifetimes_by_id.find(entry.tensor_id);
            const auto first = lifetime_it == lifetimes_by_id.end() ? 0u : lifetime_it->second.first_operation_index;
            const auto last = lifetime_it == lifetimes_by_id.end()
                                  ? static_cast<std::uint32_t>(optimization.workload_graph.operations.size() - 1u)
                                  : lifetime_it->second.last_operation_index;
            if (duplicate == device_tensors.end()) {
                device_tensors.push_back(SequencedTensor{
                    entry.tensor_id,
                    entry.device_uid,
                    entry.bytes,
                    first,
                    last,
                    entry.persistent});
            } else {
                duplicate->bytes = std::max(duplicate->bytes, entry.bytes);
                duplicate->first = std::min(duplicate->first, first);
                duplicate->last = std::max(duplicate->last, last);
                duplicate->persistent = duplicate->persistent || entry.persistent;
            }
        }
    }

    std::ostringstream summary;
    for (const auto& [device_uid, tensors] : tensors_by_device) {
        const auto* graph = find_graph_by_uid(devices_, device_uid);
        const auto effective_capacity =
            graph == nullptr ? 0u : effective_capacity_bytes(*graph, options_.product.memory);
        const auto safe_capacity = static_cast<std::uint64_t>(
            static_cast<double>(effective_capacity) * options_.product.memory.max_pressure_ratio);

        std::unordered_map<std::string, SequencedTensor> live_tensors;
        std::unordered_set<std::string> spilled_tensors;
        std::uint64_t live_bytes = 0u;

        for (std::uint32_t operation_index = 0u;
             operation_index < optimization.workload_graph.operations.size();
             ++operation_index) {
            const auto& operation = optimization.workload_graph.operations[operation_index];
            std::set<std::string> current_needed;
            current_needed.insert(operation.input_tensor_ids.begin(), operation.input_tensor_ids.end());
            current_needed.insert(operation.output_tensor_ids.begin(), operation.output_tensor_ids.end());
            current_needed.insert(operation.temporary_tensor_ids.begin(), operation.temporary_tensor_ids.end());

            for (const auto& tensor : tensors) {
                if (tensor.first > operation_index || tensor.last < operation_index) {
                    continue;
                }
                if (live_tensors.find(tensor.tensor_id) != live_tensors.end()) {
                    continue;
                }
                const bool spilled = spilled_tensors.find(tensor.tensor_id) != spilled_tensors.end();
                const bool needed_now = current_needed.find(tensor.tensor_id) != current_needed.end() ||
                                        tensor.first == operation_index || tensor.persistent;
                if (!needed_now) {
                    continue;
                }
                report.actions.push_back(ResidencyAction{
                    spilled ? ResidencyActionKind::reload : ResidencyActionKind::prefetch,
                    tensor.tensor_id,
                    device_uid,
                    operation.name,
                    operation_index,
                    tensor.bytes,
                    tensor.persistent});
                live_tensors.emplace(tensor.tensor_id, tensor);
                live_bytes += tensor.bytes;
                if (spilled) {
                    report.reload_bytes += tensor.bytes;
                    spilled_tensors.erase(tensor.tensor_id);
                }
            }

            report.peak_live_bytes = std::max(report.peak_live_bytes, live_bytes);
            while (safe_capacity > 0u && live_bytes > safe_capacity) {
                auto candidate_it = live_tensors.end();
                for (auto it = live_tensors.begin(); it != live_tensors.end(); ++it) {
                    if (it->second.persistent || current_needed.find(it->first) != current_needed.end()) {
                        continue;
                    }
                    if (candidate_it == live_tensors.end() || it->second.last > candidate_it->second.last) {
                        candidate_it = it;
                    }
                }
                if (candidate_it == live_tensors.end()) {
                    ++report.forced_spill_count;
                    report.viable_without_spill = false;
                    break;
                }
                report.actions.push_back(ResidencyAction{
                    ResidencyActionKind::spill,
                    candidate_it->second.tensor_id,
                    device_uid,
                    operation.name,
                    operation_index,
                    candidate_it->second.bytes,
                    candidate_it->second.persistent});
                report.spill_bytes += candidate_it->second.bytes;
                live_bytes -= candidate_it->second.bytes;
                spilled_tensors.insert(candidate_it->second.tensor_id);
                live_tensors.erase(candidate_it);
            }

            for (auto it = live_tensors.begin(); it != live_tensors.end();) {
                if (!it->second.persistent && it->second.last == operation_index) {
                    report.actions.push_back(ResidencyAction{
                        ResidencyActionKind::evict,
                        it->second.tensor_id,
                        device_uid,
                        operation.name,
                        operation_index,
                        it->second.bytes,
                        false});
                    live_bytes -= it->second.bytes;
                    it = live_tensors.erase(it);
                } else {
                    ++it;
                }
            }
        }

        if (report.forced_spill_count > 0u) {
            if (summary.tellp() > 0) {
                summary << "; ";
            }
            summary << device_uid << " forced_spills=" << report.forced_spill_count;
        }
    }

    if (report.spill_bytes > 0u && summary.tellp() == 0) {
        summary << "spill=" << report.spill_bytes << " reload=" << report.reload_bytes;
    }
    report.summary = summary.str();
    return report;
}

KernelCoverageReport Runtime::build_kernel_coverage(const OptimizationReport& optimization) const {
    KernelCoverageReport report;
    std::ostringstream summary;
    for (const auto& optimized : optimization.operations) {
        const auto* graph = find_graph_by_uid(devices_, optimized.config.primary_device_uid);
        if (graph == nullptr) {
            continue;
        }
        std::string reason;
        if (!backend_supports_operation(*graph, optimized.operation.op_class, &reason)) {
            report.all_supported = false;
            report.issues.push_back(KernelCoverageIssue{
                optimized.operation.name,
                graph->uid,
                backend_name_for_graph(*graph),
                optimized.operation.op_class,
                false,
                reason});
            if (summary.tellp() > 0) {
                summary << "; ";
            }
            summary << optimized.operation.name << '@' << graph->probe << ": " << reason;
        }
    }
    report.summary = summary.str();
    return report;
}

AssetPrefetchReport Runtime::build_asset_prefetch(
    const WorkloadManifest& manifest,
    const OptimizationReport& optimization) const {
    const auto queue_hint_for_asset = [&](const WorkloadAsset& asset, const std::string& device_uid) {
        const auto* target_graph = device_uid.empty() ? nullptr : find_graph_by_uid(devices_, device_uid);
        const bool target_is_host = target_graph != nullptr && target_graph->probe == "host";
        if (device_uid.empty() || target_is_host || asset.preferred_residency == "host") {
            return std::string("host_io");
        }
        if (asset.preferred_residency == "device" || asset.preferred_residency == "accelerator") {
            return std::string("host_to_device");
        }
        return std::string("host_to_device");
    };

    AssetPrefetchReport report;
    if (manifest.assets.empty()) {
        return report;
    }

    std::unordered_map<std::string, std::vector<std::string>> tensor_devices;
    for (const auto& optimized : optimization.operations) {
        for (const auto& entry : optimized.graph.residency_plan) {
            if (!entry.persistent) {
                continue;
            }
            auto& devices = tensor_devices[entry.tensor_id];
            if (std::find(devices.begin(), devices.end(), entry.device_uid) == devices.end()) {
                devices.push_back(entry.device_uid);
            }
        }
    }

    std::ostringstream summary;
    for (const auto& asset : manifest.assets) {
        const bool exists = asset.path.empty() ? false : std::filesystem::exists(asset.path);
        if (asset.preload_required && !exists) {
            report.missing_required_assets = true;
            if (summary.tellp() > 0) {
                summary << "; ";
            }
            summary << "missing asset " << asset.id;
        }
        if (asset.tensor_ids.empty()) {
            const auto queue_hint = queue_hint_for_asset(asset, std::string());
            report.entries.push_back(AssetPrefetchEntry{
                asset.id,
                asset.path,
                std::string(),
                std::string(),
                asset.file_offset,
                asset.bytes,
                queue_hint,
                asset.preferred_residency,
                exists,
                asset.preload_required,
                asset.persistent,
                asset.host_visible,
                !asset.host_visible || queue_hint != "host_io"});
            report.total_prefetch_bytes += asset.bytes;
            report.total_host_io_bytes += asset.bytes;
            continue;
        }
        const auto per_tensor_bytes = asset.bytes == 0u
                                          ? 0u
                                          : std::max<std::uint64_t>(
                                                1u,
                                                asset.bytes / static_cast<std::uint64_t>(asset.tensor_ids.size()));
        for (const auto& tensor_id : asset.tensor_ids) {
            const auto devices_it = tensor_devices.find(tensor_id);
            if (devices_it == tensor_devices.end() || devices_it->second.empty()) {
                const auto queue_hint = queue_hint_for_asset(asset, std::string());
                report.entries.push_back(AssetPrefetchEntry{
                    asset.id,
                    asset.path,
                    tensor_id,
                    std::string(),
                    asset.file_offset,
                    per_tensor_bytes,
                    queue_hint,
                    asset.preferred_residency,
                    exists,
                    asset.preload_required,
                    asset.persistent,
                    asset.host_visible,
                    !asset.host_visible || queue_hint != "host_io"});
                report.total_prefetch_bytes += per_tensor_bytes;
                report.total_host_io_bytes += per_tensor_bytes;
                continue;
            }
            for (const auto& device_uid : devices_it->second) {
                const auto queue_hint = queue_hint_for_asset(asset, device_uid);
                report.entries.push_back(AssetPrefetchEntry{
                    asset.id,
                    asset.path,
                    tensor_id,
                    device_uid,
                    asset.file_offset,
                    per_tensor_bytes,
                    queue_hint,
                    asset.preferred_residency,
                    exists,
                    asset.preload_required,
                    asset.persistent,
                    asset.host_visible,
                    !asset.host_visible || queue_hint != "host_io"});
                report.total_prefetch_bytes += per_tensor_bytes;
                if (queue_hint == "host_to_device") {
                    report.total_host_to_device_bytes += per_tensor_bytes;
                } else {
                    report.total_host_io_bytes += per_tensor_bytes;
                }
            }
        }
    }
    if (!report.missing_required_assets) {
        summary << (summary.tellp() > 0 ? "; " : "")
                << "prefetch=" << report.total_prefetch_bytes
                << " host_io=" << report.total_host_io_bytes
                << " h2d=" << report.total_host_to_device_bytes;
    }
    report.summary = summary.str();
    return report;
}

void Runtime::persist_telemetry(
    const WorkloadSpec& workload,
    const ManagedExecutionReport& report) const {
    if (!options_.product.observability.persist_telemetry) {
        return;
    }

    const auto path = telemetry_path();
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }

    const bool write_header = !std::filesystem::exists(path);
    std::ofstream output(path, std::ios::app);
    if (!output.is_open()) {
        return;
    }

    if (write_header) {
        output << "# epoch\tworkload\tkind\tphase\tshape_bucket\trequested_strategy\tselected_strategy\tfinal_strategy\tplanner_source\tplanner_confidence\tplanner_risk\texecuted\tall_succeeded\tblocked_by_memory\trolled_back_to_auto\tblacklisted_before_run\tpeak_pressure_ratio\tspill_bytes\treload_bytes\tforced_spills\tprefetch_bytes\thost_io_bytes\th2d_bytes\ttotal_runtime_us\tspeedup_vs_reference\tsummary\n";
    }

    output << execution_epoch_ << '\t'
           << workload.name << '\t'
           << to_string(workload.kind) << '\t'
           << to_string(canonical_workload_phase(workload)) << '\t'
           << canonical_workload_shape_bucket(workload) << '\t'
           << to_string(report.safety.requested_strategy) << '\t'
           << to_string(report.safety.selected_strategy) << '\t'
           << to_string(report.safety.final_strategy) << '\t'
           << to_string(report.safety.planner_strategy_source) << '\t'
           << report.safety.planner_confidence << '\t'
           << report.safety.planner_risk_score << '\t'
           << (report.executed ? 1 : 0) << '\t'
           << (report.executed && report.execution.all_succeeded ? 1 : 0) << '\t'
           << (report.safety.blocked_by_memory ? 1 : 0) << '\t'
           << (report.safety.rolled_back_to_auto ? 1 : 0) << '\t'
           << (report.safety.blacklisted_before_run ? 1 : 0) << '\t'
           << report.memory_preflight.peak_pressure_ratio << '\t'
           << report.residency_sequence.spill_bytes << '\t'
           << report.residency_sequence.reload_bytes << '\t'
           << report.residency_sequence.forced_spill_count << '\t'
           << report.asset_prefetch.total_prefetch_bytes << '\t'
           << report.asset_prefetch.total_host_io_bytes << '\t'
           << report.asset_prefetch.total_host_to_device_bytes << '\t'
           << (report.executed ? report.execution.total_runtime_us : 0.0) << '\t'
           << (report.executed ? report.execution.speedup_vs_reference : 0.0) << '\t'
           << report.safety.summary << '\n';
}

bool Runtime::should_include_descriptor(const HardwareGraph& candidate) const {
    return std::none_of(devices_.begin(), devices_.end(), [&](const HardwareGraph& existing) {
        if (existing.uid == candidate.uid) {
            return true;
        }

        const bool same_name = existing.presentation_name == candidate.presentation_name;
        const bool same_probe_shape = structural_fingerprint(existing) == structural_fingerprint(candidate);
        return same_name && same_probe_shape;
    });
}

}  // namespace jakal

