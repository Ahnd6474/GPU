#include "jakal/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace jakal {
namespace {

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

void record_partition_strategy_feedback(
    Planner& planner,
    const WorkloadSpec& requested_workload,
    const std::vector<HardwareGraph>& graphs,
    const DirectExecutionReport& report) {
    planner.ingest_strategy_feedback(
        requested_workload,
        graphs,
        StrategyFeedbackSample{
            report.optimization.partition_strategy,
            report.total_runtime_us,
            head_runtime_us(requested_workload, report),
            report.speedup_vs_reference,
            successful_operation_ratio(report),
            report.all_succeeded});
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

}  // namespace

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

    for (auto& probe : probes_) {
        if (!probe->available()) {
            continue;
        }

        for (auto& graph : probe->discover_hardware()) {
            if (should_include_descriptor(graph)) {
                devices_.push_back(std::move(graph));
            }
        }
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

    auto optimization = optimize(effective_workload, workload_graph);
    managed.safety.selected_strategy = optimization.partition_strategy;
    managed.memory_preflight = build_memory_preflight(optimization);

    auto force_auto_if_needed = [&](const char* reason) {
        if (optimization.partition_strategy == PartitionStrategy::auto_balanced) {
            return;
        }
        auto fallback_workload = workload;
        fallback_workload.partition_strategy = PartitionStrategy::auto_balanced;
        auto fallback_optimization = optimize(fallback_workload, workload_graph);
        auto fallback_memory = build_memory_preflight(fallback_optimization);
        if (fallback_memory.safe_to_run || !managed.memory_preflight.safe_to_run) {
            effective_workload = fallback_workload;
            optimization = std::move(fallback_optimization);
            managed.memory_preflight = std::move(fallback_memory);
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

    const bool explicit_strategy = managed.execution.optimization.partition_strategy != PartitionStrategy::auto_balanced;
    const bool canary_triggered =
        explicit_strategy &&
        options_.product.safety.enable_canary &&
        (!managed.execution.all_succeeded ||
         runtime_regressed(managed.execution, options_.product.safety.max_runtime_regression_ratio));

    if (canary_triggered) {
        managed.safety.canary_triggered = true;
        record_strategy_failure(workload, managed.execution.optimization.partition_strategy);
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
        record_partition_strategy_feedback(planner_, workload, devices_, managed.execution);
    }

    if (managed.memory_preflight.summary.size() > 0) {
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << managed.memory_preflight.summary;
    }
    managed.safety.summary = safety_summary.str();
    persist_telemetry(workload, managed);
    return managed;
}

ManagedExecutionReport Runtime::execute_manifest(const std::filesystem::path& manifest_path) {
    const auto manifest = load_workload_manifest(manifest_path);
    return manifest.has_graph ? execute_managed(manifest.workload, manifest.graph) : execute_managed(manifest.workload);
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
        output << "# epoch\tworkload\tkind\tphase\tshape_bucket\trequested_strategy\tselected_strategy\tfinal_strategy\texecuted\tall_succeeded\tblocked_by_memory\trolled_back_to_auto\tblacklisted_before_run\tpeak_pressure_ratio\ttotal_runtime_us\tspeedup_vs_reference\tsummary\n";
    }

    output << execution_epoch_ << '\t'
           << workload.name << '\t'
           << to_string(workload.kind) << '\t'
           << to_string(canonical_workload_phase(workload)) << '\t'
           << canonical_workload_shape_bucket(workload) << '\t'
           << to_string(report.safety.requested_strategy) << '\t'
           << to_string(report.safety.selected_strategy) << '\t'
           << to_string(report.safety.final_strategy) << '\t'
           << (report.executed ? 1 : 0) << '\t'
           << (report.executed && report.execution.all_succeeded ? 1 : 0) << '\t'
           << (report.safety.blocked_by_memory ? 1 : 0) << '\t'
           << (report.safety.rolled_back_to_auto ? 1 : 0) << '\t'
           << (report.safety.blacklisted_before_run ? 1 : 0) << '\t'
           << report.memory_preflight.peak_pressure_ratio << '\t'
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

