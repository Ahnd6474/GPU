#include "jakal/runtime.hpp"

#include <algorithm>
#include <numeric>
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
    return execution_optimizer_.optimize(workload, placement, devices_);
}

DirectExecutionReport Runtime::execute(const WorkloadSpec& workload) {
    if (devices_.empty()) {
        refresh_hardware();
    }

    const auto initial_optimization = optimize(workload);
    auto initial_report = direct_executor_.execute(initial_optimization, devices_, jakal_toolkit_index_);
    execution_optimizer_.ingest_execution_feedback(
        initial_report.optimization,
        make_feedback_records(initial_report),
        devices_);

    if (!should_retry_execution(initial_report)) {
        return initial_report;
    }

    const auto refined_optimization = optimize(workload);
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

