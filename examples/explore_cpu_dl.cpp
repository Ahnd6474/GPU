#include "jakal/runtime.hpp"
#include "jakal/workloads.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path unique_temp_file(const std::string& stem) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (stem + "-" + std::to_string(nonce) + ".tsv");
}

std::string join_devices(const std::vector<std::string>& devices) {
    if (devices.empty()) {
        return "none";
    }

    std::ostringstream stream;
    for (std::size_t index = 0; index < devices.size(); ++index) {
        if (index > 0) {
            stream << ',';
        }
        stream << devices[index];
    }
    return stream.str();
}

double mib(const std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char** argv) {
    jakal::RuntimeOptions options;
    options.cache_path = unique_temp_file("gpu-cpu-dl-plan");
    options.execution_cache_path = unique_temp_file("gpu-cpu-dl-exec");

    jakal::Runtime runtime(options);
    bool execute_workloads = false;
    bool run_all = false;
    std::optional<std::string> selected_preset_name;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--execute") {
            execute_workloads = true;
        } else if (arg == "--all") {
            run_all = true;
        } else {
            selected_preset_name = arg;
        }
    }

    const auto presets = jakal::cpu_deep_learning_exploration_presets();
    if (presets.empty()) {
        std::cerr << "No CPU deep-learning exploration presets defined.\n";
        return 1;
    }
    std::vector<jakal::CpuDeepLearningExplorationPreset> selected_presets;
    if (run_all) {
        selected_presets = presets;
    } else if (selected_preset_name.has_value()) {
        const auto it = std::find_if(
            presets.begin(),
            presets.end(),
            [&](const jakal::CpuDeepLearningExplorationPreset& preset) {
                return preset.workload.name == *selected_preset_name ||
                       preset.workload.dataset_tag == *selected_preset_name;
            });
        if (it == presets.end()) {
            std::cerr << "Unknown preset: " << *selected_preset_name << '\n';
            return 1;
        }
        selected_presets.push_back(*it);
    } else {
        const auto it = std::find_if(
            presets.begin(),
            presets.end(),
            [](const jakal::CpuDeepLearningExplorationPreset& preset) {
                return preset.workload.dataset_tag == "llm-decode-token-lite";
            });
        selected_presets.push_back(it == presets.end() ? presets.front() : *it);
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU in deep learning exploration\n";
    std::cout << "Mode: " << (execute_workloads ? "execute" : "optimize-only")
              << " preset_count=" << selected_presets.size() << '\n';
    std::cout << "Discovered devices\n";
    for (const auto& graph : runtime.devices()) {
        const auto summary = jakal::summarize_graph(graph);
        std::cout << "  - " << graph.uid
                  << " probe=" << graph.probe
                  << " name=" << graph.presentation_name
                  << " exec=" << summary.execution_objects
                  << " lanes=" << summary.lanes_per_object
                  << " mem_mib=" << mib(summary.addressable_bytes)
                  << " host_link_gbps=" << std::max(summary.host_read_gbps, summary.host_write_gbps)
                  << '\n';
    }

    for (const auto& preset : selected_presets) {
        const auto optimization = runtime.optimize(preset.workload);
        double total_runtime_us = 0.0;
        double total_reference_runtime_us = 0.0;
        double speedup_vs_reference = 0.0;
        std::size_t host_ops = 0;
        std::size_t opencl_ops = 0;
        std::size_t mixed_ops = 0;

        std::vector<jakal::OperationExecutionRecord> execution_records;
        if (execute_workloads) {
            const auto report = runtime.execute(preset.workload);
            if (!report.all_succeeded) {
                std::cerr << "Execution failed for " << preset.workload.name << ".\n";
                return 1;
            }
            total_runtime_us = report.total_runtime_us;
            total_reference_runtime_us = report.total_reference_runtime_us;
            speedup_vs_reference = report.speedup_vs_reference;
            execution_records = report.operations;
            for (const auto& operation : report.operations) {
                host_ops += operation.used_host ? 1u : 0u;
                opencl_ops += operation.used_opencl ? 1u : 0u;
                mixed_ops += operation.used_multiple_devices ? 1u : 0u;
            }
        } else {
            total_runtime_us = std::accumulate(
                optimization.operations.begin(),
                optimization.operations.end(),
                0.0,
                [](const double total, const jakal::OperationOptimizationResult& operation) {
                    return total + operation.graph.predicted_latency_us;
                });
        }

        const auto persistent_bytes = std::accumulate(
            optimization.workload_graph.tensors.begin(),
            optimization.workload_graph.tensors.end(),
            std::uint64_t{0},
            [](const std::uint64_t total, const jakal::WorkloadTensor& tensor) {
                return total + (tensor.persistent ? tensor.bytes : 0ull);
            });
        const auto host_visible_bytes = std::accumulate(
            optimization.workload_graph.tensors.begin(),
            optimization.workload_graph.tensors.end(),
            std::uint64_t{0},
            [](const std::uint64_t total, const jakal::WorkloadTensor& tensor) {
                return total + (tensor.host_visible ? tensor.bytes : 0ull);
            });
        const auto total_transfer_bytes = std::accumulate(
            optimization.operations.begin(),
            optimization.operations.end(),
            std::uint64_t{0},
            [](const std::uint64_t total, const jakal::OperationOptimizationResult& operation) {
                return total + std::accumulate(
                           operation.graph.transfer_schedule.begin(),
                           operation.graph.transfer_schedule.end(),
                           std::uint64_t{0},
                           [](const std::uint64_t op_total, const jakal::TransferScheduleEntry& transfer) {
                               return op_total + transfer.bytes;
                           });
            });

        std::cout << '\n'
                  << preset.workload.name
                  << " [" << preset.workload.dataset_tag << ']'
                  << "\n  description: " << preset.description
                  << "\n  cpu_hypothesis: " << preset.cpu_hypothesis
                  << "\n  success_signal: " << preset.success_signal
                  << "\n  working_set_mib=" << mib(preset.workload.working_set_bytes)
                  << " host_exchange_mib=" << mib(preset.workload.host_exchange_bytes)
                  << " persistent_mib=" << mib(persistent_bytes)
                  << " host_visible_mib=" << mib(host_visible_bytes)
                  << " predicted_transfer_mib=" << mib(total_transfer_bytes)
                  << "\n  plan";

        for (const auto& allocation : optimization.placement.allocations) {
            std::cout << ' '
                      << allocation.device.uid
                      << '(' << allocation.device.probe
                      << ",ratio=" << allocation.ratio
                      << ",score=" << allocation.score
                      << ')';
        }

        std::cout << "\n  "
                  << (execute_workloads ? "total_runtime_us=" : "predicted_runtime_us=") << total_runtime_us
                  << " reference_us=" << total_reference_runtime_us
                  << " speedup=" << speedup_vs_reference
                  << " readiness=" << optimization.system_profile.readiness_score
                  << " stability=" << optimization.system_profile.stability_score;
        if (execute_workloads) {
            std::cout << "\n  backend_counts host=" << host_ops
                      << " opencl=" << opencl_ops
                      << " mixed=" << mixed_ops;
        }
        std::cout
                  << "\n  operations\n";

        for (std::size_t index = 0; index < optimization.operations.size(); ++index) {
            const auto& optimized = optimization.operations[index];
            const auto op_transfer_bytes = std::accumulate(
                optimized.graph.transfer_schedule.begin(),
                optimized.graph.transfer_schedule.end(),
                std::uint64_t{0},
                [](const std::uint64_t total, const jakal::TransferScheduleEntry& transfer) {
                    return total + transfer.bytes;
                });
            std::cout << "    "
                      << optimized.operation.name
                      << " class=" << jakal::to_string(optimized.operation.op_class)
                      << " strategy=" << jakal::to_string(optimized.config.strategy)
                      << " primary=" << optimized.config.primary_device_uid
                      << " devices=" << join_devices(optimized.config.participating_devices)
                      << " parts=" << optimized.config.logical_partitions
                      << " peak_mib=" << mib(optimized.graph.peak_resident_bytes)
                      << " transfer_mib=" << mib(op_transfer_bytes)
                      << " transfer_us=" << optimized.graph.predicted_transfer_latency_us;
            if (execute_workloads) {
                const auto& executed = execution_records[index];
                std::cout << " runtime_us=" << executed.runtime_us
                          << " backend=" << executed.backend_name
                          << " host=" << executed.used_host
                          << " opencl=" << executed.used_opencl
                          << " multi=" << executed.used_multiple_devices;
                if (!executed.backend_error.empty()) {
                    std::cout << " error=" << executed.backend_error;
                }
            } else {
                std::cout << " predicted_us=" << optimized.graph.predicted_latency_us;
            }
            std::cout << '\n';
        }
    }

    return 0;
}

