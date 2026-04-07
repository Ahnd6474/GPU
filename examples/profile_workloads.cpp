#include "jakal/runtime.hpp"
#include "jakal/workloads.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <unordered_map>

int main() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    jakal::RuntimeOptions options;
    options.cache_path = std::filesystem::temp_directory_path() /
                         ("gpu-profile-plan-" + std::to_string(nonce) + ".tsv");
    options.execution_cache_path = std::filesystem::temp_directory_path() /
                                   ("gpu-profile-exec-" + std::to_string(nonce) + ".tsv");
    jakal::Runtime runtime(options);
    const auto presets = jakal::canonical_workload_presets();

    if (presets.empty()) {
        std::cerr << "No canonical workloads defined.\n";
        return 1;
    }

    std::cout << "Canonical workload profile\n";
    for (const auto& preset : presets) {
        const auto report = runtime.execute(preset.workload);
        if (!report.all_succeeded) {
            std::cerr << "Execution failed for " << preset.workload.name << ".\n";
            return 1;
        }
        const auto warm_report = runtime.execute(preset.workload);
        if (!warm_report.all_succeeded) {
            std::cerr << "Warm execution failed for " << preset.workload.name << ".\n";
            return 1;
        }

        std::size_t opencl_ops = 0;
        std::size_t host_ops = 0;
        std::unordered_map<std::string, std::size_t> requested_backend_counts;
        for (const auto& operation : warm_report.operations) {
            opencl_ops += operation.used_opencl ? 1u : 0u;
            host_ops += operation.used_host ? 1u : 0u;
            const auto key = operation.requested_gpu_backend.empty()
                                 ? std::string("host")
                                 : (operation.requested_gpu_vendor + ":" + operation.requested_gpu_backend);
            ++requested_backend_counts[key];
        }

        std::cout << '\n'
                  << preset.workload.name
                  << " [" << preset.workload.dataset_tag << "]"
                  << "\n  kind=" << jakal::to_string(preset.workload.kind)
                  << " ops=" << warm_report.operations.size()
                  << "\n  pass0=" << std::fixed << std::setprecision(3) << report.total_runtime_us << "us"
                  << " readiness0=" << report.optimization.system_profile.readiness_score
                  << " stability0=" << report.optimization.system_profile.stability_score
                  << "\n  pass1=" << warm_report.total_runtime_us << "us"
                  << " readiness1=" << warm_report.optimization.system_profile.readiness_score
                  << " stability1=" << warm_report.optimization.system_profile.stability_score
                  << " reference=" << warm_report.total_reference_runtime_us << "us"
                  << " exec_improvement=" << warm_report.speedup_vs_reference << "x"
                  << " learning_gain=" << (report.total_runtime_us / std::max(warm_report.total_runtime_us, 1.0)) << "x"
                  << "\n  graph_opt=" << warm_report.optimization.graph_optimization.optimizer_name
                  << " passes=" << warm_report.optimization.graph_optimization.passes.size()
                  << " logical_parts=" << warm_report.optimization.graph_optimization.total_logical_partitions
                  << "\n  desc=" << preset.description
                  << "\n  backends host=" << host_ops
                  << " opencl=" << opencl_ops;

        std::unordered_map<std::string, std::size_t> optimizer_counts;
        for (const auto& operation : warm_report.optimization.operations) {
            ++optimizer_counts[operation.benchmark.optimizer_name];
        }
        std::cout << "\n  optimizers";
        for (const auto& [name, count] : optimizer_counts) {
            std::cout << " " << name << '=' << count;
        }
        std::cout << "\n  requested_backends";
        for (const auto& [name, count] : requested_backend_counts) {
            std::cout << " " << name << '=' << count;
        }
        std::cout
                  << '\n';
    }

    return 0;
}

