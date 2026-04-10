#include "jakal/runtime.hpp"
#include "jakal/workloads.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    bool smoke = false;
    bool host_only = false;
};

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--smoke") {
            options.smoke = true;
        } else if (arg == "--host-only") {
            options.host_only = true;
        }
    }
    return options;
}

double measure_ms(const auto& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::optional<jakal::WorkloadSpec> find_cpu_preset(const std::string& dataset_tag) {
    for (const auto& preset : jakal::cpu_deep_learning_exploration_presets()) {
        if (preset.workload.dataset_tag == dataset_tag) {
            return preset.workload;
        }
    }
    return std::nullopt;
}

std::optional<jakal::WorkloadSpec> find_canonical_preset(const std::string& dataset_tag) {
    for (const auto& preset : jakal::canonical_workload_presets()) {
        if (preset.workload.dataset_tag == dataset_tag) {
            return preset.workload;
        }
    }
    return std::nullopt;
}

bool run_case(jakal::Runtime& runtime, const std::string& label, const jakal::WorkloadSpec& workload) {
    jakal::OptimizationReport report;
    const double optimize_ms = measure_ms([&]() {
        report = runtime.optimize(workload);
    });
    if (report.operations.empty()) {
        std::cerr << "benchmark failed: optimize returned no operations for " << label << '\n';
        return false;
    }

    jakal::DirectExecutionReport execution;
    const double execute_ms = measure_ms([&]() {
        execution = runtime.execute(workload);
    });
    if (!execution.all_succeeded || execution.operations.empty()) {
        std::cerr << "benchmark failed: execute did not succeed for " << label << '\n';
        return false;
    }

    std::size_t host_ops = 0u;
    std::size_t gpu_ops = 0u;
    std::size_t mixed_ops = 0u;
    for (const auto& operation : execution.operations) {
        const bool host = operation.backend_name.find("host") != std::string::npos;
        const bool mixed = operation.backend_name.find("mixed") != std::string::npos;
        if (mixed) {
            ++mixed_ops;
        } else if (host) {
            ++host_ops;
        } else {
            ++gpu_ops;
        }
    }

    std::cout << label
              << " optimize_ms=" << optimize_ms
              << " execute_ms=" << execute_ms
              << " ops=" << execution.operations.size()
              << " host_ops=" << host_ops
              << " gpu_ops=" << gpu_ops
              << " mixed_ops=" << mixed_ops
              << " cache=" << (report.loaded_from_cache ? "hit" : "miss")
              << '\n';
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const auto cli = parse_args(argc, argv);

    jakal::RuntimeOptions options;
    if (cli.host_only) {
        options.enable_opencl_probe = false;
        options.enable_level_zero_probe = false;
        options.enable_vulkan_probe = false;
        options.enable_cuda_probe = false;
        options.enable_rocm_probe = false;
        options.enable_vulkan_status = false;
    }
    jakal::Runtime runtime(options);

    const auto decode = find_cpu_preset("llm-decode-token-lite");
    const auto prefill = find_cpu_preset("llm-prefill-context-lite");
    const auto vision = find_canonical_preset("ai-vision-inference-224");
    const auto upscale = find_canonical_preset("gaming-fsr-like-720p-to-1080p");
    if (!decode.has_value() || !prefill.has_value() || !vision.has_value() || !upscale.has_value()) {
        std::cerr << "benchmark failed: missing built-in workload presets\n";
        return 1;
    }

    std::vector<std::pair<std::string, jakal::WorkloadSpec>> cases = {
        {"decode", *decode},
        {"prefill", *prefill},
        {"vision", *vision},
        {"upscale", *upscale},
    };
    if (cli.smoke) {
        cases.resize(2u);
    }

    for (const auto& [label, workload] : cases) {
        if (!run_case(runtime, label, workload)) {
            return 1;
        }
    }

    return 0;
}
