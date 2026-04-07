#include "jakal/runtime.hpp"

#include <iomanip>
#include <iostream>
#include <string>

namespace {

void print_devices(const jakal::Runtime& runtime) {
    std::cout << "devices=" << runtime.devices().size() << '\n';
    for (const auto& graph : runtime.devices()) {
        const auto summary = jakal::summarize_graph(graph);
        std::cout << "  uid=" << graph.uid
                  << " probe=" << graph.probe
                  << " name=" << graph.presentation_name
                  << " exec=" << summary.execution_objects
                  << " mem_mib=" << (summary.addressable_bytes / (1024ull * 1024ull))
                  << '\n';
    }
}

jakal::RuntimeOptions parse_options(int argc, char** argv) {
    jakal::RuntimeOptions options;
    options.enable_host_probe = true;
    options.enable_opencl_probe = false;
    options.enable_level_zero_probe = false;
    options.enable_cuda_probe = false;
    options.enable_rocm_probe = false;

    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--opencl-only") {
            options.enable_opencl_probe = true;
            options.enable_level_zero_probe = false;
        } else if (arg == "--level-zero-only") {
            options.enable_opencl_probe = false;
            options.enable_level_zero_probe = true;
        } else if (arg == "--both-intel") {
            options.enable_opencl_probe = true;
            options.enable_level_zero_probe = true;
        } else if (arg == "--host-only") {
            options.enable_opencl_probe = false;
            options.enable_level_zero_probe = false;
        }
    }
    return options;
}

bool parse_low_precision(int argc, char** argv) {
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--highp") {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const bool low_precision = parse_low_precision(argc, argv);
        std::cout << "runtime-init\n" << std::flush;
        jakal::Runtime runtime(options);
        print_devices(runtime);

        const jakal::WorkloadSpec workload{
            "live-backend-smoke",
            jakal::WorkloadKind::tensor,
            "",
            128ull * 1024ull * 1024ull,
            64ull * 1024ull * 1024ull,
            2.0e11,
            4,
            false,
            false,
            low_precision};

        std::cout << "plan\n" << std::flush;
        const auto plan = runtime.plan(workload);
        std::cout << "  allocations=" << plan.allocations.size()
                  << " strategy=" << jakal::to_string(plan.resolved_partition_strategy)
                  << " source=" << jakal::to_string(plan.strategy_source)
                  << " confidence=" << std::fixed << std::setprecision(3) << plan.strategy_confidence
                  << '\n';
        for (const auto& allocation : plan.allocations) {
            std::cout << "    device=" << allocation.device.uid
                      << " ratio=" << allocation.ratio
                      << " score=" << allocation.score
                      << '\n';
        }

        std::cout << "execute\n" << std::flush;
        const auto report = runtime.execute(workload);
        std::cout << "  success=" << (report.all_succeeded ? "yes" : "no")
                  << " total_us=" << report.total_runtime_us
                  << " reference_us=" << report.total_reference_runtime_us
                  << " speedup=" << report.speedup_vs_reference
                  << '\n';
        for (const auto& operation : report.operations) {
            std::cout << "    op=" << operation.operation_name
                      << " backend=" << operation.backend_name
                      << " requested=";
            if (operation.requested_gpu_backend.empty()) {
                std::cout << "host";
            } else {
                std::cout << operation.requested_gpu_vendor << ':' << operation.requested_gpu_backend;
            }
            std::cout << " host=" << operation.used_host
                      << " opencl=" << operation.used_opencl
                      << " verified=" << operation.verified;
            if (!operation.backend_error.empty()) {
                std::cout << " error=" << operation.backend_error;
            }
            std::cout << '\n';
        }

        std::cout << "live backend smoke ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "exception: " << error.what() << '\n';
        return 1;
    }
}
