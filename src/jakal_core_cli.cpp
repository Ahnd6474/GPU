#include "jakal/runtime.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage() {
    std::cout
        << "Usage: jakal_core_cli <command> [options]\n"
        << "Commands:\n"
        << "  doctor [--host-only] [--runtime-root PATH]\n"
        << "  devices [--host-only] [--runtime-root PATH]\n"
        << "  paths [--runtime-root PATH]\n"
        << "  smoke [--host-only] [--runtime-root PATH]\n"
        << "  run-manifest <path> [--host-only] [--runtime-root PATH]\n";
}

struct CliOptions {
    std::string command;
    std::filesystem::path runtime_root;
    std::filesystem::path manifest_path;
    bool host_only = false;
};

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;
    if (argc > 1) {
        options.command = argv[1];
    }
    for (int index = 2; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--host-only") {
            options.host_only = true;
            continue;
        }
        if (arg == "--runtime-root" && index + 1 < argc) {
            options.runtime_root = argv[++index];
            continue;
        }
        if (options.manifest_path.empty()) {
            options.manifest_path = arg;
        }
    }
    return options;
}

jakal::RuntimeOptions make_options(const CliOptions& cli) {
    auto options = jakal::make_runtime_options_for_install(cli.runtime_root);
    options.product.observability.persist_telemetry = false;
    if (cli.host_only) {
        options.enable_opencl_probe = false;
        options.enable_level_zero_probe = false;
        options.enable_vulkan_probe = false;
        options.enable_cuda_probe = false;
        options.enable_rocm_probe = false;
        options.enable_vulkan_status = false;
    }
    return options;
}

void print_paths(const jakal::RuntimeInstallPaths& paths) {
    std::cout << "install_root=" << paths.install_root.string() << '\n';
    std::cout << "writable_root=" << paths.writable_root.string() << '\n';
    std::cout << "config_dir=" << paths.config_dir.string() << '\n';
    std::cout << "cache_dir=" << paths.cache_dir.string() << '\n';
    std::cout << "logs_dir=" << paths.logs_dir.string() << '\n';
    std::cout << "telemetry_path=" << paths.telemetry_path.string() << '\n';
    std::cout << "planner_cache_path=" << paths.planner_cache_path.string() << '\n';
    std::cout << "execution_cache_path=" << paths.execution_cache_path.string() << '\n';
    std::cout << "python_dir=" << paths.python_dir.string() << '\n';
}

void print_devices(const jakal::Runtime& runtime) {
    std::cout << "devices=" << runtime.devices().size() << '\n';
    for (const auto& graph : runtime.devices()) {
        const auto summary = jakal::summarize_graph(graph);
        std::cout << "  uid=" << graph.uid
                  << " probe=" << graph.probe
                  << " name=" << graph.presentation_name
                  << " exec=" << summary.execution_objects
                  << " vec=" << summary.native_vector_bits
                  << " mem_mib=" << (summary.addressable_bytes / (1024ull * 1024ull))
                  << '\n';
    }
}

void print_statuses(const jakal::Runtime& runtime) {
    for (const auto& status : runtime.backend_statuses()) {
        std::cout << "backend=" << status.backend_name
                  << " code=" << jakal::to_string(status.code)
                  << " enabled=" << (status.enabled ? "yes" : "no")
                  << " available=" << (status.available ? "yes" : "no")
                  << " direct=" << (status.direct_execution ? "yes" : "no")
                  << " modeled=" << (status.modeled_fallback ? "yes" : "no");
        if (!status.device_uid.empty()) {
            std::cout << " device=" << status.device_uid;
        }
        if (!status.detail.empty()) {
            std::cout << " detail=" << status.detail;
        }
        std::cout << '\n';
    }
}

int run_doctor(const CliOptions& cli) {
    jakal::Runtime runtime(make_options(cli));
    print_paths(runtime.install_paths());
    print_statuses(runtime);
    print_devices(runtime);
    const bool has_host = std::any_of(runtime.devices().begin(), runtime.devices().end(), [](const jakal::HardwareGraph& graph) {
        return graph.probe == "host";
    });
    return has_host ? 0 : 2;
}

int run_devices(const CliOptions& cli) {
    jakal::Runtime runtime(make_options(cli));
    print_devices(runtime);
    return 0;
}

int run_paths(const CliOptions& cli) {
    const auto options = make_options(cli);
    const auto paths = jakal::resolve_runtime_install_paths(cli.runtime_root);
    (void)options;
    print_paths(paths);
    return 0;
}

int run_smoke(const CliOptions& cli) {
    jakal::Runtime runtime(make_options(cli));
    const jakal::WorkloadSpec workload{
        "cli-smoke",
        jakal::WorkloadKind::tensor,
        "cli-smoke-lite",
        128ull * 1024ull * 1024ull,
        64ull * 1024ull * 1024ull,
        2.0e11,
        4,
        false,
        false,
        true,
        jakal::PartitionStrategy::auto_balanced,
        jakal::WorkloadPhase::prefill,
        "b4-lite"};
    const auto report = runtime.execute(workload);
    std::cout << "operations=" << report.operations.size()
              << " total_us=" << report.total_runtime_us
              << " speedup=" << report.speedup_vs_reference
              << " success=" << (report.all_succeeded ? "yes" : "no") << '\n';
    return report.all_succeeded ? 0 : 3;
}

int run_manifest(const CliOptions& cli) {
    if (cli.manifest_path.empty()) {
        std::cerr << "run-manifest requires a manifest path\n";
        return 2;
    }
    jakal::Runtime runtime(make_options(cli));
    const auto report = runtime.execute_manifest(cli.manifest_path);
    std::cout << "executed=" << (report.executed ? "yes" : "no")
              << " operations=" << report.execution.operations.size()
              << " runtime_us=" << report.execution.total_runtime_us
              << " telemetry=" << report.telemetry_path.string() << '\n';
    if (!report.safety.summary.empty()) {
        std::cout << "safety=" << report.safety.summary << '\n';
    }
    return report.executed ? 0 : 4;
}

}  // namespace

int main(int argc, char** argv) {
    const auto cli = parse_args(argc, argv);
    if (cli.command.empty() || cli.command == "--help" || cli.command == "-h") {
        print_usage();
        return cli.command.empty() ? 1 : 0;
    }

    try {
        if (cli.command == "doctor") {
            return run_doctor(cli);
        }
        if (cli.command == "devices") {
            return run_devices(cli);
        }
        if (cli.command == "paths") {
            return run_paths(cli);
        }
        if (cli.command == "smoke") {
            return run_smoke(cli);
        }
        if (cli.command == "run-manifest") {
            return run_manifest(cli);
        }
        std::cerr << "unknown command: " << cli.command << '\n';
        print_usage();
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "jakal_core_cli error: " << error.what() << '\n';
        return 10;
    }
}
