#include "jakal/runtime.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void print_usage() {
    std::cout
        << "Usage: jakal_core_cli <command> [options]\n"
        << "Commands:\n"
        << "  doctor [--host-only] [--json] [--runtime-root PATH]\n"
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
    bool json = false;
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
        if (arg == "--json") {
            options.json = true;
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

struct BackendAggregate {
    std::string name;
    bool enabled = false;
    bool available = false;
    bool ready_direct = false;
    bool ready_modeled = false;
    bool has_disabled_entry = false;
};

struct RecommendationOption {
    std::string id;
    std::string label;
    std::string description;
    std::string reason;
    bool available = false;
    bool recommended = false;
    std::vector<std::string> prerequisite_ids;
};

struct PrerequisiteOption {
    std::string id;
    std::string label;
    std::string description;
    std::string reason;
    bool available = true;
    bool recommended = false;
    bool selected_by_default = false;
    std::vector<std::string> support_urls;
};

struct DoctorSummary {
    std::vector<RecommendationOption> recommendations;
    std::vector<PrerequisiteOption> prerequisites;
    std::string recommended_backend_id;
    std::size_t ready_direct_count = 0u;
    std::size_t ready_modeled_count = 0u;
    std::size_t unavailable_count = 0u;
};

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

const BackendAggregate* find_backend_aggregate(
    const std::unordered_map<std::string, BackendAggregate>& aggregates,
    const std::string& name) {
    const auto it = aggregates.find(name);
    return it == aggregates.end() ? nullptr : &it->second;
}

bool backend_ready(const BackendAggregate* aggregate) {
    return aggregate != nullptr && (aggregate->ready_direct || aggregate->ready_modeled);
}

std::unordered_map<std::string, BackendAggregate> build_backend_aggregates(const jakal::Runtime& runtime) {
    std::unordered_map<std::string, BackendAggregate> aggregates;
    for (const auto& status : runtime.backend_statuses()) {
        auto& aggregate = aggregates[status.backend_name];
        aggregate.name = status.backend_name;
        aggregate.enabled = aggregate.enabled || status.enabled;
        aggregate.available = aggregate.available || status.available;
        aggregate.ready_direct = aggregate.ready_direct || status.code == jakal::RuntimeBackendStatusCode::ready_direct;
        aggregate.ready_modeled =
            aggregate.ready_modeled || status.code == jakal::RuntimeBackendStatusCode::ready_modeled;
        aggregate.has_disabled_entry =
            aggregate.has_disabled_entry || status.code == jakal::RuntimeBackendStatusCode::disabled;
    }
    return aggregates;
}

bool has_accelerator_vendor(const jakal::Runtime& runtime, const std::string& needle) {
    return std::any_of(runtime.devices().begin(), runtime.devices().end(), [&](const jakal::HardwareGraph& graph) {
        return graph.probe != "host" && lower_copy(graph.presentation_name).find(needle) != std::string::npos;
    });
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            escaped << "\\\\";
            break;
        case '"':
            escaped << "\\\"";
            break;
        case '\b':
            escaped << "\\b";
            break;
        case '\f':
            escaped << "\\f";
            break;
        case '\n':
            escaped << "\\n";
            break;
        case '\r':
            escaped << "\\r";
            break;
        case '\t':
            escaped << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20u) {
                escaped << "\\u"
                        << std::hex
                        << std::uppercase
                        << std::setw(4)
                        << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch))
                        << std::dec
                        << std::nouppercase
                        << std::setfill(' ');
            } else {
                escaped << ch;
            }
            break;
        }
    }
    return escaped.str();
}

void append_json_string(std::ostringstream& output, const std::string& value) {
    output << '"' << json_escape(value) << '"';
}

void append_json_string_array(std::ostringstream& output, const std::vector<std::string>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0u) {
            output << ',';
        }
        append_json_string(output, values[index]);
    }
    output << ']';
}

DoctorSummary summarize_doctor(const jakal::Runtime& runtime) {
    DoctorSummary summary;
    const auto aggregates = build_backend_aggregates(runtime);
    const auto* host = find_backend_aggregate(aggregates, "host");
    const auto* opencl = find_backend_aggregate(aggregates, "opencl");
    const auto* level_zero = find_backend_aggregate(aggregates, "level-zero");
    const auto* vulkan = find_backend_aggregate(aggregates, "vulkan");

    const bool host_ready = backend_ready(host);
    const bool opencl_ready = backend_ready(opencl);
    const bool level_zero_ready = backend_ready(level_zero);
    const bool vulkan_ready = backend_ready(vulkan);
    const bool opencl_ready_direct = opencl != nullptr && opencl->ready_direct;
    const bool level_zero_ready_direct = level_zero != nullptr && level_zero->ready_direct;
    const bool vulkan_ready_direct = vulkan != nullptr && vulkan->ready_direct;
    const bool has_any_accelerator =
        std::any_of(runtime.devices().begin(), runtime.devices().end(), [](const jakal::HardwareGraph& graph) {
            return graph.probe != "host";
        });
    const bool has_intel_accelerator = has_accelerator_vendor(runtime, "intel");

    for (const auto& status : runtime.backend_statuses()) {
        switch (status.code) {
        case jakal::RuntimeBackendStatusCode::ready_direct:
            ++summary.ready_direct_count;
            break;
        case jakal::RuntimeBackendStatusCode::ready_modeled:
            ++summary.ready_modeled_count;
            break;
        case jakal::RuntimeBackendStatusCode::unavailable:
        case jakal::RuntimeBackendStatusCode::no_devices:
            ++summary.unavailable_count;
            break;
        case jakal::RuntimeBackendStatusCode::disabled:
            break;
        }
    }

    summary.recommendations = {
        RecommendationOption{
            "auto",
            "Auto (recommended)",
            "Keep host and viable accelerators enabled so Jakal can choose per workload.",
            has_any_accelerator ? "Accelerator backends are present, so automatic placement is the best default."
                                : "No accelerator is currently usable, so auto will behave like host-only until drivers are added.",
            host_ready,
            false,
            {}},
        RecommendationOption{
            "cpu-only",
            "CPU only",
            "Run entirely on the host-native backend.",
            host_ready ? "Host execution is always available." : "Host probe is unavailable.",
            host_ready,
            false,
            {}},
        RecommendationOption{
            "intel-level-zero",
            "Intel GPU + Level Zero",
            "Prefer Intel GPU execution through Level Zero with host fallback.",
            level_zero_ready ? "Level Zero is ready on an Intel accelerator."
                             : "Intel accelerator detected but Level Zero is not ready yet.",
            has_intel_accelerator || level_zero_ready,
            false,
            {"intel-level-zero-runtime"}},
        RecommendationOption{
            "vulkan-runtime",
            "Vulkan runtime only",
            "Use the Vulkan backend path when direct driver support is present.",
            vulkan_ready_direct
                ? "Vulkan direct backend is ready on this machine."
                : (vulkan_ready
                       ? "Vulkan is only in modeled mode right now, so it is not the best primary accelerator choice."
                       : "Vulkan support is not ready yet and would need repair or driver install."),
            has_any_accelerator || vulkan_ready,
            false,
            {"vulkan-support"}},
        RecommendationOption{
            "opencl-fallback",
            "OpenCL fallback",
            "Use OpenCL as the accelerator fallback path with host backup.",
            opencl_ready_direct
                ? "OpenCL direct backend is ready and can act as the primary accelerator path."
                : (opencl_ready
                       ? "OpenCL is usable, but only as a softer compatibility path."
                       : "OpenCL is not ready yet and would need a vendor runtime."),
            has_any_accelerator || opencl_ready,
            false,
            {"opencl-runtime"}}};

    if (!has_any_accelerator || (!level_zero_ready && !opencl_ready && !vulkan_ready)) {
        summary.recommended_backend_id = "cpu-only";
    } else if (has_intel_accelerator && level_zero_ready_direct) {
        summary.recommended_backend_id = "intel-level-zero";
    } else if (opencl_ready_direct && (!vulkan_ready_direct || !level_zero_ready_direct)) {
        summary.recommended_backend_id = "opencl-fallback";
    } else if (vulkan_ready_direct && !opencl_ready_direct && !level_zero_ready_direct) {
        summary.recommended_backend_id = "vulkan-runtime";
    } else if (opencl_ready && !level_zero_ready) {
        summary.recommended_backend_id = "opencl-fallback";
    } else {
        summary.recommended_backend_id = "auto";
    }

    for (auto& recommendation : summary.recommendations) {
        recommendation.recommended = recommendation.id == summary.recommended_backend_id;
    }

    const bool vulkan_needs_repair = has_any_accelerator && !vulkan_ready_direct;
    const bool level_zero_needs_repair = has_intel_accelerator && !level_zero_ready;
    const bool opencl_needs_repair = has_any_accelerator && !opencl_ready;
    const bool skip_selected = !vulkan_needs_repair && !level_zero_needs_repair && !opencl_needs_repair;

    summary.prerequisites = {
        PrerequisiteOption{
            "vulkan-support",
            "Install/repair Vulkan support",
            "Repair or install GPU vendor Vulkan components used by Jakal's Vulkan backend.",
            vulkan_needs_repair
                ? "Vulkan is not currently ready for direct execution on this machine."
                : "Vulkan direct execution already looks usable.",
            true,
            vulkan_needs_repair,
            false,
            {
                "https://www.nvidia.com/Download/index.aspx",
                "https://www.intel.com/content/www/us/en/download-center/home.html",
                "https://www.amd.com/en/support/download/drivers.html"}} ,
        PrerequisiteOption{
            "intel-level-zero-runtime",
            "Install Intel Level Zero runtime",
            "Install or repair Intel GPU runtime support for Level Zero.",
            level_zero_needs_repair ? "Intel accelerator detected without a ready Level Zero path."
                                    : "Level Zero is already ready or no Intel accelerator was detected.",
            has_intel_accelerator || level_zero_ready,
            level_zero_needs_repair,
            false,
            {"https://www.intel.com/content/www/us/en/download-center/home.html"}} ,
        PrerequisiteOption{
            "opencl-runtime",
            "Install OpenCL runtime",
            "Install or repair a vendor OpenCL runtime as accelerator fallback.",
            opencl_needs_repair ? "OpenCL is not currently ready on this machine."
                                : "OpenCL is already ready or no accelerator was detected.",
            has_any_accelerator || opencl_ready,
            opencl_needs_repair && (!level_zero_needs_repair || !has_intel_accelerator),
            false,
            {
                "https://www.intel.com/content/www/us/en/download-center/home.html",
                "https://www.nvidia.com/Download/index.aspx",
                "https://www.amd.com/en/support/download/drivers.html"}} ,
        PrerequisiteOption{
            "skip-existing-drivers",
            "Skip and use existing drivers",
            "Keep the current machine drivers and install only the Jakal runtime.",
            skip_selected ? "Existing drivers already cover the recommended backend."
                          : "Useful when driver updates are managed outside the Jakal installer.",
            true,
            skip_selected,
            skip_selected,
            {}}};

    return summary;
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

void print_recommendations(const DoctorSummary& summary) {
    std::cout << "recommended_backend=" << summary.recommended_backend_id << '\n';
    for (const auto& recommendation : summary.recommendations) {
        std::cout << "recommendation=" << recommendation.id
                  << " available=" << (recommendation.available ? "yes" : "no")
                  << " recommended=" << (recommendation.recommended ? "yes" : "no")
                  << " label=" << recommendation.label
                  << " reason=" << recommendation.reason
                  << '\n';
    }
    for (const auto& prerequisite : summary.prerequisites) {
        std::cout << "prerequisite=" << prerequisite.id
                  << " available=" << (prerequisite.available ? "yes" : "no")
                  << " recommended=" << (prerequisite.recommended ? "yes" : "no")
                  << " default=" << (prerequisite.selected_by_default ? "yes" : "no")
                  << " label=" << prerequisite.label
                  << " reason=" << prerequisite.reason
                  << '\n';
    }
}

std::string build_doctor_json(const jakal::Runtime& runtime, const DoctorSummary& summary) {
    std::ostringstream output;
    const auto& paths = runtime.install_paths();
    output << '{';
    output << "\"schema_version\":1,";
    output << "\"recommended_backend_id\":";
    append_json_string(output, summary.recommended_backend_id);
    output << ",\"paths\":{";
    output << "\"install_root\":";
    append_json_string(output, paths.install_root.string());
    output << ",\"writable_root\":";
    append_json_string(output, paths.writable_root.string());
    output << ",\"config_dir\":";
    append_json_string(output, paths.config_dir.string());
    output << ",\"cache_dir\":";
    append_json_string(output, paths.cache_dir.string());
    output << ",\"logs_dir\":";
    append_json_string(output, paths.logs_dir.string());
    output << ",\"telemetry_path\":";
    append_json_string(output, paths.telemetry_path.string());
    output << ",\"planner_cache_path\":";
    append_json_string(output, paths.planner_cache_path.string());
    output << ",\"execution_cache_path\":";
    append_json_string(output, paths.execution_cache_path.string());
    output << ",\"python_dir\":";
    append_json_string(output, paths.python_dir.string());
    output << "},\"summary\":{";
    output << "\"device_count\":" << runtime.devices().size();
    output << ",\"ready_direct_count\":" << summary.ready_direct_count;
    output << ",\"ready_modeled_count\":" << summary.ready_modeled_count;
    output << ",\"unavailable_count\":" << summary.unavailable_count;
    output << "},\"backends\":[";
    for (std::size_t index = 0; index < runtime.backend_statuses().size(); ++index) {
        const auto& status = runtime.backend_statuses()[index];
        if (index > 0u) {
            output << ',';
        }
        output << '{';
        output << "\"backend_name\":";
        append_json_string(output, status.backend_name);
        output << ",\"code\":";
        append_json_string(output, jakal::to_string(status.code));
        output << ",\"enabled\":" << (status.enabled ? "true" : "false");
        output << ",\"available\":" << (status.available ? "true" : "false");
        output << ",\"direct_execution\":" << (status.direct_execution ? "true" : "false");
        output << ",\"modeled_fallback\":" << (status.modeled_fallback ? "true" : "false");
        output << ",\"device_uid\":";
        append_json_string(output, status.device_uid);
        output << ",\"detail\":";
        append_json_string(output, status.detail);
        output << '}';
    }
    output << "],\"devices\":[";
    for (std::size_t index = 0; index < runtime.devices().size(); ++index) {
        const auto& graph = runtime.devices()[index];
        const auto summary_graph = jakal::summarize_graph(graph);
        if (index > 0u) {
            output << ',';
        }
        output << '{';
        output << "\"uid\":";
        append_json_string(output, graph.uid);
        output << ",\"probe\":";
        append_json_string(output, graph.probe);
        output << ",\"name\":";
        append_json_string(output, graph.presentation_name);
        output << ",\"execution_objects\":" << summary_graph.execution_objects;
        output << ",\"native_vector_bits\":" << summary_graph.native_vector_bits;
        output << ",\"addressable_bytes\":" << summary_graph.addressable_bytes;
        output << '}';
    }
    output << "],\"recommendations\":[";
    for (std::size_t index = 0; index < summary.recommendations.size(); ++index) {
        const auto& recommendation = summary.recommendations[index];
        if (index > 0u) {
            output << ',';
        }
        output << '{';
        output << "\"id\":";
        append_json_string(output, recommendation.id);
        output << ",\"label\":";
        append_json_string(output, recommendation.label);
        output << ",\"description\":";
        append_json_string(output, recommendation.description);
        output << ",\"reason\":";
        append_json_string(output, recommendation.reason);
        output << ",\"available\":" << (recommendation.available ? "true" : "false");
        output << ",\"recommended\":" << (recommendation.recommended ? "true" : "false");
        output << ",\"prerequisite_ids\":";
        append_json_string_array(output, recommendation.prerequisite_ids);
        output << '}';
    }
    output << "],\"prerequisite_choices\":[";
    for (std::size_t index = 0; index < summary.prerequisites.size(); ++index) {
        const auto& prerequisite = summary.prerequisites[index];
        if (index > 0u) {
            output << ',';
        }
        output << '{';
        output << "\"id\":";
        append_json_string(output, prerequisite.id);
        output << ",\"label\":";
        append_json_string(output, prerequisite.label);
        output << ",\"description\":";
        append_json_string(output, prerequisite.description);
        output << ",\"reason\":";
        append_json_string(output, prerequisite.reason);
        output << ",\"available\":" << (prerequisite.available ? "true" : "false");
        output << ",\"recommended\":" << (prerequisite.recommended ? "true" : "false");
        output << ",\"selected_by_default\":" << (prerequisite.selected_by_default ? "true" : "false");
        output << ",\"support_urls\":";
        append_json_string_array(output, prerequisite.support_urls);
        output << '}';
    }
    output << "]}";
    return output.str();
}

int run_doctor(const CliOptions& cli) {
    jakal::Runtime runtime(make_options(cli));
    const auto summary = summarize_doctor(runtime);
    if (cli.json) {
        std::cout << build_doctor_json(runtime, summary) << '\n';
    } else {
        print_paths(runtime.install_paths());
        print_statuses(runtime);
        print_devices(runtime);
        print_recommendations(summary);
    }
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
