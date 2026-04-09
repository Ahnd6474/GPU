#include "jakal/executors/direct_backends.hpp"
#include "jakal/runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

#if defined(_WIN32)
using LibraryHandle = HMODULE;

LibraryHandle load_library(const char* name) {
    return LoadLibraryA(name);
}

void close_library(LibraryHandle library) {
    if (library != nullptr) {
        FreeLibrary(library);
    }
}
#else
using LibraryHandle = void*;

LibraryHandle load_library(const char* name) {
    return dlopen(name, RTLD_LAZY);
}

void close_library(LibraryHandle library) {
    if (library != nullptr) {
        dlclose(library);
    }
}
#endif

struct InstallLayout {
    std::filesystem::path executable_path;
    std::filesystem::path install_root;
    std::filesystem::path bin_dir;
    std::filesystem::path share_dir;
    std::filesystem::path docs_dir;
    std::filesystem::path update_dir;
    std::filesystem::path remove_dir;
    std::filesystem::path state_dir;
    std::filesystem::path self_check_marker;
};

struct BackendSupportRow {
    std::string backend;
    bool configured = true;
    bool detected = false;
    bool usable = false;
    std::string detail;
};

bool library_present(const std::initializer_list<const char*> candidates) {
    for (const auto* candidate : candidates) {
        auto library = load_library(candidate);
        if (library != nullptr) {
            close_library(library);
            return true;
        }
    }
    return false;
}

std::vector<std::filesystem::path> split_search_path(const char* value) {
    std::vector<std::filesystem::path> paths;
    if (value == nullptr || *value == '\0') {
        return paths;
    }

#if defined(_WIN32)
    constexpr char separator = ';';
#else
    constexpr char separator = ':';
#endif

    std::stringstream stream(value);
    std::string field;
    while (std::getline(stream, field, separator)) {
        if (!field.empty()) {
            paths.emplace_back(field);
        }
    }
    return paths;
}

std::optional<std::filesystem::path> find_file_with_prefixes(
    const std::vector<std::filesystem::path>& roots,
    const std::initializer_list<std::string_view> prefixes) {
    for (const auto& root : roots) {
        std::error_code ec;
        if (root.empty() || !std::filesystem::exists(root, ec) || !std::filesystem::is_directory(root, ec)) {
            continue;
        }
        for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
            if (ec || !entry.is_regular_file()) {
                continue;
            }
            const auto filename = entry.path().filename().string();
            for (const auto prefix : prefixes) {
                if (filename.rfind(prefix.data(), 0u) == 0u) {
                    return entry.path();
                }
            }
        }
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> find_program_in_roots(
    const std::vector<std::filesystem::path>& roots,
    const std::initializer_list<std::string_view> names) {
    for (const auto& root : roots) {
        for (const auto name : names) {
            std::error_code ec;
            const auto candidate = root / std::string(name);
            if (std::filesystem::exists(candidate, ec)) {
                return candidate;
            }
        }
    }
    return std::nullopt;
}

bool command_exists(const std::string& command) {
#if defined(_WIN32)
    const std::string probe = "where \"" + command + "\" >nul 2>nul";
#else
    const std::string probe = "command -v \"" + command + "\" >/dev/null 2>&1";
#endif
    return std::system(probe.c_str()) == 0;
}

bool is_build_config_dir(const std::filesystem::path& path) {
    const auto name = path.filename().string();
    return name == "Debug" || name == "Release" || name == "RelWithDebInfo" || name == "MinSizeRel";
}

std::filesystem::path resolve_executable_path(const char* arg0) {
    std::error_code ec;
    auto path = std::filesystem::absolute(arg0 == nullptr ? std::filesystem::path() : std::filesystem::path(arg0), ec);
    if (ec) {
        path = std::filesystem::current_path(ec);
    }
    const auto canonical = std::filesystem::weakly_canonical(path, ec);
    return ec ? path : canonical;
}

std::filesystem::path user_state_root() {
#if defined(_WIN32)
    if (const char* local_app_data = std::getenv("LOCALAPPDATA")) {
        return std::filesystem::path(local_app_data) / "Jakal-Core";
    }
#else
    if (const char* xdg_state_home = std::getenv("XDG_STATE_HOME")) {
        return std::filesystem::path(xdg_state_home) / "jakal-core";
    }
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".local" / "state" / "jakal-core";
    }
#endif
    return std::filesystem::temp_directory_path() / "jakal-core";
}

InstallLayout resolve_install_layout(const char* argv0) {
    InstallLayout layout;
    layout.executable_path = resolve_executable_path(argv0);
    auto executable_dir = layout.executable_path.parent_path();
    layout.install_root = executable_dir;
    if (is_build_config_dir(executable_dir)) {
        layout.install_root = executable_dir.parent_path();
        layout.bin_dir = executable_dir;
    } else if (executable_dir.filename() == "bin") {
        layout.install_root = executable_dir.parent_path();
        layout.bin_dir = executable_dir;
    } else {
        layout.bin_dir = executable_dir;
    }

    layout.share_dir = layout.install_root / "share" / "jakal-core";
    layout.docs_dir = layout.install_root / "share" / "doc" / "JakalCore";
    layout.update_dir = layout.share_dir / "update";
    layout.remove_dir = layout.share_dir / "remove";
    layout.state_dir = user_state_root() / "state";
    layout.self_check_marker = layout.state_dir / "bootstrap-self-check.txt";
    return layout;
}

std::string yes_no(const bool value) {
    return value ? "yes" : "no";
}

std::string now_string() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &time);
#else
    localtime_r(&time, &local_time);
#endif
    std::ostringstream stream;
    stream << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S");
    return stream.str();
}

std::string supported_ops_for(const jakal::HardwareGraph& graph) {
    constexpr std::array<jakal::OperationClass, 5> kOps = {
        jakal::OperationClass::elementwise_map,
        jakal::OperationClass::reduction,
        jakal::OperationClass::matmul,
        jakal::OperationClass::convolution_2d,
        jakal::OperationClass::resample_2d};

    std::vector<std::string> supported;
    for (const auto op : kOps) {
        std::string reason;
        if (jakal::runtime_backend_supports_operation(graph, op, &reason)) {
            supported.push_back(jakal::to_string(op));
        }
    }

    std::ostringstream stream;
    for (std::size_t index = 0; index < supported.size(); ++index) {
        if (index > 0u) {
            stream << ", ";
        }
        stream << supported[index];
    }
    return stream.str();
}

BackendSupportRow make_host_row() {
    return BackendSupportRow{"host", true, true, true, "built into jakal_core"};
}

BackendSupportRow make_opencl_row(const jakal::RuntimeOptions& options) {
    const bool detected =
#if defined(_WIN32)
        library_present({"OpenCL.dll"});
#elif defined(__APPLE__)
        library_present({"/System/Library/Frameworks/OpenCL.framework/OpenCL", "libOpenCL.dylib"});
#else
        library_present({"libOpenCL.so", "libOpenCL.so.1"});
#endif
    return BackendSupportRow{"opencl", options.enable_opencl_probe, detected, options.enable_opencl_probe && detected, detected ? "OpenCL loader detected" : "OpenCL loader missing"};
}

BackendSupportRow make_level_zero_row(const jakal::RuntimeOptions& options) {
    const bool detected =
#if defined(_WIN32)
        library_present({"ze_loader.dll"});
#else
        library_present({"libze_loader.so", "libze_loader.so.1"});
#endif
    return BackendSupportRow{"level-zero", options.enable_level_zero_probe, detected, options.enable_level_zero_probe && detected, detected ? "Level Zero loader detected" : "Level Zero loader missing"};
}

BackendSupportRow make_cuda_row(const jakal::RuntimeOptions& options) {
    const bool driver_detected =
#if defined(_WIN32)
        library_present({"nvcuda.dll"});
#else
        library_present({"libcuda.so", "libcuda.so.1"});
#endif

    std::vector<std::filesystem::path> search_roots;
    if (const char* cuda_path = std::getenv("CUDA_PATH")) {
        search_roots.emplace_back(std::filesystem::path(cuda_path) / "bin");
    }
    const auto nvrtc = find_file_with_prefixes(search_roots, {"nvrtc64_", "nvrtc"});

    BackendSupportRow row{"cuda", options.enable_cuda_probe, driver_detected && nvrtc.has_value(), options.enable_cuda_probe && driver_detected && nvrtc.has_value(), ""};
    if (!driver_detected) {
        row.detail = "NVIDIA driver missing";
    } else if (!nvrtc.has_value()) {
        row.detail = "NVRTC missing";
    } else {
        row.detail = "NVIDIA driver and NVRTC detected";
    }
    return row;
}

BackendSupportRow make_rocm_row(const jakal::RuntimeOptions& options) {
    const bool runtime_detected =
#if defined(_WIN32)
        library_present({"amdhip64.dll"});
#else
        library_present({"libamdhip64.so", "libamdhip64.so.6"});
#endif

    std::vector<std::filesystem::path> search_roots;
    if (const char* rocm_path = std::getenv("ROCM_PATH")) {
        search_roots.emplace_back(std::filesystem::path(rocm_path) / "bin");
        search_roots.emplace_back(std::filesystem::path(rocm_path) / "lib");
    }
    if (const char* hip_path = std::getenv("HIP_PATH")) {
        search_roots.emplace_back(std::filesystem::path(hip_path) / "bin");
        search_roots.emplace_back(std::filesystem::path(hip_path) / "lib");
    }
    const auto hiprtc = find_file_with_prefixes(search_roots, {"hiprtc", "libhiprtc"});

    BackendSupportRow row{"rocm", options.enable_rocm_probe, runtime_detected && hiprtc.has_value(), options.enable_rocm_probe && runtime_detected && hiprtc.has_value(), ""};
    if (!runtime_detected) {
        row.detail = "HIP runtime missing";
    } else if (!hiprtc.has_value()) {
        row.detail = "HIPRTC missing";
    } else {
        row.detail = "HIP runtime and HIPRTC detected";
    }
    return row;
}

BackendSupportRow make_vulkan_row() {
    const bool loader_detected =
#if defined(_WIN32)
        library_present({"vulkan-1.dll"});
#else
        library_present({"libvulkan.so", "libvulkan.so.1"});
#endif

    std::vector<std::filesystem::path> compiler_roots;
    if (const char* vulkan_sdk = std::getenv("VULKAN_SDK")) {
#if defined(_WIN32)
        compiler_roots.emplace_back(std::filesystem::path(vulkan_sdk) / "Bin");
#else
        compiler_roots.emplace_back(std::filesystem::path(vulkan_sdk) / "bin");
#endif
    }
    const auto compiler = find_program_in_roots(
                              compiler_roots,
                              {"glslangValidator.exe", "glslc.exe", "glslangValidator", "glslc"})
                              .has_value() ||
                          command_exists("glslangValidator") ||
                          command_exists("glslc");
    const bool direct_available = jakal::executors::vulkan_direct_backend_available();

    BackendSupportRow row{"vulkan-direct", true, loader_detected && compiler, direct_available, ""};
    if (!loader_detected) {
        row.detail = "Vulkan loader missing";
    } else if (!compiler) {
        row.detail = "Shader compiler missing";
    } else if (!direct_available) {
        row.detail = "Loader/compiler found but direct backend not active";
    } else {
        row.detail = "Vulkan loader and shader compiler detected";
    }
    return row;
}

std::vector<BackendSupportRow> collect_supported_backends(const jakal::RuntimeOptions& options) {
    std::vector<BackendSupportRow> rows;
    rows.push_back(make_host_row());
    rows.push_back(make_opencl_row(options));
    rows.push_back(make_level_zero_row(options));
    rows.push_back(make_cuda_row(options));
    rows.push_back(make_rocm_row(options));
    rows.push_back(make_vulkan_row());
    return rows;
}

void print_install_layout(const InstallLayout& layout) {
    std::cout << "Install Layout\n";
    std::cout << "  root:    " << layout.install_root.string() << '\n';
    std::cout << "  bin:     " << layout.bin_dir.string() << '\n';
    std::cout << "  docs:    " << layout.docs_dir.string() << '\n';
    std::cout << "  update:  " << layout.update_dir.string() << '\n';
    std::cout << "  remove:  " << layout.remove_dir.string() << '\n';
    std::cout << "  state:   " << layout.state_dir.string() << '\n';
}

void print_supported_backends(const std::vector<BackendSupportRow>& rows) {
    std::cout << "\nSupported Backends\n";
    for (const auto& row : rows) {
        std::cout << "  " << std::setw(14) << std::left << row.backend
                  << " configured=" << std::setw(3) << yes_no(row.configured)
                  << " detected=" << std::setw(3) << yes_no(row.detected)
                  << " usable=" << std::setw(3) << yes_no(row.usable)
                  << " detail=" << row.detail << '\n';
    }
}

void print_active_backends(const jakal::Runtime& runtime) {
    std::cout << "\nActive Backends\n";
    if (runtime.devices().empty()) {
        std::cout << "  no hardware graphs discovered\n";
        return;
    }

    for (const auto& graph : runtime.devices()) {
        const auto summary = jakal::summarize_graph(graph);
        std::cout << "  " << graph.presentation_name
                  << " | probe=" << graph.probe
                  << " | backend=" << jakal::runtime_backend_name_for_graph(graph)
                  << " | driver=" << (graph.driver_version.empty() ? "n/a" : graph.driver_version)
                  << " | runtime=" << (graph.runtime_version.empty() ? "n/a" : graph.runtime_version)
                  << " | async=" << yes_no(summary.supports_asynchronous_dispatch)
                  << '\n';
        std::cout << "    supported_ops=" << supported_ops_for(graph) << '\n';
    }
}

void print_toolkit_variants(const jakal::Runtime& runtime) {
    std::cout << "\nToolkit Variants\n";
    if (runtime.jakal_toolkit_index().empty()) {
        std::cout << "  no toolkit variants available\n";
        return;
    }

    for (const auto& entry : runtime.jakal_toolkit_index()) {
        std::cout << "  device=" << entry.device_uid << '\n';
        for (const auto& variant : entry.variants) {
            std::cout << "    " << std::setw(18) << std::left << jakal::to_string(variant.binding.backend)
                      << " executable=" << yes_no(variant.executable)
                      << " score=" << std::fixed << std::setprecision(3) << variant.toolkit_score
                      << " rationale=" << variant.rationale << '\n';
        }
    }
}

std::string read_marker_summary(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::string line;
    std::getline(input, line);
    return line;
}

bool write_marker_summary(const std::filesystem::path& path, const std::string& summary) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open()) {
        return false;
    }
    output << summary << '\n';
    return output.good();
}

bool perform_self_check(
    const InstallLayout& layout,
    const jakal::Runtime& runtime,
    const bool persist_marker,
    const bool force,
    std::string& summary) {
    if (!force && std::filesystem::exists(layout.self_check_marker)) {
        summary = read_marker_summary(layout.self_check_marker);
        return true;
    }

    const bool has_host = std::any_of(runtime.devices().begin(), runtime.devices().end(), [](const jakal::HardwareGraph& graph) {
        return graph.probe == "host";
    });
    const bool ok = has_host && !runtime.devices().empty();

    std::set<std::string> active_backends;
    for (const auto& graph : runtime.devices()) {
        active_backends.insert(jakal::runtime_backend_name_for_graph(graph));
    }

    std::ostringstream stream;
    stream << now_string()
           << " | result=" << (ok ? "pass" : "fail")
           << " | devices=" << runtime.devices().size()
           << " | host=" << yes_no(has_host)
           << " | active=";
    bool first = true;
    for (const auto& backend : active_backends) {
        if (!first) {
            stream << ',';
        }
        first = false;
        stream << backend;
    }
    summary = stream.str();

    if (persist_marker) {
        (void)write_marker_summary(layout.self_check_marker, summary);
    }
    return ok;
}

void print_help() {
    std::cout
        << "jakal_bootstrap [--status] [--self-check] [--force-self-check] [--no-persist]\n"
        << "                [--disable-opencl] [--disable-level-zero] [--disable-cuda] [--disable-rocm]\n";
}

}  // namespace

int main(int argc, char** argv) {
    bool show_status = false;
    bool run_self_check = false;
    bool force_self_check = false;
    bool persist_marker = true;

    jakal::RuntimeOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--status") {
            show_status = true;
        } else if (arg == "--self-check") {
            run_self_check = true;
        } else if (arg == "--force-self-check") {
            run_self_check = true;
            force_self_check = true;
        } else if (arg == "--no-persist") {
            persist_marker = false;
        } else if (arg == "--disable-opencl") {
            options.enable_opencl_probe = false;
        } else if (arg == "--disable-level-zero") {
            options.enable_level_zero_probe = false;
        } else if (arg == "--disable-cuda") {
            options.enable_cuda_probe = false;
        } else if (arg == "--disable-rocm") {
            options.enable_rocm_probe = false;
        } else if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << '\n';
            print_help();
            return 1;
        }
    }

    const auto layout = resolve_install_layout(argc > 0 ? argv[0] : nullptr);
    jakal::Runtime runtime(options);
    const auto supported_backends = collect_supported_backends(options);

    const bool marker_exists = std::filesystem::exists(layout.self_check_marker);
    if (!show_status && !run_self_check) {
        show_status = true;
        run_self_check = !marker_exists;
    }

    if (show_status) {
        print_install_layout(layout);
        print_supported_backends(supported_backends);
        print_active_backends(runtime);
        print_toolkit_variants(runtime);
    }

    if (run_self_check || marker_exists) {
        std::string summary;
        const bool ok = perform_self_check(layout, runtime, persist_marker, force_self_check, summary);
        std::cout << "\nSelf Check\n";
        std::cout << "  marker: " << layout.self_check_marker.string() << '\n';
        std::cout << "  status: " << summary << '\n';
        if (run_self_check && !ok) {
            return 2;
        }
    }

    return 0;
}
