#include "jakal/c_api.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

std::filesystem::path unique_temp_file(const std::string& stem, const std::string& extension) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / (stem + "-" + std::to_string(nonce) + extension);
}

void write_manifest(const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::trunc);
    output << "[workload]\n";
    output << "name=install-smoke-manifest\n";
    output << "kind=inference\n";
    output << "dataset_tag=install-smoke-manifest-lite\n";
    output << "phase=decode\n";
    output << "working_set_bytes=8388608\n";
    output << "host_exchange_bytes=1048576\n";
    output << "estimated_flops=12000000\n";
    output << "batch_size=1\n";
    output << "latency_sensitive=true\n";
    output << "prefer_unified_memory=true\n\n";

    output << "[tensor]\n";
    output << "id=input\n";
    output << "bytes=4096\n";
    output << "host_visible=true\n";
    output << "consumers=normalize\n\n";

    output << "[tensor]\n";
    output << "id=hidden\n";
    output << "bytes=4096\n";
    output << "producer=normalize\n";
    output << "consumers=score\n\n";

    output << "[tensor]\n";
    output << "id=score-out\n";
    output << "bytes=4\n";
    output << "producer=score\n\n";

    output << "[operation]\n";
    output << "name=normalize\n";
    output << "class=elementwise_map\n";
    output << "extents=4096\n";
    output << "input_bytes=4096\n";
    output << "output_bytes=4096\n";
    output << "estimated_flops=8192\n";
    output << "parallelizable=true\n";
    output << "streaming_friendly=true\n";
    output << "inputs=input\n";
    output << "outputs=hidden\n\n";

    output << "[operation]\n";
    output << "name=score\n";
    output << "class=reduction\n";
    output << "extents=4096\n";
    output << "input_bytes=4096\n";
    output << "output_bytes=4\n";
    output << "estimated_flops=4096\n";
    output << "parallelizable=true\n";
    output << "reduction_like=true\n";
    output << "inputs=hidden\n";
    output << "outputs=score-out\n";
}

}  // namespace

int main() {
    if (jakal_core_c_api_abi_version() != JAKAL_CORE_C_API_ABI_VERSION ||
        jakal_core_runtime_telemetry_schema_version() != JAKAL_CORE_RUNTIME_TELEMETRY_SCHEMA_VERSION ||
        jakal_core_execution_performance_cache_schema_version() !=
            JAKAL_CORE_EXECUTION_PERFORMANCE_CACHE_SCHEMA_VERSION) {
        std::cerr << "runtime install smoke: c api version constants drifted\n";
        return 1;
    }

    jakal_core_runtime_options options{};
    options.enable_host_probe = 1;
    options.enable_opencl_probe = 0;
    options.enable_level_zero_probe = 0;
    options.enable_vulkan_probe = 0;
    options.enable_vulkan_status = 1;
    options.enable_cuda_probe = 0;
    options.enable_rocm_probe = 0;
    options.prefer_level_zero_over_opencl = 1;
    options.eager_hardware_refresh = 1;

    auto* runtime = jakal_core_runtime_create_with_options(&options);
    if (runtime == nullptr) {
        std::cerr << "runtime install smoke: failed to create runtime\n";
        return 1;
    }

    jakal_core_runtime_paths paths{};
    if (jakal_core_runtime_get_install_paths(runtime, &paths) != 0) {
        std::cerr << "runtime install smoke: failed to query install paths\n";
        jakal_core_runtime_destroy(runtime);
        return 1;
    }

    if (paths.cache_dir[0] == '\0' || paths.logs_dir[0] == '\0' || paths.execution_cache_path[0] == '\0') {
        std::cerr << "runtime install smoke: incomplete path policy\n";
        jakal_core_runtime_destroy(runtime);
        return 1;
    }

    const auto backend_count = jakal_core_runtime_backend_status_count(runtime);
    if (backend_count == 0u) {
        std::cerr << "runtime install smoke: missing backend statuses\n";
        jakal_core_runtime_destroy(runtime);
        return 1;
    }

    bool saw_host = false;
    bool saw_vulkan = false;
    for (size_t index = 0; index < backend_count; ++index) {
        jakal_core_backend_status_info status{};
        if (jakal_core_runtime_get_backend_status(runtime, index, &status) != 0) {
            std::cerr << "runtime install smoke: failed to read backend status\n";
            jakal_core_runtime_destroy(runtime);
            return 1;
        }
        saw_host = saw_host || std::string(status.backend_name) == "host";
        saw_vulkan = saw_vulkan || std::string(status.backend_name) == "vulkan";
    }
    if (!saw_host || !saw_vulkan) {
        std::cerr << "runtime install smoke: expected host and vulkan status entries\n";
        jakal_core_runtime_destroy(runtime);
        return 1;
    }

    jakal_core_workload_spec workload{};
    workload.name = "install-smoke";
    workload.kind = "tensor";
    workload.dataset_tag = "install-smoke-lite";
    workload.phase = "prefill";
    workload.shape_bucket = "b1-lite";
    workload.working_set_bytes = 32ull * 1024ull * 1024ull;
    workload.host_exchange_bytes = 4ull * 1024ull * 1024ull;
    workload.estimated_flops = 5.0e8;
    workload.batch_size = 1;
    workload.matrix_friendly = 1;

    jakal_core_optimization_info optimization{};
    jakal_core_operation_optimization_info ops[8]{};
    size_t count = 0;
    if (jakal_core_runtime_optimize(runtime, &workload, &optimization, ops, 8u, &count) != 0 || count == 0u) {
        std::cerr << "runtime install smoke: optimize failed\n";
        jakal_core_runtime_destroy(runtime);
        return 1;
    }

    const auto manifest_path = unique_temp_file("runtime-install-smoke", ".workload");
    write_manifest(manifest_path);
    jakal_core_execution_info execution{};
    jakal_core_execution_operation_info execution_ops[8]{};
    size_t execution_count = 0u;
    if (jakal_core_runtime_execute_manifest(
            runtime,
            manifest_path.string().c_str(),
            &execution,
            execution_ops,
            8u,
            &execution_count) != 0 ||
        execution_count == 0u) {
        std::cerr << "runtime install smoke: execute_manifest failed\n";
        std::filesystem::remove(manifest_path);
        jakal_core_runtime_destroy(runtime);
        return 1;
    }
    const auto binding_count = jakal_core_runtime_last_backend_buffer_binding_count(runtime);
    if (binding_count == 0u) {
        std::cerr << "runtime install smoke: missing backend buffer bindings\n";
        std::filesystem::remove(manifest_path);
        jakal_core_runtime_destroy(runtime);
        return 1;
    }
    jakal_core_backend_buffer_binding_info binding{};
    if (jakal_core_runtime_get_last_backend_buffer_binding(runtime, 0u, &binding) != 0) {
        std::cerr << "runtime install smoke: failed to fetch backend buffer binding\n";
        std::filesystem::remove(manifest_path);
        jakal_core_runtime_destroy(runtime);
        return 1;
    }
    const auto movement_count = jakal_core_runtime_last_residency_movement_count(runtime);
    if (movement_count > 0u) {
        jakal_core_residency_movement_info movement{};
        if (jakal_core_runtime_get_last_residency_movement(runtime, 0u, &movement) != 0) {
            std::cerr << "runtime install smoke: failed to fetch residency movement\n";
            std::filesystem::remove(manifest_path);
            jakal_core_runtime_destroy(runtime);
            return 1;
        }
    }

    std::filesystem::remove(manifest_path);
    jakal_core_runtime_destroy(runtime);
    std::cout << "runtime install smoke ok\n";
    return 0;
}
