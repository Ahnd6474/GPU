#include "jakal/c_api.h"

#include <iostream>
#include <string>

int main() {
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

    jakal_core_runtime_destroy(runtime);
    std::cout << "runtime install smoke ok\n";
    return 0;
}
