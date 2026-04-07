#include "jakal/jakal_l0.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace jakal {
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

bool library_present(const std::vector<const char*>& candidates) {
    for (const auto* candidate : candidates) {
        auto library = load_library(candidate);
        if (library != nullptr) {
            close_library(library);
            return true;
        }
    }
    return false;
}

bool contains_case_insensitive(const std::string& text, const std::string_view needle) {
    return std::search(
               text.begin(),
               text.end(),
               needle.begin(),
               needle.end(),
               [](const char left, const char right) {
                   const auto lower_left = static_cast<unsigned char>(std::tolower(static_cast<unsigned char>(left)));
                   const auto lower_right = static_cast<unsigned char>(std::tolower(static_cast<unsigned char>(right)));
                   return lower_left == lower_right;
               }) != text.end();
}

bool looks_like_gpu_graph(const HardwareGraph& graph) {
    if (graph.probe == "host") {
        return false;
    }

    const auto summary = summarize_graph(graph);
    return summary.execution_objects > 0 &&
           summary.lanes_per_object > 0 &&
           summary.addressable_bytes > (256ull * 1024ull * 1024ull);
}

JakalL0WorkloadTraits default_traits(const HardwareGraphSummary& summary) {
    JakalL0WorkloadTraits traits;
    traits.bytes = summary.addressable_bytes;
    traits.matrix_friendly = summary.matrix_units > 0;
    traits.streaming_friendly = summary.host_read_gbps >= 16.0;
    return traits;
}

double hardware_weight(const HardwareGraphSummary& summary) {
    const double execution = static_cast<double>(summary.execution_objects) *
                             std::sqrt(static_cast<double>(std::max(summary.lanes_per_object, 1u)));
    const double memory_gib = static_cast<double>(summary.addressable_bytes) / (1024.0 * 1024.0 * 1024.0);
    const double host_link = summary.host_read_gbps + summary.host_write_gbps;
    return execution + std::log2(memory_gib + 2.0) + (host_link * 0.05);
}

JakalVendorFamily infer_vendor_from_graph(const HardwareGraph& graph) {
    if (graph.probe == "level-zero") {
        return JakalVendorFamily::intel;
    }
    if (graph.probe == "cuda") {
        return JakalVendorFamily::nvidia;
    }
    if (graph.probe == "rocm") {
        return JakalVendorFamily::amd;
    }
    if (contains_case_insensitive(graph.presentation_name, "intel") ||
        contains_case_insensitive(graph.presentation_name, "iris") ||
        contains_case_insensitive(graph.presentation_name, "arc")) {
        return JakalVendorFamily::intel;
    }
    if (contains_case_insensitive(graph.presentation_name, "nvidia") ||
        contains_case_insensitive(graph.presentation_name, "geforce") ||
        contains_case_insensitive(graph.presentation_name, "rtx") ||
        contains_case_insensitive(graph.presentation_name, "gtx")) {
        return JakalVendorFamily::nvidia;
    }
    if (contains_case_insensitive(graph.presentation_name, "amd") ||
        contains_case_insensitive(graph.presentation_name, "radeon") ||
        contains_case_insensitive(graph.presentation_name, "ryzen ai")) {
        return JakalVendorFamily::amd;
    }
    return JakalVendorFamily::unknown;
}

class OpenClJakalL0Adapter final : public IJakalL0Adapter {
public:
    [[nodiscard]] std::string id() const override { return "gpu-l0.opencl"; }
    [[nodiscard]] JakalVendorFamily vendor() const override { return JakalVendorFamily::unknown; }
    [[nodiscard]] JakalBackendKind backend_kind() const override { return JakalBackendKind::opencl; }
    [[nodiscard]] bool available() const override {
#if defined(_WIN32)
        return library_present({"OpenCL.dll"});
#elif defined(__APPLE__)
        return library_present({"/System/Library/Frameworks/OpenCL.framework/OpenCL", "libOpenCL.dylib"});
#else
        return library_present({"libOpenCL.so", "libOpenCL.so.1"});
#endif
    }

    [[nodiscard]] bool matches(const HardwareGraph& graph) const override {
        return graph.probe == "opencl" || looks_like_gpu_graph(graph);
    }

    [[nodiscard]] JakalL0Binding describe(const HardwareGraph& graph) const override {
        const auto summary = summarize_graph(graph);
        JakalL0Binding binding;
        binding.device_uid = graph.uid;
        binding.graph_fingerprint = structural_fingerprint(graph);
        binding.adapter_id = id();
        binding.presentation_name = graph.presentation_name;
        binding.vendor = infer_vendor_from_graph(graph);
        binding.backend = backend_kind();
        binding.capabilities.adapter_available = available();
        binding.capabilities.direct_command_submission = false;
        binding.capabilities.explicit_memory_import = summary.unified_address_space;
        binding.capabilities.timeline_synchronization = summary.supports_asynchronous_dispatch;
        binding.capabilities.binary_module_loading = true;
        binding.capabilities.kernel_specialization = true;
        binding.capabilities.subgroup_matrix = summary.matrix_units > 0 || summary.supports_fp16;
        binding.capabilities.unified_memory = summary.unified_address_space;
        binding.capabilities.persistent_kernel_cache = true;
        binding.capabilities.asynchronous_dispatch = summary.supports_asynchronous_dispatch;
        binding.capabilities.preferred_queue_batch = summary.supports_asynchronous_dispatch ? 4u : 2u;
        binding.capabilities.preferred_workgroup = std::clamp(summary.lanes_per_object * 4u, 32u, 256u);
        binding.capabilities.estimated_launch_latency_us = std::max(summary.dispatch_latency_us, 8.0);
        binding.capabilities.estimated_host_read_gbps = summary.host_read_gbps;
        binding.capabilities.estimated_host_write_gbps = summary.host_write_gbps;

        double score = 0.70;
        score += 0.02 * std::log2(hardware_weight(summary) + 2.0);
        score += summary.supports_fp16 ? 0.04 : 0.0;
        score += summary.unified_address_space ? 0.03 : 0.0;
        score += summary.supports_asynchronous_dispatch ? 0.03 : 0.0;
        if (!binding.capabilities.adapter_available) {
            score -= 0.35;
        }
        binding.suitability_score = score;
        return binding;
    }

    [[nodiscard]] JakalL0LaunchTuning suggest_tuning(
        const HardwareGraph& graph,
        const JakalL0WorkloadTraits& workload) const override {
        const auto summary = summarize_graph(graph);
        const auto traits = workload.bytes == 0 ? default_traits(summary) : workload;
        JakalL0LaunchTuning tuning;
        tuning.workgroup_x = std::clamp(summary.lanes_per_object * 4u, 32u, 256u);
        tuning.workgroup_y = traits.streaming_friendly ? 2u : 1u;
        tuning.workgroup_z = 1u;
        tuning.queue_batch = summary.supports_asynchronous_dispatch ? 4u : 2u;
        tuning.staging_window = summary.unified_address_space ? 1u : 2u;
        tuning.prefer_vectorized_io = summary.native_vector_bits >= 128u;
        tuning.prefer_persistent_modules = true;
        tuning.prefer_shared_host_staging = summary.unified_address_space;
        return tuning;
    }
};

class LevelZeroJakalL0Adapter final : public IJakalL0Adapter {
public:
    [[nodiscard]] std::string id() const override { return "gpu-l0.level-zero"; }
    [[nodiscard]] JakalVendorFamily vendor() const override { return JakalVendorFamily::intel; }
    [[nodiscard]] JakalBackendKind backend_kind() const override { return JakalBackendKind::level_zero; }
    [[nodiscard]] bool available() const override {
#if defined(_WIN32)
        return library_present({"ze_loader.dll"});
#else
        return library_present({"libze_loader.so", "libze_loader.so.1"});
#endif
    }

    [[nodiscard]] bool matches(const HardwareGraph& graph) const override {
        if (!looks_like_gpu_graph(graph)) {
            return false;
        }
        if (graph.probe == "level-zero") {
            return true;
        }
        return contains_case_insensitive(graph.presentation_name, "intel") ||
               contains_case_insensitive(graph.presentation_name, "iris") ||
               contains_case_insensitive(graph.presentation_name, "arc") ||
               graph.probe == "opencl";
    }

    [[nodiscard]] JakalL0Binding describe(const HardwareGraph& graph) const override {
        const auto summary = summarize_graph(graph);
        JakalL0Binding binding;
        binding.device_uid = graph.uid;
        binding.graph_fingerprint = structural_fingerprint(graph);
        binding.adapter_id = id();
        binding.presentation_name = graph.presentation_name;
        binding.vendor = vendor();
        binding.backend = backend_kind();
        binding.capabilities.adapter_available = available();
        binding.capabilities.direct_command_submission = true;
        binding.capabilities.explicit_memory_import = true;
        binding.capabilities.timeline_synchronization = true;
        binding.capabilities.binary_module_loading = true;
        binding.capabilities.kernel_specialization = true;
        binding.capabilities.subgroup_matrix = summary.matrix_units > 0 || summary.supports_int8 || summary.supports_fp16;
        binding.capabilities.unified_memory = summary.unified_address_space;
        binding.capabilities.persistent_kernel_cache = true;
        binding.capabilities.asynchronous_dispatch = true;
        binding.capabilities.preferred_queue_batch = summary.supports_asynchronous_dispatch ? 6u : 4u;
        binding.capabilities.preferred_workgroup = std::clamp(summary.lanes_per_object * 8u, 32u, 512u);
        binding.capabilities.estimated_launch_latency_us = std::max(2.0, summary.dispatch_latency_us * 0.55);
        binding.capabilities.estimated_host_read_gbps = std::max(summary.host_read_gbps, 24.0);
        binding.capabilities.estimated_host_write_gbps = std::max(summary.host_write_gbps, 24.0);

        double score = 0.84;
        score += 0.03 * std::log2(hardware_weight(summary) + 2.0);
        score += summary.supports_fp16 ? 0.05 : 0.0;
        score += summary.supports_int8 ? 0.03 : 0.0;
        score += summary.unified_address_space ? 0.03 : 0.0;
        if (!binding.capabilities.adapter_available) {
            score -= 0.42;
        }
        binding.suitability_score = score;
        return binding;
    }

    [[nodiscard]] JakalL0LaunchTuning suggest_tuning(
        const HardwareGraph& graph,
        const JakalL0WorkloadTraits& workload) const override {
        const auto summary = summarize_graph(graph);
        const auto traits = workload.bytes == 0 ? default_traits(summary) : workload;
        JakalL0LaunchTuning tuning;
        tuning.workgroup_x = std::clamp(summary.lanes_per_object * 8u, 32u, 512u);
        tuning.workgroup_y = traits.matrix_friendly ? 2u : 1u;
        tuning.workgroup_z = 1u;
        tuning.queue_batch = summary.supports_asynchronous_dispatch ? 6u : 4u;
        tuning.staging_window = summary.unified_address_space ? 1u : 2u;
        tuning.prefer_vectorized_io = summary.native_vector_bits >= 128u;
        tuning.prefer_persistent_modules = true;
        tuning.prefer_shared_host_staging = summary.unified_address_space;
        return tuning;
    }
};

class VulkanJakalL0Adapter final : public IJakalL0Adapter {
public:
    [[nodiscard]] std::string id() const override { return "gpu-l0.vulkan"; }
    [[nodiscard]] JakalVendorFamily vendor() const override { return JakalVendorFamily::unknown; }
    [[nodiscard]] JakalBackendKind backend_kind() const override { return JakalBackendKind::vulkan_compute; }
    [[nodiscard]] bool available() const override {
#if defined(_WIN32)
        return library_present({"vulkan-1.dll"});
#else
        return library_present({"libvulkan.so", "libvulkan.so.1"});
#endif
    }

    [[nodiscard]] bool matches(const HardwareGraph& graph) const override {
        return looks_like_gpu_graph(graph);
    }

    [[nodiscard]] JakalL0Binding describe(const HardwareGraph& graph) const override {
        const auto summary = summarize_graph(graph);
        JakalL0Binding binding;
        binding.device_uid = graph.uid;
        binding.graph_fingerprint = structural_fingerprint(graph);
        binding.adapter_id = id();
        binding.presentation_name = graph.presentation_name;
        binding.vendor = infer_vendor_from_graph(graph);
        binding.backend = backend_kind();
        binding.capabilities.adapter_available = available();
        binding.capabilities.direct_command_submission = true;
        binding.capabilities.explicit_memory_import = true;
        binding.capabilities.timeline_synchronization = true;
        binding.capabilities.binary_module_loading = true;
        binding.capabilities.kernel_specialization = true;
        binding.capabilities.subgroup_matrix = false;
        binding.capabilities.unified_memory = summary.unified_address_space;
        binding.capabilities.persistent_kernel_cache = true;
        binding.capabilities.asynchronous_dispatch = true;
        binding.capabilities.preferred_queue_batch = 4u;
        binding.capabilities.preferred_workgroup = std::clamp(summary.lanes_per_object * 4u, 32u, 256u);
        binding.capabilities.estimated_launch_latency_us = std::max(4.0, summary.dispatch_latency_us * 0.75);
        binding.capabilities.estimated_host_read_gbps = summary.host_read_gbps;
        binding.capabilities.estimated_host_write_gbps = summary.host_write_gbps;

        double score = 0.68;
        score += summary.supports_asynchronous_dispatch ? 0.05 : 0.0;
        score += summary.unified_address_space ? 0.02 : 0.0;
        if (!binding.capabilities.adapter_available) {
            score -= 0.36;
        }
        binding.suitability_score = score;
        return binding;
    }

    [[nodiscard]] JakalL0LaunchTuning suggest_tuning(
        const HardwareGraph& graph,
        const JakalL0WorkloadTraits& workload) const override {
        const auto summary = summarize_graph(graph);
        JakalL0LaunchTuning tuning;
        tuning.workgroup_x = std::clamp(summary.lanes_per_object * 4u, 32u, 256u);
        tuning.workgroup_y = workload.op_class == OperationClass::resample_2d ? 4u : 1u;
        tuning.workgroup_z = 1u;
        tuning.queue_batch = workload.latency_sensitive ? 2u : 4u;
        tuning.staging_window = summary.unified_address_space ? 1u : 2u;
        tuning.prefer_vectorized_io = true;
        tuning.prefer_persistent_modules = true;
        tuning.prefer_shared_host_staging = summary.unified_address_space;
        return tuning;
    }
};

class CudaJakalL0Adapter final : public IJakalL0Adapter {
public:
    [[nodiscard]] std::string id() const override { return "gpu-l0.cuda"; }
    [[nodiscard]] JakalVendorFamily vendor() const override { return JakalVendorFamily::nvidia; }
    [[nodiscard]] JakalBackendKind backend_kind() const override { return JakalBackendKind::cuda; }
    [[nodiscard]] bool available() const override {
#if defined(_WIN32)
        return library_present({"nvcuda.dll"});
#else
        return library_present({"libcuda.so", "libcuda.so.1"});
#endif
    }

    [[nodiscard]] bool matches(const HardwareGraph& graph) const override {
        if (!looks_like_gpu_graph(graph)) {
            return false;
        }
        if (graph.probe == "cuda") {
            return true;
        }
        return contains_case_insensitive(graph.presentation_name, "nvidia") ||
               contains_case_insensitive(graph.presentation_name, "geforce") ||
               contains_case_insensitive(graph.presentation_name, "rtx") ||
               contains_case_insensitive(graph.presentation_name, "cuda");
    }

    [[nodiscard]] JakalL0Binding describe(const HardwareGraph& graph) const override {
        const auto summary = summarize_graph(graph);
        JakalL0Binding binding;
        binding.device_uid = graph.uid;
        binding.graph_fingerprint = structural_fingerprint(graph);
        binding.adapter_id = id();
        binding.presentation_name = graph.presentation_name;
        binding.vendor = vendor();
        binding.backend = backend_kind();
        binding.capabilities.adapter_available = available();
        binding.capabilities.direct_command_submission = true;
        binding.capabilities.explicit_memory_import = true;
        binding.capabilities.timeline_synchronization = true;
        binding.capabilities.binary_module_loading = true;
        binding.capabilities.kernel_specialization = true;
        binding.capabilities.subgroup_matrix = summary.matrix_units > 0 || summary.supports_fp16 || summary.supports_int8;
        binding.capabilities.unified_memory = summary.unified_address_space;
        binding.capabilities.persistent_kernel_cache = true;
        binding.capabilities.asynchronous_dispatch = true;
        binding.capabilities.preferred_queue_batch = 6u;
        binding.capabilities.preferred_workgroup = std::clamp(summary.lanes_per_object * 8u, 64u, 512u);
        binding.capabilities.estimated_launch_latency_us = std::max(3.0, summary.dispatch_latency_us * 0.60);
        binding.capabilities.estimated_host_read_gbps = std::max(summary.host_read_gbps, 24.0);
        binding.capabilities.estimated_host_write_gbps = std::max(summary.host_write_gbps, 24.0);

        double score = 0.88;
        score += summary.supports_fp16 ? 0.05 : 0.0;
        score += summary.supports_int8 ? 0.04 : 0.0;
        if (!binding.capabilities.adapter_available) {
            score -= 0.45;
        }
        binding.suitability_score = score;
        return binding;
    }

    [[nodiscard]] JakalL0LaunchTuning suggest_tuning(
        const HardwareGraph& graph,
        const JakalL0WorkloadTraits& workload) const override {
        const auto summary = summarize_graph(graph);
        JakalL0LaunchTuning tuning;
        tuning.workgroup_x = std::clamp(summary.lanes_per_object * 8u, 64u, 512u);
        tuning.workgroup_y = workload.matrix_friendly ? 2u : 1u;
        tuning.workgroup_z = 1u;
        tuning.queue_batch = workload.latency_sensitive ? 3u : 6u;
        tuning.staging_window = summary.unified_address_space ? 1u : 3u;
        tuning.prefer_vectorized_io = true;
        tuning.prefer_persistent_modules = true;
        tuning.prefer_shared_host_staging = summary.unified_address_space;
        return tuning;
    }
};

class RocmJakalL0Adapter final : public IJakalL0Adapter {
public:
    [[nodiscard]] std::string id() const override { return "gpu-l0.rocm"; }
    [[nodiscard]] JakalVendorFamily vendor() const override { return JakalVendorFamily::amd; }
    [[nodiscard]] JakalBackendKind backend_kind() const override { return JakalBackendKind::rocm; }
    [[nodiscard]] bool available() const override {
#if defined(_WIN32)
        return library_present({"amdhip64.dll"});
#else
        return library_present({"libamdhip64.so", "libamdhip64.so.6", "libhiprtc.so"});
#endif
    }

    [[nodiscard]] bool matches(const HardwareGraph& graph) const override {
        if (!looks_like_gpu_graph(graph)) {
            return false;
        }
        if (graph.probe == "rocm") {
            return true;
        }
        return contains_case_insensitive(graph.presentation_name, "amd") ||
               contains_case_insensitive(graph.presentation_name, "radeon") ||
               contains_case_insensitive(graph.presentation_name, "instinct") ||
               contains_case_insensitive(graph.presentation_name, "rocm") ||
               contains_case_insensitive(graph.presentation_name, "hip");
    }

    [[nodiscard]] JakalL0Binding describe(const HardwareGraph& graph) const override {
        const auto summary = summarize_graph(graph);
        JakalL0Binding binding;
        binding.device_uid = graph.uid;
        binding.graph_fingerprint = structural_fingerprint(graph);
        binding.adapter_id = id();
        binding.presentation_name = graph.presentation_name;
        binding.vendor = vendor();
        binding.backend = backend_kind();
        binding.capabilities.adapter_available = available();
        binding.capabilities.direct_command_submission = true;
        binding.capabilities.explicit_memory_import = true;
        binding.capabilities.timeline_synchronization = true;
        binding.capabilities.binary_module_loading = true;
        binding.capabilities.kernel_specialization = true;
        binding.capabilities.subgroup_matrix = summary.matrix_units > 0 || summary.supports_fp16 || summary.supports_int8;
        binding.capabilities.unified_memory = summary.unified_address_space;
        binding.capabilities.persistent_kernel_cache = true;
        binding.capabilities.asynchronous_dispatch = true;
        binding.capabilities.preferred_queue_batch = 6u;
        binding.capabilities.preferred_workgroup = std::clamp(summary.lanes_per_object * 8u, 64u, 512u);
        binding.capabilities.estimated_launch_latency_us = std::max(3.0, summary.dispatch_latency_us * 0.62);
        binding.capabilities.estimated_host_read_gbps = std::max(summary.host_read_gbps, 24.0);
        binding.capabilities.estimated_host_write_gbps = std::max(summary.host_write_gbps, 24.0);

        double score = 0.86;
        score += summary.supports_fp16 ? 0.05 : 0.0;
        score += summary.supports_int8 ? 0.04 : 0.0;
        if (!binding.capabilities.adapter_available) {
            score -= 0.45;
        }
        binding.suitability_score = score;
        return binding;
    }

    [[nodiscard]] JakalL0LaunchTuning suggest_tuning(
        const HardwareGraph& graph,
        const JakalL0WorkloadTraits& workload) const override {
        const auto summary = summarize_graph(graph);
        JakalL0LaunchTuning tuning;
        tuning.workgroup_x = std::clamp(summary.lanes_per_object * 8u, 64u, 512u);
        tuning.workgroup_y = workload.matrix_friendly ? 2u : 1u;
        tuning.workgroup_z = 1u;
        tuning.queue_batch = workload.latency_sensitive ? 3u : 6u;
        tuning.staging_window = summary.unified_address_space ? 1u : 3u;
        tuning.prefer_vectorized_io = true;
        tuning.prefer_persistent_modules = true;
        tuning.prefer_shared_host_staging = summary.unified_address_space;
        return tuning;
    }
};

}  // namespace

std::string to_string(const JakalVendorFamily vendor_family) {
    switch (vendor_family) {
    case JakalVendorFamily::intel:
        return "intel";
    case JakalVendorFamily::amd:
        return "amd";
    case JakalVendorFamily::nvidia:
        return "nvidia";
    case JakalVendorFamily::unknown:
        return "unknown";
    }
    return "unknown";
}

std::string to_string(const JakalBackendKind backend_kind) {
    switch (backend_kind) {
    case JakalBackendKind::opencl:
        return "opencl";
    case JakalBackendKind::level_zero:
        return "level-zero";
    case JakalBackendKind::vulkan_compute:
        return "vulkan-compute";
    case JakalBackendKind::cuda:
        return "cuda";
    case JakalBackendKind::rocm:
        return "rocm";
    }
    return "unknown";
}

std::vector<std::unique_ptr<IJakalL0Adapter>> make_default_jakal_l0_adapters() {
    std::vector<std::unique_ptr<IJakalL0Adapter>> adapters;
    adapters.push_back(std::make_unique<LevelZeroJakalL0Adapter>());
    adapters.push_back(std::make_unique<CudaJakalL0Adapter>());
    adapters.push_back(std::make_unique<RocmJakalL0Adapter>());
    adapters.push_back(std::make_unique<VulkanJakalL0Adapter>());
    adapters.push_back(std::make_unique<OpenClJakalL0Adapter>());
    return adapters;
}

}  // namespace jakal

