#include "jakal/backend.hpp"

#include "jakal/device.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

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

void* load_symbol(LibraryHandle library, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(library, name));
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

void* load_symbol(LibraryHandle library, const char* name) {
    return dlsym(library, name);
}

void close_library(LibraryHandle library) {
    if (library != nullptr) {
        dlclose(library);
    }
}
#endif

std::string sanitize_id_fragment(const std::string& text) {
    std::string result = text;
    for (char& ch : result) {
        const auto value = static_cast<unsigned char>(ch);
        if (!std::isalnum(value) && ch != '_' && ch != '-') {
            ch = '_';
        }
    }
    return result;
}

std::string format_compact_version(const int version) {
    if (version <= 0) {
        return {};
    }
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

std::string format_packed_api_version(const std::uint32_t version) {
    if (version == 0u) {
        return {};
    }
    const auto major = static_cast<std::uint32_t>((version >> 16u) & 0xffffu);
    const auto minor = static_cast<std::uint32_t>(version & 0xffffu);
    if (major == 0u && minor == 0u) {
        return {};
    }
    return std::to_string(major) + "." + std::to_string(minor);
}

HardwareGraph make_native_gpu_graph(
    const std::string& uid_prefix,
    const std::string& probe_name,
    const std::string& presentation_name,
    const std::string& driver_version,
    const std::string& runtime_version,
    const std::string& compiler_version,
    const std::uint32_t ordinal,
    const std::uint32_t execution_objects,
    const std::uint32_t lanes_per_object,
    const std::uint32_t matrix_units,
    const std::uint32_t clock_mhz,
    const std::uint64_t global_memory_bytes,
    const std::uint64_t local_memory_bytes,
    const std::uint64_t cache_bytes,
    const bool unified_memory,
    const bool supports_fp16,
    const bool supports_int8) {
    HardwareGraph graph;
    graph.uid = uid_prefix + ":" + std::to_string(ordinal);
    graph.probe = probe_name;
    graph.presentation_name = presentation_name;
    graph.driver_version = driver_version;
    graph.runtime_version = runtime_version;
    graph.compiler_version = compiler_version;
    graph.ordinal = ordinal;

    const std::string root_id = graph.uid + "/root";
    const std::string scheduler_id = graph.uid + "/control/scheduler";
    const std::string queue_id = graph.uid + "/control/queue";
    const std::string cluster_id = graph.uid + "/compute/cluster/0";
    const std::string global_memory_id = graph.uid + "/storage/global-memory";
    const std::string cache_id = graph.uid + "/storage/cache";
    const std::string host_link_id = graph.uid + "/transfer/host-link";

    graph.nodes.push_back({root_id, "device-root", "", HardwareObjectDomain::control, HardwareObjectRole::root});
    graph.nodes.push_back(
        {scheduler_id, "device-scheduler", root_id, HardwareObjectDomain::control, HardwareObjectRole::scheduler});
    graph.nodes.back().control.queue_slots = std::max(execution_objects, 1u);
    graph.nodes.back().control.supports_asynchronous_dispatch = true;
    graph.nodes.push_back({queue_id, "command-queue", root_id, HardwareObjectDomain::control, HardwareObjectRole::queue});
    graph.nodes.back().control.queue_slots = std::max(execution_objects * 2u, 1u);
    graph.nodes.back().control.supports_asynchronous_dispatch = true;
    graph.nodes.push_back({cluster_id, "compute-cluster", root_id, HardwareObjectDomain::compute, HardwareObjectRole::cluster});
    graph.nodes.back().compute.execution_width = std::max(execution_objects, 1u);
    graph.nodes.back().compute.resident_contexts = std::max(execution_objects, 1u);
    graph.nodes.back().compute.matrix_engines = matrix_units;
    graph.nodes.back().compute.clock_mhz = clock_mhz;
    graph.nodes.back().compute.native_vector_bits = std::max(lanes_per_object, 1u) * 32u;
    graph.nodes.back().compute.supports_fp16 = supports_fp16;
    graph.nodes.back().compute.supports_int8 = supports_int8;
    graph.nodes.push_back(
        {global_memory_id, "global-memory", root_id, HardwareObjectDomain::storage, HardwareObjectRole::global_memory});
    graph.nodes.back().storage.capacity_bytes = global_memory_bytes;
    graph.nodes.back().storage.directly_attached_bytes = unified_memory ? 0ull : global_memory_bytes;
    graph.nodes.back().storage.shared_host_bytes = unified_memory ? global_memory_bytes : 0ull;
    graph.nodes.back().storage.coherent_with_host = unified_memory;
    graph.nodes.back().storage.unified_address_space = unified_memory;
    graph.nodes.push_back({cache_id, "global-cache", root_id, HardwareObjectDomain::storage, HardwareObjectRole::cache});
    graph.nodes.back().storage.capacity_bytes = cache_bytes;
    graph.nodes.push_back(
        {host_link_id, "host-link", root_id, HardwareObjectDomain::transfer, HardwareObjectRole::transfer_link});
    graph.nodes.back().transfer.read_bandwidth_gbps = unified_memory ? 96.0 : 32.0;
    graph.nodes.back().transfer.write_bandwidth_gbps = unified_memory ? 96.0 : 32.0;
    graph.nodes.back().transfer.dispatch_latency_us = unified_memory ? 8.0 : 16.0;
    graph.nodes.back().transfer.synchronization_latency_us = unified_memory ? 6.0 : 12.0;

    graph.edges.push_back({root_id, scheduler_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
    graph.edges.push_back({root_id, queue_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
    graph.edges.push_back({root_id, cluster_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
    graph.edges.push_back({root_id, global_memory_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
    graph.edges.push_back({root_id, cache_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
    graph.edges.push_back({root_id, host_link_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
    graph.edges.push_back({scheduler_id, queue_id, GraphEdgeSemantics::controls, true, 1.0, 0.0, 0.0});
    graph.edges.push_back({queue_id, cluster_id, GraphEdgeSemantics::dispatches, true, 1.0, 0.0, unified_memory ? 8.0 : 16.0});
    graph.edges.push_back({host_link_id, global_memory_id, GraphEdgeSemantics::transfers_to, true, 1.0, unified_memory ? 96.0 : 32.0, unified_memory ? 8.0 : 16.0});
    graph.edges.push_back({global_memory_id, host_link_id, GraphEdgeSemantics::writes_to, true, 1.0, unified_memory ? 96.0 : 32.0, unified_memory ? 6.0 : 12.0});

    for (std::uint32_t unit_index = 0; unit_index < std::max(execution_objects, 1u); ++unit_index) {
        const std::string tile_id = graph.uid + "/compute/tile/" + std::to_string(unit_index);
        const std::string pipeline_id = tile_id + "/pipeline";
        const std::string scratch_id = tile_id + "/scratchpad";
        graph.nodes.push_back(
            {tile_id, "tile-" + std::to_string(unit_index), cluster_id, HardwareObjectDomain::compute, HardwareObjectRole::tile});
        graph.nodes.back().compute.execution_width = std::max(lanes_per_object, 1u);
        graph.nodes.back().compute.resident_contexts = 1u;
        graph.nodes.back().compute.matrix_engines = matrix_units > 0 ? 1u : 0u;
        graph.nodes.back().compute.clock_mhz = clock_mhz;
        graph.nodes.back().compute.native_vector_bits = std::max(lanes_per_object, 1u) * 32u;
        graph.nodes.back().compute.supports_fp16 = supports_fp16;
        graph.nodes.back().compute.supports_int8 = supports_int8;
        graph.nodes.push_back(
            {pipeline_id, "pipeline-" + std::to_string(unit_index), tile_id, HardwareObjectDomain::compute, HardwareObjectRole::pipeline});
        graph.nodes.back().compute.execution_width = std::max(lanes_per_object, 1u);
        graph.nodes.back().compute.clock_mhz = clock_mhz;
        graph.nodes.back().compute.native_vector_bits = std::max(lanes_per_object, 1u) * 32u;
        graph.nodes.back().compute.supports_fp16 = supports_fp16;
        graph.nodes.back().compute.supports_int8 = supports_int8;
        graph.edges.push_back({cluster_id, tile_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
        graph.edges.push_back({tile_id, pipeline_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
        graph.edges.push_back({queue_id, pipeline_id, GraphEdgeSemantics::dispatches, true, 1.0, 0.0, unified_memory ? 8.0 : 16.0});
        graph.edges.push_back({global_memory_id, pipeline_id, GraphEdgeSemantics::reads_from, true, 1.0, unified_memory ? 96.0 : 32.0, unified_memory ? 6.0 : 12.0});
        graph.edges.push_back({pipeline_id, global_memory_id, GraphEdgeSemantics::writes_to, true, 1.0, unified_memory ? 96.0 : 32.0, unified_memory ? 6.0 : 12.0});

        if (local_memory_bytes > 0) {
            graph.nodes.push_back(
                {scratch_id, "scratchpad-" + std::to_string(unit_index), tile_id, HardwareObjectDomain::storage, HardwareObjectRole::scratchpad});
            graph.nodes.back().storage.capacity_bytes = local_memory_bytes;
            graph.nodes.back().storage.directly_attached_bytes = local_memory_bytes;
            graph.edges.push_back({tile_id, scratch_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
            graph.edges.push_back({global_memory_id, scratch_id, GraphEdgeSemantics::transfers_to, true, 1.0, unified_memory ? 96.0 : 32.0, unified_memory ? 6.0 : 12.0});
            graph.edges.push_back({scratch_id, pipeline_id, GraphEdgeSemantics::feeds, true, 1.0, 0.0, 0.0});
        }
    }

    materialize_graph_costs(graph);
    return graph;
}

class LevelZeroApi {
public:
    using ze_result_t = std::int32_t;
    using ze_driver_handle_t = void*;
    using ze_device_handle_t = void*;
    using ze_api_version_t = std::uint32_t;
    using ze_init_fn = ze_result_t (*)(std::uint32_t);
    using ze_driver_get_fn = ze_result_t (*)(std::uint32_t*, ze_driver_handle_t*);
    using ze_device_get_fn = ze_result_t (*)(ze_driver_handle_t, std::uint32_t*, ze_device_handle_t*);
    using ze_driver_get_api_version_fn = ze_result_t (*)(ze_driver_handle_t, ze_api_version_t*);

    LevelZeroApi() {
#if defined(_WIN32)
        library_ = load_library("ze_loader.dll");
#else
        library_ = load_library("libze_loader.so");
        if (library_ == nullptr) {
            library_ = load_library("libze_loader.so.1");
        }
#endif
        if (library_ == nullptr) {
            return;
        }
        ze_init_ = reinterpret_cast<ze_init_fn>(load_symbol(library_, "zeInit"));
        ze_driver_get_ = reinterpret_cast<ze_driver_get_fn>(load_symbol(library_, "zeDriverGet"));
        ze_device_get_ = reinterpret_cast<ze_device_get_fn>(load_symbol(library_, "zeDeviceGet"));
        ze_driver_get_api_version_ =
            reinterpret_cast<ze_driver_get_api_version_fn>(load_symbol(library_, "zeDriverGetApiVersion"));
        loaded_ = ze_init_ != nullptr && ze_driver_get_ != nullptr && ze_device_get_ != nullptr;
    }

    ~LevelZeroApi() {
        close_library(library_);
    }

    [[nodiscard]] bool loaded() const { return loaded_; }

    [[nodiscard]] std::vector<HardwareGraph> discover() const {
        std::vector<HardwareGraph> graphs;
        if (!loaded_ || ze_init_(0u) != 0) {
            return graphs;
        }
        std::uint32_t driver_count = 0;
        if (ze_driver_get_(&driver_count, nullptr) != 0 || driver_count == 0) {
            return graphs;
        }
        std::vector<ze_driver_handle_t> drivers(driver_count, nullptr);
        if (ze_driver_get_(&driver_count, drivers.data()) != 0) {
            return graphs;
        }

        std::uint32_t ordinal = 0;
        for (const auto driver : drivers) {
            ze_api_version_t api_version = 0u;
            if (ze_driver_get_api_version_ != nullptr) {
                (void)ze_driver_get_api_version_(driver, &api_version);
            }
            std::uint32_t device_count = 0;
            if (ze_device_get_(driver, &device_count, nullptr) != 0 || device_count == 0) {
                continue;
            }
            std::vector<ze_device_handle_t> devices(device_count, nullptr);
            if (ze_device_get_(driver, &device_count, devices.data()) != 0) {
                continue;
            }
            for (const auto device : devices) {
                (void)device;
                const std::string name = "level-zero-device-" + std::to_string(ordinal);
                graphs.push_back(make_native_gpu_graph(
                    "level-zero:" + sanitize_id_fragment(name),
                    "level-zero",
                    name,
                    format_packed_api_version(api_version),
                    format_packed_api_version(api_version),
                    "ocloc",
                    ordinal++,
                    64u,
                    16u,
                    16u,
                    1400u,
                    8ull * 1024ull * 1024ull * 1024ull,
                    64ull * 1024ull,
                    4ull * 1024ull * 1024ull,
                    true,
                    true,
                    true));
            }
        }
        return graphs;
    }

private:
    LibraryHandle library_ = nullptr;
    bool loaded_ = false;
    ze_init_fn ze_init_ = nullptr;
    ze_driver_get_fn ze_driver_get_ = nullptr;
    ze_device_get_fn ze_device_get_ = nullptr;
    ze_driver_get_api_version_fn ze_driver_get_api_version_ = nullptr;
};

class CudaApi {
public:
    using CUresult = int;
    using CUdevice = int;
    using cu_init_fn = CUresult (*)(unsigned int);
    using cu_device_get_count_fn = CUresult (*)(int*);
    using cu_device_get_fn = CUresult (*)(CUdevice*, int);
    using cu_device_get_name_fn = CUresult (*)(char*, int, CUdevice);
    using cu_device_total_mem_fn = CUresult (*)(std::size_t*, CUdevice);
    using cu_device_get_attribute_fn = CUresult (*)(int*, int, CUdevice);
    using cu_driver_get_version_fn = CUresult (*)(int*);

    CudaApi() {
#if defined(_WIN32)
        library_ = load_library("nvcuda.dll");
#else
        library_ = load_library("libcuda.so");
        if (library_ == nullptr) {
            library_ = load_library("libcuda.so.1");
        }
#endif
        if (library_ == nullptr) {
            return;
        }
        cu_init_ = reinterpret_cast<cu_init_fn>(load_symbol(library_, "cuInit"));
        cu_device_get_count_ = reinterpret_cast<cu_device_get_count_fn>(load_symbol(library_, "cuDeviceGetCount"));
        cu_device_get_ = reinterpret_cast<cu_device_get_fn>(load_symbol(library_, "cuDeviceGet"));
        cu_device_get_name_ = reinterpret_cast<cu_device_get_name_fn>(load_symbol(library_, "cuDeviceGetName"));
        cu_device_total_mem_ = reinterpret_cast<cu_device_total_mem_fn>(load_symbol(library_, "cuDeviceTotalMem_v2"));
        cu_device_get_attribute_ =
            reinterpret_cast<cu_device_get_attribute_fn>(load_symbol(library_, "cuDeviceGetAttribute"));
        cu_driver_get_version_ =
            reinterpret_cast<cu_driver_get_version_fn>(load_symbol(library_, "cuDriverGetVersion"));
        loaded_ = cu_init_ != nullptr && cu_device_get_count_ != nullptr && cu_device_get_ != nullptr &&
                  cu_device_get_name_ != nullptr && cu_device_total_mem_ != nullptr && cu_device_get_attribute_ != nullptr;
    }

    ~CudaApi() {
        close_library(library_);
    }

    [[nodiscard]] bool loaded() const { return loaded_; }

    [[nodiscard]] std::vector<HardwareGraph> discover() const {
        std::vector<HardwareGraph> graphs;
        if (!loaded_ || cu_init_(0u) != 0) {
            return graphs;
        }
        int device_count = 0;
        if (cu_device_get_count_(&device_count) != 0 || device_count <= 0) {
            return graphs;
        }
        constexpr int kMultiProcessorCount = 16;
        constexpr int kClockRateKhz = 13;
        constexpr int kWarpSize = 10;
        constexpr int kL2CacheSize = 38;
        constexpr int kManagedMemory = 83;
        int driver_version = 0;
        if (cu_driver_get_version_ != nullptr) {
            (void)cu_driver_get_version_(&driver_version);
        }
        for (int ordinal = 0; ordinal < device_count; ++ordinal) {
            CUdevice device = 0;
            if (cu_device_get_(&device, ordinal) != 0) {
                continue;
            }
            char name[256]{};
            std::size_t total_mem = 0;
            int multiprocessors = 0;
            int clock_rate_khz = 0;
            int warp_size = 32;
            int l2_cache = 0;
            int managed_memory = 0;
            cu_device_get_name_(name, static_cast<int>(std::size(name)), device);
            cu_device_total_mem_(&total_mem, device);
            cu_device_get_attribute_(&multiprocessors, kMultiProcessorCount, device);
            cu_device_get_attribute_(&clock_rate_khz, kClockRateKhz, device);
            cu_device_get_attribute_(&warp_size, kWarpSize, device);
            cu_device_get_attribute_(&l2_cache, kL2CacheSize, device);
            cu_device_get_attribute_(&managed_memory, kManagedMemory, device);
            const std::string device_name = name[0] == '\0' ? "cuda-device" : name;
            graphs.push_back(make_native_gpu_graph(
                "cuda:" + sanitize_id_fragment(device_name),
                "cuda",
                device_name,
                format_compact_version(driver_version),
                format_compact_version(driver_version),
                "nvrtc",
                static_cast<std::uint32_t>(ordinal),
                std::max(multiprocessors, 1),
                std::max(warp_size, 1),
                std::max(multiprocessors / 2, 1),
                std::max(clock_rate_khz / 1000, 1000),
                total_mem == 0 ? (8ull * 1024ull * 1024ull * 1024ull) : static_cast<std::uint64_t>(total_mem),
                64ull * 1024ull,
                std::max(l2_cache, 4 * 1024 * 1024),
                managed_memory != 0,
                true,
                true));
        }
        return graphs;
    }

private:
    LibraryHandle library_ = nullptr;
    bool loaded_ = false;
    cu_init_fn cu_init_ = nullptr;
    cu_device_get_count_fn cu_device_get_count_ = nullptr;
    cu_device_get_fn cu_device_get_ = nullptr;
    cu_device_get_name_fn cu_device_get_name_ = nullptr;
    cu_device_total_mem_fn cu_device_total_mem_ = nullptr;
    cu_device_get_attribute_fn cu_device_get_attribute_ = nullptr;
    cu_driver_get_version_fn cu_driver_get_version_ = nullptr;
};

class RocmApi {
public:
    using hipError_t = int;
    using hipDevice_t = int;
    using hip_init_fn = hipError_t (*)(unsigned int);
    using hip_get_device_count_fn = hipError_t (*)(int*);
    using hip_device_get_fn = hipError_t (*)(hipDevice_t*, int);
    using hip_device_get_name_fn = hipError_t (*)(char*, int, hipDevice_t);
    using hip_device_total_mem_fn = hipError_t (*)(std::size_t*, hipDevice_t);
    using hip_device_get_attribute_fn = hipError_t (*)(int*, int, hipDevice_t);
    using hip_driver_get_version_fn = hipError_t (*)(int*);
    using hip_runtime_get_version_fn = hipError_t (*)(int*);

    RocmApi() {
#if defined(_WIN32)
        library_ = load_library("amdhip64.dll");
#else
        library_ = load_library("libamdhip64.so");
        if (library_ == nullptr) {
            library_ = load_library("libamdhip64.so.6");
        }
#endif
        if (library_ == nullptr) {
            return;
        }
        hip_init_ = reinterpret_cast<hip_init_fn>(load_symbol(library_, "hipInit"));
        hip_get_device_count_ =
            reinterpret_cast<hip_get_device_count_fn>(load_symbol(library_, "hipGetDeviceCount"));
        hip_device_get_ = reinterpret_cast<hip_device_get_fn>(load_symbol(library_, "hipDeviceGet"));
        hip_device_get_name_ = reinterpret_cast<hip_device_get_name_fn>(load_symbol(library_, "hipDeviceGetName"));
        hip_device_total_mem_ =
            reinterpret_cast<hip_device_total_mem_fn>(load_symbol(library_, "hipDeviceTotalMem"));
        hip_device_get_attribute_ =
            reinterpret_cast<hip_device_get_attribute_fn>(load_symbol(library_, "hipDeviceGetAttribute"));
        hip_driver_get_version_ =
            reinterpret_cast<hip_driver_get_version_fn>(load_symbol(library_, "hipDriverGetVersion"));
        hip_runtime_get_version_ =
            reinterpret_cast<hip_runtime_get_version_fn>(load_symbol(library_, "hipRuntimeGetVersion"));
        loaded_ = hip_init_ != nullptr && hip_get_device_count_ != nullptr && hip_device_get_ != nullptr &&
                  hip_device_get_name_ != nullptr && hip_device_total_mem_ != nullptr && hip_device_get_attribute_ != nullptr;
    }

    ~RocmApi() {
        close_library(library_);
    }

    [[nodiscard]] bool loaded() const { return loaded_; }

    [[nodiscard]] std::vector<HardwareGraph> discover() const {
        std::vector<HardwareGraph> graphs;
        if (!loaded_ || hip_init_(0u) != 0) {
            return graphs;
        }
        int device_count = 0;
        if (hip_get_device_count_(&device_count) != 0 || device_count <= 0) {
            return graphs;
        }
        constexpr int kMultiprocessorCount = 8;
        constexpr int kClockRateKhz = 13;
        constexpr int kWarpSize = 10;
        constexpr int kL2CacheSize = 38;
        constexpr int kManagedMemory = 83;
        int driver_version = 0;
        int runtime_version = 0;
        if (hip_driver_get_version_ != nullptr) {
            (void)hip_driver_get_version_(&driver_version);
        }
        if (hip_runtime_get_version_ != nullptr) {
            (void)hip_runtime_get_version_(&runtime_version);
        }
        for (int ordinal = 0; ordinal < device_count; ++ordinal) {
            hipDevice_t device = 0;
            if (hip_device_get_(&device, ordinal) != 0) {
                continue;
            }
            char name[256]{};
            std::size_t total_mem = 0;
            int multiprocessors = 0;
            int clock_rate_khz = 0;
            int warp_size = 64;
            int l2_cache = 0;
            int managed_memory = 0;
            hip_device_get_name_(name, static_cast<int>(std::size(name)), device);
            hip_device_total_mem_(&total_mem, device);
            hip_device_get_attribute_(&multiprocessors, kMultiprocessorCount, device);
            hip_device_get_attribute_(&clock_rate_khz, kClockRateKhz, device);
            hip_device_get_attribute_(&warp_size, kWarpSize, device);
            hip_device_get_attribute_(&l2_cache, kL2CacheSize, device);
            hip_device_get_attribute_(&managed_memory, kManagedMemory, device);
            const std::string device_name = name[0] == '\0' ? "rocm-device" : name;
            graphs.push_back(make_native_gpu_graph(
                "rocm:" + sanitize_id_fragment(device_name),
                "rocm",
                device_name,
                format_compact_version(driver_version),
                format_compact_version(runtime_version),
                "hiprtc",
                static_cast<std::uint32_t>(ordinal),
                std::max(multiprocessors, 1),
                std::max(warp_size, 1),
                std::max(multiprocessors / 2, 1),
                std::max(clock_rate_khz / 1000, 1000),
                total_mem == 0 ? (8ull * 1024ull * 1024ull * 1024ull) : static_cast<std::uint64_t>(total_mem),
                64ull * 1024ull,
                std::max(l2_cache, 4 * 1024 * 1024),
                managed_memory != 0,
                true,
                true));
        }
        return graphs;
    }

private:
    LibraryHandle library_ = nullptr;
    bool loaded_ = false;
    hip_init_fn hip_init_ = nullptr;
    hip_get_device_count_fn hip_get_device_count_ = nullptr;
    hip_device_get_fn hip_device_get_ = nullptr;
    hip_device_get_name_fn hip_device_get_name_ = nullptr;
    hip_device_total_mem_fn hip_device_total_mem_ = nullptr;
    hip_device_get_attribute_fn hip_device_get_attribute_ = nullptr;
    hip_driver_get_version_fn hip_driver_get_version_ = nullptr;
    hip_runtime_get_version_fn hip_runtime_get_version_ = nullptr;
};

class LevelZeroProbe final : public IDeviceProbe {
public:
    [[nodiscard]] std::string name() const override { return "level-zero"; }
    [[nodiscard]] bool available() const override { return api_.loaded(); }
    std::vector<HardwareGraph> discover_hardware() override { return api_.discover(); }

private:
    LevelZeroApi api_;
};

class CudaProbe final : public IDeviceProbe {
public:
    [[nodiscard]] std::string name() const override { return "cuda"; }
    [[nodiscard]] bool available() const override { return api_.loaded(); }
    std::vector<HardwareGraph> discover_hardware() override { return api_.discover(); }

private:
    CudaApi api_;
};

class RocmProbe final : public IDeviceProbe {
public:
    [[nodiscard]] std::string name() const override { return "rocm"; }
    [[nodiscard]] bool available() const override { return api_.loaded(); }
    std::vector<HardwareGraph> discover_hardware() override { return api_.discover(); }

private:
    RocmApi api_;
};

class VulkanApi {
public:
    using vk_get_instance_proc_addr_fn = PFN_vkGetInstanceProcAddr;
    using vk_create_instance_fn = PFN_vkCreateInstance;
    using vk_enumerate_physical_devices_fn = PFN_vkEnumeratePhysicalDevices;
    using vk_get_physical_device_properties_fn = PFN_vkGetPhysicalDeviceProperties;
    using vk_get_physical_device_memory_properties_fn = PFN_vkGetPhysicalDeviceMemoryProperties;
    using vk_destroy_instance_fn = PFN_vkDestroyInstance;

    VulkanApi() {
#if defined(_WIN32)
        library_ = load_library("vulkan-1.dll");
#else
        library_ = load_library("libvulkan.so");
        if (library_ == nullptr) {
            library_ = load_library("libvulkan.so.1");
        }
#endif
        if (library_ == nullptr) {
            return;
        }
        vk_get_instance_proc_addr_ =
            reinterpret_cast<vk_get_instance_proc_addr_fn>(load_symbol(library_, "vkGetInstanceProcAddr"));
        if (vk_get_instance_proc_addr_ == nullptr) {
            return;
        }
        vk_create_instance_ =
            reinterpret_cast<vk_create_instance_fn>(vk_get_instance_proc_addr_(nullptr, "vkCreateInstance"));
        loaded_ = vk_create_instance_ != nullptr;
    }

    ~VulkanApi() {
        close_library(library_);
    }

    [[nodiscard]] bool loaded() const { return loaded_; }

    [[nodiscard]] std::vector<HardwareGraph> discover() const {
        std::vector<HardwareGraph> graphs;
        if (!loaded_) {
            return graphs;
        }

        const VkApplicationInfo app_info{
            VK_STRUCTURE_TYPE_APPLICATION_INFO,
            nullptr,
            "JakalRuntime",
            1u,
            "JakalRuntime",
            1u,
            VK_API_VERSION_1_0};
        const VkInstanceCreateInfo create_info{
            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            nullptr,
            0u,
            &app_info,
            0u,
            nullptr,
            0u,
            nullptr};

        VkInstance instance = VK_NULL_HANDLE;
        if (vk_create_instance_(&create_info, nullptr, &instance) != VK_SUCCESS || instance == VK_NULL_HANDLE) {
            return graphs;
        }

        const auto vk_enumerate_physical_devices =
            reinterpret_cast<vk_enumerate_physical_devices_fn>(
                vk_get_instance_proc_addr_(instance, "vkEnumeratePhysicalDevices"));
        const auto vk_get_physical_device_properties =
            reinterpret_cast<vk_get_physical_device_properties_fn>(
                vk_get_instance_proc_addr_(instance, "vkGetPhysicalDeviceProperties"));
        const auto vk_get_physical_device_memory_properties =
            reinterpret_cast<vk_get_physical_device_memory_properties_fn>(
                vk_get_instance_proc_addr_(instance, "vkGetPhysicalDeviceMemoryProperties"));
        const auto vk_destroy_instance =
            reinterpret_cast<vk_destroy_instance_fn>(vk_get_instance_proc_addr_(instance, "vkDestroyInstance"));

        if (vk_enumerate_physical_devices == nullptr ||
            vk_get_physical_device_properties == nullptr ||
            vk_get_physical_device_memory_properties == nullptr ||
            vk_destroy_instance == nullptr) {
            if (vk_destroy_instance != nullptr) {
                vk_destroy_instance(instance, nullptr);
            }
            return graphs;
        }

        std::uint32_t device_count = 0u;
        if (vk_enumerate_physical_devices(instance, &device_count, nullptr) != VK_SUCCESS || device_count == 0u) {
            vk_destroy_instance(instance, nullptr);
            return graphs;
        }
        std::vector<VkPhysicalDevice> devices(device_count, VK_NULL_HANDLE);
        if (vk_enumerate_physical_devices(instance, &device_count, devices.data()) != VK_SUCCESS) {
            vk_destroy_instance(instance, nullptr);
            return graphs;
        }

        std::uint32_t ordinal = 0u;
        for (const auto device : devices) {
            VkPhysicalDeviceProperties props{};
            VkPhysicalDeviceMemoryProperties memory{};
            vk_get_physical_device_properties(device, &props);
            vk_get_physical_device_memory_properties(device, &memory);

            std::uint64_t total_memory = 0ull;
            bool host_visible = false;
            bool host_coherent = false;
            for (std::uint32_t heap_index = 0u; heap_index < memory.memoryHeapCount; ++heap_index) {
                if ((memory.memoryHeaps[heap_index].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0u) {
                    total_memory += memory.memoryHeaps[heap_index].size;
                }
            }
            for (std::uint32_t type_index = 0u; type_index < memory.memoryTypeCount; ++type_index) {
                const auto flags = memory.memoryTypes[type_index].propertyFlags;
                host_visible = host_visible || ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0u);
                host_coherent = host_coherent || ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u);
            }

            const auto vendor = props.vendorID;
            const bool likely_fp16 = vendor == 0x10DEu || vendor == 0x8086u || vendor == 0x1002u || vendor == 0x1022u;
            const auto driver = std::to_string(VK_VERSION_MAJOR(props.driverVersion)) + "." +
                                std::to_string(VK_VERSION_MINOR(props.driverVersion));
            const auto api = std::to_string(VK_VERSION_MAJOR(props.apiVersion)) + "." +
                             std::to_string(VK_VERSION_MINOR(props.apiVersion));
            const auto name = props.deviceName[0] == '\0' ? ("vulkan-device-" + std::to_string(ordinal))
                                                           : std::string(props.deviceName);

            graphs.push_back(make_native_gpu_graph(
                "vulkan:" + sanitize_id_fragment(name),
                "vulkan",
                name,
                driver,
                api,
                "glslang",
                ordinal++,
                std::max<std::uint32_t>(16u, props.limits.maxComputeWorkGroupInvocations / 8u),
                std::max<std::uint32_t>(8u, props.limits.maxComputeWorkGroupInvocations / 16u),
                likely_fp16 ? 8u : 4u,
                1400u,
                total_memory == 0ull ? (4ull * 1024ull * 1024ull * 1024ull) : total_memory,
                64ull * 1024ull,
                2ull * 1024ull * 1024ull,
                host_visible && host_coherent,
                likely_fp16,
                true));
        }

        vk_destroy_instance(instance, nullptr);
        return graphs;
    }

private:
    LibraryHandle library_ = nullptr;
    bool loaded_ = false;
    vk_get_instance_proc_addr_fn vk_get_instance_proc_addr_ = nullptr;
    vk_create_instance_fn vk_create_instance_ = nullptr;
};

class VulkanProbe final : public IDeviceProbe {
public:
    [[nodiscard]] std::string name() const override { return "vulkan"; }
    [[nodiscard]] bool available() const override { return api_.loaded(); }
    std::vector<HardwareGraph> discover_hardware() override { return api_.discover(); }

private:
    VulkanApi api_;
};

}  // namespace

std::unique_ptr<IDeviceProbe> make_level_zero_probe() {
    return std::make_unique<LevelZeroProbe>();
}

std::unique_ptr<IDeviceProbe> make_cuda_probe() {
    return std::make_unique<CudaProbe>();
}

std::unique_ptr<IDeviceProbe> make_vulkan_probe() {
    return std::make_unique<VulkanProbe>();
}

std::unique_ptr<IDeviceProbe> make_rocm_probe() {
    return std::make_unique<RocmProbe>();
}

}  // namespace jakal

