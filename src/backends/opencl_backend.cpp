#include "jakal/backend.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
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

using cl_int = std::int32_t;
using cl_uint = std::uint32_t;
using cl_ulong = std::uint64_t;
using cl_bitfield = cl_ulong;
using cl_device_type = cl_bitfield;
using cl_platform_info = cl_uint;
using cl_device_info = cl_uint;
using cl_bool = cl_uint;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_device_type CL_DEVICE_TYPE_ALL = 0xFFFFFFFFu;

constexpr cl_platform_info CL_PLATFORM_NAME = 0x0902;

constexpr cl_device_info CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002;
constexpr cl_device_info CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR = 0x1006;
constexpr cl_device_info CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C;
constexpr cl_device_info CL_DEVICE_QUEUE_PROPERTIES = 0x102A;
constexpr cl_device_info CL_DEVICE_GLOBAL_MEM_CACHE_SIZE = 0x101E;
constexpr cl_device_info CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
constexpr cl_device_info CL_DEVICE_LOCAL_MEM_SIZE = 0x1023;
constexpr cl_device_info CL_DEVICE_NAME = 0x102B;
constexpr cl_device_info CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035;
constexpr cl_device_info CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032;
constexpr cl_device_info CL_DEVICE_HALF_FP_CONFIG = 0x1033;
constexpr cl_device_info CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT = 0x103A;

using cl_get_platform_ids_fn = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
using cl_get_platform_info_fn = cl_int (*)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
using cl_get_device_ids_fn = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
using cl_get_device_info_fn = cl_int (*)(cl_device_id, cl_device_info, size_t, void*, size_t*);

class OpenClApi {
public:
    OpenClApi() {
#if defined(_WIN32)
        library_ = load_library("OpenCL.dll");
#elif defined(__APPLE__)
        library_ = load_library("/System/Library/Frameworks/OpenCL.framework/OpenCL");
        if (library_ == nullptr) {
            library_ = load_library("libOpenCL.dylib");
        }
#else
        library_ = load_library("libOpenCL.so");
        if (library_ == nullptr) {
            library_ = load_library("libOpenCL.so.1");
        }
#endif

        if (library_ == nullptr) {
            return;
        }

        get_platform_ids_ = reinterpret_cast<cl_get_platform_ids_fn>(load_symbol(library_, "clGetPlatformIDs"));
        get_platform_info_ = reinterpret_cast<cl_get_platform_info_fn>(load_symbol(library_, "clGetPlatformInfo"));
        get_device_ids_ = reinterpret_cast<cl_get_device_ids_fn>(load_symbol(library_, "clGetDeviceIDs"));
        get_device_info_ = reinterpret_cast<cl_get_device_info_fn>(load_symbol(library_, "clGetDeviceInfo"));

        loaded_ = get_platform_ids_ != nullptr &&
                  get_platform_info_ != nullptr &&
                  get_device_ids_ != nullptr &&
                  get_device_info_ != nullptr;
    }

    ~OpenClApi() {
        close_library(library_);
    }

    OpenClApi(const OpenClApi&) = delete;
    OpenClApi& operator=(const OpenClApi&) = delete;

    [[nodiscard]] bool loaded() const {
        return loaded_;
    }

    [[nodiscard]] cl_get_platform_ids_fn get_platform_ids() const {
        return get_platform_ids_;
    }

    [[nodiscard]] cl_get_platform_info_fn get_platform_info() const {
        return get_platform_info_;
    }

    [[nodiscard]] cl_get_device_ids_fn get_device_ids() const {
        return get_device_ids_;
    }

    [[nodiscard]] cl_get_device_info_fn get_device_info() const {
        return get_device_info_;
    }

private:
    LibraryHandle library_ = nullptr;
    cl_get_platform_ids_fn get_platform_ids_ = nullptr;
    cl_get_platform_info_fn get_platform_info_ = nullptr;
    cl_get_device_ids_fn get_device_ids_ = nullptr;
    cl_get_device_info_fn get_device_info_ = nullptr;
    bool loaded_ = false;
};

std::string sanitize_id_fragment(const std::string& text) {
    std::string result = text;
    for (char& ch : result) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-')) {
            ch = '_';
        }
    }
    return result;
}

template <typename T>
T get_scalar_info(cl_get_device_info_fn getter, cl_device_id device, cl_device_info key, const T fallback = T{}) {
    T value{};
    if (getter(device, key, sizeof(T), &value, nullptr) != CL_SUCCESS) {
        return fallback;
    }
    return value;
}

std::string get_string_info(cl_get_device_info_fn getter, cl_device_id device, cl_device_info key) {
    size_t size = 0;
    if (getter(device, key, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return {};
    }

    std::string text(size, '\0');
    if (getter(device, key, size, text.data(), nullptr) != CL_SUCCESS) {
        return {};
    }

    if (!text.empty() && text.back() == '\0') {
        text.pop_back();
    }
    return text;
}

std::string get_platform_string(cl_get_platform_info_fn getter, cl_platform_id platform, cl_platform_info key) {
    size_t size = 0;
    if (getter(platform, key, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return {};
    }

    std::string text(size, '\0');
    if (getter(platform, key, size, text.data(), nullptr) != CL_SUCCESS) {
        return {};
    }

    if (!text.empty() && text.back() == '\0') {
        text.pop_back();
    }
    return text;
}

class OpenClProbe final : public IDeviceProbe {
public:
    [[nodiscard]] std::string name() const override {
        return "opencl";
    }

    [[nodiscard]] bool available() const override {
        return api_.loaded();
    }

    std::vector<HardwareGraph> discover_hardware() override {
        std::vector<HardwareGraph> graphs;
        if (!api_.loaded()) {
            return graphs;
        }

        cl_uint platform_count = 0;
        if (api_.get_platform_ids()(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0) {
            return graphs;
        }

        std::vector<cl_platform_id> platforms(platform_count, nullptr);
        if (api_.get_platform_ids()(platform_count, platforms.data(), nullptr) != CL_SUCCESS) {
            return graphs;
        }

        std::uint32_t ordinal = 0;
        for (const auto platform : platforms) {
            const std::string platform_name = get_platform_string(api_.get_platform_info(), platform, CL_PLATFORM_NAME);
            const std::string platform_fragment = sanitize_id_fragment(platform_name);

            cl_uint device_count = 0;
            if (api_.get_device_ids()(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count) != CL_SUCCESS || device_count == 0) {
                continue;
            }

            std::vector<cl_device_id> platform_devices(device_count, nullptr);
            if (api_.get_device_ids()(platform, CL_DEVICE_TYPE_ALL, device_count, platform_devices.data(), nullptr) != CL_SUCCESS) {
                continue;
            }

            for (const auto device_id : platform_devices) {
                const std::string device_name = get_string_info(api_.get_device_info(), device_id, CL_DEVICE_NAME);
                const bool unified_memory = get_scalar_info<cl_bool>(
                    api_.get_device_info(),
                    device_id,
                    CL_DEVICE_HOST_UNIFIED_MEMORY,
                    0) != 0;
                const cl_bitfield queue_properties = get_scalar_info<cl_bitfield>(
                    api_.get_device_info(),
                    device_id,
                    CL_DEVICE_QUEUE_PROPERTIES,
                    0);
                const std::uint32_t compute_units = get_scalar_info<cl_uint>(
                    api_.get_device_info(),
                    device_id,
                    CL_DEVICE_MAX_COMPUTE_UNITS,
                    1);
                const std::uint32_t native_float_width = std::max(
                    get_scalar_info<cl_uint>(api_.get_device_info(), device_id, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, 0),
                    1u);
                const std::uint32_t clock_mhz = get_scalar_info<cl_uint>(
                    api_.get_device_info(),
                    device_id,
                    CL_DEVICE_MAX_CLOCK_FREQUENCY,
                    0);
                const std::uint64_t global_memory = get_scalar_info<cl_ulong>(
                    api_.get_device_info(),
                    device_id,
                    CL_DEVICE_GLOBAL_MEM_SIZE,
                    0);
                const std::uint64_t local_memory = get_scalar_info<cl_ulong>(
                    api_.get_device_info(),
                    device_id,
                    CL_DEVICE_LOCAL_MEM_SIZE,
                    0);
                const std::uint64_t cache_bytes = get_scalar_info<cl_ulong>(
                    api_.get_device_info(),
                    device_id,
                    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                    0);
                const bool supports_fp64 = get_scalar_info<cl_bitfield>(
                    api_.get_device_info(),
                    device_id,
                    CL_DEVICE_DOUBLE_FP_CONFIG,
                    0) != 0;
                const bool supports_fp16 = get_scalar_info<cl_bitfield>(
                    api_.get_device_info(),
                    device_id,
                    CL_DEVICE_HALF_FP_CONFIG,
                    0) != 0;
                const bool supports_int8 = get_scalar_info<cl_uint>(
                    api_.get_device_info(),
                    device_id,
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                    0) > 0;

                HardwareGraph graph;
                graph.uid = "opencl:" + platform_fragment + ":" + std::to_string(ordinal);
                graph.probe = "opencl";
                graph.presentation_name = device_name.empty() ? "opencl-device" : device_name;
                graph.ordinal = ordinal;

                const std::string root_id = graph.uid + "/root";
                const std::string scheduler_id = graph.uid + "/control/scheduler";
                const std::string queue_id = graph.uid + "/control/queue";
                const std::string cluster_id = graph.uid + "/compute/cluster/0";
                const std::string global_memory_id = graph.uid + "/storage/global-memory";
                const std::string cache_id = graph.uid + "/storage/cache";
                const std::string host_link_id = graph.uid + "/transfer/host-link";
                const std::string host_memory_id = graph.uid + "/storage/host-aperture";

                graph.nodes.push_back(HardwareObjectNode{
                    root_id,
                    "device-root",
                    "",
                    HardwareObjectDomain::control,
                    HardwareObjectRole::root,
                    HardwareObjectResolution::coarse,
                    0,
                    {},
                    {},
                    {},
                    {0, 0, false, queue_properties != 0}});
                graph.nodes.push_back(HardwareObjectNode{
                    scheduler_id,
                    "device-scheduler",
                    root_id,
                    HardwareObjectDomain::control,
                    HardwareObjectRole::scheduler,
                    HardwareObjectResolution::medium,
                    0,
                    {},
                    {},
                    {},
                    {compute_units, 0, false, queue_properties != 0}});
                graph.nodes.push_back(HardwareObjectNode{
                    queue_id,
                    "command-queue",
                    root_id,
                    HardwareObjectDomain::control,
                    HardwareObjectRole::queue,
                    HardwareObjectResolution::medium,
                    0,
                    {},
                    {},
                    {},
                    {queue_properties == 0 ? 1u : compute_units, 0, false, queue_properties != 0}});
                graph.nodes.push_back(HardwareObjectNode{
                    cluster_id,
                    "compute-cluster",
                    root_id,
                    HardwareObjectDomain::compute,
                    HardwareObjectRole::cluster,
                    HardwareObjectResolution::medium,
                    0,
                    {compute_units, compute_units, 0, clock_mhz, native_float_width * 32u, supports_fp64, true, supports_fp16, false, supports_int8},
                    {},
                    {},
                    {}});
                graph.nodes.push_back(HardwareObjectNode{
                    global_memory_id,
                    "global-memory",
                    root_id,
                    HardwareObjectDomain::storage,
                    HardwareObjectRole::global_memory,
                    HardwareObjectResolution::medium,
                    0,
                    {},
                    {global_memory, unified_memory ? 0ull : global_memory, unified_memory ? global_memory : 0ull, unified_memory, unified_memory},
                    {},
                    {}});
                graph.nodes.push_back(HardwareObjectNode{
                    cache_id,
                    "global-cache",
                    root_id,
                    HardwareObjectDomain::storage,
                    HardwareObjectRole::cache,
                    HardwareObjectResolution::medium,
                    0,
                    {},
                    {cache_bytes, 0, 0, false, false},
                    {},
                    {}});
                graph.nodes.push_back(HardwareObjectNode{
                    host_link_id,
                    "host-link",
                    root_id,
                    HardwareObjectDomain::transfer,
                    HardwareObjectRole::transfer_link,
                    HardwareObjectResolution::medium,
                    0,
                    {},
                    {},
                    {unified_memory ? 96.0 : 24.0, unified_memory ? 96.0 : 24.0, unified_memory ? 10.0 : 28.0, unified_memory ? 8.0 : 18.0},
                    {}});

                graph.edges.push_back(HardwareGraphEdge{root_id, scheduler_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
                graph.edges.push_back(HardwareGraphEdge{root_id, queue_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
                graph.edges.push_back(HardwareGraphEdge{root_id, cluster_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
                graph.edges.push_back(HardwareGraphEdge{root_id, global_memory_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
                graph.edges.push_back(HardwareGraphEdge{root_id, cache_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
                graph.edges.push_back(HardwareGraphEdge{root_id, host_link_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
                graph.edges.push_back(HardwareGraphEdge{scheduler_id, queue_id, GraphEdgeSemantics::controls, true, 1.0, 0.0, 0.0});
                graph.edges.push_back(HardwareGraphEdge{
                    host_link_id,
                    global_memory_id,
                    GraphEdgeSemantics::transfers_to,
                    true,
                    1.0,
                    unified_memory ? 96.0 : 24.0,
                    unified_memory ? 10.0 : 28.0});
                graph.edges.push_back(HardwareGraphEdge{
                    global_memory_id,
                    host_link_id,
                    GraphEdgeSemantics::writes_to,
                    true,
                    1.0,
                    unified_memory ? 96.0 : 24.0,
                    unified_memory ? 8.0 : 18.0});

                if (unified_memory) {
                    graph.nodes.push_back(HardwareObjectNode{
                        host_memory_id,
                        "host-aperture",
                        root_id,
                        HardwareObjectDomain::storage,
                        HardwareObjectRole::host_memory,
                        HardwareObjectResolution::medium,
                        0,
                        {},
                        {global_memory, 0, global_memory, true, true},
                        {},
                        {}});
                    graph.edges.push_back(HardwareGraphEdge{root_id, host_memory_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
                    graph.edges.push_back(HardwareGraphEdge{
                        host_memory_id,
                        host_link_id,
                        GraphEdgeSemantics::transfers_to,
                        true,
                        1.0,
                        96.0,
                        10.0});
                }

                for (std::uint32_t unit_index = 0; unit_index < compute_units; ++unit_index) {
                    const std::string tile_id = graph.uid + "/compute/tile/" + std::to_string(unit_index);
                    const std::string pipeline_id = tile_id + "/pipeline";
                    const std::string scratch_id = tile_id + "/scratchpad";

                    graph.nodes.push_back(HardwareObjectNode{
                        tile_id,
                        "tile-" + std::to_string(unit_index),
                        cluster_id,
                        HardwareObjectDomain::compute,
                        HardwareObjectRole::tile,
                        HardwareObjectResolution::medium,
                        unit_index,
                        {native_float_width, 1, 0, clock_mhz, native_float_width * 32u, supports_fp64, true, supports_fp16, false, supports_int8},
                        {},
                        {},
                        {}});
                    graph.nodes.push_back(HardwareObjectNode{
                        pipeline_id,
                        "pipeline-" + std::to_string(unit_index),
                        tile_id,
                        HardwareObjectDomain::compute,
                        HardwareObjectRole::pipeline,
                        HardwareObjectResolution::aggressive,
                        unit_index,
                        {native_float_width, 1, 0, clock_mhz, native_float_width * 32u, supports_fp64, true, supports_fp16, false, supports_int8},
                        {},
                        {},
                        {}});

                    graph.edges.push_back(HardwareGraphEdge{cluster_id, tile_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
                    graph.edges.push_back(HardwareGraphEdge{tile_id, pipeline_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
                    graph.edges.push_back(HardwareGraphEdge{
                        queue_id,
                        pipeline_id,
                        GraphEdgeSemantics::dispatches,
                        true,
                        1.0,
                        0.0,
                        unified_memory ? 10.0 : 28.0});
                    graph.edges.push_back(HardwareGraphEdge{
                        global_memory_id,
                        pipeline_id,
                        GraphEdgeSemantics::reads_from,
                        true,
                        1.0,
                        unified_memory ? 96.0 : 24.0,
                        unified_memory ? 8.0 : 18.0});
                    graph.edges.push_back(HardwareGraphEdge{
                        pipeline_id,
                        global_memory_id,
                        GraphEdgeSemantics::writes_to,
                        true,
                        1.0,
                        unified_memory ? 96.0 : 24.0,
                        unified_memory ? 8.0 : 18.0});

                    if (local_memory > 0) {
                        graph.nodes.push_back(HardwareObjectNode{
                            scratch_id,
                            "scratchpad-" + std::to_string(unit_index),
                            tile_id,
                            HardwareObjectDomain::storage,
                            HardwareObjectRole::scratchpad,
                            HardwareObjectResolution::aggressive,
                            unit_index,
                            {},
                            {local_memory, local_memory, 0, false, false},
                            {},
                            {}});
                        graph.edges.push_back(HardwareGraphEdge{tile_id, scratch_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
                        graph.edges.push_back(HardwareGraphEdge{
                            global_memory_id,
                            scratch_id,
                            GraphEdgeSemantics::transfers_to,
                            true,
                            1.0,
                            unified_memory ? 96.0 : 24.0,
                            unified_memory ? 8.0 : 18.0});
                        graph.edges.push_back(HardwareGraphEdge{
                            scratch_id,
                            pipeline_id,
                            GraphEdgeSemantics::feeds,
                            true,
                            1.0,
                            0.0,
                            0.0});
                    }

                    for (std::uint32_t lane_index = 0; lane_index < native_float_width; ++lane_index) {
                        const std::string lane_id = tile_id + "/lane/" + std::to_string(lane_index);
                        graph.nodes.push_back(HardwareObjectNode{
                            lane_id,
                            "lane-" + std::to_string(unit_index) + "-" + std::to_string(lane_index),
                            tile_id,
                            HardwareObjectDomain::compute,
                            HardwareObjectRole::lane,
                            HardwareObjectResolution::aggressive,
                            lane_index,
                            {1, 1, 0, clock_mhz, 32u, supports_fp64, true, supports_fp16, false, supports_int8},
                            {},
                            {},
                            {}});
                        graph.edges.push_back(HardwareGraphEdge{tile_id, lane_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
                        graph.edges.push_back(HardwareGraphEdge{pipeline_id, lane_id, GraphEdgeSemantics::feeds, true, 1.0, 0.0, 0.0});
                    }
                }

                materialize_graph_costs(graph);
                graphs.push_back(std::move(graph));
                ++ordinal;
            }
        }

        return graphs;
    }

private:
    OpenClApi api_;
};

}  // namespace

std::unique_ptr<IDeviceProbe> make_opencl_probe() {
    return std::make_unique<OpenClProbe>();
}

}  // namespace jakal

