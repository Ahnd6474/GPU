#include "jakal/backend.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <intrin.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

#if !defined(_WIN32) && (defined(__i386__) || defined(__x86_64__))
#include <cpuid.h>
#endif

namespace jakal {
namespace {

struct HostStructuralSeed {
    std::string brand = "host-execution-domain";
    std::uint32_t native_vector_bits = 128;
    bool supports_fp64 = true;
    bool supports_fp16 = false;
    bool supports_bf16 = false;
    bool supports_int8 = false;
};

std::string trim_copy(std::string value) {
    auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(), std::find_if_not(value.begin(), value.end(), is_space));
    value.erase(std::find_if_not(value.rbegin(), value.rend(), is_space).base(), value.end());
    return value;
}

HostStructuralSeed detect_host_seed() {
    HostStructuralSeed seed;

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    int registers[4] = {};
    __cpuid(registers, 0x80000000);
    const int max_extended_leaf = registers[0];
    if (max_extended_leaf >= 0x80000004) {
        int brand[12] = {};
        __cpuid(brand + 0, 0x80000002);
        __cpuid(brand + 4, 0x80000003);
        __cpuid(brand + 8, 0x80000004);

        char name[49] = {};
        std::memcpy(name, brand, sizeof(brand));
        seed.brand = trim_copy(name);
    }

    __cpuid(registers, 1);
    const bool has_sse = (registers[3] & (1 << 25)) != 0;
    const bool has_avx = (registers[2] & (1 << 28)) != 0;
    const bool has_f16c = (registers[2] & (1 << 29)) != 0;

    seed.native_vector_bits = has_avx ? 256u : has_sse ? 128u : 64u;
    seed.supports_fp16 = has_f16c;

    __cpuidex(registers, 7, 0);
    const bool has_avx2 = (registers[1] & (1 << 5)) != 0;
    const bool has_avx512f = (registers[1] & (1 << 16)) != 0;
    const bool has_avx512_bf16 = false;

    if (has_avx512f) {
        seed.native_vector_bits = 512u;
    } else if (has_avx2) {
        seed.native_vector_bits = 256u;
    }

    seed.supports_bf16 = has_avx512_bf16;
    seed.supports_int8 = has_avx2 || has_avx512f;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;

    if (__get_cpuid_max(0x80000000, nullptr) >= 0x80000004) {
        unsigned int brand[12] = {};
        __get_cpuid(0x80000002, &brand[0], &brand[1], &brand[2], &brand[3]);
        __get_cpuid(0x80000003, &brand[4], &brand[5], &brand[6], &brand[7]);
        __get_cpuid(0x80000004, &brand[8], &brand[9], &brand[10], &brand[11]);

        char name[49] = {};
        std::memcpy(name, brand, sizeof(brand));
        seed.brand = trim_copy(name);
    }

    if (__get_cpuid_max(0, nullptr) >= 1) {
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        const bool has_sse = (edx & (1u << 25u)) != 0;
        const bool has_avx = (ecx & (1u << 28u)) != 0;
        const bool has_f16c = (ecx & (1u << 29u)) != 0;

        seed.native_vector_bits = has_avx ? 256u : has_sse ? 128u : 64u;
        seed.supports_fp16 = has_f16c;
    }

    if (__get_cpuid_max(0, nullptr) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        const bool has_avx2 = (ebx & (1u << 5u)) != 0;
        const bool has_avx512f = (ebx & (1u << 16u)) != 0;

        if (has_avx512f) {
            seed.native_vector_bits = 512u;
        } else if (has_avx2) {
            seed.native_vector_bits = 256u;
        }

        seed.supports_int8 = has_avx2 || has_avx512f;
    }
#endif

    return seed;
}

std::uint64_t detect_total_memory() {
#if defined(_WIN32)
    MEMORYSTATUSEX status = {};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        return status.ullTotalPhys;
    }
#elif defined(__APPLE__)
    std::uint64_t memory = 0;
    std::size_t size = sizeof(memory);
    if (sysctlbyname("hw.memsize", &memory, &size, nullptr, 0) == 0) {
        return memory;
    }
#elif defined(__linux__)
    struct sysinfo info = {};
    if (sysinfo(&info) == 0) {
        return static_cast<std::uint64_t>(info.totalram) * static_cast<std::uint64_t>(info.mem_unit);
    }
#endif
    return 0;
}

class HostProbe final : public IDeviceProbe {
public:
    [[nodiscard]] std::string name() const override {
        return "host";
    }

    [[nodiscard]] bool available() const override {
        return true;
    }

    std::vector<HardwareGraph> discover_hardware() override {
        const auto seed = detect_host_seed();
        const auto logical_cores = std::max(1u, std::thread::hardware_concurrency());
        const auto total_memory = detect_total_memory();
        const auto lane_count = std::max(1u, seed.native_vector_bits / 32u);

        HardwareGraph graph;
        graph.uid = "host:0";
        graph.probe = "host";
        graph.presentation_name = seed.brand;
        graph.ordinal = 0;

        const std::string root_id = graph.uid + "/root";
        const std::string scheduler_id = graph.uid + "/control/scheduler";
        const std::string queue_id = graph.uid + "/control/queue";
        const std::string cluster_id = graph.uid + "/compute/cluster/0";
        const std::string memory_id = graph.uid + "/storage/host-memory";
        const std::string fabric_id = graph.uid + "/transfer/coherent-fabric";

        graph.nodes.push_back(HardwareObjectNode{
            root_id,
            "host-root",
            "",
            HardwareObjectDomain::control,
            HardwareObjectRole::root,
            HardwareObjectResolution::coarse,
            0,
            {},
            {},
            {},
            {0, 0, true, true}});
        graph.nodes.push_back(HardwareObjectNode{
            scheduler_id,
            "host-scheduler",
            root_id,
            HardwareObjectDomain::control,
            HardwareObjectRole::scheduler,
            HardwareObjectResolution::medium,
            0,
            {},
            {},
            {},
            {logical_cores, 0, true, true}});
        graph.nodes.push_back(HardwareObjectNode{
            queue_id,
            "dispatch-queue",
            root_id,
            HardwareObjectDomain::control,
            HardwareObjectRole::queue,
            HardwareObjectResolution::medium,
            0,
            {},
            {},
            {},
            {logical_cores, 0, true, true}});
        graph.nodes.push_back(HardwareObjectNode{
            cluster_id,
            "core-cluster",
            root_id,
            HardwareObjectDomain::compute,
            HardwareObjectRole::cluster,
            HardwareObjectResolution::medium,
            0,
            {logical_cores, logical_cores, 0, 0, seed.native_vector_bits, seed.supports_fp64, true, seed.supports_fp16, seed.supports_bf16, seed.supports_int8},
            {},
            {},
            {}});
        graph.nodes.push_back(HardwareObjectNode{
            memory_id,
            "host-memory",
            root_id,
            HardwareObjectDomain::storage,
            HardwareObjectRole::host_memory,
            HardwareObjectResolution::medium,
            0,
            {},
            {total_memory, 0, total_memory, true, true},
            {},
            {}});
        graph.nodes.push_back(HardwareObjectNode{
            fabric_id,
            "coherent-fabric",
            root_id,
            HardwareObjectDomain::transfer,
            HardwareObjectRole::transfer_link,
            HardwareObjectResolution::medium,
            0,
            {},
            {},
            {256.0, 256.0, 2.0, 1.0},
            {}});

        graph.edges.push_back(HardwareGraphEdge{root_id, scheduler_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
        graph.edges.push_back(HardwareGraphEdge{root_id, queue_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
        graph.edges.push_back(HardwareGraphEdge{root_id, cluster_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
        graph.edges.push_back(HardwareGraphEdge{root_id, memory_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
        graph.edges.push_back(HardwareGraphEdge{root_id, fabric_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
        graph.edges.push_back(HardwareGraphEdge{scheduler_id, queue_id, GraphEdgeSemantics::controls, true, 1.0, 0.0, 0.0});
        graph.edges.push_back(HardwareGraphEdge{memory_id, fabric_id, GraphEdgeSemantics::transfers_to, true, 256.0, 256.0, 1.0});
        graph.edges.push_back(HardwareGraphEdge{fabric_id, memory_id, GraphEdgeSemantics::writes_to, true, 256.0, 256.0, 1.0});

        for (std::uint32_t core_index = 0; core_index < logical_cores; ++core_index) {
            const std::string tile_id = graph.uid + "/compute/tile/" + std::to_string(core_index);
            const std::string pipeline_id = tile_id + "/pipeline";

            graph.nodes.push_back(HardwareObjectNode{
                tile_id,
                "core-tile-" + std::to_string(core_index),
                cluster_id,
                HardwareObjectDomain::compute,
                HardwareObjectRole::tile,
                HardwareObjectResolution::medium,
                core_index,
                {lane_count, 1, 0, 0, seed.native_vector_bits, seed.supports_fp64, true, seed.supports_fp16, seed.supports_bf16, seed.supports_int8},
                {},
                {},
                {}});
            graph.nodes.push_back(HardwareObjectNode{
                pipeline_id,
                "vector-pipeline-" + std::to_string(core_index),
                tile_id,
                HardwareObjectDomain::compute,
                HardwareObjectRole::pipeline,
                HardwareObjectResolution::aggressive,
                core_index,
                {lane_count, 1, 0, 0, seed.native_vector_bits, seed.supports_fp64, true, seed.supports_fp16, seed.supports_bf16, seed.supports_int8},
                {},
                {},
                {}});

            graph.edges.push_back(HardwareGraphEdge{cluster_id, tile_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
            graph.edges.push_back(HardwareGraphEdge{tile_id, pipeline_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
            graph.edges.push_back(HardwareGraphEdge{queue_id, pipeline_id, GraphEdgeSemantics::dispatches, true, 1.0, 0.0, 2.0});
            graph.edges.push_back(HardwareGraphEdge{
                fabric_id,
                pipeline_id,
                GraphEdgeSemantics::transfers_to,
                true,
                1.0,
                256.0 / static_cast<double>(logical_cores),
                2.0});
            graph.edges.push_back(HardwareGraphEdge{
                pipeline_id,
                fabric_id,
                GraphEdgeSemantics::writes_to,
                true,
                1.0,
                256.0 / static_cast<double>(logical_cores),
                1.0});

            for (std::uint32_t lane_index = 0; lane_index < lane_count; ++lane_index) {
                const std::string lane_id = tile_id + "/lane/" + std::to_string(lane_index);
                graph.nodes.push_back(HardwareObjectNode{
                    lane_id,
                    "lane-" + std::to_string(core_index) + "-" + std::to_string(lane_index),
                    tile_id,
                    HardwareObjectDomain::compute,
                    HardwareObjectRole::lane,
                    HardwareObjectResolution::aggressive,
                    lane_index,
                    {1, 1, 0, 0, 32u, seed.supports_fp64, true, seed.supports_fp16, seed.supports_bf16, seed.supports_int8},
                    {},
                    {},
                    {}});
                graph.edges.push_back(HardwareGraphEdge{tile_id, lane_id, GraphEdgeSemantics::contains, true, 1.0, 0.0, 0.0});
                graph.edges.push_back(HardwareGraphEdge{pipeline_id, lane_id, GraphEdgeSemantics::feeds, true, 1.0, 0.0, 0.0});
            }
        }

        materialize_graph_costs(graph);
        return {graph};
    }
};

}  // namespace

std::unique_ptr<IDeviceProbe> make_host_probe() {
    return std::make_unique<HostProbe>();
}

}  // namespace jakal

