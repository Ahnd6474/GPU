#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace jakal {

enum class HardwareObjectDomain {
    compute,
    storage,
    transfer,
    control
};

enum class HardwareObjectRole {
    root,
    cluster,
    tile,
    lane,
    pipeline,
    global_memory,
    host_memory,
    cache,
    scratchpad,
    transfer_link,
    queue,
    scheduler,
    router
};

enum class HardwareObjectResolution {
    coarse,
    medium,
    aggressive
};

enum class GraphEdgeSemantics {
    contains,
    controls,
    dispatches,
    transfers_to,
    reads_from,
    writes_to,
    feeds,
    synchronizes_with
};

struct ComputeDomainInfo {
    std::uint32_t execution_width = 0;
    std::uint32_t resident_contexts = 0;
    std::uint32_t matrix_engines = 0;
    std::uint32_t clock_mhz = 0;
    std::uint32_t native_vector_bits = 0;
    bool supports_fp64 = false;
    bool supports_fp32 = true;
    bool supports_fp16 = false;
    bool supports_bf16 = false;
    bool supports_int8 = false;
};

struct StorageDomainInfo {
    std::uint64_t capacity_bytes = 0;
    std::uint64_t directly_attached_bytes = 0;
    std::uint64_t shared_host_bytes = 0;
    bool coherent_with_host = false;
    bool unified_address_space = false;
};

struct TransferDomainInfo {
    double read_bandwidth_gbps = 0.0;
    double write_bandwidth_gbps = 0.0;
    double dispatch_latency_us = 0.0;
    double synchronization_latency_us = 0.0;
};

struct ControlDomainInfo {
    std::uint32_t queue_slots = 0;
    std::uint32_t numa_domain = 0;
    bool host_visible = false;
    bool supports_asynchronous_dispatch = false;
};

struct HardwareObjectNode {
    std::string id;
    std::string label;
    std::string parent_id;
    HardwareObjectDomain domain = HardwareObjectDomain::control;
    HardwareObjectRole role = HardwareObjectRole::root;
    HardwareObjectResolution resolution = HardwareObjectResolution::coarse;
    std::uint32_t ordinal = 0;
    ComputeDomainInfo compute;
    StorageDomainInfo storage;
    TransferDomainInfo transfer;
    ControlDomainInfo control;
};

struct HardwareGraphEdge {
    std::string source_id;
    std::string target_id;
    GraphEdgeSemantics semantics = GraphEdgeSemantics::contains;
    bool directed = true;
    double weight = 0.0;
    double bandwidth_gbps = 0.0;
    double latency_us = 0.0;
};

struct HardwareGraph {
    std::string uid;
    std::string probe;
    std::string presentation_name;
    std::string driver_version;
    std::string runtime_version;
    std::string compiler_version;
    std::uint32_t ordinal = 0;
    std::vector<HardwareObjectNode> nodes;
    std::vector<HardwareGraphEdge> edges;
};

struct HardwareGraphSummary {
    std::uint32_t execution_objects = 0;
    std::uint32_t lanes_per_object = 0;
    std::uint32_t resident_contexts = 0;
    std::uint32_t matrix_units = 0;
    std::uint32_t clock_mhz = 0;
    std::uint32_t queue_slots = 0;
    std::uint32_t numa_domain = 0;
    std::uint32_t native_vector_bits = 0;
    std::uint64_t addressable_bytes = 0;
    std::uint64_t directly_attached_bytes = 0;
    std::uint64_t shared_host_bytes = 0;
    std::uint64_t local_scratch_bytes = 0;
    std::uint64_t cache_bytes = 0;
    double host_read_gbps = 0.0;
    double host_write_gbps = 0.0;
    double dispatch_latency_us = 0.0;
    double synchronization_latency_us = 0.0;
    double average_transfer_cost_us = 0.0;
    double average_dispatch_cost_us = 0.0;
    double average_feed_cost_us = 0.0;
    double average_hierarchy_cost_us = 0.0;
    bool coherent_with_host = false;
    bool unified_address_space = false;
    bool host_visible = false;
    bool supports_asynchronous_dispatch = false;
    bool supports_fp64 = false;
    bool supports_fp32 = false;
    bool supports_fp16 = false;
    bool supports_bf16 = false;
    bool supports_int8 = false;
};

[[nodiscard]] std::string to_string(HardwareObjectDomain domain);
[[nodiscard]] std::string to_string(HardwareObjectRole role);
[[nodiscard]] std::string to_string(HardwareObjectResolution resolution);
[[nodiscard]] std::string to_string(GraphEdgeSemantics semantics);
void materialize_graph_costs(HardwareGraph& graph);
[[nodiscard]] HardwareGraphSummary summarize_graph(const HardwareGraph& graph);
[[nodiscard]] std::string structural_fingerprint(const HardwareGraph& graph);

}  // namespace jakal

