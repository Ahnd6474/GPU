#include "jakal/device.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jakal {
namespace {

constexpr double kBytesPerGb = 1.0e9;
constexpr double kMicrosecondsPerSecond = 1.0e6;
constexpr double kCanonicalTransferBytes = 256.0 * 1024.0;
constexpr double kCanonicalFeedBytes = 16.0 * 1024.0;
constexpr double kDefaultOnChipBandwidthGbps = 512.0;

bool is_execution_node(const HardwareObjectNode& node) {
    return node.domain == HardwareObjectDomain::compute && node.role == HardwareObjectRole::tile;
}

bool is_lane_node(const HardwareObjectNode& node) {
    return node.domain == HardwareObjectDomain::compute && node.role == HardwareObjectRole::lane;
}

bool is_pipeline_node(const HardwareObjectNode& node) {
    return node.domain == HardwareObjectDomain::compute && node.role == HardwareObjectRole::pipeline;
}

bool is_latency_candidate(const HardwareObjectNode& node) {
    return node.domain == HardwareObjectDomain::transfer || node.domain == HardwareObjectDomain::control;
}

template <typename T>
T min_positive(const T current, const T candidate) {
    if (candidate <= static_cast<T>(0)) {
        return current;
    }
    if (current <= static_cast<T>(0)) {
        return candidate;
    }
    return std::min(current, candidate);
}

double transfer_time_us(const double bytes, const double bandwidth_gbps) {
    if (bytes <= 0.0 || bandwidth_gbps <= 0.0) {
        return 0.0;
    }
    return (bytes / (bandwidth_gbps * kBytesPerGb)) * kMicrosecondsPerSecond;
}

double default_bandwidth_for_feed(const HardwareObjectNode* source, const HardwareObjectNode* target) {
    if (source != nullptr) {
        if (source->role == HardwareObjectRole::scratchpad || source->role == HardwareObjectRole::cache) {
            return 1024.0;
        }
        if (source->role == HardwareObjectRole::global_memory || source->role == HardwareObjectRole::host_memory) {
            return 128.0;
        }
    }
    if (target != nullptr) {
        if (target->role == HardwareObjectRole::lane) {
            return 1024.0;
        }
        if (target->role == HardwareObjectRole::pipeline) {
            return 768.0;
        }
    }
    return kDefaultOnChipBandwidthGbps;
}

double average_or_zero(const double total, const std::size_t count) {
    return count == 0 ? 0.0 : total / static_cast<double>(count);
}

}  // namespace

std::string to_string(const HardwareObjectDomain domain) {
    switch (domain) {
    case HardwareObjectDomain::compute:
        return "compute";
    case HardwareObjectDomain::storage:
        return "storage";
    case HardwareObjectDomain::transfer:
        return "transfer";
    case HardwareObjectDomain::control:
    default:
        return "control";
    }
}

std::string to_string(const HardwareObjectRole role) {
    switch (role) {
    case HardwareObjectRole::root:
        return "root";
    case HardwareObjectRole::cluster:
        return "cluster";
    case HardwareObjectRole::tile:
        return "tile";
    case HardwareObjectRole::lane:
        return "lane";
    case HardwareObjectRole::pipeline:
        return "pipeline";
    case HardwareObjectRole::global_memory:
        return "global_memory";
    case HardwareObjectRole::host_memory:
        return "host_memory";
    case HardwareObjectRole::cache:
        return "cache";
    case HardwareObjectRole::scratchpad:
        return "scratchpad";
    case HardwareObjectRole::transfer_link:
        return "transfer_link";
    case HardwareObjectRole::queue:
        return "queue";
    case HardwareObjectRole::scheduler:
        return "scheduler";
    case HardwareObjectRole::router:
    default:
        return "router";
    }
}

std::string to_string(const HardwareObjectResolution resolution) {
    switch (resolution) {
    case HardwareObjectResolution::coarse:
        return "coarse";
    case HardwareObjectResolution::medium:
        return "medium";
    case HardwareObjectResolution::aggressive:
    default:
        return "aggressive";
    }
}

std::string to_string(const GraphEdgeSemantics semantics) {
    switch (semantics) {
    case GraphEdgeSemantics::contains:
        return "contains";
    case GraphEdgeSemantics::controls:
        return "controls";
    case GraphEdgeSemantics::dispatches:
        return "dispatches";
    case GraphEdgeSemantics::transfers_to:
        return "transfers_to";
    case GraphEdgeSemantics::reads_from:
        return "reads_from";
    case GraphEdgeSemantics::writes_to:
        return "writes_to";
    case GraphEdgeSemantics::feeds:
        return "feeds";
    case GraphEdgeSemantics::synchronizes_with:
    default:
        return "synchronizes_with";
    }
}

void materialize_graph_costs(HardwareGraph& graph) {
    std::unordered_map<std::string, const HardwareObjectNode*> node_lookup;
    node_lookup.reserve(graph.nodes.size());
    for (const auto& node : graph.nodes) {
        node_lookup.emplace(node.id, &node);
    }

    std::unordered_map<std::string, std::vector<std::string>> children;
    children.reserve(graph.nodes.size());
    for (const auto& node : graph.nodes) {
        if (!node.parent_id.empty()) {
            children[node.parent_id].push_back(node.id);
        }
    }

    std::unordered_map<std::string, std::vector<std::size_t>> outgoing_edges;
    outgoing_edges.reserve(graph.nodes.size());
    for (std::size_t edge_index = 0; edge_index < graph.edges.size(); ++edge_index) {
        outgoing_edges[graph.edges[edge_index].source_id].push_back(edge_index);
    }

    for (auto& edge : graph.edges) {
        const auto source_it = node_lookup.find(edge.source_id);
        const auto target_it = node_lookup.find(edge.target_id);
        const HardwareObjectNode* source = source_it == node_lookup.end() ? nullptr : source_it->second;
        const HardwareObjectNode* target = target_it == node_lookup.end() ? nullptr : target_it->second;

        switch (edge.semantics) {
        case GraphEdgeSemantics::transfers_to:
        case GraphEdgeSemantics::reads_from:
        case GraphEdgeSemantics::writes_to: {
            const double bandwidth = edge.bandwidth_gbps > 0.0 ? edge.bandwidth_gbps : 32.0;
            edge.weight = edge.latency_us + transfer_time_us(kCanonicalTransferBytes, bandwidth);
            break;
        }
        case GraphEdgeSemantics::dispatches:
            edge.weight = edge.latency_us > 0.0 ? edge.latency_us : 2.0;
            break;
        case GraphEdgeSemantics::feeds: {
            const double bandwidth =
                edge.bandwidth_gbps > 0.0 ? edge.bandwidth_gbps : default_bandwidth_for_feed(source, target);
            const double base_latency = edge.latency_us > 0.0 ? edge.latency_us : 0.10;
            edge.weight = base_latency + transfer_time_us(kCanonicalFeedBytes, bandwidth);
            break;
        }
        case GraphEdgeSemantics::controls:
            edge.weight = edge.latency_us > 0.0 ? edge.latency_us : 0.25;
            break;
        case GraphEdgeSemantics::synchronizes_with:
            edge.weight = edge.latency_us > 0.0 ? edge.latency_us : 0.50;
            break;
        case GraphEdgeSemantics::contains:
        default:
            edge.weight = 0.0;
            break;
        }
    }

    std::unordered_map<std::string, double> subtree_cost_cache;
    subtree_cost_cache.reserve(graph.nodes.size());

    std::function<double(const std::string&)> subtree_cost = [&](const std::string& node_id) -> double {
        if (const auto cached = subtree_cost_cache.find(node_id); cached != subtree_cost_cache.end()) {
            return cached->second;
        }

        double direct_total = 0.0;
        std::size_t direct_count = 0;
        if (const auto out_it = outgoing_edges.find(node_id); out_it != outgoing_edges.end()) {
            for (const auto edge_index : out_it->second) {
                const auto& edge = graph.edges[edge_index];
                if (edge.semantics == GraphEdgeSemantics::contains) {
                    continue;
                }
                direct_total += edge.weight;
                ++direct_count;
            }
        }

        double child_total = 0.0;
        std::size_t child_count = 0;
        if (const auto child_it = children.find(node_id); child_it != children.end()) {
            for (const auto& child_id : child_it->second) {
                child_total += subtree_cost(child_id);
                ++child_count;
            }
        }

        const double direct_component = average_or_zero(direct_total, direct_count);
        const double child_component = average_or_zero(child_total, child_count);
        const double aggregate_cost = direct_component + (child_component * 0.35);
        subtree_cost_cache.emplace(node_id, aggregate_cost);
        return aggregate_cost;
    };

    for (auto& edge : graph.edges) {
        const double target_subtree_cost = subtree_cost(edge.target_id);
        const double source_subtree_cost = subtree_cost(edge.source_id);

        switch (edge.semantics) {
        case GraphEdgeSemantics::contains:
            edge.weight = target_subtree_cost;
            break;
        case GraphEdgeSemantics::controls:
            edge.weight += target_subtree_cost * 0.20;
            break;
        case GraphEdgeSemantics::dispatches:
            edge.weight += target_subtree_cost * 0.10;
            break;
        case GraphEdgeSemantics::feeds:
            edge.weight += (source_subtree_cost + target_subtree_cost) * 0.05;
            break;
        case GraphEdgeSemantics::transfers_to:
        case GraphEdgeSemantics::reads_from:
        case GraphEdgeSemantics::writes_to:
            edge.weight += target_subtree_cost * 0.03;
            break;
        case GraphEdgeSemantics::synchronizes_with:
            edge.weight += (source_subtree_cost + target_subtree_cost) * 0.10;
            break;
        default:
            break;
        }
    }
}

HardwareGraphSummary summarize_graph(const HardwareGraph& graph) {
    HardwareGraphSummary summary;
    bool has_numa_domain = false;

    std::unordered_map<std::string, std::uint32_t> lanes_per_parent;
    for (const auto& node : graph.nodes) {
        if (is_lane_node(node) && !node.parent_id.empty()) {
            ++lanes_per_parent[node.parent_id];
        }
    }

    std::uint32_t fallback_compute_nodes = 0;
    std::uint32_t fallback_lanes_per_object = 0;
    std::uint32_t fallback_resident_contexts = 0;
    std::uint32_t fallback_matrix_units = 0;
    double transfer_cost_total = 0.0;
    double dispatch_cost_total = 0.0;
    double feed_cost_total = 0.0;
    double hierarchy_cost_total = 0.0;
    std::size_t transfer_cost_count = 0;
    std::size_t dispatch_cost_count = 0;
    std::size_t feed_cost_count = 0;
    std::size_t hierarchy_cost_count = 0;

    for (const auto& node : graph.nodes) {
        if (node.domain == HardwareObjectDomain::compute) {
            summary.clock_mhz = std::max(summary.clock_mhz, node.compute.clock_mhz);
            summary.native_vector_bits = std::max(summary.native_vector_bits, node.compute.native_vector_bits);
            summary.supports_fp64 = summary.supports_fp64 || node.compute.supports_fp64;
            summary.supports_fp32 = summary.supports_fp32 || node.compute.supports_fp32;
            summary.supports_fp16 = summary.supports_fp16 || node.compute.supports_fp16;
            summary.supports_bf16 = summary.supports_bf16 || node.compute.supports_bf16;
            summary.supports_int8 = summary.supports_int8 || node.compute.supports_int8;

            if (is_execution_node(node)) {
                ++summary.execution_objects;
                summary.resident_contexts += std::max(node.compute.resident_contexts, 1u);
                summary.matrix_units += node.compute.matrix_engines;
                summary.lanes_per_object = std::max(
                    summary.lanes_per_object,
                    std::max(lanes_per_parent[node.id], node.compute.execution_width));
            } else if (node.role == HardwareObjectRole::cluster) {
                ++fallback_compute_nodes;
                fallback_resident_contexts += std::max(node.compute.resident_contexts, 1u);
                fallback_lanes_per_object = std::max(fallback_lanes_per_object, node.compute.execution_width);
                fallback_matrix_units += node.compute.matrix_engines;
            } else if (is_pipeline_node(node) && node.compute.matrix_engines > 0) {
                summary.matrix_units += node.compute.matrix_engines;
            }
        }

        if (node.domain == HardwareObjectDomain::storage) {
            summary.coherent_with_host = summary.coherent_with_host || node.storage.coherent_with_host;
            summary.unified_address_space = summary.unified_address_space || node.storage.unified_address_space;

            if (node.role == HardwareObjectRole::global_memory) {
                summary.addressable_bytes += node.storage.capacity_bytes;
                summary.directly_attached_bytes += node.storage.directly_attached_bytes;
                summary.shared_host_bytes += node.storage.shared_host_bytes;
            } else if (node.role == HardwareObjectRole::host_memory) {
                if (summary.addressable_bytes == 0) {
                    summary.addressable_bytes += node.storage.capacity_bytes;
                    summary.shared_host_bytes += node.storage.shared_host_bytes;
                }
            } else if (node.role == HardwareObjectRole::scratchpad) {
                summary.local_scratch_bytes += node.storage.capacity_bytes;
            } else if (node.role == HardwareObjectRole::cache) {
                summary.cache_bytes += node.storage.capacity_bytes;
            }
        }

        if (node.domain == HardwareObjectDomain::transfer) {
            summary.host_read_gbps = std::max(summary.host_read_gbps, node.transfer.read_bandwidth_gbps);
            summary.host_write_gbps = std::max(summary.host_write_gbps, node.transfer.write_bandwidth_gbps);
        }

        if (is_latency_candidate(node)) {
            summary.dispatch_latency_us = min_positive(summary.dispatch_latency_us, node.transfer.dispatch_latency_us);
            summary.synchronization_latency_us =
                min_positive(summary.synchronization_latency_us, node.transfer.synchronization_latency_us);
        }

        if (node.domain == HardwareObjectDomain::control) {
            summary.queue_slots = std::max(summary.queue_slots, node.control.queue_slots);
            if (!has_numa_domain) {
                summary.numa_domain = node.control.numa_domain;
                has_numa_domain = true;
            } else {
                summary.numa_domain = std::min(summary.numa_domain, node.control.numa_domain);
            }
            summary.host_visible = summary.host_visible || node.control.host_visible;
            summary.supports_asynchronous_dispatch =
                summary.supports_asynchronous_dispatch || node.control.supports_asynchronous_dispatch;
        }
    }

    for (const auto& edge : graph.edges) {
        switch (edge.semantics) {
        case GraphEdgeSemantics::transfers_to:
        case GraphEdgeSemantics::reads_from:
        case GraphEdgeSemantics::writes_to:
            if (edge.weight > 0.0) {
                transfer_cost_total += edge.weight;
                ++transfer_cost_count;
            }
            break;
        case GraphEdgeSemantics::dispatches:
            if (edge.weight > 0.0) {
                dispatch_cost_total += edge.weight;
                ++dispatch_cost_count;
                summary.dispatch_latency_us = min_positive(summary.dispatch_latency_us, edge.weight);
            }
            break;
        case GraphEdgeSemantics::feeds:
            if (edge.weight > 0.0) {
                feed_cost_total += edge.weight;
                ++feed_cost_count;
            }
            break;
        case GraphEdgeSemantics::contains:
        case GraphEdgeSemantics::controls:
            if (edge.weight > 0.0) {
                hierarchy_cost_total += edge.weight;
                ++hierarchy_cost_count;
            }
            break;
        case GraphEdgeSemantics::synchronizes_with:
            if (edge.weight > 0.0) {
                summary.synchronization_latency_us = min_positive(summary.synchronization_latency_us, edge.weight);
            }
            break;
        default:
            break;
        }
    }

    summary.average_transfer_cost_us = average_or_zero(transfer_cost_total, transfer_cost_count);
    summary.average_dispatch_cost_us = average_or_zero(dispatch_cost_total, dispatch_cost_count);
    summary.average_feed_cost_us = average_or_zero(feed_cost_total, feed_cost_count);
    summary.average_hierarchy_cost_us = average_or_zero(hierarchy_cost_total, hierarchy_cost_count);

    if (summary.execution_objects == 0 && fallback_compute_nodes > 0) {
        summary.execution_objects = fallback_compute_nodes;
        summary.resident_contexts = std::max(summary.resident_contexts, fallback_resident_contexts);
        summary.lanes_per_object = std::max(summary.lanes_per_object, fallback_lanes_per_object);
        summary.matrix_units = std::max(summary.matrix_units, fallback_matrix_units);
    }

    if (summary.lanes_per_object == 0 && summary.native_vector_bits > 0) {
        summary.lanes_per_object = std::max(1u, summary.native_vector_bits / 32u);
    }

    if (summary.dispatch_latency_us <= 0.0) {
        summary.dispatch_latency_us = summary.host_visible ? 2.0 : 24.0;
    }
    if (summary.synchronization_latency_us <= 0.0) {
        summary.synchronization_latency_us = summary.host_visible ? 1.0 : 16.0;
    }

    return summary;
}

std::string structural_fingerprint(const HardwareGraph& graph) {
    const auto summary = summarize_graph(graph);

    std::ostringstream stream;
    stream << summary.execution_objects << ':'
           << summary.lanes_per_object << ':'
           << summary.resident_contexts << ':'
           << summary.matrix_units << ':'
           << summary.clock_mhz << ':'
           << summary.queue_slots << ':'
           << summary.numa_domain << ':'
           << summary.native_vector_bits << ':'
           << summary.addressable_bytes << ':'
           << summary.directly_attached_bytes << ':'
           << summary.shared_host_bytes << ':'
           << summary.local_scratch_bytes << ':'
           << summary.cache_bytes << ':'
           << summary.host_read_gbps << ':'
           << summary.host_write_gbps << ':'
           << summary.dispatch_latency_us << ':'
           << summary.synchronization_latency_us << ':'
           << summary.average_transfer_cost_us << ':'
           << summary.average_dispatch_cost_us << ':'
           << summary.average_feed_cost_us << ':'
           << summary.average_hierarchy_cost_us << ':'
           << summary.coherent_with_host << ':'
           << summary.unified_address_space << ':'
           << summary.host_visible << ':'
           << summary.supports_asynchronous_dispatch << ':'
           << summary.supports_fp64 << ':'
           << summary.supports_fp32 << ':'
           << summary.supports_fp16 << ':'
           << summary.supports_bf16 << ':'
           << summary.supports_int8 << ':'
           << graph.nodes.size() << ':'
           << graph.edges.size();
    return stream.str();
}

}  // namespace jakal

