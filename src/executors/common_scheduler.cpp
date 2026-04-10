#include "jakal/executors/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace jakal::executors {

namespace {

std::uint32_t head_group_count(const OperationOptimizationResult& operation);

double log_bucket(const double value) {
    return value <= 0.0 ? 0.0 : std::log2(value + 1.0);
}

std::size_t semantic_item_quantum(
    const OperationOptimizationResult& operation,
    const std::size_t total_items) {
    std::size_t quantum = 1u;
    if (operation.config.semantic_token_partitioning &&
        operation.operation.preferred_token_block > 0u) {
        quantum = std::max<std::size_t>(
            quantum,
            static_cast<std::size_t>(operation.operation.preferred_token_block));
    }
    if (operation.config.semantic_head_partitioning &&
        operation.operation.attention_head_count > 0u) {
        const auto groups = std::max<std::uint32_t>(1u, head_group_count(operation));
        const auto head_group_span = std::max<std::size_t>(1u, total_items / static_cast<std::size_t>(groups));
        if (head_group_span > 1u && head_group_span < total_items) {
            quantum = std::max(quantum, head_group_span);
        }
    }
    return std::max<std::size_t>(1u, quantum);
}

std::size_t align_semantic_count(
    const std::size_t candidate,
    const std::size_t quantum,
    const std::size_t remaining,
    const bool final_device) {
    if (final_device || quantum <= 1u || remaining <= quantum) {
        return remaining;
    }
    std::size_t aligned = (candidate / quantum) * quantum;
    if (aligned == 0u && remaining >= quantum) {
        aligned = quantum;
    }
    if (aligned >= remaining) {
        aligned = (remaining / quantum) * quantum;
        if (aligned == 0u) {
            aligned = std::min(remaining, quantum);
        }
    }
    return std::min(aligned, remaining);
}

double inference_bias(const OperationOptimizationResult& operation, const HardwareGraph& graph) {
    const auto summary = summarize_graph(graph);
    const double execution_width =
        static_cast<double>(std::max(summary.execution_objects, 1u)) *
        static_cast<double>(std::max(summary.lanes_per_object, 1u));
    const double clock_scale = static_cast<double>(std::max(summary.clock_mhz, 1u)) / 1000.0;
    double bias = 1.0 + std::min(log_bucket((execution_width * clock_scale) / 16.0) * 0.10, 0.40);

    if (operation.operation.matrix_friendly || operation.operation.op_class == OperationClass::matmul) {
        bias += std::min(log_bucket(static_cast<double>(summary.matrix_units) + 1.0) * 0.12, 0.55);
        bias += std::min(log_bucket(static_cast<double>(summary.cache_bytes) / (256.0 * 1024.0)) * 0.08, 0.30);
        bias += std::min(log_bucket(static_cast<double>(summary.local_scratch_bytes) / (32.0 * 1024.0)) * 0.08, 0.24);
        if (operation.config.use_low_precision) {
            bias += (summary.supports_fp16 || summary.supports_bf16 || summary.supports_int8) ? 0.10 : -0.08;
        }
    } else if (operation.operation.op_class == OperationClass::reduction) {
        bias += std::min(log_bucket(static_cast<double>(summary.cache_bytes) / (128.0 * 1024.0)) * 0.07, 0.24);
    } else if (operation.operation.streaming_friendly || operation.operation.op_class == OperationClass::resample_2d) {
        const double transfer_gbps = std::max(summary.host_read_gbps, summary.host_write_gbps);
        bias += std::min(log_bucket(transfer_gbps / 16.0) * 0.08, 0.25);
    }

    if (summary.supports_asynchronous_dispatch && operation.config.overlap_transfers) {
        bias += 0.08;
    }
    if (operation.config.telemetry_staging_hit_rate > 0.0) {
        const double staging = std::clamp(operation.config.telemetry_staging_hit_rate, 0.0, 1.0);
        if (graph.probe == "host") {
            bias += (1.0 - staging) * 0.12;
        } else {
            bias += staging * 0.10;
        }
    }
    if (operation.config.telemetry_cross_device_sync_cost_us > 0.0 &&
        operation.config.participating_devices.size() > 1u) {
        const double sync_penalty =
            std::clamp(operation.config.telemetry_cross_device_sync_cost_us / 200.0, 0.0, 1.0);
        if (graph.probe == "host") {
            bias += sync_penalty * 0.10;
        } else {
            bias -= sync_penalty * 0.12;
        }
    }
    if (operation.config.telemetry_residency_pressure > 0.0) {
        const double pressure = std::clamp(operation.config.telemetry_residency_pressure, 0.0, 1.5);
        if (graph.probe == "host" && summary.coherent_with_host) {
            bias += pressure * 0.05;
        } else if (graph.probe != "host") {
            bias -= pressure * 0.08;
        }
    }
    bias += std::min(std::log2(static_cast<double>(std::max(operation.config.queue_depth, 1u))) * 0.035, 0.10);
    bias += std::min(std::log2(static_cast<double>(std::max(operation.config.stages, 1u))) * 0.030, 0.08);
    if (operation.config.logical_partitions > 1u && operation.operation.parallelizable) {
        bias += std::min(
            std::log2(static_cast<double>(operation.config.logical_partitions)) * 0.025,
            0.08);
    }
    if (operation.graph.predicted_memory_pressure > 0.0) {
        const double pressure = std::clamp(operation.graph.predicted_memory_pressure, 0.0, 2.0);
        if (graph.probe == "host") {
            bias += pressure < 0.85 ? 0.04 : -0.05;
        } else {
            bias -= std::clamp((pressure - 0.80) * 0.08, 0.0, 0.12);
        }
    }
    if (operation.operation.attention_head_count > 0u) {
        const double head_scale =
            std::log2(static_cast<double>(std::max(operation.operation.attention_head_count, 1u)) + 1.0);
        if (graph.probe == "host") {
            bias += operation.operation.attention_head_group_size <= 4u
                        ? std::min(head_scale * 0.04, 0.18)
                        : std::min(head_scale * 0.02, 0.10);
            if (operation.operation.preferred_kv_residency == "host" ||
                operation.operation.preferred_kv_residency == "shared") {
                bias += 0.10;
            }
        } else {
            bias += std::min(head_scale * 0.05, 0.24);
            if (operation.operation.preferred_token_block >= 64u) {
                bias += 0.08;
            }
            if (operation.operation.preferred_kv_residency == "accelerator" ||
                operation.operation.preferred_kv_residency == "shared") {
                bias += 0.06;
            }
        }
    }
    if (operation.operation.preferred_kv_residency == "shared") {
        bias += summary.coherent_with_host ? 0.07 : -0.04;
        bias += summary.unified_address_space ? 0.04 : 0.0;
    }
    if (operation.operation.residency_sensitive_fusion) {
        if (graph.probe == "host" &&
            (operation.operation.preferred_kv_residency == "host" ||
             operation.operation.preferred_kv_residency == "shared")) {
            bias += 0.08;
        } else if (graph.probe != "host" &&
                   operation.operation.preferred_kv_residency == "accelerator") {
            bias += 0.06;
        }
    }
    if ((operation.operation.input_bytes + operation.operation.output_bytes) >= (16ull * 1024ull * 1024ull) &&
        !summary.coherent_with_host &&
        !summary.unified_address_space) {
        bias -= 0.08;
    }
    if (graph.probe == "host" && (operation.operation.matrix_friendly || operation.operation.op_class == OperationClass::matmul)) {
        bias -= 0.10;
    }

    return std::clamp(bias, 0.45, 2.50);
}

std::uint32_t head_group_count(const OperationOptimizationResult& operation) {
    if (operation.operation.attention_head_count == 0u) {
        return 1u;
    }
    const auto group_size = std::max(operation.operation.attention_head_group_size, 1u);
    return std::max<std::uint32_t>(1u, (operation.operation.attention_head_count + group_size - 1u) / group_size);
}

std::uint32_t token_block_count(
    const OperationOptimizationResult& operation,
    const std::size_t total_items) {
    if (operation.operation.preferred_token_block == 0u || total_items == 0u) {
        return 1u;
    }
    return std::max<std::uint32_t>(
        1u,
        static_cast<std::uint32_t>(
            (total_items + static_cast<std::size_t>(operation.operation.preferred_token_block) - 1u) /
            static_cast<std::size_t>(operation.operation.preferred_token_block)));
}

}  // namespace

std::vector<DeviceAssignment> DefaultIntraDeviceScheduler::make_assignments(
    const OptimizationReport& optimization,
    const OperationOptimizationResult& operation,
    const std::vector<HardwareGraph>& graphs,
    const std::size_t total_items) const {
    std::unordered_map<std::string, const HardwareGraph*> graph_lookup;
    for (const auto& graph : graphs) {
        graph_lookup.emplace(graph.uid, &graph);
    }

    std::unordered_map<std::string, double> ratios;
    for (const auto& allocation : optimization.placement.allocations) {
        if (std::find(
                operation.config.participating_devices.begin(),
                operation.config.participating_devices.end(),
                allocation.device.uid) != operation.config.participating_devices.end()) {
            ratios[allocation.device.uid] = allocation.ratio;
        }
    }

    if (ratios.empty()) {
        for (const auto& uid : operation.config.participating_devices) {
            ratios[uid] = 1.0;
        }
    }

    std::unordered_map<std::string, double> weighted_ratios;
    double total_ratio = 0.0;
    for (const auto& uid : operation.config.participating_devices) {
        const auto graph_it = graph_lookup.find(uid);
        if (graph_it == graph_lookup.end()) {
            continue;
        }
        const double base_ratio = ratios.contains(uid) ? ratios.at(uid) : 1.0;
        const double weighted_ratio = base_ratio * inference_bias(operation, *graph_it->second);
        weighted_ratios[uid] = weighted_ratio;
        total_ratio += weighted_ratio;
    }

    if (total_ratio <= 0.0) {
        total_ratio = static_cast<double>(std::max<std::size_t>(operation.config.participating_devices.size(), 1u));
        for (const auto& uid : operation.config.participating_devices) {
            weighted_ratios[uid] = 1.0;
        }
    }

    std::vector<DeviceAssignment> assignments;
    assignments.reserve(
        operation.config.participating_devices.size() *
        static_cast<std::size_t>(std::max(operation.config.logical_partitions, 1u)));
    const auto semantic_head_groups = head_group_count(operation);
    const auto semantic_token_blocks = token_block_count(operation, total_items);
    const auto semantic_quantum = semantic_item_quantum(operation, total_items);
    std::size_t consumed = 0;

    for (std::size_t index = 0; index < operation.config.participating_devices.size(); ++index) {
        const auto& uid = operation.config.participating_devices[index];
        const auto graph_it = graph_lookup.find(uid);
        if (graph_it == graph_lookup.end()) {
            continue;
        }

        const double ratio = weighted_ratios.contains(uid) ? weighted_ratios.at(uid) / total_ratio : (1.0 / total_ratio);
        std::size_t count = 0;
        if (index + 1 == operation.config.participating_devices.size()) {
            count = total_items - consumed;
        } else {
            count = static_cast<std::size_t>(std::llround(static_cast<double>(total_items) * ratio));
            count = std::min(count, total_items - consumed);
            count = align_semantic_count(count, semantic_quantum, total_items - consumed, false);
        }

        const auto partitions = std::max(operation.config.logical_partitions, 1u);
        std::size_t local_consumed = 0;
        for (std::uint32_t partition = 0; partition < partitions; ++partition) {
            std::size_t partition_count = 0;
            if (partition + 1 == partitions) {
                partition_count = count - local_consumed;
            } else {
                partition_count = static_cast<std::size_t>(
                    std::llround(static_cast<double>(count) / static_cast<double>(partitions)));
                partition_count = std::min(partition_count, count - local_consumed);
                partition_count = align_semantic_count(
                    partition_count,
                    semantic_quantum,
                    count - local_consumed,
                    false);
            }

            assignments.push_back(DeviceAssignment{
                graph_it->second,
                ratio / static_cast<double>(partitions),
                {consumed + local_consumed, partition_count},
                partition,
                partitions,
                !operation.config.semantic_head_partitioning || semantic_head_groups == 0u
                    ? 0u
                    : static_cast<std::uint32_t>(
                          ((consumed + local_consumed) / std::max<std::size_t>(semantic_quantum, 1u)) %
                          semantic_head_groups),
                semantic_head_groups,
                !operation.config.semantic_token_partitioning || operation.operation.preferred_token_block == 0u
                    ? 0u
                    : static_cast<std::uint32_t>(
                          (consumed + local_consumed) /
                          static_cast<std::size_t>(operation.operation.preferred_token_block)),
                semantic_token_blocks});
            local_consumed += partition_count;
        }

        consumed += count;
    }

    if (!assignments.empty() && consumed < total_items) {
        assignments.back().shard.count += total_items - consumed;
    }

    assignments.erase(
        std::remove_if(assignments.begin(), assignments.end(), [](const DeviceAssignment& assignment) {
            return assignment.graph == nullptr || assignment.shard.count == 0;
        }),
        assignments.end());

    return assignments;
}

}  // namespace jakal::executors

