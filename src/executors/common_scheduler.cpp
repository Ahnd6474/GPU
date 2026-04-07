#include "jakal/executors/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace jakal::executors {

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
    double total_ratio = 0.0;
    for (const auto& allocation : optimization.placement.allocations) {
        if (std::find(
                operation.config.participating_devices.begin(),
                operation.config.participating_devices.end(),
                allocation.device.uid) != operation.config.participating_devices.end()) {
            ratios[allocation.device.uid] = allocation.ratio;
            total_ratio += allocation.ratio;
        }
    }

    if (total_ratio <= 0.0) {
        total_ratio = static_cast<double>(std::max<std::size_t>(operation.config.participating_devices.size(), 1u));
        for (const auto& uid : operation.config.participating_devices) {
            ratios[uid] = 1.0;
        }
    }

    std::vector<DeviceAssignment> assignments;
    assignments.reserve(
        operation.config.participating_devices.size() *
        static_cast<std::size_t>(std::max(operation.config.logical_partitions, 1u)));
    std::size_t consumed = 0;

    for (std::size_t index = 0; index < operation.config.participating_devices.size(); ++index) {
        const auto& uid = operation.config.participating_devices[index];
        const auto graph_it = graph_lookup.find(uid);
        if (graph_it == graph_lookup.end()) {
            continue;
        }

        const double ratio = ratios.contains(uid) ? ratios.at(uid) / total_ratio : (1.0 / total_ratio);
        std::size_t count = 0;
        if (index + 1 == operation.config.participating_devices.size()) {
            count = total_items - consumed;
        } else {
            count = static_cast<std::size_t>(std::llround(static_cast<double>(total_items) * ratio));
            count = std::min(count, total_items - consumed);
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
            }

            assignments.push_back(DeviceAssignment{
                graph_it->second,
                ratio / static_cast<double>(partitions),
                {consumed + local_consumed, partition_count},
                partition,
                partitions});
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

