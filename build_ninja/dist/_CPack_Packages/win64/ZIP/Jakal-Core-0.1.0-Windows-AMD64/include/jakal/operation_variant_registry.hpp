#pragma once

#include "jakal/execution.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace jakal {

enum class OperationVariantScope {
    per_allocation,
    placement_sharded,
    accelerator_sharded
};

struct OperationVariantRequest {
    const WorkloadSpec& workload;
    const OperationSpec& operation;
    std::size_t placement_device_count = 0;
    std::size_t accelerator_device_count = 0;
    bool low_spec_mode = false;
    bool activation_driven = false;
    bool tolerates_low_precision = false;
    bool allow_placement_sharding = false;
    bool allow_accelerator_sharding = false;
};

struct OperationVariantSpec {
    std::string id;
    std::string description;
    OperationVariantScope scope = OperationVariantScope::per_allocation;
    std::vector<OperationClass> applicable_classes;
    ExecutionStrategy strategy = ExecutionStrategy::single_device;
    bool overlap_transfers = false;
    bool use_low_precision = false;
    bool requires_parallelizable = false;
    bool requires_streaming_friendly = false;
    bool requires_low_precision_tolerance = false;
    bool requires_multiple_devices = false;
    bool requires_multiple_accelerators = false;
    bool latency_only = false;
    bool exclude_when_latency_sensitive = false;
    std::optional<std::uint32_t> forced_queue_depth;
    std::optional<std::uint32_t> forced_stages;
    std::optional<std::uint32_t> forced_logical_partitions;
    std::uint32_t min_queue_depth = 0;
    std::uint32_t min_stages = 0;
    double activation_bias = 0.0;
    int sort_order = 0;
};

class OperationVariantRegistry {
public:
    void register_variant(OperationVariantSpec spec);

    [[nodiscard]] std::vector<OperationVariantSpec> resolve(const OperationVariantRequest& request) const;

    [[nodiscard]] static const OperationVariantRegistry& builtin();

private:
    std::vector<OperationVariantSpec> variants_;
};

}  // namespace jakal
