#pragma once

#include "jakal/backend.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace jakal {

enum class WorkloadKind {
    custom,
    inference,
    image,
    tensor,
    gaming,
    training
};

enum class WorkloadPhase {
    unknown,
    prefill,
    decode,
    cache_update,
    dequantize,
    training_step
};

enum class PartitionStrategy {
    auto_balanced,
    blind_sharded,
    role_split,
    reduce_on_gpu,
    projection_sharded,
    tpu_like
};

struct WorkloadSpec {
    std::string name;
    WorkloadKind kind = WorkloadKind::custom;
    std::string dataset_tag;
    std::uint64_t working_set_bytes = 0;
    std::uint64_t host_exchange_bytes = 0;
    double estimated_flops = 0.0;
    std::uint32_t batch_size = 1;
    bool latency_sensitive = false;
    bool prefer_unified_memory = false;
    bool matrix_friendly = false;
    PartitionStrategy partition_strategy = PartitionStrategy::auto_balanced;
    WorkloadPhase phase = WorkloadPhase::unknown;
    std::string shape_bucket;
};

struct PlanAllocation {
    HardwareGraph device;
    double ratio = 0.0;
    double score = 0.0;
};

struct ExecutionPlan {
    std::string signature;
    std::vector<PlanAllocation> allocations;
    PartitionStrategy resolved_partition_strategy = PartitionStrategy::auto_balanced;
    bool loaded_from_cache = false;
};

struct StrategyFeedbackSample {
    PartitionStrategy strategy = PartitionStrategy::auto_balanced;
    double total_runtime_us = 0.0;
    double head_runtime_us = 0.0;
    double speedup_vs_reference = 1.0;
    double successful_operation_ratio = 1.0;
    bool all_succeeded = false;
};

std::string to_string(WorkloadKind kind);
std::string to_string(WorkloadPhase phase);
std::string to_string(PartitionStrategy strategy);
[[nodiscard]] WorkloadPhase canonical_workload_phase(const WorkloadSpec& workload);
[[nodiscard]] std::string canonical_workload_shape_bucket(const WorkloadSpec& workload);

class Planner {
public:
    [[nodiscard]] static std::filesystem::path default_cache_path();

    explicit Planner(std::filesystem::path cache_path = default_cache_path());

    [[nodiscard]] ExecutionPlan build_plan(
        const WorkloadSpec& workload,
        const std::vector<HardwareGraph>& graphs);
    void ingest_strategy_feedback(
        const WorkloadSpec& workload,
        const std::vector<HardwareGraph>& graphs,
        const StrategyFeedbackSample& feedback);

private:
    struct CachedAllocation {
        std::string device_uid;
        double ratio = 0.0;
        double score = 0.0;
    };

    struct StrategyStats {
        std::uint32_t observations = 0;
        std::uint32_t successful_observations = 0;
        double average_runtime_us = 0.0;
        double average_head_runtime_us = 0.0;
        double average_speedup_vs_reference = 1.0;
        double average_successful_operation_ratio = 1.0;
        std::uint64_t last_update_epoch = 0;
    };

    [[nodiscard]] PartitionStrategy resolve_partition_strategy(
        const WorkloadSpec& workload,
        const std::vector<HardwareGraph>& graphs) const;

    void load_cache();
    void persist_cache() const;

    std::filesystem::path cache_path_;
    bool cache_loaded_ = false;
    std::uint64_t feedback_epoch_ = 0;
    std::unordered_map<std::string, std::vector<CachedAllocation>> cache_;
    std::unordered_map<std::string, std::unordered_map<std::string, StrategyStats>> strategy_stats_;
    std::unordered_map<std::string, std::unordered_map<std::string, StrategyStats>> family_strategy_stats_;
};

}  // namespace jakal

