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
};

struct PlanAllocation {
    HardwareGraph device;
    double ratio = 0.0;
    double score = 0.0;
};

struct ExecutionPlan {
    std::string signature;
    std::vector<PlanAllocation> allocations;
    bool loaded_from_cache = false;
};

std::string to_string(WorkloadKind kind);

class Planner {
public:
    [[nodiscard]] static std::filesystem::path default_cache_path();

    explicit Planner(std::filesystem::path cache_path = default_cache_path());

    [[nodiscard]] ExecutionPlan build_plan(
        const WorkloadSpec& workload,
        const std::vector<HardwareGraph>& graphs);

private:
    struct CachedAllocation {
        std::string device_uid;
        double ratio = 0.0;
        double score = 0.0;
    };

    void load_cache();
    void persist_cache() const;

    std::filesystem::path cache_path_;
    bool cache_loaded_ = false;
    std::unordered_map<std::string, std::vector<CachedAllocation>> cache_;
};

}  // namespace jakal

