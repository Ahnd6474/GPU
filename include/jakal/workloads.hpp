#pragma once

#include "jakal/planner.hpp"

#include <string>
#include <vector>

namespace jakal {

struct CanonicalWorkloadPreset {
    WorkloadSpec workload;
    std::string description;
    std::string baseline_label;
};

struct CpuDeepLearningExplorationPreset {
    WorkloadSpec workload;
    std::string description;
    std::string cpu_hypothesis;
    std::string success_signal;
};

[[nodiscard]] std::vector<CanonicalWorkloadPreset> canonical_workload_presets();
[[nodiscard]] std::vector<CpuDeepLearningExplorationPreset> cpu_deep_learning_exploration_presets();

}  // namespace jakal

