#pragma once

#include "jakal/execution.hpp"

#include <filesystem>
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

struct WorkloadManifest {
    WorkloadSpec workload;
    WorkloadGraph graph;
    bool has_graph = false;
    std::filesystem::path source_path;
};

[[nodiscard]] std::vector<CanonicalWorkloadPreset> canonical_workload_presets();
[[nodiscard]] std::vector<CpuDeepLearningExplorationPreset> cpu_deep_learning_exploration_presets();
[[nodiscard]] WorkloadManifest load_workload_manifest(const std::filesystem::path& path);
void normalize_workload_graph(WorkloadGraph& graph);

}  // namespace jakal

