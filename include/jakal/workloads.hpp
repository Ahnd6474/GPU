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

struct WorkloadAsset {
    std::string id;
    std::filesystem::path path;
    std::vector<std::string> tensor_ids;
    std::uint64_t bytes = 0;
    bool persistent = true;
    bool host_visible = false;
    bool preload_required = true;
    std::string preferred_residency = "auto";
};

struct WorkloadManifest {
    WorkloadSpec workload;
    WorkloadGraph graph;
    std::vector<WorkloadAsset> assets;
    bool has_graph = false;
    std::filesystem::path source_path;
    std::string source_format = "manifest";
    std::string source_entry;
    bool imported = false;
};

[[nodiscard]] std::vector<CanonicalWorkloadPreset> canonical_workload_presets();
[[nodiscard]] std::vector<CpuDeepLearningExplorationPreset> cpu_deep_learning_exploration_presets();
[[nodiscard]] WorkloadManifest load_workload_source(const std::filesystem::path& path);
[[nodiscard]] WorkloadManifest load_workload_manifest(const std::filesystem::path& path);
void normalize_workload_graph(WorkloadGraph& graph);

}  // namespace jakal

