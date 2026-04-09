#pragma once

#include "jakal/execution.hpp"

#include <filesystem>
#include <string>
#include <unordered_map>
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
    std::uint64_t file_offset = 0;
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

struct CompiledTensorRef {
    std::string id;
    std::string alias_group;
    std::uint64_t bytes = 0;
    std::uint32_t lifetime_index = 0;
    std::uint32_t first_operation_index = 0;
    std::uint32_t last_operation_index = 0;
    bool persistent = false;
    bool temporary = false;
    bool host_visible = false;
    bool has_lifetime = false;
};

struct CompiledOperationRef {
    std::string name;
    std::uint32_t operation_index = 0;
    std::vector<std::uint32_t> input_tensor_indices;
    std::vector<std::uint32_t> output_tensor_indices;
    std::vector<std::uint32_t> temporary_tensor_indices;
    std::vector<std::uint32_t> dependency_operation_indices;
};

struct CompiledWorkloadGraph {
    std::string signature;
    std::vector<CompiledTensorRef> tensors;
    std::vector<CompiledOperationRef> operations;
    std::unordered_map<std::string, std::uint32_t> tensor_indices;
    std::unordered_map<std::string, std::uint32_t> operation_indices;
};

[[nodiscard]] std::vector<CanonicalWorkloadPreset> canonical_workload_presets();
[[nodiscard]] std::vector<CpuDeepLearningExplorationPreset> cpu_deep_learning_exploration_presets();
[[nodiscard]] WorkloadManifest load_workload_source(const std::filesystem::path& path);
[[nodiscard]] WorkloadManifest load_workload_manifest(const std::filesystem::path& path);
[[nodiscard]] CompiledWorkloadGraph compile_workload_graph(const WorkloadGraph& graph);
void normalize_workload_graph(WorkloadGraph& graph);

}  // namespace jakal

