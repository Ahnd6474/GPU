#include "jakal/jakal_toolkit.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

namespace jakal {
namespace {

double workload_bias(const JakalToolkitVariant& variant, const JakalL0WorkloadTraits& workload) {
    double bias = 0.0;

    if (workload.matrix_friendly && variant.binding.capabilities.subgroup_matrix) {
        bias += 0.08;
    }
    if (workload.streaming_friendly && variant.binding.capabilities.unified_memory) {
        bias += 0.04;
    }
    if (workload.latency_sensitive) {
        bias += std::max(0.0, 0.04 - (variant.binding.capabilities.estimated_launch_latency_us * 0.002));
    }

    switch (workload.op_class) {
    case OperationClass::matmul:
        bias += variant.binding.capabilities.subgroup_matrix ? 0.06 : -0.02;
        break;
    case OperationClass::convolution_2d:
        bias += variant.binding.capabilities.kernel_specialization ? 0.04 : 0.0;
        break;
    case OperationClass::resample_2d:
        bias += variant.binding.backend == JakalBackendKind::vulkan_compute ? 0.05 : 0.0;
        break;
    case OperationClass::reduction:
        bias += variant.binding.capabilities.asynchronous_dispatch ? 0.03 : 0.0;
        break;
    case OperationClass::elementwise_map:
    default:
        break;
    }

    const double bytes_gib = static_cast<double>(workload.bytes) / (1024.0 * 1024.0 * 1024.0);
    bias += std::min(0.08, bytes_gib * 0.01);
    return bias;
}

std::string make_rationale(const JakalToolkitVariant& variant) {
    std::string rationale = to_string(variant.binding.vendor);
    rationale += ":";
    rationale += to_string(variant.binding.backend);
    rationale += variant.executable ? ":ready" : ":planned";
    rationale += variant.binding.capabilities.unified_memory ? ":unified" : ":staged";
    if (variant.binding.capabilities.subgroup_matrix) {
        rationale += ":matrix";
    }
    if (variant.binding.capabilities.direct_command_submission) {
        rationale += ":direct-submit";
    }
    return rationale;
}

}  // namespace

JakalToolkit::JakalToolkit() : adapters_(make_default_jakal_l0_adapters()) {}

JakalToolkit::JakalToolkit(std::vector<std::unique_ptr<IJakalL0Adapter>> adapters)
    : adapters_(std::move(adapters)) {}

std::vector<JakalToolkitVariant> JakalToolkit::rank_variants(
    const HardwareGraph& graph,
    const JakalL0WorkloadTraits& workload) const {
    std::vector<JakalToolkitVariant> variants;
    for (const auto& adapter : adapters_) {
        if (!adapter->matches(graph)) {
            continue;
        }

        JakalToolkitVariant variant;
        variant.binding = adapter->describe(graph);
        variant.tuning = adapter->suggest_tuning(graph, workload);
        variant.executable = variant.binding.capabilities.adapter_available;
        variant.toolkit_score = variant.binding.suitability_score + workload_bias(variant, workload);
        if (!variant.executable) {
            variant.toolkit_score -= 0.25;
        }
        variant.rationale = make_rationale(variant);
        variants.push_back(std::move(variant));
    }

    std::sort(variants.begin(), variants.end(), [](const JakalToolkitVariant& left, const JakalToolkitVariant& right) {
        if (left.executable != right.executable) {
            return left.executable && !right.executable;
        }
        if (std::abs(left.toolkit_score - right.toolkit_score) > 1.0e-9) {
            return left.toolkit_score > right.toolkit_score;
        }
        return left.binding.adapter_id < right.binding.adapter_id;
    });
    return variants;
}

std::optional<JakalToolkitVariant> JakalToolkit::select_best(
    const HardwareGraph& graph,
    const JakalL0WorkloadTraits& workload) const {
    auto variants = rank_variants(graph, workload);
    if (variants.empty()) {
        return std::nullopt;
    }
    return variants.front();
}

std::vector<JakalToolkitIndexEntry> JakalToolkit::build_index(
    const std::vector<HardwareGraph>& graphs,
    const JakalL0WorkloadTraits& workload) const {
    std::vector<JakalToolkitIndexEntry> index;
    index.reserve(graphs.size());
    for (const auto& graph : graphs) {
        auto variants = rank_variants(graph, workload);
        if (variants.empty()) {
            continue;
        }

        index.push_back(JakalToolkitIndexEntry{
            graph.uid,
            structural_fingerprint(graph),
            std::move(variants)});
    }
    return index;
}

}  // namespace jakal

