#include "jakal/planner.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace jakal {
namespace {

constexpr double kMinShardRatio = 0.05;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kMiB = 1024.0 * 1024.0;
constexpr double kKiB = 1024.0;
constexpr double kLearnedStrategyMargin = 0.03;
constexpr double kLearnedFamilyStrategyMargin = 0.05;
constexpr double kFamilyFreshnessHalfLifeEpochs = 10.0;
constexpr std::uint32_t kMinimumAutoBaselineObservations = 1u;
constexpr std::uint32_t kMinimumFamilyAutoBaselineObservations = 2u;
constexpr std::uint32_t kStrategyExplorationTargetObservations = 1u;
constexpr std::uint64_t kMaximumFamilyStalenessEpochs = 32u;
constexpr std::string_view kPlannerHeuristicRevision = "planner-r2-device-agnostic";

struct PartitionAllocationBias {
    double host_ratio = 0.0;
    double accelerator_ratio = 1.0;
};

struct ScoreComponents {
    double parallel_score = 0.0;
    double memory_score = 0.0;
    double graph_bonus = 1.0;
    double transfer_penalty = 0.0;
    double memory_fit_cap = 1.0;
};

std::vector<std::string> split_tab(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) {
        fields.push_back(field);
    }
    return fields;
}

std::string ascii_lower_copy(const std::string& input) {
    std::string lowered = input;
    std::transform(
        lowered.begin(),
        lowered.end(),
        lowered.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
    return lowered;
}

bool contains_phase_token(const std::string& haystack, const std::string& needle) {
    return !needle.empty() && haystack.find(needle) != std::string::npos;
}

std::uint64_t next_power_of_two(const std::uint64_t value) {
    if (value <= 1ull) {
        return 1ull;
    }
    std::uint64_t bucket = 1ull;
    while (bucket < value && bucket < (std::numeric_limits<std::uint64_t>::max() >> 1u)) {
        bucket <<= 1u;
    }
    return bucket;
}

std::uint64_t bucket_bytes_mib(const std::uint64_t bytes) {
    if (bytes == 0ull) {
        return 0ull;
    }
    const std::uint64_t mib = (bytes + static_cast<std::uint64_t>(kMiB) - 1ull) / static_cast<std::uint64_t>(kMiB);
    return next_power_of_two(std::max<std::uint64_t>(1ull, mib));
}

std::uint64_t bucket_flops_giga(const double flops) {
    if (!(flops > 0.0)) {
        return 0ull;
    }
    const auto giga = static_cast<std::uint64_t>(std::ceil(flops / 1.0e9));
    return next_power_of_two(std::max<std::uint64_t>(1ull, giga));
}

std::string topology_family_token(const HardwareGraph& graph);

std::string build_signature(const WorkloadSpec& workload, const std::vector<HardwareGraph>& graphs) {
    std::vector<std::string> fingerprints;
    fingerprints.reserve(graphs.size());
    for (const auto& graph : graphs) {
        fingerprints.push_back(graph.uid + ":" + structural_fingerprint(graph));
    }
    std::sort(fingerprints.begin(), fingerprints.end());

    std::ostringstream signature;
    signature << kPlannerHeuristicRevision << '|'
              << workload.name << '|'
              << to_string(workload.kind) << '|'
              << workload.dataset_tag << '|'
              << to_string(canonical_workload_phase(workload)) << '|'
              << canonical_workload_shape_bucket(workload) << '|'
              << workload.working_set_bytes << '|'
              << workload.host_exchange_bytes << '|'
              << std::fixed << std::setprecision(3) << workload.estimated_flops << '|'
              << workload.batch_size << '|'
              << workload.latency_sensitive << '|'
              << workload.prefer_unified_memory << '|'
              << workload.matrix_friendly << '|'
              << to_string(workload.partition_strategy);

    for (const auto& fingerprint : fingerprints) {
        signature << '|' << fingerprint;
    }

    return signature.str();
}

std::string build_learning_key(const WorkloadSpec& workload, const std::vector<HardwareGraph>& graphs) {
    std::vector<std::string> fingerprints;
    fingerprints.reserve(graphs.size());
    for (const auto& graph : graphs) {
        fingerprints.push_back(graph.uid + ":" + structural_fingerprint(graph));
    }
    std::sort(fingerprints.begin(), fingerprints.end());

    std::ostringstream key;
    key << kPlannerHeuristicRevision << '|'
        << workload.name << '|'
        << to_string(workload.kind) << '|'
        << workload.dataset_tag << '|'
        << to_string(canonical_workload_phase(workload)) << '|'
        << canonical_workload_shape_bucket(workload) << '|'
        << workload.working_set_bytes << '|'
        << workload.host_exchange_bytes << '|'
        << std::fixed << std::setprecision(3) << workload.estimated_flops << '|'
        << workload.batch_size << '|'
        << workload.latency_sensitive << '|'
        << workload.prefer_unified_memory << '|'
        << workload.matrix_friendly;
    for (const auto& fingerprint : fingerprints) {
        key << '|' << fingerprint;
    }
    return key.str();
}

std::string build_family_learning_key(const WorkloadSpec& workload, const std::vector<HardwareGraph>& graphs) {
    std::vector<std::string> family_tokens;
    family_tokens.reserve(graphs.size());
    for (const auto& graph : graphs) {
        family_tokens.push_back(topology_family_token(graph));
    }
    std::sort(family_tokens.begin(), family_tokens.end());

    std::ostringstream key;
    key << kPlannerHeuristicRevision << '|'
        << workload.name << '|'
        << to_string(workload.kind) << '|'
        << workload.dataset_tag << '|'
        << to_string(canonical_workload_phase(workload)) << '|'
        << canonical_workload_shape_bucket(workload) << '|'
        << workload.working_set_bytes << '|'
        << workload.host_exchange_bytes << '|'
        << std::fixed << std::setprecision(3) << workload.estimated_flops << '|'
        << workload.batch_size << '|'
        << workload.latency_sensitive << '|'
        << workload.prefer_unified_memory << '|'
        << workload.matrix_friendly;
    for (const auto& token : family_tokens) {
        key << '|' << token;
    }
    return key.str();
}

std::filesystem::path strategy_cache_path_for(const std::filesystem::path& cache_path) {
    auto path = cache_path;
    path += ".strategy";
    return path;
}

std::filesystem::path family_strategy_cache_path_for(const std::filesystem::path& cache_path) {
    auto path = cache_path;
    path += ".strategy_family";
    return path;
}

std::filesystem::path confidence_calibration_cache_path_for(const std::filesystem::path& cache_path) {
    auto path = cache_path;
    path += ".confidence";
    return path;
}

std::uint32_t confidence_bucket_index(const double confidence) {
    return static_cast<std::uint32_t>(std::clamp(std::floor(std::clamp(confidence, 0.0, 0.999) * 10.0), 0.0, 9.0));
}

std::string build_confidence_calibration_key(
    const WorkloadSpec& workload,
    const PlanStrategySource source,
    const double confidence) {
    std::ostringstream key;
    key << to_string(workload.kind) << '|'
        << to_string(canonical_workload_phase(workload)) << '|'
        << to_string(source) << '|'
        << confidence_bucket_index(confidence);
    return key.str();
}

bool is_host_graph(const HardwareGraph& graph) {
    return graph.probe == "host";
}

bool has_partitionable_topology(const std::vector<HardwareGraph>& graphs) {
    bool saw_host = false;
    bool saw_accelerator = false;
    for (const auto& graph : graphs) {
        if (is_host_graph(graph)) {
            saw_host = true;
        } else {
            saw_accelerator = true;
        }
        if (saw_host && saw_accelerator) {
            return true;
        }
    }
    return false;
}

bool uses_explicit_partition_strategy(const WorkloadSpec& workload) {
    return workload.partition_strategy != PartitionStrategy::auto_balanced;
}

bool append_strategy_once(std::vector<PartitionStrategy>& strategies, const PartitionStrategy strategy) {
    if (std::find(strategies.begin(), strategies.end(), strategy) != strategies.end()) {
        return false;
    }
    strategies.push_back(strategy);
    return true;
}

std::vector<PartitionStrategy> exploration_order_for(const WorkloadSpec& workload) {
    std::vector<PartitionStrategy> order;
    if (workload.heuristic_partition_hint.has_value()) {
        append_strategy_once(order, *workload.heuristic_partition_hint);
    }

    if (workload.kind == WorkloadKind::inference && workload.latency_sensitive && workload.matrix_friendly) {
        for (const auto strategy : {
                 PartitionStrategy::role_split,
                 PartitionStrategy::tpu_like,
                 PartitionStrategy::reduce_on_gpu,
                 PartitionStrategy::projection_sharded,
                 PartitionStrategy::blind_sharded}) {
            append_strategy_once(order, strategy);
        }
        return order;
    }
    if (workload.matrix_friendly) {
        for (const auto strategy : {
                 PartitionStrategy::projection_sharded,
                 PartitionStrategy::reduce_on_gpu,
                 PartitionStrategy::role_split,
                 PartitionStrategy::tpu_like,
                 PartitionStrategy::blind_sharded}) {
            append_strategy_once(order, strategy);
        }
        return order;
    }
    for (const auto strategy : {
             PartitionStrategy::role_split,
             PartitionStrategy::reduce_on_gpu,
             PartitionStrategy::blind_sharded,
             PartitionStrategy::projection_sharded,
             PartitionStrategy::tpu_like}) {
        append_strategy_once(order, strategy);
    }
    return order;
}

std::uint64_t bucket_positive_u32(const std::uint32_t value) {
    return next_power_of_two(std::max<std::uint32_t>(1u, value));
}

std::uint64_t bucket_bandwidth_gbps(const double bandwidth) {
    if (!(bandwidth > 0.0)) {
        return 0ull;
    }
    return next_power_of_two(std::max<std::uint64_t>(1ull, static_cast<std::uint64_t>(std::ceil(bandwidth))));
}

std::string topology_family_token(const HardwareGraph& graph) {
    const auto summary = summarize_graph(graph);
    std::ostringstream token;
    token << (is_host_graph(graph) ? "host" : "accel") << ':'
          << graph.probe << ':'
          << bucket_positive_u32(summary.execution_objects) << ':'
          << bucket_positive_u32(summary.lanes_per_object) << ':'
          << bucket_positive_u32(summary.matrix_units == 0u ? 1u : summary.matrix_units) << ':'
          << bucket_bytes_mib(summary.directly_attached_bytes) << ':'
          << bucket_bytes_mib(summary.shared_host_bytes) << ':'
          << bucket_bandwidth_gbps(std::max(summary.host_read_gbps, summary.host_write_gbps)) << ':'
          << (summary.unified_address_space ? 'u' : 'd')
          << (summary.coherent_with_host ? 'c' : 'n')
          << (summary.supports_fp16 ? 'h' : 'n')
          << (summary.supports_bf16 ? 'b' : 'n')
          << (summary.supports_int8 ? 'i' : 'n');
    return token.str();
}

PartitionAllocationBias partition_allocation_bias(const PartitionStrategy strategy) {
    switch (strategy) {
    case PartitionStrategy::blind_sharded:
        return {0.35, 0.65};
    case PartitionStrategy::role_split:
        return {0.30, 0.70};
    case PartitionStrategy::reduce_on_gpu:
        return {0.20, 0.80};
    case PartitionStrategy::projection_sharded:
        return {0.25, 0.75};
    case PartitionStrategy::tpu_like:
        return {0.12, 0.88};
    case PartitionStrategy::auto_balanced:
    default:
        return {0.0, 1.0};
    }
}

double workload_host_exchange_ratio(const WorkloadSpec& workload) {
    const auto working_set = std::max<std::uint64_t>(workload.working_set_bytes, 1ull);
    return std::clamp(
        static_cast<double>(workload.host_exchange_bytes) / static_cast<double>(working_set),
        0.0,
        4.0);
}

double workload_compute_intensity(const WorkloadSpec& workload) {
    if (workload.working_set_bytes == 0ull || workload.estimated_flops <= 0.0) {
        return workload.matrix_friendly ? 48.0 : 12.0;
    }
    return workload.estimated_flops / static_cast<double>(workload.working_set_bytes);
}

double workload_dispatch_sensitivity(const WorkloadSpec& workload) {
    double sensitivity = 0.0;
    if (workload.latency_sensitive) {
        sensitivity += workload.batch_size <= 1u ? 0.55 : 0.28;
    }
    sensitivity += std::clamp(workload_host_exchange_ratio(workload) * 0.45, 0.0, 0.30);
    if (!workload.matrix_friendly) {
        sensitivity += 0.18;
    }
    if (workload.prefer_unified_memory) {
        sensitivity += 0.08;
    }
    return std::clamp(sensitivity, 0.0, 1.0);
}

double auto_host_ratio_floor(const WorkloadSpec& workload) {
    double floor = 0.0;
    if (workload.latency_sensitive) {
        floor += workload.batch_size <= 1u ? 0.10 : 0.05;
    }
    floor += std::clamp(workload_host_exchange_ratio(workload) * 0.40, 0.0, 0.16);
    if (!workload.matrix_friendly) {
        floor += 0.08;
    }
    if (workload.prefer_unified_memory) {
        floor += 0.03;
    }
    return std::clamp(floor, 0.0, 0.32);
}

double auto_accelerator_ratio_floor(const WorkloadSpec& workload) {
    double floor = 0.0;
    if (workload.matrix_friendly) {
        floor += workload.latency_sensitive ? 0.42 : 0.55;
    } else if (workload.estimated_flops >= 5.0e10) {
        floor += 0.12;
    }
    return std::clamp(floor, 0.0, 0.60);
}

double strategy_selection_score(
    const WorkloadSpec& workload,
    const std::uint32_t observations,
    const std::uint32_t successful_observations,
    const double average_runtime_us,
    const double average_head_runtime_us,
    const double average_speedup_vs_reference,
    const double average_successful_operation_ratio,
    const double baseline_runtime_us,
    const double baseline_head_runtime_us,
    const double freshness_weight = 1.0) {
    if (observations == 0u) {
        return 0.0;
    }
    const double success_rate =
        static_cast<double>(successful_observations) / static_cast<double>(observations);
    const double operation_success = std::clamp(average_successful_operation_ratio, 0.0, 1.0);
    const double freshness = std::clamp(freshness_weight, 0.20, 1.0);
    const double reliability = std::clamp((success_rate * 0.70) + (operation_success * 0.30), 0.0, 1.0) * freshness;
    const double confidence = std::min(1.0, static_cast<double>(observations) / 3.0) * freshness;
    const double runtime_gain =
        std::clamp((std::max(baseline_runtime_us, 1.0) / std::max(average_runtime_us, 1.0)) - 1.0, -0.40, 0.60);
    const double head_gain = std::clamp(
        (std::max(baseline_head_runtime_us, 1.0) / std::max(average_head_runtime_us, 1.0)) - 1.0,
        -0.45,
        0.75);
    const double centered_speedup = std::clamp(average_speedup_vs_reference - 1.0, -0.35, 0.50);

    double metric_gain = 0.0;
    if (workload.latency_sensitive) {
        metric_gain = (head_gain * 0.55) + (runtime_gain * 0.25) + (centered_speedup * 0.20);
    } else {
        metric_gain = (centered_speedup * 0.45) + (runtime_gain * 0.35) + (head_gain * 0.20);
    }

    return reliability * (1.0 + (metric_gain * confidence));
}

double compute_weight_from_workload(const WorkloadSpec& workload) {
    if (workload.working_set_bytes == 0 || workload.estimated_flops <= 0.0) {
        double fallback = workload.matrix_friendly ? 0.72 : 0.60;
        fallback -= workload_dispatch_sensitivity(workload) * (workload.matrix_friendly ? 0.10 : 0.18);
        return std::clamp(fallback, 0.18, workload.matrix_friendly ? 0.82 : 0.70);
    }

    const double intensity = workload_compute_intensity(workload);
    const double dispatch_sensitivity = workload_dispatch_sensitivity(workload);
    const double exchange_ratio = workload_host_exchange_ratio(workload);

    double weight = std::clamp(intensity / 64.0, 0.22, workload.matrix_friendly ? 0.88 : 0.76);
    weight -= dispatch_sensitivity * (workload.matrix_friendly ? 0.14 : 0.22);
    weight -= std::clamp(exchange_ratio * 0.12, 0.0, 0.18);
    if (workload.latency_sensitive && workload.batch_size <= 1u && !workload.matrix_friendly) {
        weight -= 0.08;
    }
    return std::clamp(weight, 0.18, workload.matrix_friendly ? 0.82 : 0.70);
}

double effective_host_link_gbps(const HardwareGraphSummary& summary) {
    const double link = std::max(summary.host_read_gbps, summary.host_write_gbps);
    if (link > 0.0) {
        return link;
    }
    if (summary.unified_address_space) {
        return 128.0;
    }
    if (summary.coherent_with_host) {
        return 48.0;
    }
    return 16.0;
}

double effective_dispatch_latency_us(const HardwareGraphSummary& summary) {
    if (summary.average_dispatch_cost_us > 0.0) {
        return summary.average_dispatch_cost_us;
    }
    if (summary.dispatch_latency_us > 0.0) {
        return summary.dispatch_latency_us;
    }
    return summary.supports_asynchronous_dispatch ? 12.0 : 30.0;
}

double effective_sync_latency_us(const HardwareGraphSummary& summary) {
    if (summary.synchronization_latency_us > 0.0) {
        return summary.synchronization_latency_us;
    }
    return summary.supports_asynchronous_dispatch ? 8.0 : 20.0;
}

ScoreComponents score_graph(const HardwareGraph& graph, const WorkloadSpec& workload) {
    const auto summary = summarize_graph(graph);
    const bool host_graph = is_host_graph(graph);
    const double execution_objects = static_cast<double>(std::max(summary.execution_objects, 1u));
    const double lanes_per_object = static_cast<double>(std::max(summary.lanes_per_object, 1u));
    const double resident_contexts = static_cast<double>(std::max(summary.resident_contexts, 1u));
    const std::uint32_t fallback_clock_mhz = summary.host_visible ? 2500u : 1000u;
    const double clock_ghz = static_cast<double>(summary.clock_mhz == 0 ? fallback_clock_mhz : summary.clock_mhz) / 1000.0;
    const double matrix_units = static_cast<double>(summary.matrix_units);

    const double attached_gib = static_cast<double>(summary.directly_attached_bytes) / kGiB;
    const double shared_gib = static_cast<double>(summary.shared_host_bytes) / kGiB;
    const double addressable_gib = static_cast<double>(summary.addressable_bytes) / kGiB;
    const double scratch_kib = static_cast<double>(summary.local_scratch_bytes) / kKiB;
    const double cache_mib = static_cast<double>(summary.cache_bytes) / kMiB;
    const double dispatch_sensitivity = workload_dispatch_sensitivity(workload);
    const double exchange_ratio = workload_host_exchange_ratio(workload);
    const double compute_intensity = workload_compute_intensity(workload);

    const double structural_parallelism =
        execution_objects * std::sqrt(lanes_per_object) * std::log2(resident_contexts + 1.0) * clock_ghz;

    double memory_score =
        std::sqrt(std::max(1.0, (attached_gib * 32.0) + (shared_gib * 8.0) + std::max(addressable_gib, 1.0))) *
        (1.0 + (std::log2((scratch_kib / 32.0) + 1.0) * 0.15)) *
        (1.0 + (std::log2(cache_mib + 1.0) * 0.05));

    if (summary.coherent_with_host) {
        memory_score *= 1.05;
    }

    if (summary.unified_address_space && workload.prefer_unified_memory) {
        memory_score *= 1.10;
    }
    if (host_graph && dispatch_sensitivity > 0.0) {
        memory_score *= 1.0 + (dispatch_sensitivity * 0.05);
    }

    double graph_bonus = 1.0;
    if (workload.matrix_friendly) {
        graph_bonus += std::log2(matrix_units + 1.0) * 0.30;
        if (summary.supports_fp16) {
            graph_bonus += 0.10;
        }
        if (summary.supports_bf16) {
            graph_bonus += 0.12;
        }
        if (summary.supports_int8) {
            graph_bonus += 0.08;
        }
    }
    if (host_graph) {
        graph_bonus += dispatch_sensitivity * 0.28;
        graph_bonus += std::clamp(exchange_ratio * 0.30, 0.0, 0.28);
        if (workload.latency_sensitive && workload.batch_size <= 1u) {
            graph_bonus += 0.12;
        }
        if (workload.matrix_friendly && compute_intensity >= 48.0) {
            graph_bonus -= 0.22;
            memory_score *= 0.96;
        }
    } else {
        graph_bonus -= dispatch_sensitivity * (workload.matrix_friendly ? 0.06 : 0.16);
        if (!workload.matrix_friendly && compute_intensity < 24.0) {
            graph_bonus -= 0.06;
        }
        if (workload.matrix_friendly && compute_intensity >= 48.0) {
            graph_bonus += 0.12;
        }
    }

    // Favor graphs that expose more useful structure for mapping and scheduling.
    graph_bonus += std::log2(static_cast<double>(graph.nodes.size()) + 1.0) * 0.02;
    graph_bonus += std::log2(static_cast<double>(graph.edges.size()) + 1.0) * 0.01;
    graph_bonus += std::log2((summary.average_feed_cost_us > 0.0 ? 1.0 / summary.average_feed_cost_us : 1.0) + 1.0) * 0.03;

    double transfer_penalty =
        (effective_dispatch_latency_us(summary) + effective_sync_latency_us(summary)) / 100.0;

    if (summary.average_transfer_cost_us > 0.0) {
        transfer_penalty += summary.average_transfer_cost_us / (workload.latency_sensitive ? 40.0 : 80.0);
    }

    if (summary.average_hierarchy_cost_us > 0.0) {
        transfer_penalty += summary.average_hierarchy_cost_us / (workload.latency_sensitive ? 120.0 : 240.0);
    }

    if (workload.host_exchange_bytes > 0) {
        transfer_penalty +=
            (static_cast<double>(workload.host_exchange_bytes) / kGiB) / effective_host_link_gbps(summary);
    } else if (!summary.unified_address_space) {
        transfer_penalty += workload.latency_sensitive ? 0.30 : 0.10;
    }
    if (host_graph) {
        transfer_penalty *= workload.host_exchange_bytes > 0 ? 0.45 : 0.60;
    } else {
        transfer_penalty += exchange_ratio * (workload.latency_sensitive ? 0.90 : 0.45);
        if (dispatch_sensitivity > 0.0 && !summary.unified_address_space) {
            transfer_penalty += dispatch_sensitivity * 0.35;
        }
        if (workload.matrix_friendly && compute_intensity >= 48.0) {
            transfer_penalty *= 0.92;
        }
    }

    if (workload.prefer_unified_memory && summary.unified_address_space) {
        transfer_penalty *= 0.75;
    }

    double memory_fit_cap = 1.0;
    if (workload.working_set_bytes > 0) {
        std::uint64_t fit_bytes = summary.directly_attached_bytes;
        if (summary.unified_address_space || summary.coherent_with_host) {
            fit_bytes += summary.shared_host_bytes;
        }
        if (fit_bytes == 0) {
            fit_bytes = summary.addressable_bytes;
        }
        if (fit_bytes > 0) {
            memory_fit_cap = std::min(
                1.0,
                0.90 * static_cast<double>(fit_bytes) / static_cast<double>(workload.working_set_bytes));
        }
    }

    return ScoreComponents{
        structural_parallelism,
        memory_score,
        std::max(0.55, graph_bonus),
        transfer_penalty,
        memory_fit_cap};
}

}  // namespace

std::string to_string(WorkloadKind kind) {
    switch (kind) {
    case WorkloadKind::inference:
        return "inference";
    case WorkloadKind::image:
        return "image";
    case WorkloadKind::tensor:
        return "tensor";
    case WorkloadKind::gaming:
        return "gaming";
    case WorkloadKind::training:
        return "training";
    case WorkloadKind::custom:
    default:
        return "custom";
    }
}

std::string to_string(const WorkloadPhase phase) {
    switch (phase) {
    case WorkloadPhase::prefill:
        return "prefill";
    case WorkloadPhase::decode:
        return "decode";
    case WorkloadPhase::cache_update:
        return "cache_update";
    case WorkloadPhase::dequantize:
        return "dequantize";
    case WorkloadPhase::training_step:
        return "training_step";
    case WorkloadPhase::unknown:
    default:
        return "unknown";
    }
}

std::string to_string(const PartitionStrategy strategy) {
    switch (strategy) {
    case PartitionStrategy::blind_sharded:
        return "blind_sharded";
    case PartitionStrategy::role_split:
        return "role_split";
    case PartitionStrategy::reduce_on_gpu:
        return "reduce_on_gpu";
    case PartitionStrategy::projection_sharded:
        return "projection_sharded";
    case PartitionStrategy::tpu_like:
        return "tpu_like";
    case PartitionStrategy::auto_balanced:
    default:
        return "auto_balanced";
    }
}

std::string to_string(const PlanStrategySource source) {
    switch (source) {
    case PlanStrategySource::explicit_request:
        return "explicit_request";
    case PlanStrategySource::exact_learning:
        return "exact_learning";
    case PlanStrategySource::family_learning:
        return "family_learning";
    case PlanStrategySource::exploration:
        return "exploration";
    case PlanStrategySource::heuristic_auto:
    default:
        return "heuristic_auto";
    }
}

WorkloadPhase canonical_workload_phase(const WorkloadSpec& workload) {
    if (workload.phase != WorkloadPhase::unknown) {
        return workload.phase;
    }

    const auto phase_hint = ascii_lower_copy(workload.dataset_tag + "|" + workload.name);
    if (contains_phase_token(phase_hint, "prefill")) {
        return WorkloadPhase::prefill;
    }
    if (contains_phase_token(phase_hint, "decode")) {
        return WorkloadPhase::decode;
    }
    if (contains_phase_token(phase_hint, "kv-cache") ||
        contains_phase_token(phase_hint, "cache-update") ||
        contains_phase_token(phase_hint, "cache_update")) {
        return WorkloadPhase::cache_update;
    }
    if (contains_phase_token(phase_hint, "dequant") ||
        contains_phase_token(phase_hint, "int4") ||
        contains_phase_token(phase_hint, "quant")) {
        return WorkloadPhase::dequantize;
    }
    if (workload.kind == WorkloadKind::training ||
        contains_phase_token(phase_hint, "train") ||
        contains_phase_token(phase_hint, "backward") ||
        contains_phase_token(phase_hint, "optimizer")) {
        return WorkloadPhase::training_step;
    }
    return WorkloadPhase::unknown;
}

std::string canonical_workload_shape_bucket(const WorkloadSpec& workload) {
    if (!workload.shape_bucket.empty()) {
        return workload.shape_bucket;
    }

    std::ostringstream stream;
    stream << "phase=" << to_string(canonical_workload_phase(workload))
           << "|b=" << next_power_of_two(std::max<std::uint32_t>(1u, workload.batch_size))
           << "|ws_mib=" << bucket_bytes_mib(workload.working_set_bytes)
           << "|hx_mib=" << bucket_bytes_mib(workload.host_exchange_bytes)
           << "|flops_g=" << bucket_flops_giga(workload.estimated_flops);
    return stream.str();
}

double family_freshness_weight(const std::uint64_t current_epoch, const std::uint64_t last_update_epoch) {
    if (last_update_epoch == 0u || current_epoch <= last_update_epoch) {
        return 1.0;
    }
    const double age = static_cast<double>(current_epoch - last_update_epoch);
    const double freshness = std::exp(-age / kFamilyFreshnessHalfLifeEpochs);
    return std::clamp(freshness, 0.20, 1.0);
}

double strategy_decision_confidence(
    const std::uint32_t observations,
    const std::uint32_t successful_observations,
    const double average_successful_operation_ratio,
    const double score_delta,
    const bool family,
    const double freshness_weight) {
    if (observations == 0u) {
        return family ? 0.20 : 0.25;
    }

    const double observation_confidence = std::min(1.0, static_cast<double>(observations) / 4.0);
    const double success_rate =
        static_cast<double>(successful_observations) / static_cast<double>(observations);
    const double success_confidence =
        std::clamp((success_rate * 0.65) + (average_successful_operation_ratio * 0.35), 0.0, 1.0);
    const double margin_confidence = std::clamp(score_delta / 0.15, 0.0, 1.0);
    double confidence =
        (observation_confidence * 0.45) + (success_confidence * 0.35) + (margin_confidence * 0.20);
    confidence *= std::clamp(freshness_weight, 0.20, 1.0);
    if (family) {
        confidence *= 0.85;
    }
    return std::clamp(confidence, family ? 0.20 : 0.25, family ? 0.90 : 0.98);
}

double apply_confidence_calibration(
    const std::unordered_map<std::string, ConfidenceCalibrationStats>& calibration_store,
    const WorkloadSpec& workload,
    const PlanStrategySource source,
    const double raw_confidence) {
    if (source == PlanStrategySource::explicit_request) {
        return 1.0;
    }

    const auto it = calibration_store.find(build_confidence_calibration_key(workload, source, raw_confidence));
    if (it == calibration_store.end() || it->second.observations < 2u) {
        return std::clamp(raw_confidence, 0.0, 1.0);
    }

    const auto& stats = it->second;
    const double success_rate =
        static_cast<double>(stats.successful_observations + 1u) / static_cast<double>(stats.observations + 2u);
    const double rollback_penalty =
        static_cast<double>(stats.rollback_observations) / static_cast<double>(std::max<std::uint32_t>(1u, stats.observations));
    const double empirical_confidence = std::clamp(
        (success_rate * 0.55) +
            (stats.average_successful_operation_ratio * 0.30) +
            ((1.0 / std::max(1.0, stats.average_runtime_regression)) * 0.15) -
            (rollback_penalty * 0.25),
        0.05,
        0.99);
    const double calibration_weight = std::min(0.75, static_cast<double>(stats.observations) / 8.0);
    return std::clamp(
        (raw_confidence * (1.0 - calibration_weight)) + (empirical_confidence * calibration_weight),
        0.05,
        0.99);
}

Planner::Planner(std::filesystem::path cache_path)
    : cache_path_(std::move(cache_path)) {}

Planner::ResolvedStrategyDecision Planner::resolve_partition_strategy(
    const WorkloadSpec& workload,
    const std::vector<HardwareGraph>& graphs) const {
    if (workload.partition_strategy != PartitionStrategy::auto_balanced) {
        return {
            workload.partition_strategy,
            PlanStrategySource::explicit_request,
            1.0,
            "explicit partition strategy request"};
    }

    struct StrategyResolution {
        PartitionStrategy strategy = PartitionStrategy::auto_balanced;
        PlanStrategySource source = PlanStrategySource::heuristic_auto;
        double confidence = 0.40;
        std::string reason;
        bool has_baseline = false;
    };

    const auto exact_it = strategy_stats_.find(build_learning_key(workload, graphs));
    const auto family_it = family_strategy_stats_.find(build_family_learning_key(workload, graphs));

    const auto resolve_from_bucket =
        [&](const auto* bucket,
            const bool allow_exploration,
            const std::uint32_t minimum_auto_baseline_observations,
            const double margin,
            const bool apply_family_aging) -> StrategyResolution {
        if (bucket == nullptr) {
            return {};
        }

        const auto auto_it = bucket->find(to_string(PartitionStrategy::auto_balanced));
        const std::uint32_t auto_observations = auto_it == bucket->end() ? 0u : auto_it->second.observations;
        if (auto_observations < minimum_auto_baseline_observations) {
            return {};
        }
        if (apply_family_aging &&
            auto_it != bucket->end() &&
            feedback_epoch_ > auto_it->second.last_update_epoch &&
            (feedback_epoch_ - auto_it->second.last_update_epoch) > kMaximumFamilyStalenessEpochs) {
            return {};
        }

        if (allow_exploration && has_partitionable_topology(graphs)) {
            for (const auto strategy : exploration_order_for(workload)) {
                const auto strategy_it = bucket->find(to_string(strategy));
                const std::uint32_t observations =
                    strategy_it == bucket->end() ? 0u : strategy_it->second.observations;
                if (observations < kStrategyExplorationTargetObservations) {
                    return {
                        strategy,
                        PlanStrategySource::exploration,
                        0.25,
                        "exploring missing observations for " + to_string(strategy),
                        true};
                }
            }
        }

        const double auto_score =
            auto_it == bucket->end()
                ? 1.0
                : strategy_selection_score(
                      workload,
                      auto_it->second.observations,
                      auto_it->second.successful_observations,
                      auto_it->second.average_runtime_us,
                      auto_it->second.average_head_runtime_us,
                      auto_it->second.average_speedup_vs_reference,
                      auto_it->second.average_successful_operation_ratio,
                      auto_it->second.average_runtime_us,
                      auto_it->second.average_head_runtime_us,
                      apply_family_aging
                          ? family_freshness_weight(feedback_epoch_, auto_it->second.last_update_epoch)
                          : 1.0);

        PartitionStrategy best_strategy = PartitionStrategy::auto_balanced;
        double best_score = auto_score;
        double best_speedup = 1.0;
        std::uint32_t best_observations = 0u;

        for (const auto strategy : {
                 PartitionStrategy::blind_sharded,
                 PartitionStrategy::role_split,
                 PartitionStrategy::reduce_on_gpu,
                 PartitionStrategy::projection_sharded,
                 PartitionStrategy::tpu_like}) {
            const auto strategy_it = bucket->find(to_string(strategy));
            if (strategy_it == bucket->end()) {
                continue;
            }

            const auto& stats = strategy_it->second;
            const double baseline_runtime_us =
                auto_it == bucket->end() ? stats.average_runtime_us : auto_it->second.average_runtime_us;
            const double baseline_head_runtime_us =
                auto_it == bucket->end() ? stats.average_head_runtime_us : auto_it->second.average_head_runtime_us;
            const double score = strategy_selection_score(
                workload,
                stats.observations,
                stats.successful_observations,
                stats.average_runtime_us,
                stats.average_head_runtime_us,
                stats.average_speedup_vs_reference,
                stats.average_successful_operation_ratio,
                baseline_runtime_us,
                baseline_head_runtime_us,
                apply_family_aging ? family_freshness_weight(feedback_epoch_, stats.last_update_epoch) : 1.0);
            if (score > best_score ||
                (score == best_score && stats.average_speedup_vs_reference > best_speedup)) {
                best_strategy = strategy;
                best_score = score;
                best_speedup = stats.average_speedup_vs_reference;
                best_observations = stats.observations;
            }
        }

        if (best_strategy == PartitionStrategy::auto_balanced) {
            return {
                PartitionStrategy::auto_balanced,
                PlanStrategySource::heuristic_auto,
                auto_it == bucket->end() ? 0.35 : 0.60,
                auto_it == bucket->end() ? "no learned strategy baseline" : "learned baseline favored auto",
                true};
        }
        if (best_score <= (auto_score + margin)) {
            return {
                PartitionStrategy::auto_balanced,
                PlanStrategySource::heuristic_auto,
                auto_it == bucket->end() ? 0.35 : 0.55,
                "learned alternative margin too small",
                true};
        }
        if (best_observations == 0u) {
            return {
                PartitionStrategy::auto_balanced,
                PlanStrategySource::heuristic_auto,
                auto_it == bucket->end() ? 0.35 : 0.50,
                "candidate strategy lacked observations",
                true};
        }
        const auto best_it = bucket->find(to_string(best_strategy));
        const auto freshness =
            apply_family_aging && best_it != bucket->end()
                ? family_freshness_weight(feedback_epoch_, best_it->second.last_update_epoch)
                : 1.0;
        const double confidence =
            best_it == bucket->end()
                ? 0.40
                : strategy_decision_confidence(
                      best_it->second.observations,
                      best_it->second.successful_observations,
                      best_it->second.average_successful_operation_ratio,
                      best_score - auto_score,
                      apply_family_aging,
                      freshness);
        return {
            best_strategy,
            apply_family_aging ? PlanStrategySource::family_learning : PlanStrategySource::exact_learning,
            confidence,
            std::string(apply_family_aging ? "family learned strategy " : "exact learned strategy ") +
                to_string(best_strategy),
            true};
    };

    const auto exact_resolution = resolve_from_bucket(
        exact_it == strategy_stats_.end() ? nullptr : &exact_it->second,
        true,
        kMinimumAutoBaselineObservations,
        kLearnedStrategyMargin,
        false);
    if (exact_resolution.strategy != PartitionStrategy::auto_balanced) {
        return {
            exact_resolution.strategy,
            exact_resolution.source,
            apply_confidence_calibration(
                confidence_calibration_stats_,
                workload,
                exact_resolution.source,
                exact_resolution.confidence),
            exact_resolution.reason};
    }

    const auto family_resolution = resolve_from_bucket(
        family_it == family_strategy_stats_.end() ? nullptr : &family_it->second,
        false,
        kMinimumFamilyAutoBaselineObservations,
        kLearnedFamilyStrategyMargin,
        true);
    if (family_resolution.strategy != PartitionStrategy::auto_balanced) {
        return {
            family_resolution.strategy,
            family_resolution.source,
            apply_confidence_calibration(
                confidence_calibration_stats_,
                workload,
                family_resolution.source,
                family_resolution.confidence),
            family_resolution.reason};
    }

    const auto resolve_from_heuristic_hint = [&]() -> std::optional<ResolvedStrategyDecision> {
        if (!has_partitionable_topology(graphs) ||
            !workload.heuristic_partition_hint.has_value() ||
            *workload.heuristic_partition_hint == PartitionStrategy::auto_balanced) {
            return std::nullopt;
        }

        const double confidence = std::clamp(
            workload.heuristic_partition_hint_confidence > 0.0 ? workload.heuristic_partition_hint_confidence : 0.58,
            0.35,
            0.88);
        return ResolvedStrategyDecision{
            *workload.heuristic_partition_hint,
            PlanStrategySource::heuristic_auto,
            confidence,
            workload.heuristic_partition_hint_reason.empty()
                ? ("graph-aware heuristic favored " + to_string(*workload.heuristic_partition_hint))
                : workload.heuristic_partition_hint_reason};
    };

    if (!exact_resolution.has_baseline && family_resolution.has_baseline) {
        if (const auto hinted = resolve_from_heuristic_hint(); hinted.has_value()) {
            return {
                hinted->strategy,
                hinted->source,
                apply_confidence_calibration(
                    confidence_calibration_stats_,
                    workload,
                    hinted->source,
                    hinted->confidence),
                hinted->reason};
        }
    }

    if (exact_resolution.has_baseline || family_resolution.has_baseline) {
        if (exact_resolution.has_baseline) {
            return {
                PartitionStrategy::auto_balanced,
                exact_resolution.source,
                apply_confidence_calibration(
                    confidence_calibration_stats_,
                    workload,
                    exact_resolution.source,
                    exact_resolution.confidence),
                exact_resolution.reason};
        }
        return {
            PartitionStrategy::auto_balanced,
            family_resolution.source,
            apply_confidence_calibration(
                confidence_calibration_stats_,
                workload,
                family_resolution.source,
                family_resolution.confidence),
            family_resolution.reason};
    }

    if (const auto hinted = resolve_from_heuristic_hint(); hinted.has_value()) {
        return {
            hinted->strategy,
            hinted->source,
            apply_confidence_calibration(
                confidence_calibration_stats_,
                workload,
                hinted->source,
                hinted->confidence),
            hinted->reason};
    }

    return {
        PartitionStrategy::auto_balanced,
        PlanStrategySource::heuristic_auto,
        apply_confidence_calibration(
            confidence_calibration_stats_,
            workload,
            PlanStrategySource::heuristic_auto,
            0.30),
        "no learning history, using heuristic auto"};
}

std::filesystem::path Planner::default_cache_path() {
    try {
        return std::filesystem::temp_directory_path() / "jakal_core_plan_cache.tsv";
    } catch (const std::exception&) {
        return std::filesystem::path("jakal_core_plan_cache.tsv");
    }
}

ExecutionPlan Planner::build_plan(const WorkloadSpec& workload, const std::vector<HardwareGraph>& graphs) {
    load_cache();

    const auto strategy_decision = resolve_partition_strategy(workload, graphs);
    const auto effective_strategy = strategy_decision.strategy;
    auto effective_workload = workload;
    effective_workload.partition_strategy = effective_strategy;

    ExecutionPlan plan;
    plan.signature = build_signature(effective_workload, graphs);
    plan.resolved_partition_strategy = effective_strategy;
    plan.strategy_source = strategy_decision.source;
    plan.strategy_confidence = strategy_decision.confidence;
    plan.strategy_reason = strategy_decision.reason;

    std::unordered_map<std::string, HardwareGraph> graph_lookup;
    for (const auto& graph : graphs) {
        graph_lookup.emplace(graph.uid, graph);
    }

    if (const auto it = cache_.find(plan.signature); it != cache_.end()) {
        bool cache_hit = true;
        for (const auto& cached : it->second) {
            if (!graph_lookup.contains(cached.device_uid)) {
                cache_hit = false;
                break;
            }
        }
        if (cache_hit) {
            plan.loaded_from_cache = true;
            for (const auto& cached : it->second) {
                plan.allocations.push_back(PlanAllocation{
                    graph_lookup.at(cached.device_uid),
                    cached.ratio,
                    cached.score});
            }
            return plan;
        }
    }

    struct Candidate {
        HardwareGraph graph;
        double score = 0.0;
        double ratio = 0.0;
        double cap = 1.0;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(graphs.size());

    const double compute_weight = compute_weight_from_workload(effective_workload);
    const double transfer_weight = effective_workload.latency_sensitive ? 0.25 : 0.12;
    const double memory_weight = std::max(0.10, 1.0 - compute_weight - transfer_weight);

    for (const auto& graph : graphs) {
        const auto components = score_graph(graph, effective_workload);

        double score = (components.parallel_score * compute_weight) +
                       (components.memory_score * memory_weight);
        score *= components.graph_bonus;
        score /= (1.0 + (components.transfer_penalty * (effective_workload.latency_sensitive ? 1.4 : 0.8)));

        if (components.memory_fit_cap < 0.05) {
            score *= components.memory_fit_cap;
        }

        if (score > 0.0) {
            candidates.push_back(Candidate{
                graph,
                score,
                0.0,
                std::max(0.0, components.memory_fit_cap)});
        }
    }

    if (candidates.empty() && !graphs.empty()) {
        plan.allocations.push_back(PlanAllocation{graphs.front(), 1.0, 1.0});
        cache_[plan.signature] = {
            CachedAllocation{graphs.front().uid, 1.0, 1.0}};
        persist_cache();
        return plan;
    }

    if (candidates.empty()) {
        return plan;
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& left, const Candidate& right) {
        return left.score > right.score;
    });

    const Candidate fallback_candidate = candidates.front();

    if (effective_workload.latency_sensitive && candidates.size() > 2) {
        candidates.resize(2);
    }

    const auto normalize_candidate_ratios = [&]() {
        double total_ratio = 0.0;
        for (const auto& candidate : candidates) {
            total_ratio += candidate.ratio;
        }

        if (total_ratio <= 0.0 && !candidates.empty()) {
            candidates.front().ratio = 1.0;
            total_ratio = 1.0;
        }

        if (total_ratio > 0.0) {
            for (auto& candidate : candidates) {
                candidate.ratio /= total_ratio;
            }
        }
    };

    double total_score = 0.0;
    for (const auto& candidate : candidates) {
        total_score += candidate.score;
    }
    if (total_score <= 0.0) {
        total_score = 1.0;
    }

    for (auto& candidate : candidates) {
        candidate.ratio = std::min(candidate.score / total_score, candidate.cap);
    }

    normalize_candidate_ratios();

    if (effective_workload.partition_strategy == PartitionStrategy::auto_balanced && candidates.size() > 1u) {
        const auto enforce_group_floor = [&](const auto& predicate, const double target_floor) {
            if (!(target_floor > 0.0) || !(target_floor < 1.0)) {
                return;
            }

            double group_ratio = 0.0;
            double other_ratio = 0.0;
            for (const auto& candidate : candidates) {
                if (predicate(candidate)) {
                    group_ratio += candidate.ratio;
                } else {
                    other_ratio += candidate.ratio;
                }
            }

            if (group_ratio >= target_floor || group_ratio <= 0.0 || other_ratio <= 0.0) {
                return;
            }

            const double scale_group = target_floor / group_ratio;
            const double scale_other = (1.0 - target_floor) / other_ratio;
            for (auto& candidate : candidates) {
                candidate.ratio *= predicate(candidate) ? scale_group : scale_other;
            }
        };

        const bool has_host = std::any_of(candidates.begin(), candidates.end(), [](const Candidate& candidate) {
            return is_host_graph(candidate.graph);
        });
        const bool has_accelerator =
            std::any_of(candidates.begin(), candidates.end(), [](const Candidate& candidate) {
                return !is_host_graph(candidate.graph);
            });

        if (has_host && has_accelerator) {
            enforce_group_floor(
                [](const Candidate& candidate) { return is_host_graph(candidate.graph); },
                auto_host_ratio_floor(effective_workload));
            enforce_group_floor(
                [](const Candidate& candidate) { return !is_host_graph(candidate.graph); },
                auto_accelerator_ratio_floor(effective_workload));
            normalize_candidate_ratios();
        }
    }

    if (candidates.size() > 1) {
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(), [](const Candidate& candidate) {
                return candidate.ratio < kMinShardRatio;
            }),
            candidates.end());
    }

    if (candidates.empty()) {
        candidates.push_back(fallback_candidate);
        candidates.front().ratio = 1.0;
    }

    normalize_candidate_ratios();

    if (uses_explicit_partition_strategy(effective_workload)) {
        const auto host_graph_it = std::find_if(graphs.begin(), graphs.end(), [](const HardwareGraph& graph) {
            return is_host_graph(graph);
        });
        const auto accelerator_graph_it = std::find_if(graphs.begin(), graphs.end(), [](const HardwareGraph& graph) {
            return !is_host_graph(graph);
        });
        if (host_graph_it != graphs.end() && accelerator_graph_it != graphs.end()) {
            const auto score_for_graph = [&](const HardwareGraph& graph) {
                const auto candidate_it = std::find_if(candidates.begin(), candidates.end(), [&](const Candidate& candidate) {
                    return candidate.graph.uid == graph.uid;
                });
                if (candidate_it != candidates.end()) {
                    return std::max(candidate_it->score, candidate_it->cap);
                }

                const auto components = score_graph(graph, effective_workload);
                double score = (components.parallel_score * compute_weight) +
                               (components.memory_score * memory_weight);
                score *= components.graph_bonus;
                score /= (1.0 + (components.transfer_penalty * (effective_workload.latency_sensitive ? 1.4 : 0.8)));
                if (components.memory_fit_cap < 0.05) {
                    score *= components.memory_fit_cap;
                }
                return std::max(score, components.memory_fit_cap);
            };

            const auto bias = partition_allocation_bias(effective_workload.partition_strategy);
            const double total = std::max(1.0e-9, bias.host_ratio + bias.accelerator_ratio);
            const double host_score = score_for_graph(*host_graph_it);
            const double accelerator_score = score_for_graph(*accelerator_graph_it);

            candidates.clear();
            candidates.push_back(Candidate{
                *host_graph_it,
                host_score,
                bias.host_ratio / total,
                1.0});
            candidates.push_back(Candidate{
                *accelerator_graph_it,
                accelerator_score,
                bias.accelerator_ratio / total,
                1.0});
        }
    }

    std::vector<CachedAllocation> cached_allocations;
    cached_allocations.reserve(candidates.size());

    for (const auto& candidate : candidates) {
        plan.allocations.push_back(PlanAllocation{
            candidate.graph,
            candidate.ratio,
            candidate.score});
        cached_allocations.push_back(CachedAllocation{
            candidate.graph.uid,
            candidate.ratio,
            candidate.score});
    }

    cache_[plan.signature] = std::move(cached_allocations);
    persist_cache();
    return plan;
}

void Planner::ingest_strategy_feedback(
    const WorkloadSpec& workload,
    const std::vector<HardwareGraph>& graphs,
    const StrategyFeedbackSample& feedback) {
    load_cache();
    ++feedback_epoch_;

    const auto ingest_into = [&](auto& store, const std::string& key) {
        auto& stats = store[key][to_string(feedback.strategy)];
        ++stats.observations;
        if (feedback.all_succeeded) {
            ++stats.successful_observations;
        }
        stats.last_update_epoch = feedback_epoch_;

        const double sample_count = static_cast<double>(stats.observations);
        const double runtime_sample = std::max(feedback.total_runtime_us, 0.0);
        const double head_runtime_sample =
            feedback.head_runtime_us > 0.0 ? feedback.head_runtime_us : std::max(feedback.total_runtime_us, 0.0);
        stats.average_runtime_us += (runtime_sample - stats.average_runtime_us) / sample_count;
        stats.average_head_runtime_us += (head_runtime_sample - stats.average_head_runtime_us) / sample_count;
        stats.average_speedup_vs_reference +=
            (feedback.speedup_vs_reference - stats.average_speedup_vs_reference) / sample_count;
        const double operation_success_ratio = std::clamp(feedback.successful_operation_ratio, 0.0, 1.0);
        stats.average_successful_operation_ratio +=
            (operation_success_ratio - stats.average_successful_operation_ratio) / sample_count;
    };

    ingest_into(strategy_stats_, build_learning_key(workload, graphs));
    ingest_into(family_strategy_stats_, build_family_learning_key(workload, graphs));

    if (feedback.strategy_source != PlanStrategySource::explicit_request) {
        auto& stats = confidence_calibration_stats_[build_confidence_calibration_key(
            workload,
            feedback.strategy_source,
            feedback.planned_confidence)];
        ++stats.observations;
        if (feedback.all_succeeded && !feedback.runtime_regressed) {
            ++stats.successful_observations;
        }
        if (feedback.rolled_back_to_auto) {
            ++stats.rollback_observations;
        }
        const double sample_count = static_cast<double>(stats.observations);
        const double op_success = std::clamp(feedback.successful_operation_ratio, 0.0, 1.0);
        stats.average_successful_operation_ratio +=
            (op_success - stats.average_successful_operation_ratio) / sample_count;
        const double runtime_regression = feedback.runtime_regressed
                                              ? std::max(1.05, 1.0 / std::max(0.01, feedback.speedup_vs_reference))
                                              : 1.0;
        stats.average_runtime_regression +=
            (runtime_regression - stats.average_runtime_regression) / sample_count;
    }

    persist_cache();
}

void Planner::load_cache() {
    if (cache_loaded_) {
        return;
    }
    cache_loaded_ = true;

    std::ifstream input(cache_path_);
    std::string line;
    if (input.is_open()) {
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }

            const auto fields = split_tab(line);
            if (fields.size() != 4) {
                continue;
            }

            try {
                cache_[fields[0]].push_back(CachedAllocation{
                    fields[1],
                    std::stod(fields[2]),
                    std::stod(fields[3])});
            } catch (const std::exception&) {
                continue;
            }
        }
    }

    std::ifstream strategy_input(strategy_cache_path_for(cache_path_));
    if (!strategy_input.is_open()) {
        strategy_input.clear();
    }

    const auto load_strategy_store = [&](std::ifstream& stream, auto& store) {
        std::string strategy_line;
        while (std::getline(stream, strategy_line)) {
            if (strategy_line.empty() || strategy_line[0] == '#') {
                continue;
            }

            const auto fields = split_tab(strategy_line);
            if (fields.size() != 6 && fields.size() != 8 && fields.size() != 9) {
                continue;
            }

            try {
                StrategyStats stats;
                stats.observations = static_cast<std::uint32_t>(std::stoul(fields[2]));
                stats.successful_observations = static_cast<std::uint32_t>(std::stoul(fields[3]));
                stats.average_runtime_us = std::stod(fields[4]);
                if (fields.size() == 9) {
                    stats.average_head_runtime_us = std::stod(fields[5]);
                    stats.average_speedup_vs_reference = std::stod(fields[6]);
                    stats.average_successful_operation_ratio = std::stod(fields[7]);
                    stats.last_update_epoch = static_cast<std::uint64_t>(std::stoull(fields[8]));
                } else if (fields.size() == 8) {
                    stats.average_head_runtime_us = std::stod(fields[5]);
                    stats.average_speedup_vs_reference = std::stod(fields[6]);
                    stats.average_successful_operation_ratio = std::stod(fields[7]);
                    stats.last_update_epoch = stats.observations;
                } else {
                    stats.average_head_runtime_us = stats.average_runtime_us;
                    stats.average_speedup_vs_reference = std::stod(fields[5]);
                    stats.average_successful_operation_ratio =
                        stats.observations == 0u
                            ? 1.0
                            : static_cast<double>(stats.successful_observations) / static_cast<double>(stats.observations);
                    stats.last_update_epoch = stats.observations;
                }
                store[fields[0]][fields[1]] = stats;
                feedback_epoch_ = std::max(feedback_epoch_, stats.last_update_epoch);
            } catch (const std::exception&) {
                continue;
            }
        }
    };

    if (strategy_input.is_open()) {
        load_strategy_store(strategy_input, strategy_stats_);
    }

    std::ifstream family_strategy_input(family_strategy_cache_path_for(cache_path_));
    if (family_strategy_input.is_open()) {
        load_strategy_store(family_strategy_input, family_strategy_stats_);
    }

    std::ifstream calibration_input(confidence_calibration_cache_path_for(cache_path_));
    if (calibration_input.is_open()) {
        std::string calibration_line;
        while (std::getline(calibration_input, calibration_line)) {
            if (calibration_line.empty() || calibration_line[0] == '#') {
                continue;
            }
            const auto fields = split_tab(calibration_line);
            if (fields.size() != 6) {
                continue;
            }
            try {
                confidence_calibration_stats_[fields[0]] = ConfidenceCalibrationStats{
                    static_cast<std::uint32_t>(std::stoul(fields[1])),
                    static_cast<std::uint32_t>(std::stoul(fields[2])),
                    static_cast<std::uint32_t>(std::stoul(fields[3])),
                    std::stod(fields[4]),
                    std::stod(fields[5])};
            } catch (const std::exception&) {
                continue;
            }
        }
    }
}

void Planner::persist_cache() const {
    const auto parent = cache_path_.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }

    std::ofstream output(cache_path_, std::ios::trunc);
    if (!output.is_open()) {
        return;
    }

    output << "# signature\tdevice_uid\tratio\tscore\n";
    for (const auto& [signature, allocations] : cache_) {
        for (const auto& allocation : allocations) {
            output << signature << '\t'
                   << allocation.device_uid << '\t'
                   << allocation.ratio << '\t'
                   << allocation.score << '\n';
        }
    }

    std::ofstream strategy_output(strategy_cache_path_for(cache_path_), std::ios::trunc);
    if (!strategy_output.is_open()) {
        return;
    }

    const auto persist_strategy_store = [](std::ofstream& output, const auto& store) {
        output
            << "# learning_key\tstrategy\tobservations\tsuccesses\tavg_runtime_us\tavg_head_runtime_us\tavg_speedup\tavg_operation_success\tlast_update_epoch\n";
        for (const auto& [learning_key, strategies] : store) {
            for (const auto& [strategy, stats] : strategies) {
                output << learning_key << '\t'
                       << strategy << '\t'
                       << stats.observations << '\t'
                       << stats.successful_observations << '\t'
                       << stats.average_runtime_us << '\t'
                       << stats.average_head_runtime_us << '\t'
                       << stats.average_speedup_vs_reference << '\t'
                       << stats.average_successful_operation_ratio << '\t'
                       << stats.last_update_epoch << '\n';
            }
        }
    };

    persist_strategy_store(strategy_output, strategy_stats_);

    std::ofstream family_strategy_output(family_strategy_cache_path_for(cache_path_), std::ios::trunc);
    if (!family_strategy_output.is_open()) {
        return;
    }
    persist_strategy_store(family_strategy_output, family_strategy_stats_);

    std::ofstream calibration_output(confidence_calibration_cache_path_for(cache_path_), std::ios::trunc);
    if (!calibration_output.is_open()) {
        return;
    }
    calibration_output
        << "# calibration_key\tobservations\tsuccesses\trollbacks\tavg_operation_success\tavg_runtime_regression\n";
    for (const auto& [calibration_key, stats] : confidence_calibration_stats_) {
        calibration_output << calibration_key << '\t'
                           << stats.observations << '\t'
                           << stats.successful_observations << '\t'
                           << stats.rollback_observations << '\t'
                           << stats.average_successful_operation_ratio << '\t'
                           << stats.average_runtime_regression << '\n';
    }
}

}  // namespace jakal

