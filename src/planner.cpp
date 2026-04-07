#include "jakal/planner.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
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

std::string build_signature(const WorkloadSpec& workload, const std::vector<HardwareGraph>& graphs) {
    std::vector<std::string> fingerprints;
    fingerprints.reserve(graphs.size());
    for (const auto& graph : graphs) {
        fingerprints.push_back(graph.uid + ":" + structural_fingerprint(graph));
    }
    std::sort(fingerprints.begin(), fingerprints.end());

    std::ostringstream signature;
    signature << workload.name << '|'
              << to_string(workload.kind) << '|'
              << workload.dataset_tag << '|'
              << workload.working_set_bytes << '|'
              << workload.host_exchange_bytes << '|'
              << std::fixed << std::setprecision(3) << workload.estimated_flops << '|'
              << workload.batch_size << '|'
              << workload.latency_sensitive << '|'
              << workload.prefer_unified_memory << '|'
              << workload.matrix_friendly;

    for (const auto& fingerprint : fingerprints) {
        signature << '|' << fingerprint;
    }

    return signature.str();
}

double compute_weight_from_workload(const WorkloadSpec& workload) {
    if (workload.working_set_bytes == 0 || workload.estimated_flops <= 0.0) {
        return workload.matrix_friendly ? 0.72 : 0.60;
    }

    const double intensity = workload.estimated_flops / static_cast<double>(workload.working_set_bytes);
    return std::clamp(intensity / 64.0, 0.25, workload.matrix_friendly ? 0.88 : 0.82);
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
        graph_bonus,
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

Planner::Planner(std::filesystem::path cache_path)
    : cache_path_(std::move(cache_path)) {}

std::filesystem::path Planner::default_cache_path() {
    try {
        return std::filesystem::temp_directory_path() / "jakal_core_plan_cache.tsv";
    } catch (const std::exception&) {
        return std::filesystem::path("jakal_core_plan_cache.tsv");
    }
}

ExecutionPlan Planner::build_plan(const WorkloadSpec& workload, const std::vector<HardwareGraph>& graphs) {
    load_cache();

    ExecutionPlan plan;
    plan.signature = build_signature(workload, graphs);

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

    const double compute_weight = compute_weight_from_workload(workload);
    const double transfer_weight = workload.latency_sensitive ? 0.25 : 0.12;
    const double memory_weight = std::max(0.10, 1.0 - compute_weight - transfer_weight);

    for (const auto& graph : graphs) {
        const auto components = score_graph(graph, workload);

        double score = (components.parallel_score * compute_weight) +
                       (components.memory_score * memory_weight);
        score *= components.graph_bonus;
        score /= (1.0 + (components.transfer_penalty * (workload.latency_sensitive ? 1.4 : 0.8)));

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

    if (workload.latency_sensitive && candidates.size() > 2) {
        candidates.resize(2);
    }

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

    total_ratio = 0.0;
    for (const auto& candidate : candidates) {
        total_ratio += candidate.ratio;
    }
    if (total_ratio > 0.0) {
        for (auto& candidate : candidates) {
            candidate.ratio /= total_ratio;
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

void Planner::load_cache() {
    if (cache_loaded_) {
        return;
    }
    cache_loaded_ = true;

    std::ifstream input(cache_path_);
    if (!input.is_open()) {
        return;
    }

    std::string line;
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
}

}  // namespace jakal

