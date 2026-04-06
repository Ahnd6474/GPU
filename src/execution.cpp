#include "gpu/execution.hpp"

#include "gpu/device.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace gpu {
namespace {

constexpr std::uint64_t kKiB = 1024ull;
constexpr std::uint64_t kMiB = 1024ull * 1024ull;
constexpr std::uint64_t kGiB = 1024ull * 1024ull * 1024ull;
constexpr double kBytesPerGb = 1.0e9;
constexpr double kMicrosecondsPerSecond = 1.0e6;
constexpr std::uint32_t kLearningWarmupSamples = 3u;

struct ValidationResult {
    double reference_latency_us = 0.0;
    double candidate_latency_us = 0.0;
    double relative_error = 0.0;
    double reference_spread_us = 0.0;
    double candidate_spread_us = 0.0;
    std::uint32_t samples = 0;
};

struct SampleProfile {
    double average_us = 0.0;
    double spread_us = 0.0;
    std::uint32_t samples = 0;
};

std::string join_csv(const std::vector<std::string>& values);

std::filesystem::path performance_cache_path_for(const std::filesystem::path& cache_path) {
    if (cache_path.empty()) {
        return std::filesystem::path("gpu_runtime_execution_perf_cache.tsv");
    }
    auto path = cache_path;
    path += ".perf";
    return path;
}

std::string shape_bucket_for(const OperationSpec& operation) {
    std::ostringstream stream;
    stream << to_string(operation.op_class);
    for (const auto extent : operation.extents) {
        std::uint64_t bucket = 1;
        while (bucket < extent) {
            bucket <<= 1u;
        }
        stream << ':' << bucket;
    }
    const auto bytes_bucket = std::max<std::uint64_t>(1ull, operation.input_bytes / (4ull * kMiB));
    stream << "|b" << bytes_bucket;
    return stream.str();
}

std::string summarize_graph_set(const std::vector<HardwareGraph>& graphs) {
    std::vector<std::string> fingerprints;
    fingerprints.reserve(graphs.size());
    for (const auto& graph : graphs) {
        fingerprints.push_back(structural_fingerprint(graph));
    }
    std::sort(fingerprints.begin(), fingerprints.end());
    std::ostringstream stream;
    for (const auto& fingerprint : fingerprints) {
        stream << fingerprint << '|';
    }
    return stream.str();
}

std::string performance_key(
    const std::string& graph_set_signature,
    const WorkloadSpec& workload,
    const SystemProfile& system,
    const std::string& shape_bucket,
    const ExecutionConfig& config) {
    std::ostringstream stream;
    stream << graph_set_signature << '|'
           << to_string(workload.kind) << '|'
           << workload.dataset_tag << '|'
           << shape_bucket << '|'
           << system.low_spec_mode << '|'
           << system.on_battery << '|'
           << system.battery_saver << '|'
           << (system.free_memory_ratio < 0.25 ? "mem-low" : system.free_memory_ratio < 0.5 ? "mem-mid" : "mem-high")
           << '|'
           << to_string(config.strategy) << '|'
           << config.primary_device_uid << '|'
           << join_csv(config.participating_devices) << '|'
           << config.queue_depth << '|'
           << config.stages << '|'
           << config.tile_x << '|'
           << config.tile_y << '|'
           << config.tile_k << '|'
           << config.logical_partitions << '|'
           << config.overlap_transfers << '|'
           << config.use_low_precision << '|'
           << std::fixed << std::setprecision(3)
           << config.queue_depth_scale << '|'
           << config.stage_scale << '|'
           << config.tile_scale << '|'
           << config.overlap_ratio << '|'
           << config.partition_intensity << '|'
           << config.precision_mix;
    return stream.str();
}

std::string backend_penalty_key(
    const std::string& graph_set_signature,
    const WorkloadSpec& workload,
    const std::string& operation_name,
    const ExecutionConfig& config) {
    std::ostringstream stream;
    stream << graph_set_signature << '|'
           << to_string(workload.kind) << '|'
           << workload.dataset_tag << '|'
           << operation_name << '|'
           << config.primary_device_uid << '|'
           << join_csv(config.participating_devices) << '|'
           << to_string(config.strategy) << '|'
           << config.logical_partitions;
    return stream.str();
}

double preset_trace_weight(const WorkloadSpec& workload) {
    double weight = workload.dataset_tag.empty() ? 1.0 : 1.35;
    switch (workload.kind) {
    case WorkloadKind::gaming:
        weight *= 1.20;
        break;
    case WorkloadKind::training:
        weight *= 1.15;
        break;
    case WorkloadKind::inference:
        weight *= 1.10;
        break;
    case WorkloadKind::image:
    case WorkloadKind::tensor:
    case WorkloadKind::custom:
    default:
        break;
    }
    return weight;
}

double clamp_unit(const double value) {
    return std::clamp(value, 0.0, 1.0);
}

double clamp_raw(const double value) {
    return std::clamp(value, -6.0, 6.0);
}

ContinuousExecutionState clamp_state(ContinuousExecutionState state) {
    state.queue_depth_raw = clamp_raw(state.queue_depth_raw);
    state.stage_raw = clamp_raw(state.stage_raw);
    state.tile_raw = clamp_raw(state.tile_raw);
    state.overlap_raw = clamp_raw(state.overlap_raw);
    state.partition_raw = clamp_raw(state.partition_raw);
    state.precision_raw = clamp_raw(state.precision_raw);
    state.single_device_logit = clamp_raw(state.single_device_logit);
    state.sharded_logit = clamp_raw(state.sharded_logit);
    state.streaming_logit = clamp_raw(state.streaming_logit);
    state.overlapped_logit = clamp_raw(state.overlapped_logit);
    return state;
}

double sigmoid(const double value) {
    const double clamped = clamp_raw(value);
    if (clamped >= 0.0) {
        const double z = std::exp(-clamped);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(clamped);
    return z / (1.0 + z);
}

double positive_scale_from_raw(const double raw) {
    return 0.50 + (1.50 * sigmoid(raw));
}

std::array<double, 4> strategy_soft_weights(const ContinuousExecutionState& state) {
    std::array<double, 4> logits{
        state.single_device_logit,
        state.sharded_logit,
        state.streaming_logit,
        state.overlapped_logit};
    const double max_logit = *std::max_element(logits.begin(), logits.end());
    double total = 0.0;
    for (auto& value : logits) {
        value = std::exp(value - max_logit);
        total += value;
    }
    if (total <= 0.0) {
        return {1.0, 0.0, 0.0, 0.0};
    }
    for (auto& value : logits) {
        value /= total;
    }
    return logits;
}

double aggregate_execution_score(const std::vector<HardwareGraph>& graphs) {
    double score = 0.0;
    for (const auto& graph : graphs) {
        const auto summary = summarize_graph(graph);
        score += static_cast<double>(std::max(summary.execution_objects, 1u)) *
                 static_cast<double>(std::max(summary.lanes_per_object, 1u));
    }
    return score;
}

SystemProfile capture_system_profile(const WorkloadSpec& workload, const std::vector<HardwareGraph>& graphs) {
    SystemProfile profile;

#ifdef _WIN32
    MEMORYSTATUSEX memory_status{};
    memory_status.dwLength = sizeof(memory_status);
    if (GlobalMemoryStatusEx(&memory_status) != 0) {
        profile.available_memory_bytes = memory_status.ullAvailPhys;
        if (memory_status.ullTotalPhys > 0) {
            profile.free_memory_ratio =
                static_cast<double>(memory_status.ullAvailPhys) / static_cast<double>(memory_status.ullTotalPhys);
        }
    }

    SYSTEM_POWER_STATUS power_status{};
    if (GetSystemPowerStatus(&power_status) != 0) {
        profile.on_battery = power_status.ACLineStatus == 0;
        profile.battery_saver = power_status.SystemStatusFlag != 0;
        if (power_status.BatteryLifePercent != 255) {
            profile.battery_percent = static_cast<double>(power_status.BatteryLifePercent);
        }
    }
#else
    (void)workload;
#endif

    std::uint64_t addressable_bytes = 0;
    std::uint64_t directly_attached_bytes = 0;
    for (const auto& graph : graphs) {
        const auto summary = summarize_graph(graph);
        addressable_bytes += summary.addressable_bytes;
        directly_attached_bytes += summary.directly_attached_bytes;
    }

    const bool low_memory_machine =
        (profile.available_memory_bytes > 0 && profile.available_memory_bytes < 6ull * kGiB) ||
        (addressable_bytes > 0 && addressable_bytes < 12ull * kGiB);
    const bool low_compute_machine = aggregate_execution_score(graphs) < 512.0;
    profile.low_spec_mode = low_memory_machine || low_compute_machine;

    const double demand_bytes =
        static_cast<double>(workload.working_set_bytes + workload.host_exchange_bytes + (workload.batch_size * 4ull * kMiB));
    if (profile.available_memory_bytes > 0) {
        const double available = static_cast<double>(profile.available_memory_bytes);
        profile.paging_risk = std::clamp((demand_bytes - (available * 0.65)) / std::max(available, 1.0), 0.0, 2.0);
    }

    if (profile.on_battery) {
        profile.sustained_slowdown *= profile.battery_percent < 25.0 ? 1.30 : 1.12;
    }
    if (profile.battery_saver) {
        profile.sustained_slowdown *= 1.20;
    }
    if (profile.free_memory_ratio < 0.20) {
        profile.sustained_slowdown *= 1.25;
    }
    if (profile.low_spec_mode) {
        profile.sustained_slowdown *= 1.15;
    }

    profile.amortization_gain = profile.low_spec_mode ? 0.80 : 0.90;
    profile.initialization_penalty_us = profile.low_spec_mode ? 1800.0 : 900.0;
    profile.stability_score =
        clamp_unit((1.0 / profile.sustained_slowdown) * (1.0 - (0.30 * std::min(profile.paging_risk, 1.0))));
    return profile;
}

std::vector<std::string> split_tab(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) {
        fields.push_back(field);
    }
    return fields;
}

std::vector<std::string> split_csv(const std::string& text) {
    if (text.empty()) {
        return {};
    }

    std::vector<std::string> fields;
    std::stringstream stream(text);
    std::string field;
    while (std::getline(stream, field, ',')) {
        if (!field.empty()) {
            fields.push_back(field);
        }
    }
    return fields;
}

std::string join_csv(const std::vector<std::string>& values) {
    std::ostringstream stream;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << values[index];
    }
    return stream.str();
}

std::uint64_t clamp_u64(const std::uint64_t value, const std::uint64_t min_value, const std::uint64_t max_value) {
    return std::min(std::max(value, min_value), max_value);
}

std::uint32_t round_down_to_multiple(std::uint32_t value, const std::uint32_t multiple) {
    if (multiple == 0) {
        return value;
    }
    value = std::max(value, multiple);
    return value - (value % multiple);
}

std::uint32_t round_up_to_multiple(std::uint32_t value, const std::uint32_t multiple) {
    if (multiple == 0) {
        return value;
    }
    const auto remainder = value % multiple;
    if (remainder == 0) {
        return value;
    }
    return value + (multiple - remainder);
}

template <typename Func>
double measure_us(Func&& func) {
    const auto start = std::chrono::steady_clock::now();
    func();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

template <typename Func>
SampleProfile profile_measurements(const std::uint32_t rounds, Func&& func) {
    constexpr std::uint32_t kWarmupRounds = 1u;
    for (std::uint32_t round = 0; round < kWarmupRounds; ++round) {
        func();
    }

    SampleProfile profile;
    if (rounds == 0u) {
        return profile;
    }

    std::vector<double> samples;
    samples.reserve(rounds);
    for (std::uint32_t round = 0; round < rounds; ++round) {
        samples.push_back(measure_us(func));
    }

    const double total = std::accumulate(samples.begin(), samples.end(), 0.0);
    profile.samples = rounds;
    profile.average_us = total / static_cast<double>(rounds);
    const auto [min_it, max_it] = std::minmax_element(samples.begin(), samples.end());
    profile.spread_us = *max_it - *min_it;
    return profile;
}

float quantize_value(const float value, const bool low_precision) {
    if (!low_precision) {
        return value;
    }
    return std::round(value * 1024.0f) / 1024.0f;
}

std::vector<float> make_pattern(const std::size_t count, const float phase) {
    std::vector<float> data(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float base = static_cast<float>((static_cast<std::uint32_t>(index * 17u + 23u) % 257u) - 128u);
        data[index] = (base / 97.0f) + (phase * 0.03125f);
    }
    return data;
}

double relative_l2_error(const std::vector<float>& reference, const std::vector<float>& candidate) {
    if (reference.empty() || reference.size() != candidate.size()) {
        return 0.0;
    }

    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double ref = static_cast<double>(reference[index]);
        const double diff = ref - static_cast<double>(candidate[index]);
        numerator += diff * diff;
        denominator += ref * ref;
    }

    if (denominator <= 1.0e-18) {
        return std::sqrt(numerator);
    }
    return std::sqrt(numerator / denominator);
}

double scalar_relative_error(const double reference, const double candidate) {
    const double denominator = std::max(std::abs(reference), 1.0e-9);
    return std::abs(reference - candidate) / denominator;
}

const HardwareGraph* find_graph(
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::string& uid) {
    const auto it = graph_lookup.find(uid);
    return it == graph_lookup.end() ? nullptr : it->second;
}

const HardwareObjectNode* find_first_node(
    const HardwareGraph& graph,
    const std::initializer_list<HardwareObjectRole> roles) {
    for (const auto role : roles) {
        for (const auto& node : graph.nodes) {
            if (node.role == role) {
                return &node;
            }
        }
    }
    return graph.nodes.empty() ? nullptr : &graph.nodes.front();
}

double host_link_gbps(const HardwareGraphSummary& summary) {
    const double link = std::max(summary.host_read_gbps, summary.host_write_gbps);
    if (link > 0.0) {
        return link;
    }
    if (summary.unified_address_space) {
        return 96.0;
    }
    if (summary.coherent_with_host) {
        return 48.0;
    }
    return 16.0;
}

double transfer_cost_us(const std::uint64_t bytes, const HardwareGraphSummary& summary) {
    if (bytes == 0) {
        return 0.0;
    }

    const double bandwidth = host_link_gbps(summary);
    const double transfer_us =
        (static_cast<double>(bytes) / (bandwidth * kBytesPerGb)) * kMicrosecondsPerSecond;
    return summary.average_transfer_cost_us + transfer_us;
}

double dispatch_cost_us(const HardwareGraphSummary& summary) {
    if (summary.average_dispatch_cost_us > 0.0) {
        return summary.average_dispatch_cost_us;
    }
    return summary.dispatch_latency_us;
}

double sync_cost_us(const HardwareGraphSummary& summary) {
    if (summary.synchronization_latency_us > 0.0) {
        return summary.synchronization_latency_us;
    }
    return summary.average_hierarchy_cost_us > 0.0 ? summary.average_hierarchy_cost_us : 4.0;
}

double estimate_device_gflops(
    const HardwareGraphSummary& summary,
    const OperationSpec& operation,
    const bool low_precision) {
    const double execution_objects = static_cast<double>(std::max(summary.execution_objects, 1u));
    const double lanes = static_cast<double>(std::max(summary.lanes_per_object, 1u));
    const double contexts = static_cast<double>(std::max(summary.resident_contexts, 1u));
    const double clock_ghz =
        static_cast<double>(summary.clock_mhz == 0 ? (summary.host_visible ? 2400u : 1000u) : summary.clock_mhz) /
        1000.0;
    double ops_per_cycle = execution_objects * lanes * std::log2(contexts + 1.0);
    if (operation.matrix_friendly) {
        ops_per_cycle += static_cast<double>(summary.matrix_units) * 32.0;
    }
    if (low_precision && (summary.supports_fp16 || summary.supports_bf16 || summary.supports_int8)) {
        ops_per_cycle *= 1.35;
    }
    return std::max(1.0, ops_per_cycle * clock_ghz);
}

double expected_relative_error(const OperationSpec& operation, const ExecutionConfig& config) {
    double error = 0.0;
    if (config.use_low_precision) {
        error += operation.matrix_friendly ? 7.5e-4 : 4.0e-4;
    }
    if (config.strategy == ExecutionStrategy::sharded) {
        error += operation.reduction_like ? 3.0e-4 : 1.5e-4;
    }
    if (config.logical_partitions > 1) {
        error += operation.reduction_like ? 2.0e-4 : 5.0e-5;
    }
    if (operation.reduction_like) {
        error += 1.0e-4;
    }
    return error;
}

const WorkloadTensor* find_workload_tensor(const WorkloadGraph& workload_graph, const std::string_view tensor_id) {
    const auto it = std::find_if(
        workload_graph.tensors.begin(),
        workload_graph.tensors.end(),
        [&](const WorkloadTensor& tensor) { return tensor.id == tensor_id; });
    return it == workload_graph.tensors.end() ? nullptr : &(*it);
}

const TensorLifetime* find_tensor_lifetime(const WorkloadGraph& workload_graph, const std::string_view tensor_id) {
    const auto it = std::find_if(
        workload_graph.lifetimes.begin(),
        workload_graph.lifetimes.end(),
        [&](const TensorLifetime& lifetime) { return lifetime.tensor_id == tensor_id; });
    return it == workload_graph.lifetimes.end() ? nullptr : &(*it);
}

std::unordered_map<std::string, std::uint32_t> build_operation_index(const WorkloadGraph& workload_graph) {
    std::unordered_map<std::string, std::uint32_t> indices;
    indices.reserve(workload_graph.operations.size());
    for (std::uint32_t index = 0; index < workload_graph.operations.size(); ++index) {
        indices.emplace(workload_graph.operations[index].name, index);
    }
    return indices;
}

std::uint32_t operation_index_or_default(
    const std::unordered_map<std::string, std::uint32_t>& indices,
    const std::string& operation_name,
    const std::uint32_t fallback) {
    const auto it = indices.find(operation_name);
    return it == indices.end() ? fallback : it->second;
}

std::vector<std::string> resident_devices_for_tensor(
    const WorkloadTensor& tensor,
    const ExecutionConfig& current_config,
    const std::unordered_map<std::string, ExecutionConfig>& selected_configs) {
    if (tensor.producer_operation.empty()) {
        return {"host"};
    }

    if (const auto config_it = selected_configs.find(tensor.producer_operation); config_it != selected_configs.end()) {
        if (!config_it->second.participating_devices.empty()) {
            return config_it->second.participating_devices;
        }
        if (!config_it->second.primary_device_uid.empty()) {
            return {config_it->second.primary_device_uid};
        }
    }

    if (!current_config.participating_devices.empty()) {
        return current_config.participating_devices;
    }
    if (!current_config.primary_device_uid.empty()) {
        return {current_config.primary_device_uid};
    }
    return {"host"};
}

std::uint64_t bytes_for_tensor_mapping(
    const WorkloadTensor& tensor,
    const OperationSpec& operation,
    const ExecutionConfig& config,
    const double device_ratio,
    const std::uint32_t partition_count) {
    double share = 1.0;
    if (config.strategy == ExecutionStrategy::sharded && operation.parallelizable && !tensor.persistent) {
        share = std::clamp(device_ratio, 0.05, 1.0);
    } else if (config.strategy == ExecutionStrategy::streaming && operation.streaming_friendly && !tensor.persistent) {
        share = 0.70;
    }

    if (partition_count > 1u && (operation.parallelizable || tensor.temporary) && !tensor.persistent) {
        share /= static_cast<double>(partition_count);
    }

    share = std::clamp(share, 1.0 / static_cast<double>(std::max(partition_count, 1u)), 1.0);
    return std::max<std::uint64_t>(
        1ull,
        static_cast<std::uint64_t>(std::ceil(static_cast<double>(tensor.bytes) * share)));
}

double scheduled_transfer_latency_us(
    const std::uint64_t bytes,
    const HardwareGraphSummary* source_summary,
    const HardwareGraphSummary& target_summary,
    const bool cross_device) {
    if (bytes == 0u) {
        return 0.0;
    }

    if (source_summary == nullptr || !cross_device) {
        return transfer_cost_us(bytes, target_summary);
    }

    const double source_bw = host_link_gbps(*source_summary);
    const double target_bw = host_link_gbps(target_summary);
    const double bandwidth = std::max(8.0, std::min(source_bw, target_bw) * 0.75);
    const double base_latency =
        std::max(sync_cost_us(*source_summary), dispatch_cost_us(target_summary)) +
        target_summary.average_transfer_cost_us +
        source_summary->average_transfer_cost_us;
    return base_latency +
           ((static_cast<double>(bytes) / (bandwidth * kBytesPerGb)) * kMicrosecondsPerSecond);
}

std::string build_report_signature(
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::vector<OperationSpec>& operations) {
    std::ostringstream stream;
    stream << placement.signature << '|'
           << workload.name << '|'
           << to_string(workload.kind) << '|'
           << operations.size();
    for (const auto& operation : operations) {
        stream << '|' << operation.name << ':' << to_string(operation.op_class);
        for (const auto extent : operation.extents) {
            stream << ':' << extent;
        }
    }
    return stream.str();
}

std::string build_config_signature(const std::string& report_signature, const ExecutionConfig& config) {
    std::ostringstream stream;
    stream << report_signature << '|'
           << config.operation_name << '|'
           << to_string(config.strategy) << '|'
           << config.primary_device_uid << '|'
           << join_csv(config.participating_devices) << '|'
           << config.queue_depth << '|'
           << config.stages << '|'
           << config.tile_x << '|'
           << config.tile_y << '|'
           << config.tile_k << '|'
           << config.logical_partitions << '|'
           << config.overlap_transfers << '|'
           << config.use_low_precision << '|'
           << std::fixed << std::setprecision(6) << config.target_error_tolerance << '|'
           << std::setprecision(3)
           << config.queue_depth_scale << '|'
           << config.stage_scale << '|'
           << config.tile_scale << '|'
           << config.overlap_ratio << '|'
           << config.partition_intensity << '|'
           << config.precision_mix;
    return stream.str();
}

std::uint32_t choose_queue_depth(const HardwareGraphSummary& summary) {
    if (summary.queue_slots == 0) {
        return 1;
    }
    return std::clamp(summary.queue_slots / 32u, 1u, 8u);
}

std::uint32_t choose_logical_partitions(
    const HardwareGraphSummary& summary,
    const OperationSpec& operation,
    const double partition_intensity) {
    if (!operation.parallelizable) {
        return 1;
    }

    const std::uint32_t structural_parallelism =
        std::max({summary.execution_objects, summary.resident_contexts, summary.lanes_per_object / 8u, 1u});
    const std::uint32_t queue_parallelism = std::max(summary.queue_slots / 64u, 1u);
    const std::uint32_t max_partitions = std::clamp(std::max(structural_parallelism, queue_parallelism), 1u, 8u);
    if (max_partitions == 1u) {
        return 1u;
    }
    return 1u + static_cast<std::uint32_t>(std::llround(partition_intensity * static_cast<double>(max_partitions - 1u)));
}

bool supports_low_precision(const HardwareGraphSummary& summary);

std::uint32_t scale_tile(const std::uint32_t base, const double scale, const std::uint32_t multiple, const std::uint32_t min_value, const std::uint32_t max_value) {
    if (base == 0) {
        return 0;
    }
    auto scaled = static_cast<std::uint32_t>(std::llround(static_cast<double>(base) * scale));
    scaled = std::max(min_value, scaled);
    if (multiple > 1u) {
        scaled = round_up_to_multiple(scaled, multiple);
    }
    return std::clamp(scaled, min_value, max_value);
}

void apply_continuous_state(
    const ContinuousExecutionState& state,
    const OperationSpec& operation,
    const HardwareGraphSummary& summary,
    ExecutionConfig& config) {
    const double queue_depth_scale = positive_scale_from_raw(state.queue_depth_raw);
    const double stage_scale = positive_scale_from_raw(state.stage_raw);
    const double tile_scale = positive_scale_from_raw(state.tile_raw);
    const double overlap_ratio = sigmoid(state.overlap_raw);
    const double partition_intensity = sigmoid(state.partition_raw);
    const double precision_mix = sigmoid(state.precision_raw);

    config.queue_depth_scale = queue_depth_scale;
    config.stage_scale = stage_scale;
    config.tile_scale = tile_scale;
    config.overlap_ratio = overlap_ratio;
    config.partition_intensity = partition_intensity;
    config.precision_mix = precision_mix;

    config.queue_depth = std::clamp(
        static_cast<std::uint32_t>(std::llround(static_cast<double>(std::max(config.queue_depth, 1u)) * queue_depth_scale)),
        1u,
        8u);
    config.stages = std::clamp(
        static_cast<std::uint32_t>(std::llround(static_cast<double>(std::max(config.stages, 1u)) * stage_scale)),
        1u,
        4u);
    config.tile_x = scale_tile(config.tile_x, tile_scale, operation.matrix_friendly ? 16u : 8u, 8u, 4096u);
    config.tile_y = scale_tile(config.tile_y, tile_scale, operation.matrix_friendly ? 8u : 1u, 1u, 1024u);
    config.tile_k = scale_tile(config.tile_k, tile_scale, operation.matrix_friendly ? 16u : 4u, 4u, 1024u);
    config.logical_partitions = choose_logical_partitions(summary, operation, partition_intensity);
    config.overlap_transfers = config.overlap_transfers || overlap_ratio >= 0.35;
    if (precision_mix >= 0.55 && supports_low_precision(summary) && operation.max_relative_error >= 4.0e-4) {
        config.use_low_precision = true;
    }
}

double compute_initialization_penalty_us(
    const ExecutionConfig& config,
    const SystemProfile& system,
    const std::unordered_map<std::string, bool>& warmed_devices) {
    double penalty = 0.0;
    for (const auto& device_uid : config.participating_devices) {
        const auto warm_it = warmed_devices.find(device_uid);
        const bool warmed = warm_it != warmed_devices.end() && warm_it->second;
        if (!warmed) {
            penalty += system.initialization_penalty_us;
        }
    }
    return penalty;
}

double compute_memory_pressure_penalty_us(
    const OperationSpec& operation,
    const ExecutionConfig& config,
    const SystemProfile& system) {
    if (system.available_memory_bytes == 0) {
        return 0.0;
    }

    double replicated_input_factor = config.strategy == ExecutionStrategy::sharded ? 1.35 : 1.0;
    if (config.strategy == ExecutionStrategy::streaming) {
        replicated_input_factor *= 0.85;
    }

    const double device_copies = static_cast<double>(std::max<std::size_t>(config.participating_devices.size(), 1u));
    const double resident_bytes =
        (static_cast<double>(operation.input_bytes) * replicated_input_factor * device_copies) +
        static_cast<double>(operation.output_bytes + operation.temporary_bytes);
    const double available = static_cast<double>(system.available_memory_bytes);
    const double excess = std::max(0.0, resident_bytes - (available * 0.55));
    return (excess / static_cast<double>(kMiB)) * (system.low_spec_mode ? 12.0 : 4.0);
}

double compute_surrogate_penalty_us(
    const OperationSpec& operation,
    const ExecutionConfig& config,
    const SystemProfile& system,
    const std::unordered_map<std::string, bool>& warmed_devices) {
    double penalty = compute_initialization_penalty_us(config, system, warmed_devices);
    penalty += compute_memory_pressure_penalty_us(operation, config, system);
    penalty += system.paging_risk * (system.low_spec_mode ? 900.0 : 300.0);

    if (system.on_battery) {
        penalty += static_cast<double>(config.participating_devices.size() - 1) * 250.0;
        penalty += static_cast<double>(std::max(config.queue_depth, 1u) - 1u) * 18.0;
    }
    if (config.logical_partitions > 1u) {
        penalty += static_cast<double>(config.logical_partitions - 1u) * (system.low_spec_mode ? 55.0 : 24.0);
    }
    if (system.low_spec_mode && config.strategy == ExecutionStrategy::sharded) {
        penalty += 500.0;
    }
    if (system.low_spec_mode && config.strategy == ExecutionStrategy::streaming) {
        penalty -= 120.0;
    }
    if (system.low_spec_mode && config.overlap_transfers) {
        penalty -= 80.0;
    }
    if (system.low_spec_mode && config.use_low_precision) {
        penalty -= 60.0;
    }
    penalty -= config.overlap_ratio * (system.low_spec_mode ? 45.0 : 20.0);
    penalty += (1.0 - system.readiness_score) * (system.initialization_penalty_us * 0.15);
    penalty += (1.0 - system.stability_score) * 240.0;
    return std::max(0.0, penalty);
}

void configure_tiles(const OperationSpec& operation, const HardwareGraphSummary& summary, ExecutionConfig& config) {
    const std::uint32_t lanes = std::max(summary.lanes_per_object, 1u);
    const std::uint32_t vector_width = std::max(summary.native_vector_bits / 32u, 1u);

    switch (operation.op_class) {
    case OperationClass::matmul:
        config.tile_x = std::clamp(round_up_to_multiple(lanes * 8u, 16u), 32u, 128u);
        config.tile_y = config.tile_x;
        config.tile_k = std::clamp(round_up_to_multiple(vector_width * 8u, 8u), 16u, 128u);
        break;
    case OperationClass::convolution_2d:
    case OperationClass::resample_2d:
        config.tile_x = std::clamp(round_down_to_multiple(lanes * 4u, 8u), 16u, 64u);
        config.tile_y = std::clamp(round_down_to_multiple(lanes * 2u, 8u), 8u, 32u);
        config.tile_k = vector_width * 8u;
        break;
    case OperationClass::reduction:
    case OperationClass::elementwise_map:
    default:
        config.tile_x = std::clamp(round_up_to_multiple(lanes * 32u, 32u), 128u, 4096u);
        config.tile_y = 1;
        config.tile_k = vector_width * 4u;
        break;
    }
}

bool supports_low_precision(const HardwareGraphSummary& summary) {
    return summary.supports_fp16 || summary.supports_bf16 || summary.supports_int8;
}

std::vector<ExecutionConfig> build_candidate_configs(
    const OperationSpec& operation,
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const SystemProfile& system,
    const ContinuousExecutionState* continuous_state,
    const std::string& report_signature) {
    std::vector<ExecutionConfig> candidates;
    if (placement.allocations.empty()) {
        return candidates;
    }

    const bool activation_driven = continuous_state != nullptr;

    auto add_candidate = [&](ExecutionConfig config) {
        config.operation_name = operation.name;
        config.target_error_tolerance = operation.max_relative_error;
        if (const auto* graph = find_graph(graph_lookup, config.primary_device_uid)) {
            const auto summary = summarize_graph(*graph);
            if (config.tile_x == 0 || config.tile_y == 0 || config.tile_k == 0) {
                configure_tiles(operation, summary, config);
            }
            if (config.queue_depth == 0) {
                config.queue_depth = choose_queue_depth(summary);
            }
            if (config.stages == 0) {
                config.stages = summary.supports_asynchronous_dispatch ? 2u : 1u;
            }
            if (continuous_state != nullptr) {
                apply_continuous_state(*continuous_state, operation, summary, config);
            }
        }
        config.signature = build_config_signature(report_signature, config);
        const auto duplicate = std::find_if(
            candidates.begin(),
            candidates.end(),
            [&](const ExecutionConfig& existing) {
                return existing.signature == config.signature;
            });
        if (duplicate == candidates.end()) {
            candidates.push_back(std::move(config));
        }
    };

    const auto* primary_graph = find_graph(graph_lookup, placement.allocations.front().device.uid);
    const auto primary_summary =
        primary_graph == nullptr ? HardwareGraphSummary{} : summarize_graph(*primary_graph);

    if (activation_driven) {
        const auto strategy_weights = strategy_soft_weights(*continuous_state);
        const bool sharded_feasible =
            operation.parallelizable &&
            placement.allocations.size() > 1 &&
            !workload.latency_sensitive &&
            !(system.low_spec_mode && system.paging_risk > 0.25);
        const bool streaming_feasible = operation.streaming_friendly;
        const bool overlapped_feasible = true;

        struct WeightedStrategy {
            ExecutionStrategy strategy;
            double weight;
        };
        std::vector<WeightedStrategy> ranked{
            {ExecutionStrategy::single_device, strategy_weights[0]},
            {ExecutionStrategy::sharded, sharded_feasible ? strategy_weights[1] : -1.0},
            {ExecutionStrategy::streaming, streaming_feasible ? strategy_weights[2] : -1.0},
            {ExecutionStrategy::overlapped, overlapped_feasible ? strategy_weights[3] : -1.0}};
        std::sort(ranked.begin(), ranked.end(), [](const WeightedStrategy& left, const WeightedStrategy& right) {
            return left.weight > right.weight;
        });

        const auto apply_strategy = [&](ExecutionConfig& config, const ExecutionStrategy strategy) {
            config.strategy = strategy;
            if (strategy == ExecutionStrategy::sharded) {
                config.primary_device_uid = placement.allocations.front().device.uid;
                config.participating_devices.clear();
                for (const auto& allocation : placement.allocations) {
                    config.participating_devices.push_back(allocation.device.uid);
                }
                config.overlap_transfers = true;
            } else {
                config.participating_devices = {config.primary_device_uid};
                config.overlap_transfers =
                    strategy == ExecutionStrategy::streaming || strategy == ExecutionStrategy::overlapped;
            }
        };

        for (std::size_t rank = 0; rank < ranked.size() && rank < 2u; ++rank) {
            if (ranked[rank].weight < 0.0) {
                continue;
            }
            if (ranked[rank].strategy == ExecutionStrategy::sharded) {
                ExecutionConfig config;
                config.primary_device_uid = placement.allocations.front().device.uid;
                config.queue_depth = choose_queue_depth(primary_summary);
                config.stages = primary_summary.supports_asynchronous_dispatch ? 2u : 1u;
                if (system.low_spec_mode) {
                    config.queue_depth = std::min(config.queue_depth, 2u);
                    config.stages = 1u;
                }
                apply_strategy(config, ranked[rank].strategy);
                add_candidate(config);
            } else {
                std::size_t added_devices = 0;
                for (const auto& allocation : placement.allocations) {
                    const auto* graph = find_graph(graph_lookup, allocation.device.uid);
                    if (graph == nullptr) {
                        continue;
                    }
                    const auto summary = summarize_graph(*graph);
                    ExecutionConfig config;
                    config.primary_device_uid = allocation.device.uid;
                    config.queue_depth = choose_queue_depth(summary);
                    config.stages = summary.supports_asynchronous_dispatch ? 2u : 1u;
                    if (system.low_spec_mode) {
                        config.queue_depth = std::min(config.queue_depth, 2u);
                        config.stages = 1u;
                    }
                    apply_strategy(config, ranked[rank].strategy);
                    add_candidate(config);
                    ++added_devices;
                    if (ranked[rank].weight > 0.65 && added_devices >= 2u) {
                        break;
                    }
                }
            }
        }

        return candidates;
    }

    for (const auto& allocation : placement.allocations) {
        const auto* graph = find_graph(graph_lookup, allocation.device.uid);
        if (graph == nullptr) {
            continue;
        }

        const auto summary = summarize_graph(*graph);
        ExecutionConfig base;
        base.strategy = ExecutionStrategy::single_device;
        base.primary_device_uid = allocation.device.uid;
        base.participating_devices = {allocation.device.uid};
        base.queue_depth = choose_queue_depth(summary);
        base.stages = summary.supports_asynchronous_dispatch ? 2u : 1u;
        if (system.low_spec_mode) {
            base.queue_depth = std::min(base.queue_depth, 2u);
            base.stages = 1u;
        }
        add_candidate(base);

        if (operation.streaming_friendly &&
            (summary.supports_asynchronous_dispatch || system.low_spec_mode)) {
            ExecutionConfig streaming = base;
            streaming.strategy = ExecutionStrategy::streaming;
            streaming.stages = system.low_spec_mode ? 2u : std::max(3u, streaming.stages);
            streaming.overlap_transfers = true;
            add_candidate(streaming);
        }

        if (supports_low_precision(summary) && operation.max_relative_error >= 4.0e-4) {
            ExecutionConfig low_precision = base;
            low_precision.use_low_precision = true;
            low_precision.overlap_transfers = summary.supports_asynchronous_dispatch;
            low_precision.strategy = operation.parallelizable && summary.supports_asynchronous_dispatch
                                         ? ExecutionStrategy::overlapped
                                         : ExecutionStrategy::single_device;
            add_candidate(low_precision);
        }
    }

    if (operation.parallelizable &&
        placement.allocations.size() > 1 &&
        !workload.latency_sensitive &&
        !(system.low_spec_mode && system.paging_risk > 0.25)) {
        ExecutionConfig sharded;
        sharded.strategy = ExecutionStrategy::sharded;
        sharded.primary_device_uid = placement.allocations.front().device.uid;
        sharded.participating_devices.clear();
        for (const auto& allocation : placement.allocations) {
            sharded.participating_devices.push_back(allocation.device.uid);
        }
        sharded.overlap_transfers = true;
        sharded.queue_depth = choose_queue_depth(primary_summary);
        sharded.stages =
            system.low_spec_mode ? 2u : std::max(2u, primary_summary.supports_asynchronous_dispatch ? 2u : 1u);
        add_candidate(sharded);
    }

    return candidates;
}

std::unordered_map<std::string, double> normalized_ratios(const ExecutionPlan& placement, const ExecutionConfig& config) {
    std::unordered_map<std::string, double> ratios;
    double total = 0.0;
    for (const auto& allocation : placement.allocations) {
        if (std::find(
                config.participating_devices.begin(),
                config.participating_devices.end(),
                allocation.device.uid) != config.participating_devices.end()) {
            ratios[allocation.device.uid] = allocation.ratio;
            total += allocation.ratio;
        }
    }

    if (total <= 0.0) {
        const double uniform = config.participating_devices.empty()
                                   ? 0.0
                                   : (1.0 / static_cast<double>(config.participating_devices.size()));
        for (const auto& uid : config.participating_devices) {
            ratios[uid] = uniform;
        }
        return ratios;
    }

    for (auto& [uid, ratio] : ratios) {
        ratio /= total;
    }
    return ratios;
}

double effective_edge_latency(const ExecutionEdge& edge) {
    return edge.predicted_latency_us * (1.0 - std::clamp(edge.overlap_ratio, 0.0, 0.95));
}

double finalize_graph_latency(ExecutionGraph& graph) {
    std::unordered_map<std::string, double> node_latency;
    std::unordered_map<std::string, int> indegree;
    std::unordered_map<std::string, std::vector<const ExecutionEdge*>> outgoing;
    node_latency.reserve(graph.nodes.size());
    indegree.reserve(graph.nodes.size());
    outgoing.reserve(graph.nodes.size());

    for (const auto& node : graph.nodes) {
        node_latency.emplace(node.id, node.predicted_latency_us);
        indegree.emplace(node.id, 0);
    }

    for (const auto& edge : graph.edges) {
        ++indegree[edge.target_id];
        outgoing[edge.source_id].push_back(&edge);
    }

    std::vector<std::string> frontier;
    frontier.reserve(graph.nodes.size());
    for (const auto& [node_id, degree] : indegree) {
        if (degree == 0) {
            frontier.push_back(node_id);
        }
    }

    std::unordered_map<std::string, double> longest;
    longest.reserve(graph.nodes.size());
    for (const auto& node_id : frontier) {
        longest[node_id] = node_latency[node_id];
    }

    for (std::size_t index = 0; index < frontier.size(); ++index) {
        const auto current = frontier[index];
        const double base = longest[current];

        if (const auto out_it = outgoing.find(current); out_it != outgoing.end()) {
            for (const auto* edge : out_it->second) {
                const double candidate = base + effective_edge_latency(*edge) + node_latency[edge->target_id];
                auto& target_distance = longest[edge->target_id];
                target_distance = std::max(target_distance, candidate);
                if (--indegree[edge->target_id] == 0) {
                    frontier.push_back(edge->target_id);
                }
            }
        }
    }

    double total = 0.0;
    for (const auto& [node_id, distance] : longest) {
        total = std::max(total, distance);
    }
    graph.predicted_latency_us = total;
    return total;
}

ExecutionGraph build_execution_graph(
    const std::string& report_signature,
    const WorkloadGraph& workload_graph,
    const OperationSpec& operation,
    ExecutionConfig& config,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::unordered_map<std::string, ExecutionConfig>& selected_configs) {
    ExecutionGraph graph;
    graph.workload_signature = report_signature;
    graph.operation = operation;
    graph.participating_devices = config.participating_devices;
    config.mapped_structural_nodes.clear();

    const auto ratios = normalized_ratios(placement, config);
    const auto operation_indices = build_operation_index(workload_graph);
    const auto operation_index = operation_index_or_default(operation_indices, operation.name, 0u);
    std::unordered_map<std::string, std::uint64_t> resident_bytes_by_device;
    std::unordered_map<std::string, double> resident_pressure_by_device;

    auto add_node = [&](ExecutionNode node) {
        graph.nodes.push_back(std::move(node));
    };

    auto add_edge = [&](ExecutionEdge edge) {
        graph.edges.push_back(std::move(edge));
    };

    const std::string global_sink_id = operation.name + ".sink";
    const std::string global_sync_id = operation.name + ".sync";
    const bool needs_sync =
        config.participating_devices.size() > 1 ||
        config.logical_partitions > 1 ||
        config.strategy != ExecutionStrategy::single_device;

    if (needs_sync) {
        add_node(ExecutionNode{
            global_sync_id,
            operation.name + "-sync",
            config.primary_device_uid,
            "",
            ExecutionNodeKind::synchronize,
            operation.output_bytes,
            0.0,
            0.0});
    }

    add_node(ExecutionNode{
        global_sink_id,
        operation.name + "-sink",
        config.primary_device_uid,
        "",
        ExecutionNodeKind::sink,
        operation.output_bytes,
        0.0,
        0.0});

    for (const auto& device_uid : config.participating_devices) {
        const auto* hardware = find_graph(graph_lookup, device_uid);
        if (hardware == nullptr) {
            continue;
        }

        const auto summary = summarize_graph(*hardware);
        const auto* control_node = find_first_node(*hardware, {HardwareObjectRole::queue, HardwareObjectRole::scheduler});
        const auto* compute_node = find_first_node(
            *hardware,
            {HardwareObjectRole::tile, HardwareObjectRole::cluster, HardwareObjectRole::pipeline});
        const auto* storage_node = find_first_node(
            *hardware,
            {HardwareObjectRole::scratchpad,
             HardwareObjectRole::cache,
             HardwareObjectRole::global_memory,
             HardwareObjectRole::host_memory});

        const double ratio = ratios.contains(device_uid) ? ratios.at(device_uid) : 1.0;
        const auto partitions = std::max(config.logical_partitions, 1u);
        for (std::uint32_t partition = 0; partition < partitions; ++partition) {
            const double partition_ratio = ratio / static_cast<double>(partitions);
            const std::uint64_t input_share =
                static_cast<std::uint64_t>(std::ceil(static_cast<double>(operation.input_bytes) * partition_ratio));
            const std::uint64_t output_share = operation.reduction_like
                                                   ? operation.output_bytes
                                                   : static_cast<std::uint64_t>(
                                                         std::ceil(static_cast<double>(operation.output_bytes) * partition_ratio));
            const double flops_share = operation.estimated_flops * partition_ratio;
            const double compute_latency_us =
                (flops_share / (estimate_device_gflops(summary, operation, config.use_low_precision) * 1.0e9)) *
                kMicrosecondsPerSecond;

            if (control_node != nullptr) {
                config.mapped_structural_nodes.push_back(control_node->id + "#p" + std::to_string(partition));
            }
            if (compute_node != nullptr) {
                config.mapped_structural_nodes.push_back(compute_node->id + "#p" + std::to_string(partition));
            }
            if (storage_node != nullptr) {
                config.mapped_structural_nodes.push_back(storage_node->id + "#p" + std::to_string(partition));
            }

            const std::string partition_suffix =
                config.logical_partitions > 1 ? (".part" + std::to_string(partition)) : std::string();
            const std::string prefix = operation.name + "." + device_uid + partition_suffix;
            const std::string source_id = prefix + ".source";
            const std::string dispatch_id = prefix + ".dispatch";
            const std::string compute_id = prefix + ".compute";
            const std::string aggregate_id = prefix + ".aggregate";
            std::uint64_t partition_input_bytes = 0u;
            std::uint64_t partition_output_bytes = 0u;
            std::uint64_t partition_temporary_bytes = 0u;
            double input_transfer_latency_us = 0.0;
            double output_transfer_latency_us = 0.0;

            for (const auto& tensor_id : operation.input_tensor_ids) {
                const auto* tensor = find_workload_tensor(workload_graph, tensor_id);
                if (tensor == nullptr) {
                    continue;
                }

                const auto* lifetime = find_tensor_lifetime(workload_graph, tensor_id);
                const auto mapped_bytes =
                    bytes_for_tensor_mapping(*tensor, operation, config, partition_ratio, partitions);
                partition_input_bytes += mapped_bytes;

                const auto source_devices = resident_devices_for_tensor(*tensor, config, selected_configs);
                const bool resident_on_target = std::find(source_devices.begin(), source_devices.end(), device_uid) != source_devices.end();
                const std::string source_device_uid =
                    resident_on_target ? device_uid : (source_devices.empty() ? std::string("host") : source_devices.front());
                const auto* source_graph =
                    source_device_uid == "host" ? nullptr : find_graph(graph_lookup, source_device_uid);
                const auto source_summary = source_graph == nullptr ? HardwareGraphSummary{} : summarize_graph(*source_graph);
                const auto* source_summary_ptr = source_graph == nullptr ? nullptr : &source_summary;
                const bool cross_device = source_device_uid != device_uid;
                const double transfer_latency =
                    scheduled_transfer_latency_us(mapped_bytes, source_summary_ptr, summary, cross_device);
                input_transfer_latency_us = std::max(input_transfer_latency_us, transfer_latency);
                graph.predicted_transfer_latency_us += transfer_latency;

                graph.residency_plan.push_back(TensorResidencyPlanEntry{
                    tensor->id,
                    device_uid,
                    storage_node == nullptr ? std::string() : storage_node->id,
                    mapped_bytes,
                    tensor->persistent,
                    true,
                    tensor->persistent || (lifetime != nullptr && lifetime->last_operation_index > operation_index),
                    cross_device || source_device_uid == "host",
                    0.0});

                if (cross_device || source_device_uid == "host") {
                    graph.transfer_schedule.push_back(TransferScheduleEntry{
                        tensor->id,
                        source_device_uid,
                        device_uid,
                        tensor->producer_operation,
                        operation.name,
                        mapped_bytes,
                        transfer_latency,
                        std::max(config.overlap_ratio, config.overlap_transfers ? 0.35 : 0.0),
                        cross_device});
                }
            }

            for (const auto& tensor_id : operation.output_tensor_ids) {
                const auto* tensor = find_workload_tensor(workload_graph, tensor_id);
                if (tensor == nullptr) {
                    continue;
                }

                const auto* lifetime = find_tensor_lifetime(workload_graph, tensor_id);
                const auto mapped_bytes =
                    bytes_for_tensor_mapping(*tensor, operation, config, partition_ratio, partitions);
                partition_output_bytes += mapped_bytes;
                graph.residency_plan.push_back(TensorResidencyPlanEntry{
                    tensor->id,
                    device_uid,
                    storage_node == nullptr ? std::string() : storage_node->id,
                    mapped_bytes,
                    tensor->persistent,
                    false,
                    tensor->persistent || (lifetime != nullptr && lifetime->last_operation_index > operation_index),
                    false,
                    0.0});

                if (needs_sync && device_uid != config.primary_device_uid) {
                    const auto* primary_graph = find_graph(graph_lookup, config.primary_device_uid);
                    const auto primary_summary =
                        primary_graph == nullptr ? HardwareGraphSummary{} : summarize_graph(*primary_graph);
                    const double transfer_latency = primary_graph == nullptr
                                                       ? transfer_cost_us(mapped_bytes, summary)
                                                       : scheduled_transfer_latency_us(
                                                             mapped_bytes,
                                                             &summary,
                                                             primary_summary,
                                                             true);
                    output_transfer_latency_us = std::max(output_transfer_latency_us, transfer_latency);
                    graph.predicted_transfer_latency_us += transfer_latency;
                    graph.transfer_schedule.push_back(TransferScheduleEntry{
                        tensor->id,
                        device_uid,
                        config.primary_device_uid,
                        operation.name,
                        operation.name,
                        mapped_bytes,
                        transfer_latency,
                        config.overlap_ratio * 0.7,
                        true});
                }
            }

            for (const auto& tensor_id : operation.temporary_tensor_ids) {
                const auto* tensor = find_workload_tensor(workload_graph, tensor_id);
                if (tensor == nullptr) {
                    continue;
                }

                const auto mapped_bytes =
                    bytes_for_tensor_mapping(*tensor, operation, config, partition_ratio, partitions);
                partition_temporary_bytes += mapped_bytes;
                graph.residency_plan.push_back(TensorResidencyPlanEntry{
                    tensor->id,
                    device_uid,
                    storage_node == nullptr ? std::string() : storage_node->id,
                    mapped_bytes,
                    false,
                    false,
                    false,
                    false,
                    0.0});
            }

            resident_bytes_by_device[device_uid] += partition_input_bytes + partition_output_bytes + partition_temporary_bytes;
            const std::uint64_t capacity_bytes =
                summary.directly_attached_bytes > 0u
                    ? summary.directly_attached_bytes + (summary.unified_address_space ? summary.shared_host_bytes : 0u)
                    : std::max(summary.addressable_bytes, summary.shared_host_bytes);
            if (capacity_bytes > 0u) {
                resident_pressure_by_device[device_uid] = std::max(
                    resident_pressure_by_device[device_uid],
                    static_cast<double>(resident_bytes_by_device[device_uid]) / static_cast<double>(capacity_bytes));
            }

            add_node(ExecutionNode{
                source_id,
                hardware->presentation_name + partition_suffix + "-source",
                device_uid,
                storage_node == nullptr ? std::string() : storage_node->id,
                ExecutionNodeKind::source,
                partition_input_bytes == 0u ? input_share : partition_input_bytes,
                0.0,
                0.0});

            add_node(ExecutionNode{
                dispatch_id,
                hardware->presentation_name + partition_suffix + "-dispatch",
                device_uid,
                control_node == nullptr ? std::string() : control_node->id,
                ExecutionNodeKind::dispatch,
                0,
                0.0,
                dispatch_cost_us(summary)});

            add_node(ExecutionNode{
                compute_id,
                hardware->presentation_name + partition_suffix + "-compute",
                device_uid,
                compute_node == nullptr ? std::string() : compute_node->id,
                ExecutionNodeKind::compute,
                partition_output_bytes == 0u ? output_share : partition_output_bytes,
                flops_share,
                std::max(0.01, compute_latency_us)});

            add_edge(ExecutionEdge{
                source_id,
                dispatch_id,
                ExecutionEdgeKind::dataflow,
                true,
                partition_input_bytes == 0u ? input_share : partition_input_bytes,
                std::max(
                    transfer_cost_us(partition_input_bytes == 0u ? input_share : partition_input_bytes, summary),
                    input_transfer_latency_us),
                std::max(config.overlap_ratio, config.overlap_transfers ? 0.35 : 0.0)});

            add_edge(ExecutionEdge{
                dispatch_id,
                compute_id,
                ExecutionEdgeKind::control,
                true,
                0,
                dispatch_cost_us(summary),
                0.0});

            if (operation.reduction_like) {
                add_node(ExecutionNode{
                    aggregate_id,
                    hardware->presentation_name + partition_suffix + "-aggregate",
                    device_uid,
                    compute_node == nullptr ? std::string() : compute_node->id,
                    ExecutionNodeKind::aggregate,
                    operation.output_bytes,
                    static_cast<double>(operation.output_bytes),
                    sync_cost_us(summary) * 0.5});

                add_edge(ExecutionEdge{
                    compute_id,
                    aggregate_id,
                    ExecutionEdgeKind::aggregation,
                    true,
                    output_share,
                    transfer_cost_us(output_share, summary) * 0.5,
                    config.overlap_ratio * 0.5});

                if (needs_sync) {
                    add_edge(ExecutionEdge{
                        aggregate_id,
                        global_sync_id,
                        ExecutionEdgeKind::dependency,
                        true,
                        operation.output_bytes,
                        std::max(sync_cost_us(summary), output_transfer_latency_us),
                        0.0});
                } else {
                    add_edge(ExecutionEdge{
                        aggregate_id,
                        global_sink_id,
                        ExecutionEdgeKind::dataflow,
                        true,
                        operation.output_bytes,
                        transfer_cost_us(operation.output_bytes, summary),
                        0.0});
                }
            } else if (needs_sync) {
                add_edge(ExecutionEdge{
                    compute_id,
                    global_sync_id,
                    ExecutionEdgeKind::dependency,
                    true,
                    partition_output_bytes == 0u ? output_share : partition_output_bytes,
                    std::max(
                        transfer_cost_us(partition_output_bytes == 0u ? output_share : partition_output_bytes, summary),
                        output_transfer_latency_us),
                    config.overlap_ratio * 0.7});
            } else {
                add_edge(ExecutionEdge{
                    compute_id,
                    global_sink_id,
                    ExecutionEdgeKind::dataflow,
                    true,
                    output_share,
                    transfer_cost_us(output_share, summary),
                    config.overlap_ratio * 0.7});
            }
        }
    }

    if (needs_sync) {
        double sync_latency = 0.0;
        for (const auto& device_uid : config.participating_devices) {
            if (const auto* hardware = find_graph(graph_lookup, device_uid)) {
                sync_latency = std::max(sync_latency, sync_cost_us(summarize_graph(*hardware)));
            }
        }
        for (auto& node : graph.nodes) {
            if (node.id == global_sync_id) {
                node.predicted_latency_us = std::max(0.1, sync_latency);
                break;
            }
        }
        add_edge(ExecutionEdge{
            global_sync_id,
            global_sink_id,
            ExecutionEdgeKind::dataflow,
            true,
            operation.output_bytes,
            operation.output_bytes == 0 ? 0.0 : (sync_latency * 0.25),
            0.0});
    }

    for (auto& entry : graph.residency_plan) {
        if (const auto pressure_it = resident_pressure_by_device.find(entry.device_uid);
            pressure_it != resident_pressure_by_device.end()) {
            entry.pressure_ratio = pressure_it->second;
            graph.predicted_memory_pressure = std::max(graph.predicted_memory_pressure, pressure_it->second);
        }
    }
    for (const auto& [device_uid, resident_bytes] : resident_bytes_by_device) {
        (void)device_uid;
        graph.peak_resident_bytes = std::max(graph.peak_resident_bytes, resident_bytes);
    }

    graph.expected_relative_error = expected_relative_error(operation, config);
    config.signature = build_config_signature(report_signature, config);
    graph.signature = config.signature;
    finalize_graph_latency(graph);
    return graph;
}

ValidationResult validate_elementwise(const OperationSpec& operation, const ExecutionConfig& config) {
    const auto count = static_cast<std::size_t>(operation.extents.at(0));
    const auto a = make_pattern(count, 1.0f);
    const auto b = make_pattern(count, 2.0f);
    std::vector<float> reference(count, 0.0f);
    std::vector<float> candidate(count, 0.0f);

    const auto reference_profile = profile_measurements(3u, [&]() {
        for (std::size_t index = 0; index < count; ++index) {
            reference[index] = (a[index] * 1.125f) + (b[index] * 0.25f) - 0.03125f;
        }
    });

    const std::size_t tile = std::max<std::size_t>(config.tile_x, 128u);
    const auto candidate_profile = profile_measurements(3u, [&]() {
        for (std::size_t base = 0; base < count; base += tile) {
            const auto end = std::min(base + tile, count);
            for (std::size_t index = base; index < end; ++index) {
                const float lhs = quantize_value(a[index] * 1.125f, config.use_low_precision);
                const float rhs = quantize_value(b[index] * 0.25f, config.use_low_precision);
                candidate[index] = quantize_value(lhs + rhs - 0.03125f, config.use_low_precision);
            }
        }
    });

    return ValidationResult{
        reference_profile.average_us,
        candidate_profile.average_us,
        relative_l2_error(reference, candidate),
        reference_profile.spread_us,
        candidate_profile.spread_us,
        candidate_profile.samples};
}

ValidationResult validate_reduction(const OperationSpec& operation, const ExecutionConfig& config) {
    const auto count = static_cast<std::size_t>(operation.extents.at(0));
    const auto data = make_pattern(count, 3.0f);
    double reference_value = 0.0;
    double candidate_value = 0.0;

    const auto reference_profile = profile_measurements(3u, [&]() {
        double total = 0.0;
        for (const auto value : data) {
            total += value;
        }
        reference_value = total;
    });

    const std::size_t tile = std::max<std::size_t>(config.tile_x, 256u);
    const auto candidate_profile = profile_measurements(3u, [&]() {
        float total = 0.0f;
        for (std::size_t base = 0; base < count; base += tile) {
            const auto end = std::min(base + tile, count);
            float partial = 0.0f;
            for (std::size_t index = base; index < end; ++index) {
                partial = quantize_value(partial + data[index], config.use_low_precision);
            }
            total = quantize_value(total + partial, config.use_low_precision);
        }
        candidate_value = total;
    });

    return ValidationResult{
        reference_profile.average_us,
        candidate_profile.average_us,
        scalar_relative_error(reference_value, candidate_value),
        reference_profile.spread_us,
        candidate_profile.spread_us,
        candidate_profile.samples};
}

ValidationResult validate_matmul(const OperationSpec& operation, const ExecutionConfig& config) {
    const auto m = static_cast<std::size_t>(operation.extents.at(0));
    const auto n = static_cast<std::size_t>(operation.extents.at(1));
    const auto k = static_cast<std::size_t>(operation.extents.at(2));
    const auto a = make_pattern(m * k, 4.0f);
    const auto b = make_pattern(k * n, 5.0f);
    std::vector<float> reference(m * n, 0.0f);
    std::vector<float> candidate(m * n, 0.0f);

    const auto reference_profile = profile_measurements(3u, [&]() {
        for (std::size_t row = 0; row < m; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                float acc = 0.0f;
                for (std::size_t inner = 0; inner < k; ++inner) {
                    acc += a[row * k + inner] * b[inner * n + col];
                }
                reference[row * n + col] = acc;
            }
        }
    });

    const std::size_t tile_m = std::max<std::size_t>(config.tile_x, 16u);
    const std::size_t tile_n = std::max<std::size_t>(config.tile_y, 16u);
    const std::size_t tile_k = std::max<std::size_t>(config.tile_k, 8u);
    const auto candidate_profile = profile_measurements(3u, [&]() {
        for (std::size_t row_base = 0; row_base < m; row_base += tile_m) {
            for (std::size_t col_base = 0; col_base < n; col_base += tile_n) {
                for (std::size_t inner_base = 0; inner_base < k; inner_base += tile_k) {
                    const auto row_end = std::min(row_base + tile_m, m);
                    const auto col_end = std::min(col_base + tile_n, n);
                    const auto inner_end = std::min(inner_base + tile_k, k);
                    for (std::size_t row = row_base; row < row_end; ++row) {
                        for (std::size_t col = col_base; col < col_end; ++col) {
                            float acc = inner_base == 0 ? 0.0f : candidate[row * n + col];
                            for (std::size_t inner = inner_base; inner < inner_end; ++inner) {
                                const float lhs = quantize_value(a[row * k + inner], config.use_low_precision);
                                const float rhs = quantize_value(b[inner * n + col], config.use_low_precision);
                                acc = quantize_value(acc + (lhs * rhs), config.use_low_precision);
                            }
                            candidate[row * n + col] = acc;
                        }
                    }
                }
            }
        }
    });

    return ValidationResult{
        reference_profile.average_us,
        candidate_profile.average_us,
        relative_l2_error(reference, candidate),
        reference_profile.spread_us,
        candidate_profile.spread_us,
        candidate_profile.samples};
}

ValidationResult validate_convolution(const OperationSpec& operation, const ExecutionConfig& config) {
    const auto height = static_cast<std::size_t>(operation.extents.at(0));
    const auto width = static_cast<std::size_t>(operation.extents.at(1));
    const auto input = make_pattern(height * width, 6.0f);
    const std::array<float, 9> kernel{0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.125f, 0.0625f, 0.125f, 0.0625f};
    std::vector<float> reference((height - 2) * (width - 2), 0.0f);
    std::vector<float> candidate((height - 2) * (width - 2), 0.0f);

    const auto reference_profile = profile_measurements(3u, [&]() {
        for (std::size_t y = 1; y + 1 < height; ++y) {
            for (std::size_t x = 1; x + 1 < width; ++x) {
                float acc = 0.0f;
                for (std::size_t ky = 0; ky < 3; ++ky) {
                    for (std::size_t kx = 0; kx < 3; ++kx) {
                        acc += input[(y + ky - 1) * width + (x + kx - 1)] * kernel[ky * 3 + kx];
                    }
                }
                reference[(y - 1) * (width - 2) + (x - 1)] = acc;
            }
        }
    });

    const std::size_t tile_y = std::max<std::size_t>(config.tile_y, 8u);
    const std::size_t tile_x = std::max<std::size_t>(config.tile_x, 8u);
    const auto candidate_profile = profile_measurements(3u, [&]() {
        for (std::size_t y_base = 1; y_base + 1 < height; y_base += tile_y) {
            for (std::size_t x_base = 1; x_base + 1 < width; x_base += tile_x) {
                const auto y_end = std::min(y_base + tile_y, height - 1);
                const auto x_end = std::min(x_base + tile_x, width - 1);
                for (std::size_t y = y_base; y < y_end; ++y) {
                    for (std::size_t x = x_base; x < x_end; ++x) {
                        float acc = 0.0f;
                        for (std::size_t ky = 0; ky < 3; ++ky) {
                            for (std::size_t kx = 0; kx < 3; ++kx) {
                                const float value = quantize_value(
                                    input[(y + ky - 1) * width + (x + kx - 1)],
                                    config.use_low_precision);
                                acc = quantize_value(acc + (value * kernel[ky * 3 + kx]), config.use_low_precision);
                            }
                        }
                        candidate[(y - 1) * (width - 2) + (x - 1)] = acc;
                    }
                }
            }
        }
    });

    return ValidationResult{
        reference_profile.average_us,
        candidate_profile.average_us,
        relative_l2_error(reference, candidate),
        reference_profile.spread_us,
        candidate_profile.spread_us,
        candidate_profile.samples};
}

ValidationResult validate_resample(const OperationSpec& operation, const ExecutionConfig& config) {
    const auto src_h = static_cast<std::size_t>(operation.extents.at(0));
    const auto src_w = static_cast<std::size_t>(operation.extents.at(1));
    const auto dst_h = static_cast<std::size_t>(operation.extents.at(2));
    const auto dst_w = static_cast<std::size_t>(operation.extents.at(3));
    const auto input = make_pattern(src_h * src_w, 7.0f);
    std::vector<float> reference(dst_h * dst_w, 0.0f);
    std::vector<float> candidate(dst_h * dst_w, 0.0f);

    auto bilinear = [&](std::vector<float>& output, const bool low_precision) {
        for (std::size_t y = 0; y < dst_h; ++y) {
            const float src_y = (static_cast<float>(y) + 0.5f) * static_cast<float>(src_h) / static_cast<float>(dst_h) - 0.5f;
            const float clamped_y = std::clamp(std::floor(src_y), 0.0f, static_cast<float>(src_h - 1));
            const auto y0 = static_cast<std::size_t>(clamped_y);
            const auto y1 = std::min<std::size_t>(y0 + 1u, src_h - 1u);
            const float wy = static_cast<float>(src_y - std::floor(src_y));
            for (std::size_t x = 0; x < dst_w; ++x) {
                const float src_x = (static_cast<float>(x) + 0.5f) * static_cast<float>(src_w) / static_cast<float>(dst_w) - 0.5f;
                const float clamped_x = std::clamp(std::floor(src_x), 0.0f, static_cast<float>(src_w - 1));
                const auto x0 = static_cast<std::size_t>(clamped_x);
                const auto x1 = std::min<std::size_t>(x0 + 1u, src_w - 1u);
                const float wx = static_cast<float>(src_x - std::floor(src_x));
                const float v00 = quantize_value(input[y0 * src_w + x0], low_precision);
                const float v01 = quantize_value(input[y0 * src_w + x1], low_precision);
                const float v10 = quantize_value(input[y1 * src_w + x0], low_precision);
                const float v11 = quantize_value(input[y1 * src_w + x1], low_precision);
                const float top = quantize_value(v00 + ((v01 - v00) * wx), low_precision);
                const float bottom = quantize_value(v10 + ((v11 - v10) * wx), low_precision);
                output[y * dst_w + x] = quantize_value(top + ((bottom - top) * wy), low_precision);
            }
        }
    };

    const auto reference_profile = profile_measurements(3u, [&]() {
        bilinear(reference, false);
    });
    const auto candidate_profile = profile_measurements(3u, [&]() {
        bilinear(candidate, config.use_low_precision);
    });

    return ValidationResult{
        reference_profile.average_us,
        candidate_profile.average_us,
        relative_l2_error(reference, candidate),
        reference_profile.spread_us,
        candidate_profile.spread_us,
        candidate_profile.samples};
}

BenchmarkRecord benchmark_operation(
    const OperationSpec& operation,
    const ExecutionConfig& config,
    const ExecutionGraph& graph,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::string& shape_bucket,
    const double surrogate_latency_us,
    const double system_penalty_us) {
    ValidationResult validation;
    switch (operation.op_class) {
    case OperationClass::elementwise_map:
        validation = validate_elementwise(operation, config);
        break;
    case OperationClass::reduction:
        validation = validate_reduction(operation, config);
        break;
    case OperationClass::matmul:
        validation = validate_matmul(operation, config);
        break;
    case OperationClass::convolution_2d:
        validation = validate_convolution(operation, config);
        break;
    case OperationClass::resample_2d:
    default:
        validation = validate_resample(operation, config);
        break;
    }

    bool executable_on_host = config.participating_devices.size() == 1;
    if (executable_on_host) {
        if (const auto* hardware = find_graph(graph_lookup, config.primary_device_uid)) {
            executable_on_host = hardware->probe == "host";
        }
    }

    const double calibration_ratio =
        graph.predicted_latency_us > 0.0 ? (validation.candidate_latency_us / graph.predicted_latency_us) : 1.0;
    const double spread_ratio =
        (validation.reference_spread_us + validation.candidate_spread_us) /
        std::max(validation.reference_latency_us + validation.candidate_latency_us, 1.0);
    const double calibration_confidence = std::clamp(
        (static_cast<double>(std::max(validation.samples, 1u)) / 3.0) * (1.0 / (1.0 + (spread_ratio * 6.0))),
        0.0,
        1.0);
    const double calibrated_prediction =
        (surrogate_latency_us * (1.0 - calibration_confidence)) +
        (((graph.predicted_latency_us * calibration_ratio) + system_penalty_us) * calibration_confidence);
    const double effective_latency = executable_on_host ? validation.candidate_latency_us : calibrated_prediction;
    BenchmarkRecord record;
    record.operation_name = operation.name;
    record.config_signature = config.signature;
    record.shape_bucket = shape_bucket;
    record.reference_latency_us = validation.reference_latency_us;
    record.validation_latency_us = validation.candidate_latency_us;
    record.predicted_latency_us = graph.predicted_latency_us;
    record.surrogate_latency_us = surrogate_latency_us;
    record.calibrated_prediction_us = std::max(0.01, calibrated_prediction);
    record.calibration_ratio = calibration_ratio;
    record.calibration_confidence = calibration_confidence;
    record.reference_spread_us = validation.reference_spread_us;
    record.candidate_spread_us = validation.candidate_spread_us;
    record.validation_samples = validation.samples;
    record.system_penalty_us = system_penalty_us;
    record.effective_latency_us = std::max(0.01, effective_latency);
    record.speedup_vs_reference = validation.reference_latency_us / record.effective_latency_us;
    record.relative_error = validation.relative_error;
    record.accuracy_within_tolerance = validation.relative_error <= (config.target_error_tolerance + 1.0e-12);
    record.simulated = !executable_on_host;
    return record;
}

struct CandidateEvaluation {
    ExecutionConfig config;
    ExecutionGraph graph;
    BenchmarkRecord benchmark;
    OptimizationPolicy policy = OptimizationPolicy::heuristic_greedy;
    bool spsa_refined = false;
    double heuristic_objective = std::numeric_limits<double>::max();
    double learned_objective = std::numeric_limits<double>::max();
    double trace_objective = std::numeric_limits<double>::max();
    double explore_objective = std::numeric_limits<double>::max();
    double reinforce_objective = std::numeric_limits<double>::max();
    std::uint32_t observations = 0;
};

OperationOptimizationResult optimize_operation(
    const std::string& report_signature,
    const WorkloadGraph& workload_graph,
    const OperationSpec& operation,
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::unordered_map<std::string, ExecutionConfig>& selected_configs,
    const SystemProfile& system,
    const std::unordered_map<std::string, ExecutionOptimizer::PerformanceSummary>& performance_cache,
    const std::unordered_map<std::string, double>& backend_penalty_cache,
    const std::unordered_map<std::string, bool>& warmed_devices,
    const std::string& graph_set_signature,
    const ExecutionConfig* cached_config,
    const ContinuousExecutionState* continuous_state,
    const bool allow_validation);

std::optional<ExecutionConfig> make_spsa_candidate(
    const ExecutionConfig& seed,
    const std::string& report_signature,
    const WorkloadGraph& workload_graph,
    const OperationSpec& operation,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::unordered_map<std::string, ExecutionConfig>& selected_configs,
    const SystemProfile& system,
    const std::unordered_map<std::string, bool>& warmed_devices) {
    auto score_of = [&](ExecutionConfig candidate) {
        auto graph = build_execution_graph(
            report_signature,
            workload_graph,
            operation,
            candidate,
            placement,
            graph_lookup,
            selected_configs);
        const double penalty = compute_surrogate_penalty_us(operation, candidate, system, warmed_devices);
        return (graph.predicted_latency_us * system.sustained_slowdown) + penalty;
    };

    const std::uint64_t hash = std::hash<std::string>{}(seed.signature + "|" + operation.name);
    const int sign0 = (hash & 1ull) == 0 ? 1 : -1;
    const int sign1 = (hash & 2ull) == 0 ? 1 : -1;
    const int sign2 = (hash & 4ull) == 0 ? 1 : -1;
    const int sign3 = (hash & 8ull) == 0 ? 1 : -1;
    const int sign4 = (hash & 16ull) == 0 ? 1 : -1;

    auto plus = seed;
    auto minus = seed;
    const std::uint32_t tile_step = operation.matrix_friendly ? 16u : 32u;
    plus.queue_depth = std::clamp<int>(static_cast<int>(plus.queue_depth) + sign0, 1, 8);
    minus.queue_depth = std::clamp<int>(static_cast<int>(minus.queue_depth) - sign0, 1, 8);
    plus.stages = std::clamp<int>(static_cast<int>(plus.stages) + sign1, 1, 4);
    minus.stages = std::clamp<int>(static_cast<int>(minus.stages) - sign1, 1, 4);
    plus.tile_x = std::max<std::uint32_t>(8u, plus.tile_x + (sign2 > 0 ? tile_step : 0u));
    minus.tile_x = std::max<std::uint32_t>(8u, minus.tile_x + (sign2 < 0 ? tile_step : 0u));
    plus.tile_y = std::max<std::uint32_t>(1u, plus.tile_y + (sign3 > 0 ? tile_step / 2u : 0u));
    minus.tile_y = std::max<std::uint32_t>(1u, minus.tile_y + (sign3 < 0 ? tile_step / 2u : 0u));
    plus.tile_k = std::max<std::uint32_t>(4u, plus.tile_k + (sign4 > 0 ? tile_step : 0u));
    minus.tile_k = std::max<std::uint32_t>(4u, minus.tile_k + (sign4 < 0 ? tile_step : 0u));
    plus.signature = build_config_signature(report_signature, plus);
    minus.signature = build_config_signature(report_signature, minus);

    const double plus_score = score_of(plus);
    const double minus_score = score_of(minus);
    auto refined = plus_score < minus_score ? plus : minus;
    refined.signature = build_config_signature(report_signature, refined);
    if (refined.signature == seed.signature) {
        return std::nullopt;
    }
    return refined;
}

OptimizationPolicy select_policy_for_workload(
    const WorkloadSpec& workload,
    const std::vector<CandidateEvaluation>& evaluations) {
    std::uint32_t max_observations = 0;
    std::uint32_t min_observations = std::numeric_limits<std::uint32_t>::max();
    for (const auto& evaluation : evaluations) {
        max_observations = std::max(max_observations, evaluation.observations);
        min_observations = std::min(min_observations, evaluation.observations);
    }
    if (min_observations == std::numeric_limits<std::uint32_t>::max()) {
        min_observations = 0;
    }

    if (!workload.dataset_tag.empty() && max_observations >= 2u) {
        return OptimizationPolicy::trace_replay;
    }
    if (!workload.dataset_tag.empty() && max_observations >= 1) {
        return OptimizationPolicy::reinforce_softmax;
    }
    if (max_observations == 0 && min_observations == 0) {
        return OptimizationPolicy::ucb_explore;
    }
    if ((evaluations.size() > 1 && std::any_of(
            evaluations.begin(),
            evaluations.end(),
            [](const CandidateEvaluation& evaluation) { return evaluation.spsa_refined; })) &&
        max_observations == 0) {
        return OptimizationPolicy::spsa_local_search;
    }
    if (!workload.dataset_tag.empty() || max_observations > 0) {
        return OptimizationPolicy::learned_greedy;
    }
    return OptimizationPolicy::heuristic_greedy;
}

double objective_for_policy(const CandidateEvaluation& evaluation, const OptimizationPolicy policy) {
    switch (policy) {
    case OptimizationPolicy::learned_greedy:
        return evaluation.learned_objective;
    case OptimizationPolicy::trace_replay:
        return evaluation.trace_objective;
    case OptimizationPolicy::ucb_explore:
        return evaluation.explore_objective;
    case OptimizationPolicy::reinforce_softmax:
        return evaluation.reinforce_objective;
    case OptimizationPolicy::spsa_local_search:
        return evaluation.spsa_refined ? evaluation.learned_objective : (evaluation.learned_objective * 1.05);
    case OptimizationPolicy::adam_surrogate:
        return evaluation.learned_objective;
    case OptimizationPolicy::heuristic_greedy:
    default:
        return evaluation.heuristic_objective;
    }
}

struct GraphStateEvaluation {
    ContinuousExecutionState state;
    double total_objective_us = 0.0;
    std::uint32_t logical_partitions = 0;
    std::vector<OperationOptimizationResult> operations;
};

ContinuousExecutionState default_continuous_state(const WorkloadSpec& workload, const SystemProfile& system) {
    ContinuousExecutionState state;
    state.queue_depth_raw = system.low_spec_mode ? -0.4 : 0.2;
    state.stage_raw = system.low_spec_mode ? -0.2 : 0.3;
    state.tile_raw = workload.matrix_friendly ? 0.6 : 0.0;
    state.overlap_raw = workload.latency_sensitive ? -0.8 : 0.2;
    state.partition_raw = system.low_spec_mode ? -0.9 : -0.3;
    state.precision_raw = workload.matrix_friendly ? 0.5 : -0.5;
    state.single_device_logit = system.low_spec_mode ? 1.4 : 0.7;
    state.sharded_logit = workload.latency_sensitive ? -1.2 : 0.1;
    state.streaming_logit =
        (workload.kind == WorkloadKind::gaming || workload.kind == WorkloadKind::image) ? 0.6 : 0.1;
    state.overlapped_logit = 0.6;
    return clamp_state(state);
}

GraphStateEvaluation evaluate_global_state(
    const ContinuousExecutionState& state,
    const std::string& report_signature,
    const WorkloadSpec& workload,
    const WorkloadGraph& workload_graph,
    const ExecutionPlan& placement,
    const std::vector<OperationSpec>& operations,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const SystemProfile& system,
    const std::unordered_map<std::string, ExecutionOptimizer::PerformanceSummary>& performance_cache,
    const std::unordered_map<std::string, double>& backend_penalty_cache,
    const std::unordered_map<std::string, bool>& warmed_devices,
    const std::string& graph_set_signature,
    const std::unordered_map<std::string, ExecutionConfig>* cached_configs,
    const bool keep_operations) {
    GraphStateEvaluation evaluation;
    evaluation.state = clamp_state(state);
    if (keep_operations) {
        evaluation.operations.reserve(operations.size());
    }
    std::unordered_map<std::string, ExecutionConfig> selected_configs;

    for (const auto& operation : operations) {
        const ExecutionConfig* cached_config = nullptr;
        if (cached_configs != nullptr) {
            if (const auto it = cached_configs->find(operation.name); it != cached_configs->end()) {
                cached_config = &it->second;
            }
        }
        auto optimized = optimize_operation(
            report_signature,
            workload_graph,
            operation,
            workload,
            placement,
            graph_lookup,
            selected_configs,
            system,
            performance_cache,
            backend_penalty_cache,
            warmed_devices,
            graph_set_signature,
            cached_config,
            &evaluation.state,
            false);
        if (optimized.config.signature.empty()) {
            continue;
        }
        evaluation.total_objective_us += optimized.benchmark.objective_score;
        evaluation.logical_partitions += optimized.config.logical_partitions;
        selected_configs[optimized.operation.name] = optimized.config;
        if (keep_operations) {
            evaluation.operations.push_back(std::move(optimized));
        }
    }

    return evaluation;
}

GraphOptimizationSummary optimize_graph_continuous_state(
    const std::string& report_signature,
    const WorkloadSpec& workload,
    const WorkloadGraph& workload_graph,
    const ExecutionPlan& placement,
    const std::vector<OperationSpec>& operations,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const SystemProfile& system,
    const std::unordered_map<std::string, ExecutionOptimizer::PerformanceSummary>& performance_cache,
    const std::unordered_map<std::string, double>& backend_penalty_cache,
    const std::unordered_map<std::string, bool>& warmed_devices,
    const std::string& graph_set_signature,
    const std::unordered_map<std::string, ExecutionConfig>* cached_configs) {
    constexpr std::array<double, 10> kDeltas{0.35, 0.35, 0.30, 0.25, 0.25, 0.25, 0.30, 0.30, 0.30, 0.30};
    constexpr double kBeta1 = 0.9;
    constexpr double kBeta2 = 0.999;
    constexpr double kEpsilon = 1.0e-8;

    GraphOptimizationSummary summary;
    summary.optimizer_name = "adam_surrogate";
    auto state = default_continuous_state(workload, system);
    summary.initial_state = state;

    const auto initial = evaluate_global_state(
        state,
        report_signature,
        workload,
        workload_graph,
        placement,
        operations,
        graph_lookup,
        system,
        performance_cache,
        backend_penalty_cache,
        warmed_devices,
        graph_set_signature,
        cached_configs,
        false);
    summary.initial_objective_us = initial.total_objective_us;

    std::array<double, 10> m{};
    std::array<double, 10> v{};
    double learning_rate = system.low_spec_mode ? 0.06 : 0.08;
    double best_objective = initial.total_objective_us;
    auto best_state = state;

    for (std::uint32_t pass_index = 1; pass_index <= 6u; ++pass_index) {
        std::array<double, 10> gradient{};

        for (std::size_t dim = 0; dim < gradient.size(); ++dim) {
            auto plus = state;
            auto minus = state;
            switch (dim) {
            case 0:
                plus.queue_depth_raw += kDeltas[dim];
                minus.queue_depth_raw -= kDeltas[dim];
                break;
            case 1:
                plus.stage_raw += kDeltas[dim];
                minus.stage_raw -= kDeltas[dim];
                break;
            case 2:
                plus.tile_raw += kDeltas[dim];
                minus.tile_raw -= kDeltas[dim];
                break;
            case 3:
                plus.overlap_raw += kDeltas[dim];
                minus.overlap_raw -= kDeltas[dim];
                break;
            case 4:
                plus.partition_raw += kDeltas[dim];
                minus.partition_raw -= kDeltas[dim];
                break;
            case 5:
                plus.precision_raw += kDeltas[dim];
                minus.precision_raw -= kDeltas[dim];
                break;
            case 6:
                plus.single_device_logit += kDeltas[dim];
                minus.single_device_logit -= kDeltas[dim];
                break;
            case 7:
                plus.sharded_logit += kDeltas[dim];
                minus.sharded_logit -= kDeltas[dim];
                break;
            case 8:
                plus.streaming_logit += kDeltas[dim];
                minus.streaming_logit -= kDeltas[dim];
                break;
            case 9:
            default:
                plus.overlapped_logit += kDeltas[dim];
                minus.overlapped_logit -= kDeltas[dim];
                break;
            }

            const auto plus_eval = evaluate_global_state(
                clamp_state(plus),
                report_signature,
                workload,
                workload_graph,
                placement,
                operations,
                graph_lookup,
                system,
                performance_cache,
                backend_penalty_cache,
                warmed_devices,
                graph_set_signature,
                cached_configs,
                false);
            const auto minus_eval = evaluate_global_state(
                clamp_state(minus),
                report_signature,
                workload,
                workload_graph,
                placement,
                operations,
                graph_lookup,
                system,
                performance_cache,
                backend_penalty_cache,
                warmed_devices,
                graph_set_signature,
                cached_configs,
                false);
            gradient[dim] = (plus_eval.total_objective_us - minus_eval.total_objective_us) / (2.0 * kDeltas[dim]);
        }

        double gradient_norm = 0.0;
        for (double value : gradient) {
            gradient_norm += value * value;
        }
        gradient_norm = std::sqrt(gradient_norm);

        for (std::size_t dim = 0; dim < gradient.size(); ++dim) {
            m[dim] = (kBeta1 * m[dim]) + ((1.0 - kBeta1) * gradient[dim]);
            v[dim] = (kBeta2 * v[dim]) + ((1.0 - kBeta2) * gradient[dim] * gradient[dim]);
        }

        const double bias1 = 1.0 - std::pow(kBeta1, static_cast<double>(pass_index));
        const double bias2 = 1.0 - std::pow(kBeta2, static_cast<double>(pass_index));
        auto proposed = state;
        for (std::size_t dim = 0; dim < gradient.size(); ++dim) {
            const double m_hat = m[dim] / std::max(bias1, kEpsilon);
            const double v_hat = v[dim] / std::max(bias2, kEpsilon);
            const double update = learning_rate * m_hat / (std::sqrt(v_hat) + kEpsilon);
            switch (dim) {
            case 0:
                proposed.queue_depth_raw -= update;
                break;
            case 1:
                proposed.stage_raw -= update;
                break;
            case 2:
                proposed.tile_raw -= update;
                break;
            case 3:
                proposed.overlap_raw -= update;
                break;
            case 4:
                proposed.partition_raw -= update;
                break;
            case 5:
                proposed.precision_raw -= update;
                break;
            case 6:
                proposed.single_device_logit -= update;
                break;
            case 7:
                proposed.sharded_logit -= update;
                break;
            case 8:
                proposed.streaming_logit -= update;
                break;
            case 9:
            default:
                proposed.overlapped_logit -= update;
                break;
            }
        }
        proposed = clamp_state(proposed);

        const auto proposed_eval = evaluate_global_state(
            proposed,
            report_signature,
            workload,
            workload_graph,
            placement,
            operations,
            graph_lookup,
            system,
            performance_cache,
            backend_penalty_cache,
            warmed_devices,
            graph_set_signature,
            cached_configs,
            false);
        if (proposed_eval.total_objective_us < best_objective) {
            best_objective = proposed_eval.total_objective_us;
            best_state = proposed;
            state = proposed;
            learning_rate = std::min(0.12, learning_rate * 1.05);
        } else {
            learning_rate = std::max(0.02, learning_rate * 0.5);
        }

        summary.passes.push_back(GraphOptimizationPass{
            pass_index,
            proposed_eval.total_objective_us,
            gradient_norm,
            learning_rate,
            state});
    }

    summary.final_state = best_state;
    summary.final_objective_us = best_objective;
    summary.converged = best_objective <= (summary.initial_objective_us * 0.995);
    return summary;
}

ExecutionStrategy parse_strategy(const std::string_view value) {
    if (value == "sharded") {
        return ExecutionStrategy::sharded;
    }
    if (value == "streaming") {
        return ExecutionStrategy::streaming;
    }
    if (value == "overlapped") {
        return ExecutionStrategy::overlapped;
    }
    return ExecutionStrategy::single_device;
}

OperationOptimizationResult optimize_operation(
    const std::string& report_signature,
    const WorkloadGraph& workload_graph,
    const OperationSpec& operation,
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::unordered_map<std::string, ExecutionConfig>& selected_configs,
    const SystemProfile& system,
    const std::unordered_map<std::string, ExecutionOptimizer::PerformanceSummary>& performance_cache,
    const std::unordered_map<std::string, double>& backend_penalty_cache,
    const std::unordered_map<std::string, bool>& warmed_devices,
    const std::string& graph_set_signature,
    const ExecutionConfig* cached_config,
    const ContinuousExecutionState* continuous_state,
    const bool allow_validation) {
    std::vector<ExecutionConfig> candidates;
    if (cached_config != nullptr) {
        candidates.push_back(*cached_config);
    }
    const auto generated =
        build_candidate_configs(operation, workload, placement, graph_lookup, system, continuous_state, report_signature);
    for (const auto& candidate : generated) {
        const auto duplicate = std::find_if(
            candidates.begin(),
            candidates.end(),
            [&](const ExecutionConfig& existing) {
                return existing.signature == candidate.signature;
            });
        if (duplicate == candidates.end()) {
            candidates.push_back(candidate);
        }
    }

    if (candidates.empty()) {
        return {};
    }

    std::string spsa_signature;
    if (const auto spsa_candidate = make_spsa_candidate(
            candidates.front(),
            report_signature,
            workload_graph,
            operation,
            placement,
            graph_lookup,
            selected_configs,
            system,
            warmed_devices);
        spsa_candidate.has_value()) {
        const auto duplicate = std::find_if(
            candidates.begin(),
            candidates.end(),
            [&](const ExecutionConfig& existing) {
                return existing.signature == spsa_candidate->signature;
            });
        if (duplicate == candidates.end()) {
            spsa_signature = spsa_candidate->signature;
            candidates.push_back(*spsa_candidate);
        }
    }

    const auto shape_bucket = shape_bucket_for(operation);
    const double trace_weight = preset_trace_weight(workload);
    std::vector<CandidateEvaluation> evaluations;
    evaluations.reserve(candidates.size());

    double log_observation_budget = 1.0;
    for (auto candidate : candidates) {
        auto graph = build_execution_graph(
            report_signature,
            workload_graph,
            operation,
            candidate,
            placement,
            graph_lookup,
            selected_configs);
        const double system_penalty_us = compute_surrogate_penalty_us(operation, candidate, system, warmed_devices);
        double learning_scale = 1.0;
        std::uint32_t observations = 0;
        double historical_latency_us = 0.0;
        double average_reward = 0.0;
        double backend_penalty_us = 0.0;
        double calibration_ratio = 1.0;
        double validation_spread_us = 0.0;
        if (const auto it =
                performance_cache.find(performance_key(graph_set_signature, workload, system, shape_bucket, candidate));
            it != performance_cache.end()) {
            observations = it->second.observations;
            historical_latency_us = it->second.average_effective_latency_us;
            average_reward = it->second.average_reward;
            learning_scale = std::clamp(it->second.average_prediction_scale, 0.50, 2.50);
            calibration_ratio = std::clamp(it->second.average_calibration_ratio, 0.50, 2.50);
            validation_spread_us = it->second.average_validation_spread_us;
            if (it->second.observations < kLearningWarmupSamples) {
                learning_scale *= 0.97;
            }
            if (it->second.average_relative_error > candidate.target_error_tolerance) {
                learning_scale *= 1.5;
            }
        }
        if (const auto penalty_it =
                backend_penalty_cache.find(backend_penalty_key(graph_set_signature, workload, operation.name, candidate));
            penalty_it != backend_penalty_cache.end()) {
            backend_penalty_us = penalty_it->second;
        }

        const double surrogate_latency_us =
            (graph.predicted_latency_us * system.sustained_slowdown * learning_scale * calibration_ratio) + system_penalty_us +
            backend_penalty_us;
        BenchmarkRecord benchmark;
        if (allow_validation) {
            benchmark = benchmark_operation(
                operation,
                candidate,
                graph,
                graph_lookup,
                shape_bucket,
                surrogate_latency_us,
                system_penalty_us);
        } else {
            benchmark.operation_name = operation.name;
            benchmark.config_signature = candidate.signature;
            benchmark.shape_bucket = shape_bucket;
            benchmark.predicted_latency_us = graph.predicted_latency_us;
            benchmark.surrogate_latency_us = surrogate_latency_us;
            benchmark.calibrated_prediction_us = surrogate_latency_us;
            benchmark.calibration_ratio = calibration_ratio;
            benchmark.calibration_confidence = observations == 0 ? 0.15 : 0.35;
            benchmark.candidate_spread_us = validation_spread_us;
            benchmark.reference_spread_us = validation_spread_us * 0.5;
            benchmark.validation_samples = observations == 0 ? 1u : std::min(5u, observations + 1u);
            benchmark.effective_latency_us = std::max(0.01, surrogate_latency_us);
            benchmark.system_penalty_us = system_penalty_us;
            benchmark.relative_error = graph.expected_relative_error;
            benchmark.accuracy_within_tolerance =
                graph.expected_relative_error <= (candidate.target_error_tolerance + 1.0e-12);
            benchmark.simulated = true;
            benchmark.speedup_vs_reference = 1.0;
        }

        double heuristic_objective = benchmark.calibrated_prediction_us > 0.0
                                         ? benchmark.calibrated_prediction_us
                                         : surrogate_latency_us;
        if (!benchmark.simulated) {
            heuristic_objective = std::min(heuristic_objective, benchmark.validation_latency_us);
        }
        if (!benchmark.accuracy_within_tolerance) {
            const double tolerance = std::max(candidate.target_error_tolerance, 1.0e-9);
            heuristic_objective *= 10.0 + (benchmark.relative_error / tolerance);
        }

        const double learned_objective =
            (heuristic_objective * 0.45) + (benchmark.effective_latency_us * 0.45) +
            (graph.predicted_transfer_latency_us * 0.05) +
            (graph.predicted_memory_pressure * benchmark.effective_latency_us * 0.05);
        const double trace_bias = observations >= kLearningWarmupSamples ? (trace_weight + 2.0) : trace_weight;
        const double trace_objective =
            observations == 0
                ? learned_objective
                : ((historical_latency_us * trace_bias) + learned_objective) / (trace_bias + 1.0);
        log_observation_budget += static_cast<double>(observations + 1u);
        const double exploration_bonus =
            learned_objective * 0.35 *
            std::sqrt(std::log(log_observation_budget + 1.0) / static_cast<double>(observations + 1u));
        const double explore_objective = std::max(0.01, learned_objective - exploration_bonus);
        const double preference_scale = std::exp(std::clamp(average_reward, -1.5, 1.5));
        const double reinforce_objective = learned_objective / std::max(0.20, preference_scale);

        benchmark.trace_weight = trace_weight;
        evaluations.push_back(CandidateEvaluation{
            std::move(candidate),
            std::move(graph),
            std::move(benchmark),
            OptimizationPolicy::heuristic_greedy,
            !spsa_signature.empty() && candidate.signature == spsa_signature,
            heuristic_objective,
            learned_objective,
            trace_objective,
            explore_objective,
            reinforce_objective,
            observations});
    }

    if (evaluations.empty()) {
        return {};
    }

    const auto selected_policy = select_policy_for_workload(workload, evaluations);
    const auto winner = std::min_element(
        evaluations.begin(),
        evaluations.end(),
        [&](const CandidateEvaluation& left, const CandidateEvaluation& right) {
            return objective_for_policy(left, selected_policy) < objective_for_policy(right, selected_policy);
        });

    OperationOptimizationResult best;
    best.operation = operation;
    best.config = winner->config;
    best.graph = std::move(winner->graph);
    best.benchmark = std::move(winner->benchmark);
    best.benchmark.optimizer_name = to_string(selected_policy);
    if (winner->spsa_refined) {
        best.benchmark.optimizer_name += "+spsa";
    }
    best.benchmark.objective_score = objective_for_policy(*winner, selected_policy);
    return best;
}

void update_performance_summary(
    std::unordered_map<std::string, ExecutionOptimizer::PerformanceSummary>& performance_cache,
    const std::string& key,
    const std::string& shape_bucket,
    const ExecutionConfig& config,
    const double effective_latency_us,
    const double relative_error,
    const double predicted_latency_us,
    const double calibration_ratio,
    const double system_penalty_us,
    const double validation_spread_us,
    const double reward) {
    auto& summary = performance_cache[key];
    summary.shape_bucket = shape_bucket;
    summary.config = config;
    ++summary.observations;
    const double sample_count = static_cast<double>(summary.observations);
    summary.average_effective_latency_us +=
        (effective_latency_us - summary.average_effective_latency_us) / sample_count;
    summary.average_relative_error +=
        (relative_error - summary.average_relative_error) / sample_count;
    const double prediction_scale =
        predicted_latency_us > 0.0 ? (effective_latency_us / predicted_latency_us) : 1.0;
    summary.average_prediction_scale +=
        (prediction_scale - summary.average_prediction_scale) / sample_count;
    summary.average_calibration_ratio +=
        (calibration_ratio - summary.average_calibration_ratio) / sample_count;
    summary.average_system_penalty_us +=
        (system_penalty_us - summary.average_system_penalty_us) / sample_count;
    summary.average_validation_spread_us +=
        (validation_spread_us - summary.average_validation_spread_us) / sample_count;
    summary.average_reward +=
        (reward - summary.average_reward) / sample_count;
}

void update_device_learning_state(
    std::unordered_map<std::string, bool>& warmed_devices,
    std::unordered_map<std::string, double>& device_sustained_slowdown,
    const ExecutionConfig& config,
    const double effective_latency_us,
    const double predicted_latency_us) {
    for (const auto& device_uid : config.participating_devices) {
        warmed_devices[device_uid] = true;
        auto& slowdown = device_sustained_slowdown[device_uid];
        if (slowdown <= 0.0) {
            slowdown = 1.0;
        }
        const double observed_slowdown =
            predicted_latency_us > 0.0
                ? std::clamp(effective_latency_us / std::max(predicted_latency_us, 1.0), 0.5, 8.0)
                : 1.0;
        slowdown = (slowdown * 0.8) + (observed_slowdown * 0.2);
    }
}

}  // namespace

std::string to_string(const OperationClass op_class) {
    switch (op_class) {
    case OperationClass::elementwise_map:
        return "elementwise_map";
    case OperationClass::reduction:
        return "reduction";
    case OperationClass::matmul:
        return "matmul";
    case OperationClass::convolution_2d:
        return "convolution_2d";
    case OperationClass::resample_2d:
    default:
        return "resample_2d";
    }
}

std::string to_string(const ExecutionNodeKind kind) {
    switch (kind) {
    case ExecutionNodeKind::source:
        return "source";
    case ExecutionNodeKind::dispatch:
        return "dispatch";
    case ExecutionNodeKind::compute:
        return "compute";
    case ExecutionNodeKind::aggregate:
        return "aggregate";
    case ExecutionNodeKind::sink:
        return "sink";
    case ExecutionNodeKind::synchronize:
    default:
        return "synchronize";
    }
}

std::string to_string(const ExecutionEdgeKind kind) {
    switch (kind) {
    case ExecutionEdgeKind::dataflow:
        return "dataflow";
    case ExecutionEdgeKind::control:
        return "control";
    case ExecutionEdgeKind::dependency:
        return "dependency";
    case ExecutionEdgeKind::aggregation:
    default:
        return "aggregation";
    }
}

std::string to_string(const ExecutionStrategy strategy) {
    switch (strategy) {
    case ExecutionStrategy::single_device:
        return "single_device";
    case ExecutionStrategy::sharded:
        return "sharded";
    case ExecutionStrategy::streaming:
        return "streaming";
    case ExecutionStrategy::overlapped:
    default:
        return "overlapped";
    }
}

std::string to_string(const OptimizationPolicy policy) {
    switch (policy) {
    case OptimizationPolicy::heuristic_greedy:
        return "heuristic_greedy";
    case OptimizationPolicy::learned_greedy:
        return "learned_greedy";
    case OptimizationPolicy::trace_replay:
        return "trace_replay";
    case OptimizationPolicy::ucb_explore:
        return "ucb_explore";
    case OptimizationPolicy::reinforce_softmax:
        return "reinforce_softmax";
    case OptimizationPolicy::spsa_local_search:
        return "spsa_local_search";
    case OptimizationPolicy::adam_surrogate:
        return "adam_surrogate";
    default:
        return "heuristic_greedy";
    }
}

std::vector<OperationSpec> legacy_default_operation_suite_unused(const WorkloadSpec& workload) {
    const auto make_elementwise = [](
                                      const std::string& name,
                                      const std::uint64_t elements,
                                      const double tolerance = 5.0e-4) {
        return OperationSpec{
            name,
            OperationClass::elementwise_map,
            {elements},
            elements * sizeof(float) * 2ull,
            elements * sizeof(float),
            0,
            static_cast<double>(elements) * 3.0,
            tolerance,
            true,
            false,
            true,
            false};
    };

    const auto make_reduction = [](
                                    const std::string& name,
                                    const std::uint64_t elements,
                                    const double tolerance = 1.0e-3) {
        return OperationSpec{
            name,
            OperationClass::reduction,
            {elements},
            elements * sizeof(float),
            sizeof(float),
            32ull * kKiB,
            static_cast<double>(elements),
            tolerance,
            true,
            true,
            false,
            false};
    };

    const auto make_matmul = [](
                                 const std::string& name,
                                 const std::uint32_t m,
                                 const std::uint32_t n,
                                 const std::uint32_t k,
                                 const double tolerance = 2.0e-3) {
        return OperationSpec{
            name,
            OperationClass::matmul,
            {m, n, k},
            2ull * m * k * sizeof(float),
            1ull * m * n * sizeof(float),
            0,
            2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k),
            tolerance,
            true,
            false,
            false,
            true};
    };

    const auto make_convolution = [](
                                      const std::string& name,
                                      const std::uint32_t height,
                                      const std::uint32_t width,
                                      const double tolerance = 2.0e-3) {
        return OperationSpec{
            name,
            OperationClass::convolution_2d,
            {height, width},
            1ull * height * width * sizeof(float),
            1ull * (height - 2u) * (width - 2u) * sizeof(float),
            9ull * sizeof(float),
            18.0 * static_cast<double>(height - 2u) * static_cast<double>(width - 2u),
            tolerance,
            true,
            false,
            true,
            false};
    };

    const auto make_resample = [](
                                   const std::string& name,
                                   const std::uint32_t src_h,
                                   const std::uint32_t src_w,
                                   const std::uint32_t dst_h,
                                   const std::uint32_t dst_w,
                                   const double tolerance = 1.5e-3) {
        return OperationSpec{
            name,
            OperationClass::resample_2d,
            {src_h, src_w, dst_h, dst_w},
            1ull * src_h * src_w * sizeof(float),
            1ull * dst_h * dst_w * sizeof(float),
            0,
            8.0 * static_cast<double>(dst_h) * static_cast<double>(dst_w),
            tolerance,
            true,
            false,
            true,
            false};
    };

    if (workload.dataset_tag == "gaming-fsr-like-720p-to-1080p") {
        constexpr std::uint32_t src_h = 720u;
        constexpr std::uint32_t src_w = 1280u;
        constexpr std::uint32_t dst_h = 1080u;
        constexpr std::uint32_t dst_w = 1920u;
        return {
            make_elementwise("frame-pre-tonemap", src_h * src_w),
            make_elementwise("reactive-mask", src_h * src_w),
            make_reduction("exposure-luma", src_h * src_w, 1.2e-3),
            make_convolution("history-reconstruction", src_h, src_w),
            make_convolution("detail-sharpen", src_h, src_w),
            make_resample("upscale-resolve", src_h, src_w, dst_h, dst_w),
            make_elementwise("post-tonemap", dst_h * dst_w)};
    }

    if (workload.dataset_tag == "ai-vision-inference-224") {
        return {
            make_convolution("stem-conv3x3", 224u, 224u),
            make_matmul("patch-proj", 196u, 384u, 384u),
            make_matmul("attention-qkv", 196u, 384u, 384u),
            make_reduction("attention-score-reduce", 196ull * 384ull, 1.5e-3),
            make_matmul("mlp-up", 196u, 768u, 384u),
            make_elementwise("mlp-activation", 196ull * 768ull, 7.5e-4),
            make_matmul("mlp-down", 196u, 384u, 768u),
            make_reduction("token-pool", 196ull * 384ull, 1.5e-3)};
    }

    if (workload.dataset_tag == "ai-transformer-train-step-lite") {
        return {
            make_matmul("fwd-proj", 128u, 384u, 384u),
            make_elementwise("fwd-activation", 128ull * 384ull, 7.5e-4),
            make_matmul("fwd-head", 128u, 384u, 384u),
            make_reduction("loss-reduce", 128ull * 384ull, 1.5e-3),
            make_elementwise("loss-scale", 128ull * 384ull, 7.5e-4),
            make_matmul("grad-head", 384u, 384u, 128u, 2.5e-3),
            make_matmul("grad-input", 128u, 384u, 384u, 2.5e-3),
            make_reduction("grad-norm", 384ull * 384ull, 1.5e-3),
            make_elementwise("adam-moment", 384ull * 384ull, 7.5e-4),
            make_elementwise("adam-update", 384ull * 384ull, 7.5e-4)};
    }

    const std::uint64_t working_set =
        workload.working_set_bytes == 0 ? (32ull * kMiB) : workload.working_set_bytes;
    const std::uint64_t sample_bytes = clamp_u64(working_set / 12ull, 2ull * kMiB, 16ull * kMiB);
    const std::uint64_t vector_count = std::max<std::uint64_t>(sample_bytes / sizeof(float), 64ull * 1024ull);

    const auto matmul_side = round_down_to_multiple(
        static_cast<std::uint32_t>(std::clamp(
            std::sqrt(static_cast<double>(sample_bytes) / 12.0),
            32.0,
            96.0)),
        16u);
    const auto conv_side = round_down_to_multiple(
        static_cast<std::uint32_t>(std::clamp(
            std::sqrt(static_cast<double>(sample_bytes) / 8.0),
            32.0,
            80.0)),
        8u);
    const auto resample_src = round_down_to_multiple(
        static_cast<std::uint32_t>(std::clamp(
            std::sqrt(static_cast<double>(sample_bytes) / 4.0),
            64.0,
            160.0)),
        16u);
    const auto resample_dst = std::max<std::uint32_t>(96u, (resample_src * 3u) / 2u);

    const OperationSpec elementwise{
        "elementwise-map",
        OperationClass::elementwise_map,
        {vector_count},
        vector_count * sizeof(float) * 2ull,
        vector_count * sizeof(float),
        0,
        static_cast<double>(vector_count) * 3.0,
        5.0e-4,
        true,
        false,
        true,
        false};

    const OperationSpec reduction{
        "reduction-sum",
        OperationClass::reduction,
        {vector_count},
        vector_count * sizeof(float),
        sizeof(float),
        32ull * kKiB,
        static_cast<double>(vector_count),
        1.0e-3,
        true,
        true,
        false,
        false};

    const OperationSpec matmul{
        "blocked-matmul",
        OperationClass::matmul,
        {matmul_side, matmul_side, matmul_side},
        2ull * matmul_side * matmul_side * sizeof(float),
        1ull * matmul_side * matmul_side * sizeof(float),
        0,
        2.0 * static_cast<double>(matmul_side) * static_cast<double>(matmul_side) * static_cast<double>(matmul_side),
        2.0e-3,
        true,
        false,
        false,
        true};

    const OperationSpec convolution{
        "conv3x3",
        OperationClass::convolution_2d,
        {conv_side, conv_side},
        1ull * conv_side * conv_side * sizeof(float),
        1ull * (conv_side - 2u) * (conv_side - 2u) * sizeof(float),
        9ull * sizeof(float),
        18.0 * static_cast<double>(conv_side - 2u) * static_cast<double>(conv_side - 2u),
        2.0e-3,
        true,
        false,
        true,
        false};

    const OperationSpec resample{
        "bilinear-resample",
        OperationClass::resample_2d,
        {resample_src, resample_src, resample_dst, resample_dst},
        1ull * resample_src * resample_src * sizeof(float),
        1ull * resample_dst * resample_dst * sizeof(float),
        0,
        8.0 * static_cast<double>(resample_dst) * static_cast<double>(resample_dst),
        1.5e-3,
        true,
        false,
        true,
        false};

    switch (workload.kind) {
    case WorkloadKind::gaming:
        return {
            make_elementwise("frame-pre-tonemap", 1280ull * 720ull),
            make_reduction("exposure-luma", 1280ull * 720ull, 1.2e-3),
            make_convolution("history-reconstruction", 720u, 1280u),
            make_resample("upscale-resolve", 720u, 1280u, 1080u, 1920u),
            make_elementwise("post-tonemap", 1920ull * 1080ull)};
    case WorkloadKind::image:
        return {elementwise, convolution, resample};
    case WorkloadKind::inference:
        return {elementwise, reduction, matmul, convolution};
    case WorkloadKind::training:
        return {
            make_matmul("fwd-proj", 128u, matmul_side * 4u, matmul_side * 4u),
            make_elementwise("fwd-activation", 128ull * matmul_side * 4ull, 7.5e-4),
            make_matmul("grad-input", 128u, matmul_side * 4u, matmul_side * 4u, 2.5e-3),
            make_reduction("grad-norm", static_cast<std::uint64_t>(matmul_side) * matmul_side * 16ull, 1.5e-3),
            make_elementwise("adam-update", static_cast<std::uint64_t>(matmul_side) * matmul_side * 16ull, 7.5e-4)};
    case WorkloadKind::tensor:
        return {elementwise, reduction, matmul, convolution, resample};
    case WorkloadKind::custom:
    default:
        return {elementwise, reduction, matmul};
    }
}

ExecutionOptimizer::ExecutionOptimizer(std::filesystem::path cache_path)
    : cache_path_(std::move(cache_path)),
      performance_cache_path_(performance_cache_path_for(cache_path_)) {}

std::filesystem::path ExecutionOptimizer::default_cache_path() {
    try {
        return std::filesystem::temp_directory_path() / "gpu_runtime_execution_cache.tsv";
    } catch (const std::exception&) {
        return std::filesystem::path("gpu_runtime_execution_cache.tsv");
    }
}

OptimizationReport ExecutionOptimizer::optimize(
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::vector<HardwareGraph>& graphs) {
    load_cache();

    OptimizationReport report;
    report.workload_kind = workload.kind;
    report.dataset_tag = workload.dataset_tag;
    report.placement = placement;
    report.system_profile = capture_system_profile(workload, graphs);
    if (!device_sustained_slowdown_.empty()) {
        double total_slowdown = 0.0;
        for (const auto& [uid, slowdown] : device_sustained_slowdown_) {
            (void)uid;
            total_slowdown += slowdown;
        }
        report.system_profile.sustained_slowdown *=
            std::max(1.0, total_slowdown / static_cast<double>(device_sustained_slowdown_.size()));
    }
    const double warmed_fraction =
        graphs.empty() ? 0.0 : (static_cast<double>(warmed_devices_.size()) / static_cast<double>(graphs.size()));
    report.system_profile.readiness_score = clamp_unit(
        (0.12 + (0.88 * warmed_fraction * report.system_profile.amortization_gain)) /
        std::max(report.system_profile.sustained_slowdown, 1.0));
    report.system_profile.stability_score = clamp_unit(
        report.system_profile.stability_score * (0.85 + (0.15 * warmed_fraction)));

    report.workload_graph = default_workload_graph(workload);
    const auto& operations = report.workload_graph.operations;
    report.signature = build_report_signature(workload, placement, operations);
    const auto graph_set_signature = summarize_graph_set(graphs);

    std::unordered_map<std::string, const HardwareGraph*> graph_lookup;
    graph_lookup.reserve(graphs.size());
    for (const auto& graph : graphs) {
        graph_lookup.emplace(graph.uid, &graph);
    }

    std::unordered_map<std::string, ExecutionConfig> cached_by_operation;
    bool fully_cached = false;
    if (const auto cache_it = cache_.find(report.signature); cache_it != cache_.end()) {
        fully_cached = true;
        for (const auto& cached : cache_it->second) {
            if (!graph_lookup.contains(cached.config.primary_device_uid)) {
                fully_cached = false;
                break;
            }
            cached_by_operation.emplace(cached.operation_name, cached.config);
        }
        for (const auto& operation : operations) {
            if (!cached_by_operation.contains(operation.name)) {
                fully_cached = false;
                break;
            }
        }
    }

    std::vector<CachedConfig> persisted_configs;
    persisted_configs.reserve(operations.size());

    std::unordered_map<std::string, ExecutionConfig> cache_input = cached_by_operation;
    report.graph_optimization = optimize_graph_continuous_state(
        report.signature,
        workload,
        report.workload_graph,
        placement,
        operations,
        graph_lookup,
        report.system_profile,
        performance_cache_,
        backend_penalty_cache_,
        warmed_devices_,
        graph_set_signature,
        fully_cached ? &cache_input : nullptr);

    std::unordered_map<std::string, ExecutionConfig> selected_configs;
    for (const auto& operation : operations) {
        const auto cached_it = cached_by_operation.find(operation.name);
        const ExecutionConfig* cached_config = cached_it == cached_by_operation.end() ? nullptr : &cached_it->second;
        auto result = optimize_operation(
            report.signature,
            report.workload_graph,
            operation,
            workload,
            placement,
            graph_lookup,
            selected_configs,
            report.system_profile,
            performance_cache_,
            backend_penalty_cache_,
            warmed_devices_,
            graph_set_signature,
            cached_config,
            &report.graph_optimization.final_state,
            true);
        if (result.config.signature.empty()) {
            continue;
        }

        const auto perf_key = performance_key(
            graph_set_signature,
            workload,
            report.system_profile,
            result.benchmark.shape_bucket,
            result.config);
        update_performance_summary(
            performance_cache_,
            perf_key,
            result.benchmark.shape_bucket,
            result.config,
            result.benchmark.effective_latency_us,
            result.benchmark.relative_error,
            result.graph.predicted_latency_us,
            result.benchmark.calibration_ratio,
            result.benchmark.system_penalty_us,
            result.benchmark.candidate_spread_us,
            std::log(std::max(result.benchmark.speedup_vs_reference, 1.0e-6)));
        update_device_learning_state(
            warmed_devices_,
            device_sustained_slowdown_,
            result.config,
            result.benchmark.effective_latency_us,
            result.graph.predicted_latency_us);

        persisted_configs.push_back(CachedConfig{
            operation.name,
            result.config});
        report.graph_optimization.total_logical_partitions += result.config.logical_partitions;
        selected_configs[result.operation.name] = result.config;
        report.operations.push_back(std::move(result));
    }

    cache_[report.signature] = std::move(persisted_configs);
    persist_cache();
    report.loaded_from_cache = fully_cached;
    return report;
}

void ExecutionOptimizer::ingest_execution_feedback(
    const OptimizationReport& report,
    const std::vector<ExecutionFeedbackRecord>& feedback,
    const std::vector<HardwareGraph>& graphs) {
    if (feedback.empty() || report.operations.empty()) {
        return;
    }

    load_cache();

    std::unordered_map<std::string, const ExecutionFeedbackRecord*> feedback_by_operation;
    feedback_by_operation.reserve(feedback.size());
    for (const auto& record : feedback) {
        feedback_by_operation.emplace(record.operation_name, &record);
    }

    const auto graph_set_signature = summarize_graph_set(graphs);
    const WorkloadSpec feedback_workload{
        report.signature,
        report.workload_kind,
        report.dataset_tag,
        0,
        0,
        0.0,
        1,
        false,
        false,
        false};
    for (const auto& optimized : report.operations) {
        const auto feedback_it = feedback_by_operation.find(optimized.operation.name);
        if (feedback_it == feedback_by_operation.end()) {
            continue;
        }

        const auto& record = *feedback_it->second;
        const double effective_latency_us = std::max(0.01, record.runtime_us);
        const double observed_penalty_us =
            std::max(0.0, effective_latency_us - std::max(optimized.graph.predicted_latency_us, 0.0));
        const auto perf_key = performance_key(
            graph_set_signature,
            feedback_workload,
            report.system_profile,
            optimized.benchmark.shape_bucket,
            optimized.config);
        update_performance_summary(
            performance_cache_,
            perf_key,
            optimized.benchmark.shape_bucket,
            optimized.config,
            effective_latency_us,
            record.relative_error,
            optimized.graph.predicted_latency_us,
            optimized.benchmark.calibration_ratio,
            observed_penalty_us,
            optimized.benchmark.candidate_spread_us,
            std::log(std::max(record.reference_runtime_us / effective_latency_us, 1.0e-6)));
        update_device_learning_state(
            warmed_devices_,
            device_sustained_slowdown_,
            optimized.config,
            effective_latency_us,
            optimized.graph.predicted_latency_us);

        const double slowdown_vs_reference =
            record.reference_runtime_us > 0.0 ? (effective_latency_us / record.reference_runtime_us) : 1.0;
        const auto penalty_key =
            backend_penalty_key(graph_set_signature, feedback_workload, optimized.operation.name, optimized.config);
        auto& backend_penalty = backend_penalty_cache_[penalty_key];
        backend_penalty *= 0.92;
        if (record.used_host && !record.used_opencl) {
            backend_penalty *= 0.85;
        }
        if (!record.used_host && slowdown_vs_reference > 1.20) {
            const double severe_penalty =
                std::min(5000.0, std::max(300.0, (slowdown_vs_reference - 1.0) * 900.0));
            backend_penalty = std::max(backend_penalty, severe_penalty);
        } else if (slowdown_vs_reference < 0.95) {
            backend_penalty *= 0.70;
        }
        if (!record.used_host && slowdown_vs_reference > 2.50) {
            for (const auto& device_uid : optimized.config.participating_devices) {
                auto& slowdown = device_sustained_slowdown_[device_uid];
                slowdown = std::max(slowdown, std::min(12.0, slowdown_vs_reference));
            }
        }
    }

    persist_cache();
}

void ExecutionOptimizer::load_cache() {
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
        if (fields.size() != 14 && fields.size() != 21) {
            continue;
        }

        try {
            ExecutionConfig config;
            config.operation_name = fields[1];
            config.strategy = parse_strategy(fields[2]);
            config.primary_device_uid = fields[3];
            config.participating_devices = split_csv(fields[4]);
            config.mapped_structural_nodes = split_csv(fields[5]);
            config.queue_depth = static_cast<std::uint32_t>(std::stoul(fields[6]));
            config.stages = static_cast<std::uint32_t>(std::stoul(fields[7]));
            config.tile_x = static_cast<std::uint32_t>(std::stoul(fields[8]));
            config.tile_y = static_cast<std::uint32_t>(std::stoul(fields[9]));
            config.tile_k = static_cast<std::uint32_t>(std::stoul(fields[10]));
            std::size_t next = 11;
            if (fields.size() == 21) {
                config.logical_partitions = static_cast<std::uint32_t>(std::stoul(fields[next++]));
            }
            config.overlap_transfers = std::stoi(fields[next++]) != 0;
            config.use_low_precision = std::stoi(fields[next++]) != 0;
            config.target_error_tolerance = std::stod(fields[next++]);
            if (fields.size() == 21) {
                config.queue_depth_scale = std::stod(fields[next++]);
                config.stage_scale = std::stod(fields[next++]);
                config.tile_scale = std::stod(fields[next++]);
                config.overlap_ratio = std::stod(fields[next++]);
                config.partition_intensity = std::stod(fields[next++]);
                config.precision_mix = std::stod(fields[next++]);
            }
            cache_[fields[0]].push_back(CachedConfig{config.operation_name, std::move(config)});
        } catch (const std::exception&) {
            continue;
        }
    }

    std::ifstream performance_input(performance_cache_path_);
    if (!performance_input.is_open()) {
        return;
    }

    while (std::getline(performance_input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        const auto fields = split_tab(line);
        if (fields.size() != 19 && fields.size() != 26 && fields.size() != 28) {
            continue;
        }

        try {
            PerformanceSummary summary;
            summary.shape_bucket = fields[1];
            summary.config.operation_name = fields[2];
            summary.config.strategy = parse_strategy(fields[3]);
            summary.config.primary_device_uid = fields[4];
            summary.config.participating_devices = split_csv(fields[5]);
            summary.config.queue_depth = static_cast<std::uint32_t>(std::stoul(fields[6]));
            summary.config.stages = static_cast<std::uint32_t>(std::stoul(fields[7]));
            summary.config.tile_x = static_cast<std::uint32_t>(std::stoul(fields[8]));
            summary.config.tile_y = static_cast<std::uint32_t>(std::stoul(fields[9]));
            summary.config.tile_k = static_cast<std::uint32_t>(std::stoul(fields[10]));
            std::size_t next = 11;
            if (fields.size() == 26 || fields.size() == 28) {
                summary.config.logical_partitions = static_cast<std::uint32_t>(std::stoul(fields[next++]));
            }
            summary.config.overlap_transfers = std::stoi(fields[next++]) != 0;
            summary.config.use_low_precision = std::stoi(fields[next++]) != 0;
            if (fields.size() == 26 || fields.size() == 28) {
                summary.config.queue_depth_scale = std::stod(fields[next++]);
                summary.config.stage_scale = std::stod(fields[next++]);
                summary.config.tile_scale = std::stod(fields[next++]);
                summary.config.overlap_ratio = std::stod(fields[next++]);
                summary.config.partition_intensity = std::stod(fields[next++]);
                summary.config.precision_mix = std::stod(fields[next++]);
            }
            summary.observations = static_cast<std::uint32_t>(std::stoul(fields[next++]));
            summary.average_effective_latency_us = std::stod(fields[next++]);
            summary.average_relative_error = std::stod(fields[next++]);
            summary.average_prediction_scale = std::stod(fields[next++]);
            if (fields.size() == 28) {
                summary.average_calibration_ratio = std::stod(fields[next++]);
            }
            summary.average_system_penalty_us = std::stod(fields[next++]);
            if (fields.size() == 28) {
                summary.average_validation_spread_us = std::stod(fields[next++]);
            }
            summary.average_reward = std::stod(fields[next++]);
            performance_cache_[fields[0]] = std::move(summary);
        } catch (const std::exception&) {
            continue;
        }
    }
}

void ExecutionOptimizer::persist_cache() const {
    const auto parent = cache_path_.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }

    std::ofstream output(cache_path_, std::ios::trunc);
    if (!output.is_open()) {
        return;
    }

    output << "# signature\toperation\tstrategy\tprimary_device\tparticipating_devices\tmapped_nodes\tqueue_depth\tstages\ttile_x\ttile_y\ttile_k\tlogical_partitions\toverlap\tlow_precision\ttolerance\tqueue_scale\tstage_scale\ttile_scale\toverlap_ratio\tpartition_intensity\tprecision_mix\n";
    for (const auto& [signature, configs] : cache_) {
        for (const auto& cached : configs) {
            output << signature << '\t'
                   << cached.operation_name << '\t'
                   << to_string(cached.config.strategy) << '\t'
                   << cached.config.primary_device_uid << '\t'
                   << join_csv(cached.config.participating_devices) << '\t'
                   << join_csv(cached.config.mapped_structural_nodes) << '\t'
                   << cached.config.queue_depth << '\t'
                   << cached.config.stages << '\t'
                   << cached.config.tile_x << '\t'
                   << cached.config.tile_y << '\t'
                   << cached.config.tile_k << '\t'
                   << cached.config.logical_partitions << '\t'
                   << (cached.config.overlap_transfers ? 1 : 0) << '\t'
                   << (cached.config.use_low_precision ? 1 : 0) << '\t'
                   << cached.config.target_error_tolerance << '\t'
                   << cached.config.queue_depth_scale << '\t'
                   << cached.config.stage_scale << '\t'
                   << cached.config.tile_scale << '\t'
                   << cached.config.overlap_ratio << '\t'
                   << cached.config.partition_intensity << '\t'
                   << cached.config.precision_mix << '\n';
        }
    }

    std::ofstream performance_output(performance_cache_path_, std::ios::trunc);
    if (!performance_output.is_open()) {
        return;
    }

    performance_output
        << "# key\tshape_bucket\toperation\tstrategy\tprimary_device\tparticipating_devices\tqueue_depth\tstages\ttile_x\ttile_y\ttile_k\tlogical_partitions\toverlap\tlow_precision\tqueue_scale\tstage_scale\ttile_scale\toverlap_ratio\tpartition_intensity\tprecision_mix\tobservations\tavg_latency\tavg_error\tavg_scale\tavg_calibration_ratio\tavg_system_penalty\tavg_validation_spread\tavg_reward\n";
    for (const auto& [key, summary] : performance_cache_) {
        performance_output << key << '\t'
                           << summary.shape_bucket << '\t'
                           << summary.config.operation_name << '\t'
                           << to_string(summary.config.strategy) << '\t'
                           << summary.config.primary_device_uid << '\t'
                           << join_csv(summary.config.participating_devices) << '\t'
                           << summary.config.queue_depth << '\t'
                           << summary.config.stages << '\t'
                           << summary.config.tile_x << '\t'
                           << summary.config.tile_y << '\t'
                           << summary.config.tile_k << '\t'
                           << summary.config.logical_partitions << '\t'
                           << (summary.config.overlap_transfers ? 1 : 0) << '\t'
                           << (summary.config.use_low_precision ? 1 : 0) << '\t'
                           << summary.config.queue_depth_scale << '\t'
                           << summary.config.stage_scale << '\t'
                           << summary.config.tile_scale << '\t'
                           << summary.config.overlap_ratio << '\t'
                           << summary.config.partition_intensity << '\t'
                           << summary.config.precision_mix << '\t'
                           << summary.observations << '\t'
                           << summary.average_effective_latency_us << '\t'
                           << summary.average_relative_error << '\t'
                           << summary.average_prediction_scale << '\t'
                           << summary.average_calibration_ratio << '\t'
                           << summary.average_system_penalty_us << '\t'
                           << summary.average_validation_spread_us << '\t'
                           << summary.average_reward << '\n';
    }
}

}  // namespace gpu
