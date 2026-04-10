#include "jakal/execution.hpp"

#include "jakal/device.hpp"
#include "jakal/executors/host_native_kernels.hpp"
#include "jakal/operation_variant_registry.hpp"
#include "jakal/workloads.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
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

namespace jakal {
namespace {

constexpr std::uint64_t kKiB = 1024ull;
constexpr std::uint64_t kMiB = 1024ull * 1024ull;
constexpr std::uint64_t kGiB = 1024ull * 1024ull * 1024ull;
constexpr double kBytesPerGb = 1.0e9;
constexpr double kMicrosecondsPerSecond = 1.0e6;
constexpr std::uint32_t kLearningWarmupSamples = 3u;
constexpr std::uint32_t kValidationMeasurementRounds = 2u;
constexpr std::uint32_t kMinimumValidationMeasurementRounds = 1u;
constexpr std::uint32_t kGraphOptimizationPasses = 4u;
constexpr std::string_view kExecutionHeuristicRevision = "exec-r3-residency-pipeline";
constexpr std::uint64_t kDefaultCpuPackBudgetBytes = 256ull * 1024ull;

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

bool cpu_materialized_lowering_active(const OperationSpec& operation) {
    switch (operation.op_class) {
    case OperationClass::matmul:
        return operation.cpu_pack_weights || operation.cpu_pretranspose_rhs;
    case OperationClass::convolution_2d:
        return operation.cpu_input_layout.find("conv-patch9") != std::string::npos;
    case OperationClass::resample_2d:
        return operation.cpu_input_layout.find("resample-packed6") != std::string::npos;
    default:
        return false;
    }
}

bool gpu_materialized_lowering_active(const OperationSpec& operation) {
    switch (operation.op_class) {
    case OperationClass::matmul:
        return operation.gpu_pack_weights || operation.gpu_pretranspose_rhs;
    case OperationClass::convolution_2d:
        return operation.gpu_input_layout.find("conv-patch9") != std::string::npos;
    case OperationClass::resample_2d:
        return operation.gpu_input_layout.find("resample-packed6") != std::string::npos;
    default:
        return false;
    }
}

bool string_contains_any(const std::string& value, std::initializer_list<const char*> needles) {
    return std::any_of(needles.begin(), needles.end(), [&](const char* needle) {
        return value.find(needle) != std::string::npos;
    });
}

bool operation_is_attention_like(const OperationSpec& operation) {
    return operation.op_class == OperationClass::matmul &&
           string_contains_any(operation.name, {"qkv", "attn", "attention", "context", "head", "logits"});
}

bool operation_mentions_kv_cache(const OperationSpec& operation) {
    if (string_contains_any(operation.name, {"kv", "cache"})) {
        return true;
    }
    const auto tensor_mentions_kv = [](const std::vector<std::string>& tensor_ids) {
        return std::any_of(tensor_ids.begin(), tensor_ids.end(), [](const std::string& tensor_id) {
            return string_contains_any(tensor_id, {"kv", "cache"});
        });
    };
    return tensor_mentions_kv(operation.input_tensor_ids) ||
           tensor_mentions_kv(operation.output_tensor_ids) ||
           tensor_mentions_kv(operation.temporary_tensor_ids);
}

bool tensor_mentions_kv_cache(const WorkloadTensor& tensor) {
    return string_contains_any(tensor.id, {"kv", "cache"}) ||
           string_contains_any(tensor.alias_group, {"kv", "cache"}) ||
           string_contains_any(tensor.producer_operation, {"kv", "cache"});
}

std::string join_csv(const std::vector<std::string>& values);
const WorkloadTensor* find_workload_tensor(const WorkloadGraph& workload_graph, std::string_view tensor_id);
const WorkloadTensor* find_workload_tensor(const WorkloadGraph& workload_graph, const CompiledWorkloadGraph& compiled, std::string_view tensor_id);
const WorkloadTensor* find_workload_tensor(const WorkloadGraph& workload_graph, const CompiledWorkloadGraph& compiled, std::uint32_t tensor_index);
const HardwareGraph* find_graph(
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::string& uid);
bool is_host_graph(const HardwareGraph* graph);
bool is_host_device_uid(std::string_view uid);
std::string tuning_signature_suffix(const ExecutionTuningOverrides* tuning_overrides);
std::uint32_t clamp_cpu_tile_hint(
    std::uint32_t value,
    std::uint32_t extent,
    std::uint32_t multiple,
    std::uint32_t minimum);

std::filesystem::path performance_cache_path_for(const std::filesystem::path& cache_path) {
    if (cache_path.empty()) {
        return std::filesystem::path("jakal_core_execution_cache.tsv.perf");
    }
    auto path = cache_path;
    path += ".perf";
    return path;
}

std::filesystem::path graph_family_cache_path_for(const std::filesystem::path& cache_path) {
    if (cache_path.empty()) {
        return std::filesystem::path("jakal_core_execution_cache.tsv.perf.family");
    }
    auto path = cache_path;
    path += ".perf.family";
    return path;
}

std::filesystem::path cpu_runtime_hint_cache_path_for(const std::filesystem::path& cache_path) {
    if (cache_path.empty()) {
        return std::filesystem::path("jakal_core_execution_cache.tsv.cpuhint");
    }
    auto path = cache_path;
    path += ".cpuhint";
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
    stream << "|b" << bytes_bucket
           << "|cpuv:" << operation.cpu_vectorized
           << "|gput:" << operation.gpu_tensorized
           << "|cpu.in:" << operation.cpu_input_layout
           << "|cpu.w:" << operation.cpu_weight_layout
           << "|cpu.out:" << operation.cpu_output_layout
           << "|gpu.in:" << operation.gpu_input_layout
           << "|gpu.w:" << operation.gpu_weight_layout
           << "|gpu.out:" << operation.gpu_output_layout
           << "|cpu.pack:" << operation.cpu_pack_weights
           << "|gpu.pack:" << operation.gpu_pack_weights
           << "|cpu.preT:" << operation.cpu_pretranspose_rhs
           << "|gpu.preT:" << operation.gpu_pretranspose_rhs
           << "|cpu.u" << std::max(operation.cpu_micro_kernel_unroll, 1u)
           << "|cpu.tm" << std::max(operation.cpu_tile_m, 1u)
           << "|cpu.tn" << std::max(operation.cpu_tile_n, 1u)
           << "|cpu.tk" << std::max(operation.cpu_tile_k, 1u)
           << "|cpu.chunk" << std::max(operation.cpu_parallel_chunk, 1u)
           << "|cpu.avx512:" << operation.cpu_use_avx512
           << "|cpu.packb:" << std::max<std::uint64_t>(operation.cpu_pack_budget_bytes, 1ull)
           << "|cpu.st:" << std::max(operation.cpu_single_thread_cutoff, 1u)
           << "|cpu.kf:" << operation.cpu_kernel_family
           << "|cpu.lowp:" << operation.cpu_low_precision_kernel_family
           << "|heads:" << std::max(operation.attention_head_count, 1u)
           << "|headg:" << std::max(operation.attention_head_group_size, 1u)
           << "|tokb:" << std::max(operation.preferred_token_block, 1u)
           << "|kvres:" << operation.preferred_kv_residency
           << "|resfuse:" << operation.residency_sensitive_fusion
           << "|gpu.u" << std::max(operation.gpu_micro_kernel_unroll, 1u);
    for (const auto& fused : operation.fused_operation_names) {
        stream << "|f:" << fused;
    }
    return stream.str();
}

bool supports_feedback_tuned_cpu_chunk(const OperationSpec& operation) {
    return operation.op_class == OperationClass::elementwise_map ||
           operation.op_class == OperationClass::matmul ||
           operation.op_class == OperationClass::reduction ||
           operation.op_class == OperationClass::convolution_2d ||
           operation.op_class == OperationClass::resample_2d;
}

std::string cpu_runtime_hint_bucket_for(const OperationSpec& operation) {
    std::ostringstream stream;
    stream << to_string(operation.op_class);
    for (const auto extent : operation.extents) {
        std::uint64_t bucket = 1;
        while (bucket < extent) {
            bucket <<= 1u;
        }
        stream << ':' << bucket;
    }
    stream << "|cpuv:" << operation.cpu_vectorized
           << "|cpu.in:" << operation.cpu_input_layout
           << "|cpu.w:" << operation.cpu_weight_layout
           << "|cpu.out:" << operation.cpu_output_layout
           << "|cpu.u" << std::max(operation.cpu_micro_kernel_unroll, 1u)
           << "|cpu.kf:" << operation.cpu_kernel_family;
    for (const auto& fused : operation.fused_operation_names) {
        stream << "|f:" << fused;
    }
    return stream.str();
}

bool supports_feedback_tuned_cpu_tiles(const OperationSpec& operation) {
    return operation.op_class == OperationClass::matmul ||
           operation.op_class == OperationClass::convolution_2d ||
           operation.op_class == OperationClass::resample_2d;
}

bool supports_feedback_tuned_cpu_avx512(const OperationSpec& operation) {
    return operation.op_class == OperationClass::matmul;
}

std::uint32_t operation_linear_items(const OperationSpec& operation) {
    if (operation.extents.empty()) {
        return 1u;
    }
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(operation.extents.front(), 0xffffffffull));
}

std::uint32_t clamp_cpu_parallel_chunk_hint(
    const OperationSpec& operation,
    const std::uint32_t chunk) {
    return std::clamp(chunk, 1u, std::max(operation_linear_items(operation), 1u));
}

std::uint32_t clamp_cpu_single_thread_cutoff_hint(
    const OperationSpec& operation,
    const std::uint32_t cutoff) {
    const auto items = std::max(operation_linear_items(operation), 1u);
    return std::clamp(cutoff, 1u, std::max(items, cutoff));
}

std::uint32_t host_parallelism_hint(const std::vector<HardwareGraph>& graphs) {
    std::uint32_t best = 1u;
    for (const auto& graph : graphs) {
        if (graph.probe != "host") {
            continue;
        }
        const auto summary = summarize_graph(graph);
        const auto execution_objects = std::max(summary.execution_objects, 1u);
        best = std::max(best, execution_objects);
    }
    return best;
}

HardwareGraphSummary strongest_host_summary(const std::vector<HardwareGraph>& graphs) {
    HardwareGraphSummary best;
    double best_score = -1.0;
    for (const auto& graph : graphs) {
        if (graph.probe != "host") {
            continue;
        }
        const auto summary = summarize_graph(graph);
        const double score =
            (static_cast<double>(std::max(summary.native_vector_bits, 64u)) / 64.0) +
            (static_cast<double>(std::max(summary.execution_objects, 1u)) * 2.0) +
            (static_cast<double>(std::max<std::uint64_t>(summary.cache_bytes / kMiB, 1ull)) * 0.25) +
            (summary.supports_bf16 ? 1.0 : 0.0);
        if (score > best_score) {
            best = summary;
            best_score = score;
        }
    }
    return best;
}

std::uint32_t feedback_adjusted_cpu_parallel_chunk(
    const OperationSpec& operation,
    const std::vector<HardwareGraph>& graphs,
    const double effective_latency_us,
    const double predicted_latency_us) {
    const auto items = std::max(operation_linear_items(operation), 1u);
    const auto current = clamp_cpu_parallel_chunk_hint(
        operation,
        std::max(operation.cpu_parallel_chunk, 1u));
    const auto parallelism = std::max(host_parallelism_hint(graphs), 1u);
    const auto tasks = std::max<std::uint32_t>(1u, (items + current - 1u) / current);
    const bool oversharded = tasks > (parallelism * 4u);
    const bool undersharded = items > current && tasks < parallelism;
    const bool latency_regressed =
        predicted_latency_us > 0.0 && effective_latency_us > (predicted_latency_us * 1.10);

    auto candidate = current;
    if (oversharded && (latency_regressed || items >= (1u << 16u))) {
        candidate = clamp_cpu_parallel_chunk_hint(
            operation,
            std::max<std::uint32_t>(current * 2u, (items + (parallelism * 2u) - 1u) / (parallelism * 2u)));
    } else if (undersharded && latency_regressed) {
        candidate = clamp_cpu_parallel_chunk_hint(
            operation,
            std::max<std::uint32_t>(1u, current / 2u));
    }
    return candidate;
}

std::uint32_t clamp_tile_feedback_hint(
    const std::uint32_t value,
    const std::uint32_t extent,
    const std::uint32_t multiple,
    const std::uint32_t minimum) {
    return clamp_cpu_tile_hint(std::max(value, minimum), extent, multiple, minimum);
}

std::uint64_t matmul_tile_working_set_bytes(
    const std::uint32_t tile_m,
    const std::uint32_t tile_n,
    const std::uint32_t tile_k) {
    const std::uint64_t lhs_bytes = static_cast<std::uint64_t>(tile_m) * tile_k * sizeof(float);
    const std::uint64_t rhs_bytes = static_cast<std::uint64_t>(tile_k) * tile_n * sizeof(float);
    const std::uint64_t out_bytes = static_cast<std::uint64_t>(tile_m) * tile_n * sizeof(float);
    return lhs_bytes + rhs_bytes + out_bytes;
}

void fit_tiles_to_cache_budget(
    const std::uint64_t cache_budget_bytes,
    const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint32_t depth,
    std::uint32_t& tile_m,
    std::uint32_t& tile_n,
    std::uint32_t& tile_k) {
    if (cache_budget_bytes == 0u) {
        tile_m = clamp_cpu_tile_hint(tile_m, rows, 4u, 4u);
        tile_n = clamp_cpu_tile_hint(tile_n, columns, 8u, 8u);
        tile_k = clamp_cpu_tile_hint(tile_k, depth, 8u, 8u);
        return;
    }

    tile_m = clamp_cpu_tile_hint(tile_m, rows, 4u, 4u);
    tile_n = clamp_cpu_tile_hint(tile_n, columns, 8u, 8u);
    tile_k = clamp_cpu_tile_hint(tile_k, depth, 8u, 8u);

    while (matmul_tile_working_set_bytes(tile_m, tile_n, tile_k) > cache_budget_bytes) {
        if (tile_k >= tile_n && tile_k > 16u) {
            tile_k = std::max<std::uint32_t>(16u, tile_k - 8u);
        } else if (tile_n >= tile_m && tile_n > 16u) {
            tile_n = std::max<std::uint32_t>(16u, tile_n - 8u);
        } else if (tile_m > 4u) {
            tile_m = std::max<std::uint32_t>(4u, tile_m - 4u);
        } else {
            break;
        }
    }
}

void feedback_adjusted_cpu_tiles(
    const OperationSpec& operation,
    const std::vector<HardwareGraph>& graphs,
    const double effective_latency_us,
    const double predicted_latency_us,
    std::uint32_t& tile_m,
    std::uint32_t& tile_n,
    std::uint32_t& tile_k) {
    if (operation.op_class == OperationClass::convolution_2d ||
        operation.op_class == OperationClass::resample_2d) {
        const auto summary = strongest_host_summary(graphs);
        const bool wide_vectors = summary.native_vector_bits >= 512u;
        const bool latency_regressed =
            predicted_latency_us > 0.0 && effective_latency_us > (predicted_latency_us * 1.08);
        const bool latency_improved =
            predicted_latency_us > 0.0 && effective_latency_us < (predicted_latency_us * 0.92);
        const auto rows = static_cast<std::uint32_t>(
            operation.extents.size() > (operation.op_class == OperationClass::resample_2d ? 2u : 0u)
                ? operation.extents[operation.op_class == OperationClass::resample_2d ? 2u : 0u]
                : 1u);
        const auto columns = static_cast<std::uint32_t>(
            operation.extents.size() > (operation.op_class == OperationClass::resample_2d ? 3u : 1u)
                ? operation.extents[operation.op_class == OperationClass::resample_2d ? 3u : 1u]
                : 1u);
        tile_m = std::max(tile_m, 2u);
        tile_n = std::max(tile_n, 4u);
        tile_k = std::max(tile_k, operation.op_class == OperationClass::convolution_2d ? 9u : 6u);
        if (latency_regressed) {
            tile_m += wide_vectors ? 4u : 2u;
            if (columns >= 64u) {
                tile_n += wide_vectors ? 8u : 4u;
            }
        } else if (latency_improved) {
            if (rows < 32u) {
                tile_m = std::max<std::uint32_t>(2u, tile_m - 2u);
            }
            if (columns < 64u) {
                tile_n = std::max<std::uint32_t>(4u, tile_n - 4u);
            }
        }
        tile_m = clamp_tile_feedback_hint(tile_m, rows, 1u, 2u);
        tile_n = clamp_tile_feedback_hint(tile_n, columns, 1u, 4u);
        tile_k = operation.op_class == OperationClass::convolution_2d ? 9u : 6u;
        return;
    }
    const auto summary = strongest_host_summary(graphs);
    const bool wide_vectors = summary.native_vector_bits >= 512u;
    const bool large_cache = summary.cache_bytes >= (2ull * kMiB);
    const bool tight_cache = summary.cache_bytes > 0u && summary.cache_bytes < (512ull * kKiB);
    const auto parallelism = std::max(host_parallelism_hint(graphs), 1u);
    const bool latency_regressed =
        predicted_latency_us > 0.0 && effective_latency_us > (predicted_latency_us * 1.08);
    const bool latency_improved =
        predicted_latency_us > 0.0 && effective_latency_us < (predicted_latency_us * 0.92);

    const auto rows = static_cast<std::uint32_t>(operation.extents.size() > 0u ? operation.extents[0] : 1u);
    const auto columns = static_cast<std::uint32_t>(operation.extents.size() > 1u ? operation.extents[1] : 1u);
    const auto depth = static_cast<std::uint32_t>(operation.extents.size() > 2u ? operation.extents[2] : 1u);
    const bool column_heavy = columns >= (rows * 2u);
    const bool row_heavy = rows >= (columns * 2u);
    const bool depth_heavy = depth >= (std::max(rows, columns) * 2u);

    tile_m = std::max(tile_m, 4u);
    tile_n = std::max(tile_n, 8u);
    tile_k = std::max(tile_k, 8u);

    if (latency_regressed) {
        if ((rows >= 192u || row_heavy) && (large_cache || wide_vectors)) {
            tile_m += wide_vectors ? 8u : 4u;
        } else if (rows <= 32u || column_heavy) {
            tile_m = std::max<std::uint32_t>(4u, tile_m - 4u);
        }

        if ((columns >= 192u || column_heavy) && large_cache) {
            tile_n += wide_vectors ? 16u : 8u;
        } else if (columns <= 64u || row_heavy) {
            tile_n = std::max<std::uint32_t>(16u, tile_n - 8u);
        }

        if ((depth >= 192u || depth_heavy) && (large_cache || operation.cpu_micro_kernel_unroll >= 4u)) {
            tile_k += wide_vectors ? 32u : 16u;
        } else if (depth <= 64u || tight_cache) {
            tile_k = std::max<std::uint32_t>(16u, tile_k - 8u);
        }
    } else if (latency_improved) {
        if (rows < 64u) {
            tile_m = std::max<std::uint32_t>(4u, tile_m - 4u);
        }
        if (columns < 96u) {
            tile_n = std::max<std::uint32_t>(16u, tile_n - 8u);
        }
        if (depth < 96u) {
            tile_k = std::max<std::uint32_t>(16u, tile_k - 8u);
        }
    }

    if (parallelism >= 12u && rows >= 256u && tile_m < 64u) {
        tile_m += 8u;
    }
    const auto cache_budget = summary.cache_bytes == 0u
        ? 256ull * kKiB
        : std::max<std::uint64_t>(128ull * kKiB, summary.cache_bytes / (large_cache ? 2ull : 3ull));
    fit_tiles_to_cache_budget(cache_budget, rows, columns, depth, tile_m, tile_n, tile_k);
    tile_m = clamp_tile_feedback_hint(tile_m, rows, 4u, 4u);
    tile_n = clamp_tile_feedback_hint(tile_n, columns, 8u, 8u);
    tile_k = clamp_tile_feedback_hint(tile_k, depth, 8u, 8u);
}

bool feedback_adjusted_cpu_avx512_choice(
    const OperationSpec& operation,
    const std::vector<HardwareGraph>& graphs,
    const double effective_latency_us,
    const double predicted_latency_us) {
    const auto summary = strongest_host_summary(graphs);
    if (summary.native_vector_bits < 512u || !executors::host_native_kernels_compiled_with_avx512()) {
        return false;
    }
    const auto rows = static_cast<std::uint32_t>(operation.extents.size() > 0u ? operation.extents[0] : 1u);
    const auto columns = static_cast<std::uint32_t>(operation.extents.size() > 1u ? operation.extents[1] : 1u);
    const auto depth = static_cast<std::uint32_t>(operation.extents.size() > 2u ? operation.extents[2] : 1u);
    const std::uint64_t work_items =
        static_cast<std::uint64_t>(rows) * columns * depth;
    const bool small_problem =
        rows < 64u || columns < 64u || depth < 64u || work_items < (1ull << 20u);
    const bool medium_problem =
        rows >= 128u && columns >= 128u && depth >= 128u && work_items >= (1ull << 22u);
    const bool latency_regressed =
        predicted_latency_us > 0.0 && effective_latency_us > (predicted_latency_us * 1.08);

    if (operation.cpu_use_avx512) {
        if ((small_problem || work_items < (1ull << 22u)) && latency_regressed) {
            return false;
        }
        return true;
    }
    if (medium_problem && latency_regressed) {
        return true;
    }
    return operation.cpu_use_avx512;
}

template <typename T>
bool append_unique(std::vector<T>& values, const T& value) {
    if (std::find(values.begin(), values.end(), value) != values.end()) {
        return false;
    }
    values.push_back(value);
    return true;
}

HardwareGraphSummary strongest_host_summary(
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    HardwareGraphSummary best;
    double best_score = -1.0;
    for (const auto& allocation : placement.allocations) {
        const auto* graph = find_graph(graph_lookup, allocation.device.uid);
        if (!is_host_graph(graph)) {
            continue;
        }
        const auto summary = summarize_graph(*graph);
        const double score =
            (static_cast<double>(std::max(summary.native_vector_bits, 64u)) / 64.0) +
            (static_cast<double>(std::max(summary.execution_objects, 1u)) *
             static_cast<double>(std::max(summary.lanes_per_object, 1u)) / 16.0) +
            (static_cast<double>(std::max(summary.cache_bytes, 1ull)) / (512.0 * 1024.0));
        if (score > best_score) {
            best = summary;
            best_score = score;
        }
    }
    return best;
}

HardwareGraphSummary strongest_accelerator_summary(
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    HardwareGraphSummary best;
    double best_score = -1.0;
    for (const auto& allocation : placement.allocations) {
        const auto* graph = find_graph(graph_lookup, allocation.device.uid);
        if (graph == nullptr || is_host_graph(graph)) {
            continue;
        }
        const auto summary = summarize_graph(*graph);
        const double score =
            (static_cast<double>(std::max(summary.matrix_units, 1u)) * 4.0) +
            (static_cast<double>(std::max(summary.execution_objects, 1u)) *
             static_cast<double>(std::max(summary.lanes_per_object, 1u)) / 32.0) +
            (static_cast<double>(std::max(summary.local_scratch_bytes, 1ull)) / (64.0 * 1024.0)) +
            (static_cast<double>(std::max(summary.cache_bytes, 1ull)) / (1024.0 * 1024.0));
        if (score > best_score) {
            best = summary;
            best_score = score;
        }
    }
    return best;
}

WorkloadTensor* find_workload_tensor_mutable(WorkloadGraph& workload_graph, const std::string_view tensor_id) {
    const auto it = std::find_if(
        workload_graph.tensors.begin(),
        workload_graph.tensors.end(),
        [&](const WorkloadTensor& tensor) { return tensor.id == tensor_id; });
    return it == workload_graph.tensors.end() ? nullptr : &(*it);
}

OperationSpec* find_workload_operation_mutable(WorkloadGraph& workload_graph, const std::string_view operation_name) {
    const auto it = std::find_if(
        workload_graph.operations.begin(),
        workload_graph.operations.end(),
        [&](const OperationSpec& operation) { return operation.name == operation_name; });
    return it == workload_graph.operations.end() ? nullptr : &(*it);
}

std::uint64_t tensor_bytes_of(const WorkloadGraph& graph, const std::string& tensor_id) {
    const auto* tensor = find_workload_tensor(graph, tensor_id);
    return tensor == nullptr ? 0ull : tensor->bytes;
}

double operation_working_set_bytes(const OperationSpec& operation);
bool should_prefer_host_for_llm_operation(const WorkloadSpec& workload, const OperationSpec& operation);
bool should_prefer_accelerator_for_llm_operation(const WorkloadSpec& workload, const OperationSpec& operation);
bool should_prefer_host_for_phase_operation(const WorkloadSpec& workload, const OperationSpec& operation);
bool should_prefer_accelerator_for_phase_operation(const WorkloadSpec& workload, const OperationSpec& operation);
std::uint32_t choose_attention_head_count_hint(const WorkloadSpec& workload, const OperationSpec& operation);
std::uint32_t choose_attention_head_group_size_hint(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const HardwareGraphSummary& host_summary);
std::uint32_t choose_token_block_hint(const WorkloadSpec& workload, const OperationSpec& operation);
std::string choose_kv_residency_hint(const WorkloadSpec& workload, const OperationSpec& operation);
std::uint32_t desired_semantic_partition_count(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const ExecutionConfig& config);
std::string build_report_signature(
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::string& compiled_workload_signature,
    const std::vector<OperationSpec>& operations,
    const ExecutionTuningOverrides* tuning_overrides);

bool placement_has_host_and_accelerator(
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    bool has_host = false;
    bool has_accelerator = false;
    for (const auto& allocation : placement.allocations) {
        const auto it = graph_lookup.find(allocation.device.uid);
        if (it == graph_lookup.end() || it->second == nullptr) {
            continue;
        }
        has_host = has_host || it->second->probe == "host";
        has_accelerator = has_accelerator || it->second->probe != "host";
        if (has_host && has_accelerator) {
            return true;
        }
    }
    return false;
}

bool should_preserve_host_preferred_tail_boundary(
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const OperationSpec& producer,
    const OperationSpec& consumer) {
    if (!placement_has_host_and_accelerator(placement, graph_lookup)) {
        return false;
    }

    const auto phase = canonical_workload_phase(workload);
    if (phase != WorkloadPhase::decode && phase != WorkloadPhase::prefill) {
        return false;
    }

    const bool producer_accelerator_preferred =
        should_prefer_accelerator_for_llm_operation(workload, producer) ||
        should_prefer_accelerator_for_phase_operation(workload, producer);
    const bool consumer_host_preferred =
        should_prefer_host_for_llm_operation(workload, consumer) ||
        should_prefer_host_for_phase_operation(workload, consumer);
    if (!producer_accelerator_preferred || !consumer_host_preferred) {
        return false;
    }

    const bool producer_is_compute_bulk =
        producer.op_class == OperationClass::matmul ||
        producer.op_class == OperationClass::convolution_2d ||
        producer.op_class == OperationClass::resample_2d;
    const bool consumer_is_tail =
        consumer.op_class == OperationClass::elementwise_map ||
        consumer.op_class == OperationClass::reduction ||
        consumer.name.find("sample") != std::string::npos ||
        consumer.name.find("kv") != std::string::npos ||
        consumer.name.find("cache") != std::string::npos;
    return producer_is_compute_bulk && consumer_is_tail;
}

bool should_block_residency_sensitive_fusion(
    const WorkloadGraph& graph,
    const OperationSpec& producer,
    const OperationSpec& consumer) {
    if (!producer.residency_sensitive_fusion && !consumer.residency_sensitive_fusion) {
        return false;
    }
    if (producer.output_tensor_ids.empty()) {
        return false;
    }
    const auto* intermediate = find_workload_tensor(graph, producer.output_tensor_ids.front());
    if (intermediate == nullptr) {
        return false;
    }

    const bool kv_intermediate = tensor_mentions_kv_cache(*intermediate);
    const bool residency_boundary =
        producer.preferred_kv_residency != consumer.preferred_kv_residency &&
        producer.preferred_kv_residency != "auto" &&
        consumer.preferred_kv_residency != "auto";
    const bool pinned_for_host = intermediate->host_visible || intermediate->persistent;
    return kv_intermediate || residency_boundary || pinned_for_host;
}

bool can_fuse_epilogue_elementwise(
    const WorkloadGraph& graph,
    const OperationSpec& producer,
    const OperationSpec& consumer) {
    if ((producer.op_class != OperationClass::matmul &&
         producer.op_class != OperationClass::convolution_2d &&
         producer.op_class != OperationClass::resample_2d) ||
        consumer.op_class != OperationClass::elementwise_map ||
        producer.output_tensor_ids.size() != 1u ||
        consumer.output_tensor_ids.empty()) {
        return false;
    }

    const auto& intermediate_tensor = producer.output_tensor_ids.front();
    if (std::find(consumer.input_tensor_ids.begin(), consumer.input_tensor_ids.end(), intermediate_tensor) ==
        consumer.input_tensor_ids.end()) {
        return false;
    }

    std::size_t consumer_count = 0;
    for (const auto& operation : graph.operations) {
        if (operation.name == producer.name) {
            continue;
        }
        if (std::find(operation.input_tensor_ids.begin(), operation.input_tensor_ids.end(), intermediate_tensor) !=
            operation.input_tensor_ids.end()) {
            ++consumer_count;
            if (operation.name != consumer.name) {
                return false;
            }
        }
    }
    return consumer_count == 1u;
}

bool fuse_epilogue_elementwise_into_producer(
    WorkloadGraph& graph,
    const std::string& producer_name,
    const std::string& consumer_name) {
    auto* producer = find_workload_operation_mutable(graph, producer_name);
    auto* consumer = find_workload_operation_mutable(graph, consumer_name);
    if (producer == nullptr || consumer == nullptr || !can_fuse_epilogue_elementwise(graph, *producer, *consumer)) {
        return false;
    }

    const auto intermediate_tensor = producer->output_tensor_ids.front();
    for (const auto& tensor_id : consumer->input_tensor_ids) {
        if (tensor_id == intermediate_tensor) {
            continue;
        }
        if (append_unique(producer->input_tensor_ids, tensor_id)) {
            producer->input_bytes += tensor_bytes_of(graph, tensor_id);
        }
        if (auto* tensor = find_workload_tensor_mutable(graph, tensor_id)) {
            tensor->consumer_operations.erase(
                std::remove(tensor->consumer_operations.begin(), tensor->consumer_operations.end(), consumer_name),
                tensor->consumer_operations.end());
            append_unique(tensor->consumer_operations, producer_name);
        }
    }

    producer->output_tensor_ids = consumer->output_tensor_ids;
    producer->temporary_tensor_ids.insert(
        producer->temporary_tensor_ids.end(),
        consumer->temporary_tensor_ids.begin(),
        consumer->temporary_tensor_ids.end());
    producer->estimated_flops += consumer->estimated_flops;
    producer->temporary_bytes += consumer->temporary_bytes;
    producer->output_bytes = std::max(producer->output_bytes, consumer->output_bytes);
    producer->max_relative_error = std::max(producer->max_relative_error, consumer->max_relative_error);
    producer->streaming_friendly = producer->streaming_friendly || consumer->streaming_friendly;
    append_unique(producer->fused_operation_names, consumer_name);
    for (const auto& fused : consumer->fused_operation_names) {
        append_unique(producer->fused_operation_names, fused);
    }

    for (const auto& output_tensor_id : consumer->output_tensor_ids) {
        if (auto* tensor = find_workload_tensor_mutable(graph, output_tensor_id)) {
            tensor->producer_operation = producer_name;
        }
    }

    if (auto* tensor = find_workload_tensor_mutable(graph, intermediate_tensor)) {
        tensor->consumer_operations.clear();
    }
    graph.tensors.erase(
        std::remove_if(
            graph.tensors.begin(),
            graph.tensors.end(),
            [&](const WorkloadTensor& tensor) {
                return tensor.id == intermediate_tensor && !tensor.persistent && tensor.consumer_operations.empty();
            }),
        graph.tensors.end());
    graph.operations.erase(
        std::remove_if(
            graph.operations.begin(),
            graph.operations.end(),
            [&](const OperationSpec& operation) { return operation.name == consumer_name; }),
        graph.operations.end());
    return true;
}

bool can_fuse_elementwise_chain(
    const WorkloadGraph& graph,
    const OperationSpec& producer,
    const OperationSpec& consumer) {
    if (producer.op_class != OperationClass::elementwise_map ||
        consumer.op_class != OperationClass::elementwise_map ||
        producer.output_tensor_ids.size() != 1u ||
        consumer.output_tensor_ids.empty()) {
        return false;
    }

    const auto& intermediate_tensor = producer.output_tensor_ids.front();
    if (std::find(consumer.input_tensor_ids.begin(), consumer.input_tensor_ids.end(), intermediate_tensor) ==
        consumer.input_tensor_ids.end()) {
        return false;
    }

    std::size_t consumer_count = 0u;
    for (const auto& operation : graph.operations) {
        if (operation.name == producer.name) {
            continue;
        }
        if (std::find(operation.input_tensor_ids.begin(), operation.input_tensor_ids.end(), intermediate_tensor) !=
            operation.input_tensor_ids.end()) {
            ++consumer_count;
            if (operation.name != consumer.name) {
                return false;
            }
        }
    }
    return consumer_count == 1u;
}

bool fuse_elementwise_chain(
    WorkloadGraph& graph,
    const std::string& producer_name,
    const std::string& consumer_name) {
    auto* producer = find_workload_operation_mutable(graph, producer_name);
    auto* consumer = find_workload_operation_mutable(graph, consumer_name);
    if (producer == nullptr || consumer == nullptr || !can_fuse_elementwise_chain(graph, *producer, *consumer)) {
        return false;
    }

    const auto intermediate_tensor = producer->output_tensor_ids.front();
    for (const auto& tensor_id : consumer->input_tensor_ids) {
        if (tensor_id == intermediate_tensor) {
            continue;
        }
        if (append_unique(producer->input_tensor_ids, tensor_id)) {
            producer->input_bytes += tensor_bytes_of(graph, tensor_id);
        }
        if (auto* tensor = find_workload_tensor_mutable(graph, tensor_id)) {
            tensor->consumer_operations.erase(
                std::remove(tensor->consumer_operations.begin(), tensor->consumer_operations.end(), consumer_name),
                tensor->consumer_operations.end());
            append_unique(tensor->consumer_operations, producer_name);
        }
    }

    producer->output_tensor_ids = consumer->output_tensor_ids;
    producer->temporary_tensor_ids.insert(
        producer->temporary_tensor_ids.end(),
        consumer->temporary_tensor_ids.begin(),
        consumer->temporary_tensor_ids.end());
    producer->estimated_flops += consumer->estimated_flops;
    producer->temporary_bytes += consumer->temporary_bytes;
    producer->output_bytes = std::max(producer->output_bytes, consumer->output_bytes);
    producer->max_relative_error = std::max(producer->max_relative_error, consumer->max_relative_error);
    producer->streaming_friendly = producer->streaming_friendly || consumer->streaming_friendly;
    append_unique(producer->fused_operation_names, consumer_name);
    for (const auto& fused : consumer->fused_operation_names) {
        append_unique(producer->fused_operation_names, fused);
    }

    for (const auto& output_tensor_id : consumer->output_tensor_ids) {
        if (auto* tensor = find_workload_tensor_mutable(graph, output_tensor_id)) {
            tensor->producer_operation = producer_name;
        }
    }

    if (auto* tensor = find_workload_tensor_mutable(graph, intermediate_tensor)) {
        tensor->consumer_operations.clear();
    }
    graph.tensors.erase(
        std::remove_if(
            graph.tensors.begin(),
            graph.tensors.end(),
            [&](const WorkloadTensor& tensor) {
                return tensor.id == intermediate_tensor && !tensor.persistent && tensor.consumer_operations.empty();
            }),
        graph.tensors.end());
    graph.operations.erase(
        std::remove_if(
            graph.operations.begin(),
            graph.operations.end(),
            [&](const OperationSpec& operation) { return operation.name == consumer_name; }),
        graph.operations.end());
    return true;
}

bool can_fuse_elementwise_into_reduction(
    const WorkloadGraph& graph,
    const OperationSpec& producer,
    const OperationSpec& consumer) {
    if (producer.op_class != OperationClass::elementwise_map ||
        consumer.op_class != OperationClass::reduction ||
        producer.output_tensor_ids.size() != 1u ||
        consumer.output_tensor_ids.empty()) {
        return false;
    }

    const auto& intermediate_tensor = producer.output_tensor_ids.front();
    if (std::find(consumer.input_tensor_ids.begin(), consumer.input_tensor_ids.end(), intermediate_tensor) ==
        consumer.input_tensor_ids.end()) {
        return false;
    }

    std::size_t consumer_count = 0u;
    for (const auto& operation : graph.operations) {
        if (operation.name == producer.name) {
            continue;
        }
        if (std::find(operation.input_tensor_ids.begin(), operation.input_tensor_ids.end(), intermediate_tensor) !=
            operation.input_tensor_ids.end()) {
            ++consumer_count;
            if (operation.name != consumer.name) {
                return false;
            }
        }
    }
    return consumer_count == 1u;
}

bool fuse_elementwise_into_reduction(
    WorkloadGraph& graph,
    const std::string& producer_name,
    const std::string& consumer_name) {
    auto* producer = find_workload_operation_mutable(graph, producer_name);
    auto* consumer = find_workload_operation_mutable(graph, consumer_name);
    if (producer == nullptr || consumer == nullptr || !can_fuse_elementwise_into_reduction(graph, *producer, *consumer)) {
        return false;
    }

    const auto intermediate_tensor = producer->output_tensor_ids.front();
    for (const auto& tensor_id : producer->input_tensor_ids) {
        if (append_unique(consumer->input_tensor_ids, tensor_id)) {
            consumer->input_bytes += tensor_bytes_of(graph, tensor_id);
        }
        if (auto* tensor = find_workload_tensor_mutable(graph, tensor_id)) {
            tensor->consumer_operations.erase(
                std::remove(tensor->consumer_operations.begin(), tensor->consumer_operations.end(), producer_name),
                tensor->consumer_operations.end());
            append_unique(tensor->consumer_operations, consumer_name);
        }
    }

    consumer->temporary_tensor_ids.insert(
        consumer->temporary_tensor_ids.end(),
        producer->temporary_tensor_ids.begin(),
        producer->temporary_tensor_ids.end());
    consumer->estimated_flops += producer->estimated_flops;
    consumer->temporary_bytes += producer->temporary_bytes + (producer->output_bytes / 16u);
    consumer->input_bytes = std::max(consumer->input_bytes, producer->input_bytes);
    consumer->max_relative_error = std::max(consumer->max_relative_error, producer->max_relative_error);
    consumer->streaming_friendly = consumer->streaming_friendly || producer->streaming_friendly;
    append_unique(consumer->fused_operation_names, producer_name);
    for (const auto& fused : producer->fused_operation_names) {
        append_unique(consumer->fused_operation_names, fused);
    }
    for (const auto& dependency_name : producer->dependency_operation_names) {
        append_unique(consumer->dependency_operation_names, dependency_name);
    }

    if (auto* tensor = find_workload_tensor_mutable(graph, intermediate_tensor)) {
        tensor->consumer_operations.clear();
    }
    graph.tensors.erase(
        std::remove_if(
            graph.tensors.begin(),
            graph.tensors.end(),
            [&](const WorkloadTensor& tensor) {
                return tensor.id == intermediate_tensor && !tensor.persistent && tensor.consumer_operations.empty();
            }),
        graph.tensors.end());
    graph.operations.erase(
        std::remove_if(
            graph.operations.begin(),
            graph.operations.end(),
            [&](const OperationSpec& operation) { return operation.name == producer_name; }),
        graph.operations.end());
    return true;
}

bool can_fuse_reduction_epilogue_elementwise(
    const WorkloadGraph& graph,
    const OperationSpec& producer,
    const OperationSpec& consumer) {
    if (producer.op_class != OperationClass::reduction ||
        consumer.op_class != OperationClass::elementwise_map ||
        producer.output_tensor_ids.size() != 1u ||
        consumer.output_tensor_ids.empty()) {
        return false;
    }

    const auto& intermediate_tensor = producer.output_tensor_ids.front();
    if (std::find(consumer.input_tensor_ids.begin(), consumer.input_tensor_ids.end(), intermediate_tensor) ==
        consumer.input_tensor_ids.end()) {
        return false;
    }

    std::size_t consumer_count = 0u;
    for (const auto& operation : graph.operations) {
        if (operation.name == producer.name) {
            continue;
        }
        if (std::find(operation.input_tensor_ids.begin(), operation.input_tensor_ids.end(), intermediate_tensor) !=
            operation.input_tensor_ids.end()) {
            ++consumer_count;
            if (operation.name != consumer.name) {
                return false;
            }
        }
    }
    return consumer_count == 1u;
}

bool fuse_reduction_epilogue_elementwise(
    WorkloadGraph& graph,
    const std::string& producer_name,
    const std::string& consumer_name) {
    auto* producer = find_workload_operation_mutable(graph, producer_name);
    auto* consumer = find_workload_operation_mutable(graph, consumer_name);
    if (producer == nullptr || consumer == nullptr ||
        !can_fuse_reduction_epilogue_elementwise(graph, *producer, *consumer)) {
        return false;
    }

    const auto intermediate_tensor = producer->output_tensor_ids.front();
    for (const auto& tensor_id : consumer->input_tensor_ids) {
        if (tensor_id == intermediate_tensor) {
            continue;
        }
        if (append_unique(producer->input_tensor_ids, tensor_id)) {
            producer->input_bytes += tensor_bytes_of(graph, tensor_id);
        }
        if (auto* tensor = find_workload_tensor_mutable(graph, tensor_id)) {
            tensor->consumer_operations.erase(
                std::remove(tensor->consumer_operations.begin(), tensor->consumer_operations.end(), consumer_name),
                tensor->consumer_operations.end());
            append_unique(tensor->consumer_operations, producer_name);
        }
    }

    producer->output_tensor_ids = consumer->output_tensor_ids;
    producer->temporary_tensor_ids.insert(
        producer->temporary_tensor_ids.end(),
        consumer->temporary_tensor_ids.begin(),
        consumer->temporary_tensor_ids.end());
    producer->estimated_flops += consumer->estimated_flops;
    producer->temporary_bytes += consumer->temporary_bytes;
    producer->output_bytes = std::max(producer->output_bytes, consumer->output_bytes);
    producer->max_relative_error = std::max(producer->max_relative_error, consumer->max_relative_error);
    producer->streaming_friendly = producer->streaming_friendly || consumer->streaming_friendly;
    append_unique(producer->fused_operation_names, consumer_name);
    for (const auto& fused : consumer->fused_operation_names) {
        append_unique(producer->fused_operation_names, fused);
    }

    for (const auto& output_tensor_id : consumer->output_tensor_ids) {
        if (auto* tensor = find_workload_tensor_mutable(graph, output_tensor_id)) {
            tensor->producer_operation = producer_name;
        }
    }

    if (auto* tensor = find_workload_tensor_mutable(graph, intermediate_tensor)) {
        tensor->consumer_operations.clear();
    }
    graph.tensors.erase(
        std::remove_if(
            graph.tensors.begin(),
            graph.tensors.end(),
            [&](const WorkloadTensor& tensor) {
                return tensor.id == intermediate_tensor && !tensor.persistent && tensor.consumer_operations.empty();
            }),
        graph.tensors.end());
    graph.operations.erase(
        std::remove_if(
            graph.operations.begin(),
            graph.operations.end(),
            [&](const OperationSpec& operation) { return operation.name == consumer_name; }),
        graph.operations.end());
    return true;
}

bool fuse_producer_elementwise_chain(
    WorkloadGraph& graph,
    const std::string& producer_name,
    const std::string& first_consumer_name,
    const std::string& second_consumer_name) {
    if (!fuse_epilogue_elementwise_into_producer(graph, producer_name, first_consumer_name)) {
        return false;
    }
    return fuse_epilogue_elementwise_into_producer(graph, producer_name, second_consumer_name);
}

bool fuse_elementwise_chain_into_reduction(
    WorkloadGraph& graph,
    const std::string& producer_name,
    const std::string& intermediate_name,
    const std::string& reduction_name) {
    if (!fuse_elementwise_chain(graph, producer_name, intermediate_name)) {
        return false;
    }
    return fuse_elementwise_into_reduction(graph, producer_name, reduction_name);
}

std::uint32_t align_up_u32(const std::uint32_t value, const std::uint32_t multiple) {
    if (multiple == 0u) {
        return value;
    }
    return ((std::max(value, 1u) + multiple - 1u) / multiple) * multiple;
}

std::uint32_t align_down_u32(const std::uint32_t value, const std::uint32_t multiple) {
    if (multiple == 0u) {
        return value;
    }
    return std::max(multiple, (std::max(value, multiple) / multiple) * multiple);
}

std::uint32_t clamp_cpu_tile_hint(
    const std::uint32_t value,
    const std::uint32_t extent,
    const std::uint32_t multiple,
    const std::uint32_t minimum) {
    const auto aligned_extent = align_up_u32(std::max(extent, 1u), multiple);
    return std::clamp(align_up_u32(std::max(value, minimum), multiple), minimum, aligned_extent);
}

std::uint32_t choose_cpu_parallel_chunk_hint(
    const OperationSpec& operation,
    const HardwareGraphSummary& host_summary) {
    const auto base_parallelism =
        std::max<std::uint32_t>(1u, std::max(host_summary.execution_objects, 1u));
    const auto vector_bonus = host_summary.native_vector_bits >= 512u ? 2u : 1u;
    switch (operation.op_class) {
    case OperationClass::matmul: {
        const auto rows = static_cast<std::uint32_t>(operation.extents.size() > 0u ? operation.extents[0] : 1u);
        std::uint32_t chunk = rows >= 256u ? 8u : (rows >= 64u ? 4u : (rows >= 16u ? 2u : 1u));
        if (base_parallelism >= 12u) {
            chunk *= 2u;
        }
        return std::min<std::uint32_t>(std::max(chunk, 1u), std::max(rows, 1u));
    }
    case OperationClass::convolution_2d: {
        const auto out_rows =
            static_cast<std::uint32_t>(operation.extents.size() > 0u ? std::max<std::uint64_t>(operation.extents[0], 3ull) - 2ull : 1ull);
        std::uint32_t chunk = out_rows >= 96u ? 8u : (out_rows >= 32u ? 4u : (out_rows >= 12u ? 2u : 1u));
        if (base_parallelism >= 12u) {
            chunk *= 2u;
        }
        return std::min<std::uint32_t>(std::max(chunk, 1u), std::max(out_rows, 1u));
    }
    case OperationClass::resample_2d: {
        const auto rows =
            static_cast<std::uint32_t>(operation.extents.size() > 2u ? std::max<std::uint64_t>(operation.extents[2], 1ull) : 1ull);
        std::uint32_t chunk = rows >= 128u ? 8u : (rows >= 48u ? 4u : (rows >= 16u ? 2u : 1u));
        if (base_parallelism >= 12u) {
            chunk *= 2u;
        }
        return std::min<std::uint32_t>(std::max(chunk, 1u), std::max(rows, 1u));
    }
    case OperationClass::reduction: {
        const auto elements = static_cast<std::uint32_t>(
            operation.extents.empty() ? 1ull : std::min<std::uint64_t>(operation.extents[0], 1ull << 30u));
        std::uint32_t chunk = elements >= (1u << 20u) ? 8192u : (elements >= (1u << 16u) ? 4096u : 2048u);
        chunk *= vector_bonus;
        return chunk;
    }
    case OperationClass::elementwise_map:
    default: {
        const auto elements = static_cast<std::uint32_t>(
            operation.extents.empty() ? 1ull : std::min<std::uint64_t>(operation.extents[0], 1ull << 30u));
        std::uint32_t chunk = elements >= (1u << 20u) ? 2048u : (elements >= (1u << 16u) ? 1024u : 512u);
        chunk *= vector_bonus;
        return chunk;
    }
    }
}

void apply_cpu_runtime_tuning_hints(
    OperationSpec& operation,
    const HardwareGraphSummary& host_summary) {
    operation.cpu_parallel_chunk = choose_cpu_parallel_chunk_hint(operation, host_summary);
    operation.cpu_pack_budget_bytes = 0u;
    operation.cpu_single_thread_cutoff = 0u;
    operation.cpu_kernel_family = "auto";
    operation.cpu_use_avx512 = false;
    operation.cpu_low_precision_kernel_family = "auto";
    if (operation.op_class == OperationClass::convolution_2d) {
        const auto height = static_cast<std::uint32_t>(operation.extents.size() > 0u ? operation.extents[0] : 1u);
        const auto width = static_cast<std::uint32_t>(operation.extents.size() > 1u ? operation.extents[1] : 1u);
        const auto out_height = height > 2u ? height - 2u : 1u;
        const auto out_width = width > 2u ? width - 2u : 1u;
        const auto vector_scale = host_summary.native_vector_bits >= 512u ? 2u : 1u;
        operation.cpu_tile_m = clamp_cpu_tile_hint(std::max<std::uint32_t>(8u, 16u * vector_scale), out_height, 1u, 2u);
        operation.cpu_tile_n = clamp_cpu_tile_hint(std::max<std::uint32_t>(16u, 32u * vector_scale), out_width, 1u, 4u);
        operation.cpu_tile_k = 9u;
        operation.cpu_parallel_chunk =
            clamp_cpu_parallel_chunk_hint(operation, std::max(operation.cpu_parallel_chunk, operation.cpu_tile_m));
        const std::uint64_t work_items = static_cast<std::uint64_t>(out_height) * out_width * 9ull;
        operation.cpu_single_thread_cutoff =
            work_items >= (1ull << 20u) ? (1u << 16u) : (work_items >= (1ull << 17u) ? (1u << 14u) : (1u << 12u));
        operation.cpu_kernel_family =
            operation.cpu_input_layout.find("patch9") != std::string::npos ? "conv-avx2-patch9" : "conv-avx2-dense";
        return;
    }
    if (operation.op_class == OperationClass::resample_2d) {
        const auto dst_h = static_cast<std::uint32_t>(operation.extents.size() > 2u ? operation.extents[2] : 1u);
        const auto dst_w = static_cast<std::uint32_t>(operation.extents.size() > 3u ? operation.extents[3] : 1u);
        const auto vector_scale = host_summary.native_vector_bits >= 512u ? 2u : 1u;
        operation.cpu_tile_m = clamp_cpu_tile_hint(std::max<std::uint32_t>(8u, 12u * vector_scale), dst_h, 1u, 2u);
        operation.cpu_tile_n = clamp_cpu_tile_hint(std::max<std::uint32_t>(16u, 24u * vector_scale), dst_w, 1u, 4u);
        operation.cpu_tile_k = 6u;
        operation.cpu_parallel_chunk =
            clamp_cpu_parallel_chunk_hint(operation, std::max(operation.cpu_parallel_chunk, operation.cpu_tile_m));
        const std::uint64_t work_items = static_cast<std::uint64_t>(dst_h) * dst_w * 4ull;
        operation.cpu_single_thread_cutoff =
            work_items >= (1ull << 20u) ? (1u << 16u) : (work_items >= (1ull << 17u) ? (1u << 14u) : (1u << 12u));
        operation.cpu_kernel_family =
            operation.cpu_input_layout.find("packed6") != std::string::npos ? "resample-avx2-packed6" : "resample-avx2-dense";
        return;
    }
    if (operation.op_class == OperationClass::elementwise_map || operation.op_class == OperationClass::reduction) {
        const bool attention_tail =
            operation.attention_head_count > 0u ||
            operation.preferred_kv_residency == "host" ||
            operation.preferred_kv_residency == "shared" ||
            operation.residency_sensitive_fusion;
        if (attention_tail) {
            operation.cpu_kernel_family =
                operation.op_class == OperationClass::reduction ? "attention-tail-reduce" : "attention-tail-simd";
            operation.cpu_single_thread_cutoff =
                operation.op_class == OperationClass::reduction ? (1u << 13u) : (1u << 12u);
            operation.cpu_parallel_chunk = clamp_cpu_parallel_chunk_hint(
                operation,
                std::max<std::uint32_t>(
                    operation.cpu_parallel_chunk,
                    operation.preferred_token_block > 0u ? operation.preferred_token_block : 256u));
        }
        return;
    }
    if (operation.op_class != OperationClass::matmul) {
        return;
    }

    const auto rows = static_cast<std::uint32_t>(operation.extents.size() > 0u ? operation.extents[0] : 1u);
    const auto columns = static_cast<std::uint32_t>(operation.extents.size() > 1u ? operation.extents[1] : 1u);
    const auto depth = static_cast<std::uint32_t>(operation.extents.size() > 2u ? operation.extents[2] : 1u);
    const bool wide_vectors = host_summary.native_vector_bits >= 512u;
    const bool large_cache = host_summary.cache_bytes >= (2ull * kMiB);
    const bool tight_cache = host_summary.cache_bytes > 0u && host_summary.cache_bytes < (512ull * kKiB);
    const bool column_heavy = columns >= (rows * 2u);
    const bool row_heavy = rows >= (columns * 2u);
    const bool depth_heavy = depth >= (std::max(rows, columns) * 2u);
    const auto parallelism =
        std::max<std::uint32_t>(1u, std::max(host_summary.execution_objects, 1u));
    const std::uint64_t work_items =
        static_cast<std::uint64_t>(rows) * columns * depth;

    std::uint32_t tile_m = wide_vectors ? 48u : 32u;
    std::uint32_t tile_n = wide_vectors ? 64u : 48u;
    std::uint32_t tile_k = wide_vectors ? 128u : 96u;

    if (large_cache) {
        tile_m += 16u;
        tile_n += 16u;
        tile_k += 32u;
    } else if (tight_cache) {
        tile_m = std::max<std::uint32_t>(16u, tile_m - 8u);
        tile_n = std::max<std::uint32_t>(32u, tile_n - 16u);
        tile_k = std::max<std::uint32_t>(32u, tile_k - 32u);
    }

    if (column_heavy) {
        tile_m = std::max<std::uint32_t>(16u, tile_m - 8u);
        tile_n += wide_vectors ? 16u : 8u;
    } else if (row_heavy) {
        tile_m += wide_vectors ? 16u : 8u;
        tile_n = std::max<std::uint32_t>(32u, tile_n - 8u);
    }
    if (depth_heavy) {
        tile_k += wide_vectors ? 32u : 16u;
    } else if (depth <= 64u) {
        tile_k = std::max<std::uint32_t>(32u, tile_k - 16u);
    }
    if (parallelism >= 12u && rows >= 128u) {
        tile_m += 8u;
    }

    if (operation.cpu_micro_kernel_unroll >= 4u) {
        tile_k += 16u;
    }
    if (!operation.fused_operation_names.empty()) {
        tile_n = std::max<std::uint32_t>(32u, tile_n - 8u);
        tile_k = std::max<std::uint32_t>(32u, tile_k - 8u);
    }

    const auto cache_budget = host_summary.cache_bytes == 0u
        ? 256ull * kKiB
        : std::max<std::uint64_t>(128ull * kKiB, host_summary.cache_bytes / (large_cache ? 2ull : 3ull));
    fit_tiles_to_cache_budget(cache_budget, rows, columns, depth, tile_m, tile_n, tile_k);
    operation.cpu_tile_m = clamp_cpu_tile_hint(tile_m, rows, 4u, 4u);
    operation.cpu_tile_n = clamp_cpu_tile_hint(tile_n, columns, 8u, 8u);
    operation.cpu_tile_k = clamp_cpu_tile_hint(tile_k, depth, 8u, 8u);
    operation.cpu_pack_budget_bytes = std::max<std::uint64_t>(
        64ull * kKiB,
        std::min<std::uint64_t>(cache_budget, host_summary.cache_bytes == 0u ? cache_budget : host_summary.cache_bytes / 2ull));
    operation.cpu_single_thread_cutoff = work_items >= (1ull << 22u)
        ? 1u << 18u
        : (work_items >= (1ull << 18u) ? 1u << 16u : 1u << 14u);
    operation.cpu_kernel_family = operation.attention_head_count > 0u ? "matmul-attention-blocked" : "matmul-blocked";
    operation.cpu_use_avx512 =
        executors::host_native_kernels_compiled_with_avx512() &&
        wide_vectors &&
        !tight_cache &&
        rows >= 128u &&
        columns >= 128u &&
        depth >= 128u &&
        work_items >= (1ull << 23u);
    if (host_summary.supports_bf16 && host_summary.native_vector_bits >= 512u && depth >= 128u) {
        operation.cpu_low_precision_kernel_family = "bf16-blocked";
    } else if (host_summary.supports_fp16 && host_summary.native_vector_bits >= 256u) {
        operation.cpu_low_precision_kernel_family = "fp16-blocked";
    } else {
        operation.cpu_low_precision_kernel_family = "scalar-emulated";
    }
}

void apply_operation_lowering_hints(
    const WorkloadSpec& workload,
    const HardwareGraphSummary& host_summary,
    const HardwareGraphSummary& accelerator_summary,
    WorkloadGraph& graph) {
    const bool has_host = host_summary.supports_fp32 || host_summary.addressable_bytes > 0u;
    const bool has_accelerator =
        accelerator_summary.supports_fp32 || accelerator_summary.matrix_units > 0u ||
        accelerator_summary.execution_objects > 0u;

    for (auto& operation : graph.operations) {
        operation.cpu_vectorized = false;
        operation.gpu_tensorized = false;
        operation.cpu_input_layout = "native";
        operation.cpu_weight_layout = "native";
        operation.cpu_output_layout = "native";
        operation.gpu_input_layout = "native";
        operation.gpu_weight_layout = "native";
        operation.gpu_output_layout = "native";
        operation.cpu_pack_weights = false;
        operation.gpu_pack_weights = false;
        operation.cpu_pretranspose_rhs = false;
        operation.gpu_pretranspose_rhs = false;
        operation.cpu_micro_kernel_unroll = 1u;
        operation.gpu_micro_kernel_unroll = 1u;
        operation.cpu_tile_m = 0u;
        operation.cpu_tile_n = 0u;
        operation.cpu_tile_k = 0u;
        operation.cpu_parallel_chunk = 0u;
        operation.cpu_use_avx512 = false;
        operation.cpu_pack_budget_bytes = 0u;
        operation.cpu_single_thread_cutoff = 0u;
        operation.cpu_kernel_family = "auto";
        operation.cpu_low_precision_kernel_family = "auto";
        operation.attention_head_count = 0u;
        operation.attention_head_group_size = 0u;
        operation.preferred_token_block = 0u;
        operation.preferred_kv_residency = "auto";
        operation.residency_sensitive_fusion = false;

        switch (operation.op_class) {
        case OperationClass::matmul: {
            operation.cpu_pack_weights = true;
            operation.cpu_pretranspose_rhs = true;
            operation.cpu_input_layout =
                workload.kind == WorkloadKind::inference ? "cpu-token-major" : "cpu-batch-major";
            operation.cpu_weight_layout = "cpu-packed-k-major";
            operation.cpu_output_layout = "cpu-accumulator-blocked";
            operation.gpu_pack_weights = true;
            operation.gpu_pretranspose_rhs = true;
            operation.gpu_input_layout =
                workload.kind == WorkloadKind::gaming ? "gpu-frame-blocked" : "gpu-token-major";
            operation.gpu_weight_layout = "gpu-blocked-k-major";
            operation.gpu_output_layout = "gpu-accumulator-blocked";
            if (has_host && host_summary.native_vector_bits >= 128u) {
                operation.cpu_vectorized = true;
                operation.cpu_micro_kernel_unroll = std::max(
                    operation.cpu_micro_kernel_unroll,
                    std::clamp(host_summary.native_vector_bits / 128u, 1u, 8u));
            }
            if (has_accelerator &&
                (accelerator_summary.matrix_units > 0u || accelerator_summary.supports_fp16 ||
                 accelerator_summary.supports_bf16 || accelerator_summary.supports_int8)) {
                operation.gpu_tensorized = true;
                operation.gpu_output_layout = "gpu-tile-accumulator";
                operation.gpu_weight_layout = "gpu-tensorcore-tiled";
                operation.gpu_micro_kernel_unroll = std::max(
                    operation.gpu_micro_kernel_unroll,
                    std::clamp(
                        std::max(accelerator_summary.matrix_units, accelerator_summary.execution_objects),
                        2u,
                        8u));
            }
            if (!operation.fused_operation_names.empty()) {
                operation.cpu_output_layout = "cpu-fused-epilogue";
                operation.gpu_output_layout = "gpu-fused-epilogue";
                operation.temporary_bytes += operation.output_bytes / 8u;
            }
            operation.attention_head_count = choose_attention_head_count_hint(workload, operation);
            operation.attention_head_group_size =
                choose_attention_head_group_size_hint(workload, operation, host_summary);
            operation.preferred_token_block = choose_token_block_hint(workload, operation);
            operation.preferred_kv_residency = choose_kv_residency_hint(workload, operation);
            operation.residency_sensitive_fusion =
                operation.preferred_kv_residency != "auto" &&
                (operation_is_attention_like(operation) || operation_mentions_kv_cache(operation));
            break;
        }
        case OperationClass::convolution_2d:
            operation.cpu_input_layout = "cpu-conv-patch9";
            operation.cpu_output_layout =
                operation.fused_operation_names.empty() ? "cpu-conv-accumulator" : "cpu-fused-activation";
            operation.gpu_input_layout = "gpu-conv-patch9";
            operation.gpu_output_layout =
                operation.fused_operation_names.empty() ? "gpu-conv-accumulator" : "gpu-fused-activation";
            operation.cpu_pack_weights = has_host;
            operation.gpu_pack_weights = has_accelerator;
            operation.cpu_vectorized = has_host && host_summary.native_vector_bits >= 128u;
            operation.gpu_tensorized = has_accelerator &&
                                       (accelerator_summary.local_scratch_bytes >= (32ull * kKiB) ||
                                        accelerator_summary.matrix_units > 0u);
            operation.cpu_micro_kernel_unroll = operation.cpu_vectorized ? 2u : 1u;
            operation.gpu_micro_kernel_unroll = operation.gpu_tensorized ? 4u : 1u;
            break;
        case OperationClass::elementwise_map:
            operation.cpu_input_layout = operation.streaming_friendly ? "cpu-stream-linear" : "cpu-vector-major";
            operation.cpu_output_layout = operation.streaming_friendly ? "cpu-stream-linear" : "cpu-vector-major";
            operation.gpu_input_layout = operation.streaming_friendly ? "gpu-stream-linear" : "gpu-vector-major";
            operation.gpu_output_layout = operation.streaming_friendly ? "gpu-stream-linear" : "gpu-vector-major";
            operation.cpu_vectorized = has_host && host_summary.native_vector_bits >= 128u;
            operation.gpu_tensorized = has_accelerator && accelerator_summary.supports_asynchronous_dispatch;
            operation.cpu_micro_kernel_unroll = operation.cpu_vectorized ? 2u : 1u;
            operation.gpu_micro_kernel_unroll = operation.gpu_tensorized ? 2u : 1u;
            operation.preferred_kv_residency = choose_kv_residency_hint(workload, operation);
            operation.residency_sensitive_fusion =
                operation.preferred_kv_residency != "auto" && operation_mentions_kv_cache(operation);
            break;
        case OperationClass::reduction:
            operation.cpu_input_layout = "cpu-reduction-linear";
            operation.cpu_output_layout =
                operation.fused_operation_names.empty() ? "cpu-scalar-reduced" : "cpu-fused-reduction";
            operation.gpu_input_layout = "gpu-reduction-linear";
            operation.gpu_output_layout =
                operation.fused_operation_names.empty() ? "gpu-scalar-reduced" : "gpu-fused-reduction";
            operation.cpu_vectorized = has_host && host_summary.native_vector_bits >= 128u;
            operation.gpu_tensorized = has_accelerator &&
                                       (accelerator_summary.cache_bytes >= (512ull * kKiB) ||
                                        accelerator_summary.matrix_units > 0u);
            operation.cpu_micro_kernel_unroll = operation.cpu_vectorized ? 2u : 1u;
            operation.gpu_micro_kernel_unroll = operation.gpu_tensorized ? 4u : 1u;
            if (!operation.fused_operation_names.empty()) {
                operation.temporary_bytes += std::max<std::uint64_t>(operation.output_bytes / 8u, 16u);
            }
            operation.preferred_kv_residency = choose_kv_residency_hint(workload, operation);
            operation.residency_sensitive_fusion =
                operation.preferred_kv_residency != "auto" && operation_mentions_kv_cache(operation);
            break;
        case OperationClass::resample_2d:
        default:
            operation.cpu_input_layout = "cpu-resample-packed6";
            operation.cpu_output_layout = "cpu-resample-linear";
            operation.gpu_input_layout = "gpu-resample-packed6";
            operation.gpu_output_layout = "gpu-resample-linear";
            operation.cpu_vectorized = has_host && host_summary.native_vector_bits >= 128u;
            operation.gpu_tensorized = has_accelerator && accelerator_summary.local_scratch_bytes >= (16ull * kKiB);
            operation.cpu_micro_kernel_unroll = operation.cpu_vectorized ? 2u : 1u;
            operation.gpu_micro_kernel_unroll = operation.gpu_tensorized ? 2u : 1u;
            break;
        }

        apply_cpu_runtime_tuning_hints(operation, host_summary);
    }
}

void run_graph_rewrite_pass(
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    WorkloadGraph& graph,
    const ExecutionTuningOverrides* tuning_overrides) {
    const std::uint32_t graph_rewrite_level =
        tuning_overrides == nullptr ? 1u : std::max(tuning_overrides->graph_rewrite_level, 1u);
    bool changed = true;
    while (changed) {
        changed = false;
        if (graph_rewrite_level >= 1u) {
            for (std::size_t index = 0; index < graph.operations.size(); ++index) {
                const auto& producer = graph.operations[index];
                if ((producer.op_class != OperationClass::matmul &&
                     producer.op_class != OperationClass::convolution_2d &&
                     producer.op_class != OperationClass::resample_2d) ||
                    producer.output_tensor_ids.size() != 1u) {
                    continue;
                }
                const auto consumer_it = std::find_if(
                    graph.operations.begin(),
                    graph.operations.end(),
                    [&](const OperationSpec& operation) {
                        return operation.name != producer.name &&
                               std::find(
                                   operation.input_tensor_ids.begin(),
                                   operation.input_tensor_ids.end(),
                                   producer.output_tensor_ids.front()) != operation.input_tensor_ids.end();
                    });
                if (consumer_it == graph.operations.end()) {
                    continue;
                }
                if (should_preserve_host_preferred_tail_boundary(
                        workload,
                        placement,
                        graph_lookup,
                        producer,
                        *consumer_it) ||
                    should_block_residency_sensitive_fusion(
                        graph,
                        producer,
                        *consumer_it)) {
                    continue;
                }
                if (fuse_epilogue_elementwise_into_producer(
                        graph,
                        producer.name,
                        consumer_it->name)) {
                    changed = true;
                    break;
                }
            }
        }
        if (changed || graph_rewrite_level < 2u) {
            continue;
        }

        for (std::size_t index = 0; index < graph.operations.size(); ++index) {
            const auto& producer = graph.operations[index];
            if (producer.op_class != OperationClass::elementwise_map || producer.output_tensor_ids.size() != 1u) {
                continue;
            }
            const auto consumer_it = std::find_if(
                graph.operations.begin(),
                graph.operations.end(),
                [&](const OperationSpec& operation) {
                    return operation.name != producer.name &&
                           std::find(
                               operation.input_tensor_ids.begin(),
                               operation.input_tensor_ids.end(),
                               producer.output_tensor_ids.front()) != operation.input_tensor_ids.end();
                });
            if (consumer_it == graph.operations.end()) {
                continue;
            }
            if (fuse_elementwise_chain(graph, producer.name, consumer_it->name)) {
                changed = true;
                break;
            }
        }
        if (changed || graph_rewrite_level < 3u) {
            continue;
        }

        for (std::size_t index = 0; index < graph.operations.size(); ++index) {
            const auto& producer = graph.operations[index];
            if (producer.op_class != OperationClass::elementwise_map || producer.output_tensor_ids.size() != 1u) {
                continue;
            }
            const auto consumer_it = std::find_if(
                graph.operations.begin(),
                graph.operations.end(),
                [&](const OperationSpec& operation) {
                    return operation.name != producer.name &&
                           std::find(
                               operation.input_tensor_ids.begin(),
                               operation.input_tensor_ids.end(),
                               producer.output_tensor_ids.front()) != operation.input_tensor_ids.end();
                });
            if (consumer_it == graph.operations.end()) {
                continue;
            }
            if (should_block_residency_sensitive_fusion(graph, producer, *consumer_it)) {
                continue;
            }
            if (fuse_elementwise_into_reduction(graph, producer.name, consumer_it->name)) {
                changed = true;
                break;
            }
        }
        if (changed || graph_rewrite_level < 4u) {
            continue;
        }

        for (std::size_t index = 0; index < graph.operations.size(); ++index) {
            const auto& producer = graph.operations[index];
            if (producer.op_class != OperationClass::reduction || producer.output_tensor_ids.size() != 1u) {
                continue;
            }
            const auto consumer_it = std::find_if(
                graph.operations.begin(),
                graph.operations.end(),
                [&](const OperationSpec& operation) {
                    return operation.name != producer.name &&
                           std::find(
                               operation.input_tensor_ids.begin(),
                               operation.input_tensor_ids.end(),
                               producer.output_tensor_ids.front()) != operation.input_tensor_ids.end();
                });
            if (consumer_it == graph.operations.end()) {
                continue;
            }
            if (should_block_residency_sensitive_fusion(graph, producer, *consumer_it)) {
                continue;
            }
            if (fuse_reduction_epilogue_elementwise(graph, producer.name, consumer_it->name)) {
                changed = true;
                break;
            }
        }
        if (changed || graph_rewrite_level < 5u) {
            continue;
        }

        for (std::size_t index = 0; index < graph.operations.size(); ++index) {
            const auto& producer = graph.operations[index];
            if ((producer.op_class != OperationClass::matmul &&
                 producer.op_class != OperationClass::convolution_2d &&
                 producer.op_class != OperationClass::resample_2d) ||
                producer.output_tensor_ids.size() != 1u) {
                continue;
            }
            const auto first_consumer_it = std::find_if(
                graph.operations.begin(),
                graph.operations.end(),
                [&](const OperationSpec& operation) {
                    return operation.op_class == OperationClass::elementwise_map &&
                           std::find(
                               operation.input_tensor_ids.begin(),
                               operation.input_tensor_ids.end(),
                               producer.output_tensor_ids.front()) != operation.input_tensor_ids.end();
                });
            if (first_consumer_it == graph.operations.end() ||
                first_consumer_it->output_tensor_ids.size() != 1u) {
                continue;
            }
            const auto second_consumer_it = std::find_if(
                graph.operations.begin(),
                graph.operations.end(),
                [&](const OperationSpec& operation) {
                    return operation.op_class == OperationClass::elementwise_map &&
                           std::find(
                               operation.input_tensor_ids.begin(),
                               operation.input_tensor_ids.end(),
                               first_consumer_it->output_tensor_ids.front()) != operation.input_tensor_ids.end();
                });
            if (second_consumer_it == graph.operations.end()) {
                continue;
            }
            if (should_preserve_host_preferred_tail_boundary(
                    workload,
                    placement,
                    graph_lookup,
                    producer,
                    *first_consumer_it) ||
                should_block_residency_sensitive_fusion(
                    graph,
                    producer,
                    *first_consumer_it)) {
                continue;
            }
            if (fuse_producer_elementwise_chain(
                    graph,
                    producer.name,
                    first_consumer_it->name,
                    second_consumer_it->name)) {
                changed = true;
                break;
            }
        }
        if (changed) {
            continue;
        }

        for (std::size_t index = 0; index < graph.operations.size(); ++index) {
            const auto& producer = graph.operations[index];
            if (producer.op_class != OperationClass::elementwise_map ||
                producer.output_tensor_ids.size() != 1u) {
                continue;
            }
            const auto intermediate_it = std::find_if(
                graph.operations.begin(),
                graph.operations.end(),
                [&](const OperationSpec& operation) {
                    return operation.op_class == OperationClass::elementwise_map &&
                           std::find(
                               operation.input_tensor_ids.begin(),
                               operation.input_tensor_ids.end(),
                               producer.output_tensor_ids.front()) != operation.input_tensor_ids.end();
                });
            if (intermediate_it == graph.operations.end() ||
                intermediate_it->output_tensor_ids.size() != 1u) {
                continue;
            }
            const auto reduction_it = std::find_if(
                graph.operations.begin(),
                graph.operations.end(),
                [&](const OperationSpec& operation) {
                    return operation.op_class == OperationClass::reduction &&
                           std::find(
                               operation.input_tensor_ids.begin(),
                               operation.input_tensor_ids.end(),
                               intermediate_it->output_tensor_ids.front()) != operation.input_tensor_ids.end();
                });
            if (reduction_it == graph.operations.end()) {
                continue;
            }
            if (fuse_elementwise_chain_into_reduction(
                    graph,
                    producer.name,
                    intermediate_it->name,
                    reduction_it->name)) {
                changed = true;
                break;
            }
        }
    }

}

void run_lowering_hint_pass(
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    WorkloadGraph& graph) {
    normalize_workload_graph(graph);
    const auto host_summary = strongest_host_summary(placement, graph_lookup);
    const auto accelerator_summary = strongest_accelerator_summary(placement, graph_lookup);
    apply_operation_lowering_hints(workload, host_summary, accelerator_summary, graph);
}

void optimize_workload_operations_for_targets(
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    WorkloadGraph& graph,
    const ExecutionTuningOverrides* tuning_overrides) {
    run_graph_rewrite_pass(workload, placement, graph_lookup, graph, tuning_overrides);
    run_lowering_hint_pass(workload, placement, graph_lookup, graph);
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
    const std::string& report_signature,
    const std::string& graph_set_signature,
    const WorkloadSpec& workload,
    const SystemProfile& system,
    const std::string& shape_bucket,
    const ExecutionConfig& config) {
    std::ostringstream stream;
    stream << report_signature << '|'
           << graph_set_signature << '|'
           << to_string(workload.kind) << '|'
           << workload.dataset_tag << '|'
           << to_string(canonical_workload_phase(workload)) << '|'
           << canonical_workload_shape_bucket(workload) << '|'
           << to_string(workload.partition_strategy) << '|'
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
           << config.precision_mix << '|'
           << (config.semantic_head_partitioning ? 1 : 0) << '|'
           << (config.semantic_token_partitioning ? 1 : 0);
    return stream.str();
}

std::string backend_penalty_key(
    const std::string& report_signature,
    const std::string& graph_set_signature,
    const WorkloadSpec& workload,
    const std::string& operation_name,
    const ExecutionConfig& config) {
    std::ostringstream stream;
    stream << report_signature << '|'
           << graph_set_signature << '|'
           << to_string(workload.kind) << '|'
           << workload.dataset_tag << '|'
           << to_string(canonical_workload_phase(workload)) << '|'
           << canonical_workload_shape_bucket(workload) << '|'
           << to_string(workload.partition_strategy) << '|'
           << operation_name << '|'
           << config.primary_device_uid << '|'
           << join_csv(config.participating_devices) << '|'
           << to_string(config.strategy) << '|'
           << config.logical_partitions;
    return stream.str();
}

std::string performance_family_key(
    const std::string& report_signature,
    const std::string& shape_bucket,
    const ExecutionConfig& config) {
    std::ostringstream stream;
    stream << report_signature << '|'
           << shape_bucket << '|'
           << to_string(config.strategy) << '|'
           << config.primary_device_uid << '|'
           << config.participating_devices.size() << '|'
           << config.logical_partitions << '|'
           << config.overlap_transfers << '|'
           << config.use_low_precision;
    return stream.str();
}

using OptimizationDeadline = std::optional<std::chrono::steady_clock::time_point>;

OptimizationDeadline make_optimizer_deadline(const ExecutionTuningOverrides* tuning_overrides) {
    if (tuning_overrides == nullptr || !tuning_overrides->optimizer_wall_time_budget_ms.has_value()) {
        return std::nullopt;
    }
    const auto budget_ms = *tuning_overrides->optimizer_wall_time_budget_ms;
    if (budget_ms == 0u) {
        return std::nullopt;
    }
    return std::chrono::steady_clock::now() + std::chrono::milliseconds(budget_ms);
}

bool deadline_reached(const OptimizationDeadline& deadline) {
    return deadline.has_value() && std::chrono::steady_clock::now() >= *deadline;
}

std::uint32_t budget_ms_or_zero(const OptimizationDeadline& deadline) {
    if (!deadline.has_value()) {
        return 0u;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= *deadline) {
        return 0u;
    }
    return static_cast<std::uint32_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(*deadline - now).count());
}

std::string fallback_device_family_token(const std::string& uid) {
    const auto split = uid.find(':');
    return split == std::string::npos ? uid : uid.substr(0, split);
}

std::string device_family_token(const HardwareGraph* graph, const std::string& uid) {
    if (graph == nullptr) {
        return fallback_device_family_token(uid);
    }
    if (is_host_graph(graph)) {
        return "host";
    }
    return graph->probe.empty() ? fallback_device_family_token(uid) : graph->probe;
}

std::string device_family_signature_for_config(
    const ExecutionConfig& config,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    std::vector<std::string> families;
    families.reserve(config.participating_devices.size());
    for (const auto& device_uid : config.participating_devices) {
        families.push_back(device_family_token(find_graph(graph_lookup, device_uid), device_uid));
    }
    std::sort(families.begin(), families.end());
    families.erase(std::unique(families.begin(), families.end()), families.end());
    return join_csv(families);
}

std::vector<std::uint64_t> parse_shape_bucket_numbers(const std::string& bucket) {
    std::vector<std::uint64_t> values;
    std::uint64_t current = 0u;
    bool reading = false;
    for (const unsigned char ch : bucket) {
        if (std::isdigit(ch) != 0) {
            current = (current * 10u) + static_cast<std::uint64_t>(ch - static_cast<unsigned char>('0'));
            reading = true;
            continue;
        }
        if (reading) {
            values.push_back(current);
            current = 0u;
            reading = false;
        }
    }
    if (reading) {
        values.push_back(current);
    }
    return values;
}

double shape_bucket_similarity_score(const std::string& left, const std::string& right) {
    if (left == right) {
        return 1.0;
    }
    const auto left_values = parse_shape_bucket_numbers(left);
    const auto right_values = parse_shape_bucket_numbers(right);
    if (left_values.empty() || right_values.empty()) {
        return left == right ? 1.0 : 0.35;
    }
    const auto overlap = std::min(left_values.size(), right_values.size());
    double ratio_error = 0.0;
    for (std::size_t index = 0; index < overlap; ++index) {
        const double lhs = static_cast<double>(std::max<std::uint64_t>(left_values[index], 1u));
        const double rhs = static_cast<double>(std::max<std::uint64_t>(right_values[index], 1u));
        ratio_error += std::abs(std::log2(lhs / rhs));
    }
    ratio_error /= static_cast<double>(overlap);
    return std::clamp(1.0 - (ratio_error / 4.0), 0.0, 1.0);
}

const PerformanceSummary* best_graph_family_summary(
    const std::unordered_map<std::string, PerformanceSummary>& graph_family_performance_cache,
    const std::string& report_signature,
    const std::string& shape_bucket,
    const ExecutionConfig& candidate,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    const std::string key_prefix = report_signature + '|';
    const auto candidate_family = device_family_signature_for_config(candidate, graph_lookup);
    const auto candidate_primary_family =
        device_family_token(find_graph(graph_lookup, candidate.primary_device_uid), candidate.primary_device_uid);

    const PerformanceSummary* best = nullptr;
    double best_score = 0.0;
    for (const auto& [key, summary] : graph_family_performance_cache) {
        if (key.rfind(key_prefix, 0u) != 0u) {
            continue;
        }

        double score = 0.0;
        score += shape_bucket_similarity_score(shape_bucket, summary.shape_bucket) * 0.40;
        score += candidate.strategy == summary.config.strategy ? 0.18 : 0.0;
        score += candidate.logical_partitions == summary.config.logical_partitions ? 0.08 : 0.0;
        score += candidate.use_low_precision == summary.config.use_low_precision ? 0.08 : 0.0;
        score += candidate.overlap_transfers == summary.config.overlap_transfers ? 0.06 : 0.0;
        score += candidate.participating_devices.size() == summary.config.participating_devices.size() ? 0.08 : 0.0;

        const auto summary_primary_family =
            device_family_token(find_graph(graph_lookup, summary.config.primary_device_uid), summary.config.primary_device_uid);
        score += candidate_primary_family == summary_primary_family ? 0.07 : 0.0;
        score += candidate_family == summary.device_family_signature ? 0.13 : 0.0;

        if (score > best_score) {
            best_score = score;
            best = &summary;
        }
    }

    return best_score >= 0.55 ? best : nullptr;
}

struct PlacementPolicyPassResult {
    WorkloadSpec effective_workload;
    std::unordered_map<std::string, const HardwareGraph*> graph_lookup;
    std::string graph_set_signature;
};

struct ExecutionConfigSearchPassState {
    CompiledWorkloadGraph compiled_workload;
    std::unordered_map<std::string, ExecutionConfig> cached_by_operation;
    bool fully_cached = false;
    bool runtime_sensitive_path = false;
    bool lightweight_path = false;
    ValidationTier validation_tier = ValidationTier::adaptive;
    OptimizationDeadline optimization_deadline;
    std::string optimizer_route;
};

WorkloadSpec effective_workload_for_placement(const WorkloadSpec& workload, const ExecutionPlan& placement) {
    auto effective = workload;
    if (placement.strategy_source == PlanStrategySource::explicit_request) {
        effective.partition_strategy = placement.resolved_partition_strategy;
        return effective;
    }
    effective.partition_strategy = PartitionStrategy::auto_balanced;
    const bool accept_runtime_hint =
        placement.resolved_partition_strategy != PartitionStrategy::auto_balanced &&
        placement.strategy_confidence >= 0.30;
    if (accept_runtime_hint) {
        effective.heuristic_partition_hint = placement.resolved_partition_strategy;
        effective.heuristic_partition_hint_confidence =
            std::max(effective.heuristic_partition_hint_confidence, placement.strategy_confidence);
        effective.heuristic_partition_hint_reason =
            placement.strategy_reason.empty()
                ? ("runtime-managed hint for " + to_string(placement.resolved_partition_strategy))
                : placement.strategy_reason;
    }
    return effective;
}

PlacementPolicyPassResult run_placement_policy_pass(
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::vector<HardwareGraph>& graphs) {
    PlacementPolicyPassResult pass;
    pass.effective_workload = effective_workload_for_placement(workload, placement);
    pass.graph_lookup.reserve(graphs.size());
    for (const auto& graph : graphs) {
        pass.graph_lookup.emplace(graph.uid, &graph);
    }
    pass.graph_set_signature = summarize_graph_set(graphs);
    return pass;
}

ExecutionConfigSearchPassState prepare_execution_config_search_pass(
    AdaptiveExecutionOptimizer& adaptive_optimizer,
    BootstrapExecutionOptimizer& bootstrap_optimizer,
    const WorkloadSpec& effective_workload,
    const ExecutionPlan& placement,
    const WorkloadGraph& workload_graph,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const ExecutionTuningOverrides* tuning_overrides) {
    ExecutionConfigSearchPassState pass;
    pass.compiled_workload = compile_workload_graph(workload_graph);
    const auto report_signature =
        build_report_signature(
            effective_workload,
            placement,
            pass.compiled_workload.signature,
            workload_graph.operations,
            tuning_overrides);
    pass.fully_cached = bootstrap_optimizer.has_full_cache(
        report_signature,
        workload_graph.operations,
        graph_lookup,
        &pass.cached_by_operation);
    pass.runtime_sensitive_path = adaptive_optimizer.should_reoptimize(report_signature);
    pass.lightweight_path =
        adaptive_optimizer.should_use_lightweight_path(report_signature, pass.fully_cached);
    pass.validation_tier =
        tuning_overrides == nullptr ? ValidationTier::adaptive : tuning_overrides->validation_tier;
    pass.optimization_deadline = make_optimizer_deadline(tuning_overrides);
    pass.optimizer_route =
        (pass.lightweight_path || pass.runtime_sensitive_path) ? "runtime_sensitive_optimizer"
                                                               : "bootstrap_general_optimizer";
    return pass;
}

void run_feedback_retuning_pass(
    AdaptiveExecutionOptimizer& adaptive_optimizer,
    const OptimizationReport& report,
    const std::vector<ExecutionFeedbackRecord>& feedback,
    const std::vector<HardwareGraph>& graphs) {
    adaptive_optimizer.ingest_execution_feedback(report, feedback, graphs);
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

std::uint32_t choose_validation_rounds(
    const std::uint32_t observations,
    const double average_relative_error,
    const double validation_spread_us,
    const double calibration_ratio,
    const bool prefer_fast_path,
    const ValidationTier validation_tier) {
    switch (validation_tier) {
    case ValidationTier::disabled:
        return 0u;
    case ValidationTier::minimal:
        return kMinimumValidationMeasurementRounds;
    case ValidationTier::full:
        return std::max(kValidationMeasurementRounds, 3u);
    case ValidationTier::adaptive:
    default:
        break;
    }
    if (prefer_fast_path &&
        observations >= (kLearningWarmupSamples * 2u) &&
        average_relative_error <= 1.0e-4 &&
        validation_spread_us <= 12.0 &&
        std::abs(calibration_ratio - 1.0) <= 0.12) {
        return kMinimumValidationMeasurementRounds;
    }
    if (observations >= (kLearningWarmupSamples * 4u) &&
        average_relative_error <= 2.0e-4 &&
        validation_spread_us <= 18.0 &&
        std::abs(calibration_ratio - 1.0) <= 0.20) {
        return kMinimumValidationMeasurementRounds;
    }
    if (observations >= kLearningWarmupSamples &&
        (validation_spread_us >= 45.0 || std::abs(calibration_ratio - 1.0) >= 0.35)) {
        return std::max(kValidationMeasurementRounds + 1u, 3u);
    }
    return kValidationMeasurementRounds;
}

bool is_small_dispatch_bound_operation(const OperationSpec& operation) {
    const auto total_bytes =
        std::max<std::uint64_t>(operation.input_bytes + operation.output_bytes + operation.temporary_bytes, 1ull);
    const bool tiny_bytes = total_bytes <= (1ull * kMiB);
    const bool tiny_flops = operation.estimated_flops <= 6.0e6;
    const bool scalarish = operation.op_class == OperationClass::elementwise_map ||
                           operation.op_class == OperationClass::reduction;
    const bool fused_tail = !operation.fused_operation_names.empty() &&
                            operation.op_class != OperationClass::matmul &&
                            operation.op_class != OperationClass::convolution_2d;
    return (scalarish && tiny_bytes && tiny_flops) || fused_tail;
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

const WorkloadTensor* find_workload_tensor(
    const WorkloadGraph& workload_graph,
    const CompiledWorkloadGraph& compiled,
    const std::string_view tensor_id) {
    const auto it = compiled.tensor_indices.find(std::string(tensor_id));
    if (it == compiled.tensor_indices.end() || it->second >= workload_graph.tensors.size()) {
        return nullptr;
    }
    return &workload_graph.tensors[it->second];
}

const TensorLifetime* find_tensor_lifetime(const WorkloadGraph& workload_graph, const std::string_view tensor_id) {
    const auto it = std::find_if(
        workload_graph.lifetimes.begin(),
        workload_graph.lifetimes.end(),
        [&](const TensorLifetime& lifetime) { return lifetime.tensor_id == tensor_id; });
    return it == workload_graph.lifetimes.end() ? nullptr : &(*it);
}

const WorkloadTensor* find_workload_tensor(
    const WorkloadGraph& workload_graph,
    const CompiledWorkloadGraph& compiled,
    const std::uint32_t tensor_index) {
    if (tensor_index >= compiled.tensors.size() || tensor_index >= workload_graph.tensors.size()) {
        return nullptr;
    }
    return &workload_graph.tensors[tensor_index];
}

const TensorLifetime* find_tensor_lifetime(
    const WorkloadGraph& workload_graph,
    const CompiledWorkloadGraph& compiled,
    const std::string_view tensor_id) {
    const auto it = compiled.tensor_indices.find(std::string(tensor_id));
    if (it == compiled.tensor_indices.end() || it->second >= compiled.tensors.size()) {
        return nullptr;
    }
    const auto& compiled_tensor = compiled.tensors[it->second];
    if (!compiled_tensor.has_lifetime || compiled_tensor.lifetime_index >= workload_graph.lifetimes.size()) {
        return nullptr;
    }
    return &workload_graph.lifetimes[compiled_tensor.lifetime_index];
}

const TensorLifetime* find_tensor_lifetime(
    const WorkloadGraph& workload_graph,
    const CompiledWorkloadGraph& compiled,
    const std::uint32_t tensor_index) {
    if (tensor_index >= compiled.tensors.size()) {
        return nullptr;
    }
    return find_tensor_lifetime(workload_graph, compiled, compiled.tensors[tensor_index].id);
}

std::uint32_t operation_index_or_default(
    const std::unordered_map<std::string, std::uint32_t>& indices,
    const std::string& operation_name,
    const std::uint32_t fallback) {
    const auto it = indices.find(operation_name);
    return it == indices.end() ? fallback : it->second;
}

std::vector<std::string> resident_devices_for_tensor(
    const WorkloadSpec& workload,
    const WorkloadTensor& tensor,
    const OperationSpec& consumer,
    const ExecutionConfig& current_config,
    const std::unordered_map<std::string, ExecutionConfig>& selected_configs) {
    std::vector<std::string> devices;
    auto append_devices = [&](const std::vector<std::string>& source) {
        for (const auto& device_uid : source) {
            append_unique(devices, device_uid);
        }
    };

    if (!tensor.producer_operation.empty()) {
        if (const auto config_it = selected_configs.find(tensor.producer_operation); config_it != selected_configs.end()) {
            if (!config_it->second.participating_devices.empty()) {
                append_devices(config_it->second.participating_devices);
            } else if (!config_it->second.primary_device_uid.empty()) {
                append_unique(devices, config_it->second.primary_device_uid);
            }
        }
    }

    if (devices.empty() && tensor.producer_operation.empty()) {
        append_unique(devices, std::string("host"));
    }
    if (devices.empty()) {
        if (!current_config.participating_devices.empty()) {
            append_devices(current_config.participating_devices);
        } else if (!current_config.primary_device_uid.empty()) {
            append_unique(devices, current_config.primary_device_uid);
        }
    }

    const bool kv_tensor = tensor_mentions_kv_cache(tensor);
    const bool decode_like =
        canonical_workload_phase(workload) == WorkloadPhase::decode ||
        canonical_workload_phase(workload) == WorkloadPhase::cache_update;

    if ((kv_tensor || tensor.host_visible) && consumer.preferred_kv_residency == "shared") {
        if (!current_config.participating_devices.empty()) {
            append_devices(current_config.participating_devices);
        } else if (!current_config.primary_device_uid.empty()) {
            append_unique(devices, current_config.primary_device_uid);
        }
        if (tensor.host_visible || decode_like) {
            append_unique(devices, std::string("host"));
        }
    } else if (kv_tensor && consumer.preferred_kv_residency == "host") {
        devices.erase(
            std::remove_if(devices.begin(), devices.end(), [](const std::string& uid) { return !is_host_device_uid(uid); }),
            devices.end());
        if (devices.empty()) {
            append_unique(devices, std::string("host"));
        }
    } else if (kv_tensor && consumer.preferred_kv_residency == "accelerator") {
        std::vector<std::string> accelerator_devices;
        if (!current_config.participating_devices.empty()) {
            for (const auto& uid : current_config.participating_devices) {
                if (!is_host_device_uid(uid)) {
                    append_unique(accelerator_devices, uid);
                }
            }
        } else if (!current_config.primary_device_uid.empty() && !is_host_device_uid(current_config.primary_device_uid)) {
            append_unique(accelerator_devices, current_config.primary_device_uid);
        }
        if (!accelerator_devices.empty()) {
            return accelerator_devices;
        }
    } else if (tensor.host_visible && decode_like) {
        append_unique(devices, std::string("host"));
    }

    if (devices.empty()) {
        return {"host"};
    }
    return devices;
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
    const std::string& compiled_workload_signature,
    const std::vector<OperationSpec>& operations,
    const ExecutionTuningOverrides* tuning_overrides) {
    std::ostringstream stream;
    stream << kExecutionHeuristicRevision << '|'
           << placement.signature << '|'
           << "graph=" << compiled_workload_signature << '|'
           << workload.name << '|'
           << to_string(workload.kind) << '|'
           << to_string(canonical_workload_phase(workload)) << '|'
           << canonical_workload_shape_bucket(workload) << '|'
           << to_string(workload.partition_strategy) << '|'
           << operations.size();
    for (const auto& operation : operations) {
        stream << '|' << operation.name << ':' << to_string(operation.op_class);
        for (const auto extent : operation.extents) {
            stream << ':' << extent;
        }
        stream << ':' << operation.cpu_input_layout
               << ':' << operation.cpu_weight_layout
               << ':' << operation.cpu_output_layout
               << ':' << operation.gpu_input_layout
               << ':' << operation.gpu_weight_layout
               << ':' << operation.gpu_output_layout
               << ':' << operation.cpu_pack_weights
               << ':' << operation.gpu_pack_weights
               << ':' << operation.cpu_pretranspose_rhs
               << ':' << operation.gpu_pretranspose_rhs
               << ':' << operation.cpu_vectorized
               << ':' << operation.gpu_tensorized
               << ':' << std::max(operation.cpu_micro_kernel_unroll, 1u)
               << ':' << std::max(operation.cpu_tile_m, 1u)
               << ':' << std::max(operation.cpu_tile_n, 1u)
               << ':' << std::max(operation.cpu_tile_k, 1u)
               << ':' << std::max(operation.cpu_parallel_chunk, 1u)
               << ':' << operation.cpu_use_avx512
               << ':' << std::max<std::uint64_t>(operation.cpu_pack_budget_bytes, 1ull)
               << ':' << std::max(operation.cpu_single_thread_cutoff, 1u)
               << ':' << operation.cpu_low_precision_kernel_family
               << ':' << std::max(operation.attention_head_count, 1u)
               << ':' << std::max(operation.attention_head_group_size, 1u)
               << ':' << std::max(operation.preferred_token_block, 1u)
               << ':' << operation.preferred_kv_residency
               << ':' << operation.residency_sensitive_fusion
               << ':' << std::max(operation.gpu_micro_kernel_unroll, 1u);
        for (const auto& fused : operation.fused_operation_names) {
            stream << ":fused=" << fused;
        }
    }
    stream << "|tuning:" << tuning_signature_suffix(tuning_overrides);
    return stream.str();
}

double workload_host_exchange_ratio(const WorkloadSpec& workload) {
    const auto working_set = std::max<std::uint64_t>(workload.working_set_bytes, 1ull);
    return std::clamp(
        static_cast<double>(workload.host_exchange_bytes) / static_cast<double>(working_set),
        0.0,
        4.0);
}

double operation_working_set_bytes(const OperationSpec& operation) {
    return static_cast<double>(
        std::max<std::uint64_t>(
            operation.input_bytes + operation.output_bytes + operation.temporary_bytes,
            1ull));
}

double operation_compute_intensity(const OperationSpec& operation) {
    if (operation.estimated_flops <= 0.0) {
        return operation.matrix_friendly ? 32.0 : 8.0;
    }
    return operation.estimated_flops / operation_working_set_bytes(operation);
}

std::uint32_t round_down_to_power_of_two(std::uint32_t value) {
    if (value <= 1u) {
        return 1u;
    }
    std::uint32_t result = 1u;
    while ((result << 1u) <= value) {
        result <<= 1u;
    }
    return result;
}

std::uint32_t choose_attention_head_count_hint(
    const WorkloadSpec& workload,
    const OperationSpec& operation) {
    if (!operation_is_attention_like(operation)) {
        return 0u;
    }
    if (operation.extents.size() > 1u) {
        const auto width = static_cast<std::uint32_t>(std::max<std::uint64_t>(operation.extents[1], 1ull));
        if (width >= 2048u) {
            return 64u;
        }
        if (width >= 1024u) {
            return 32u;
        }
        if (width >= 512u) {
            return 16u;
        }
    }
    switch (canonical_workload_phase(workload)) {
    case WorkloadPhase::prefill:
        return 32u;
    case WorkloadPhase::decode:
        return 16u;
    default:
        return 8u;
    }
}

std::uint32_t choose_attention_head_group_size_hint(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const HardwareGraphSummary& host_summary) {
    if (operation.attention_head_count == 0u) {
        return 0u;
    }
    std::uint32_t group_size = canonical_workload_phase(workload) == WorkloadPhase::decode ? 4u : 8u;
    if (host_summary.native_vector_bits >= 512u) {
        group_size *= 2u;
    }
    if (operation.attention_head_count <= 8u) {
        group_size = std::max<std::uint32_t>(1u, operation.attention_head_count);
    }
    return round_down_to_power_of_two(std::min(group_size, std::max(operation.attention_head_count, 1u)));
}

std::uint32_t choose_token_block_hint(
    const WorkloadSpec& workload,
    const OperationSpec& operation) {
    if (!operation_is_attention_like(operation)) {
        return 0u;
    }
    const auto rows = static_cast<std::uint32_t>(operation.extents.empty() ? 1ull : std::max<std::uint64_t>(operation.extents[0], 1ull));
    if (canonical_workload_phase(workload) == WorkloadPhase::prefill) {
        if (rows >= 1024u) {
            return 128u;
        }
        if (rows >= 256u) {
            return 64u;
        }
        return 32u;
    }
    if (rows >= 256u) {
        return 32u;
    }
    if (rows >= 64u) {
        return 16u;
    }
    return 8u;
}

std::string choose_kv_residency_hint(const WorkloadSpec& workload, const OperationSpec& operation) {
    const auto phase = canonical_workload_phase(workload);
    if (operation_mentions_kv_cache(operation)) {
        if (phase == WorkloadPhase::decode || phase == WorkloadPhase::cache_update) {
            return "host";
        }
        if (phase == WorkloadPhase::prefill) {
            return "shared";
        }
    }
    if (operation_is_attention_like(operation)) {
        if (phase == WorkloadPhase::decode) {
            return "shared";
        }
        if (phase == WorkloadPhase::prefill) {
            return "accelerator";
        }
    }
    return "auto";
}

std::uint32_t attention_head_group_count(const OperationSpec& operation) {
    if (operation.attention_head_count == 0u) {
        return 1u;
    }
    const auto group_size = std::max(operation.attention_head_group_size, 1u);
    return std::max<std::uint32_t>(1u, (operation.attention_head_count + group_size - 1u) / group_size);
}

std::uint32_t attention_token_block_count(const OperationSpec& operation) {
    if (operation.preferred_token_block == 0u || operation.extents.empty()) {
        return 1u;
    }
    const auto rows = static_cast<std::uint32_t>(std::max<std::uint64_t>(operation.extents[0], 1ull));
    return std::max<std::uint32_t>(1u, (rows + operation.preferred_token_block - 1u) / operation.preferred_token_block);
}

std::uint32_t desired_semantic_partition_count(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const ExecutionConfig& config) {
    if (!operation.parallelizable) {
        return 1u;
    }
    std::uint32_t desired = 1u;
    const auto head_groups = attention_head_group_count(operation);
    const auto token_blocks = attention_token_block_count(operation);
    if (canonical_workload_phase(workload) == WorkloadPhase::decode) {
        desired = std::max(desired, head_groups);
    } else if (canonical_workload_phase(workload) == WorkloadPhase::prefill) {
        desired = std::max(desired, std::max(head_groups, token_blocks));
    }
    if (config.participating_devices.size() > 1u && operation.attention_head_count > 0u) {
        desired = std::max<std::uint32_t>(desired, static_cast<std::uint32_t>(config.participating_devices.size()));
    }
    return std::clamp(desired, 1u, 8u);
}

std::string tuning_signature_suffix(const ExecutionTuningOverrides* tuning_overrides) {
    if (tuning_overrides == nullptr) {
        return "rewrite=1|passes=auto|budget=auto|validation=adaptive|state=auto";
    }

    std::ostringstream stream;
    stream << "rewrite=" << std::max(tuning_overrides->graph_rewrite_level, 1u) << '|';
    if (tuning_overrides->graph_optimization_passes_override.has_value()) {
        stream << "passes=" << *tuning_overrides->graph_optimization_passes_override;
    } else {
        stream << "passes=auto";
    }
    stream << '|'
           << "budget=";
    if (tuning_overrides->optimizer_wall_time_budget_ms.has_value()) {
        stream << *tuning_overrides->optimizer_wall_time_budget_ms << "ms";
    } else {
        stream << "auto";
    }
    stream << '|'
           << "validation=" << to_string(tuning_overrides->validation_tier) << '|';
    if (!tuning_overrides->initial_state_override.has_value()) {
        stream << "state=auto";
        return stream.str();
    }

    const auto& state = *tuning_overrides->initial_state_override;
    stream << std::fixed << std::setprecision(2)
           << "state=" << state.queue_depth_raw << ','
           << state.stage_raw << ','
           << state.tile_raw << ','
           << state.overlap_raw << ','
           << state.partition_raw << ','
           << state.precision_raw << ','
           << state.single_device_logit << ','
           << state.sharded_logit << ','
           << state.streaming_logit << ','
           << state.overlapped_logit;
    return stream.str();
}

std::string build_config_signature(const std::string& report_signature, const ExecutionConfig& config) {
    std::ostringstream stream;
    stream << report_signature << '|'
           << config.operation_name << '|'
           << config.variant_id << '|'
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

std::uint32_t choose_matmul_tile_k_from_scratch(
    const std::uint32_t tile_x,
    const std::uint32_t tile_y,
    const std::uint64_t scratch_bytes,
    const std::uint32_t base_tile_k) {
    if (scratch_bytes == 0u) {
        return base_tile_k;
    }
    const std::uint64_t scratch_floats = scratch_bytes / sizeof(float);
    if (scratch_floats == 0u) {
        return base_tile_k;
    }
    const std::array<std::uint32_t, 9> candidates{256u, 192u, 160u, 128u, 96u, 64u, 48u, 32u, 16u};
    for (const auto candidate : candidates) {
        const std::uint64_t required =
            (static_cast<std::uint64_t>(tile_x) * candidate) +
            (static_cast<std::uint64_t>(tile_y) * candidate);
        if ((required * 5u) <= (scratch_floats * 4u)) {
            return std::max(candidate, 16u);
        }
    }
    return std::max<std::uint32_t>(16u, std::min(base_tile_k, 32u));
}

std::uint32_t choose_matmul_outer_tile_from_cache(
    const std::uint32_t base_tile,
    const std::uint64_t cache_bytes,
    const std::uint32_t tile_k,
    const std::uint32_t max_value) {
    if (cache_bytes == 0u) {
        return base_tile;
    }
    const std::uint64_t cache_floats = cache_bytes / sizeof(float);
    if (cache_floats == 0u) {
        return base_tile;
    }
    const std::array<std::uint32_t, 8> candidates{256u, 224u, 192u, 160u, 128u, 96u, 64u, 32u};
    for (const auto candidate : candidates) {
        const std::uint64_t required =
            (static_cast<std::uint64_t>(candidate) * tile_k) +
            (static_cast<std::uint64_t>(candidate) * tile_k) +
            (static_cast<std::uint64_t>(candidate) * candidate);
        if ((required * 2u) <= cache_floats) {
            return std::clamp(candidate, base_tile, max_value);
        }
    }
    return base_tile;
}

std::uint32_t choose_spatial_tile_from_local_memory(
    const std::uint32_t base_tile,
    const std::uint64_t scratch_bytes,
    const std::uint32_t min_value,
    const std::uint32_t max_value) {
    if (scratch_bytes == 0u) {
        return base_tile;
    }
    const std::uint64_t scratch_floats = scratch_bytes / sizeof(float);
    const std::array<std::uint32_t, 6> candidates{128u, 96u, 64u, 48u, 32u, 16u};
    for (const auto candidate : candidates) {
        const std::uint64_t required = static_cast<std::uint64_t>(candidate) * candidate * 3u;
        if ((required * 4u) <= (scratch_floats * 3u)) {
            return std::clamp(candidate, min_value, max_value);
        }
    }
    return std::clamp(base_tile, min_value, max_value);
}

bool supports_low_precision(const HardwareGraphSummary& summary);

bool prefer_accelerator_lowering(const HardwareGraphSummary& summary) {
    return summary.matrix_units > 0u ||
           (!summary.host_visible && summary.execution_objects > 0u) ||
           summary.local_scratch_bytes >= (32ull * kKiB);
}

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
    const bool accelerator_lowering = prefer_accelerator_lowering(summary);
    const bool vectorized = operation.cpu_vectorized;
    const bool tensorized = operation.gpu_tensorized;
    const bool pack_weights = accelerator_lowering ? operation.gpu_pack_weights : operation.cpu_pack_weights;
    const bool pretranspose_rhs = accelerator_lowering ? operation.gpu_pretranspose_rhs : operation.cpu_pretranspose_rhs;
    const std::uint32_t micro_kernel_unroll =
        std::max(
            accelerator_lowering ? operation.gpu_micro_kernel_unroll : operation.cpu_micro_kernel_unroll,
            1u);

    switch (operation.op_class) {
    case OperationClass::matmul:
        config.tile_x = accelerator_lowering
            ? std::clamp(round_up_to_multiple(lanes * 8u, 16u), 32u, 128u)
            : std::max(operation.cpu_tile_m, 16u);
        config.tile_y = accelerator_lowering
            ? config.tile_x
            : std::max(operation.cpu_tile_n, 16u);
        config.tile_k = accelerator_lowering
            ? std::clamp(round_up_to_multiple(vector_width * 8u, 8u), 16u, 128u)
            : std::max(operation.cpu_tile_k, 16u);
        if (vectorized) {
            config.tile_x = std::clamp(config.tile_x + 16u, 32u, 192u);
            config.tile_y = std::clamp(config.tile_y + 16u, 32u, 192u);
        }
        if (summary.matrix_units > 0u) {
            config.tile_x = std::clamp(config.tile_x + 32u, 32u, 192u);
            config.tile_y = std::clamp(config.tile_y + 32u, 32u, 192u);
        }
        if (tensorized) {
            config.tile_x = std::clamp(config.tile_x + 32u, 32u, 256u);
            config.tile_y = std::clamp(config.tile_y + 16u, 32u, 256u);
        }
        config.tile_k = choose_matmul_tile_k_from_scratch(
            config.tile_x,
            config.tile_y,
            summary.local_scratch_bytes,
            config.tile_k);
        config.tile_x = choose_matmul_outer_tile_from_cache(
            config.tile_x,
            summary.cache_bytes,
            config.tile_k,
            256u);
        config.tile_y = choose_matmul_outer_tile_from_cache(
            config.tile_y,
            summary.cache_bytes,
            config.tile_k,
            256u);
        if (pack_weights || pretranspose_rhs) {
            config.tile_k = std::clamp(
                round_up_to_multiple(std::max(config.tile_k, 32u) + (tensorized ? 32u : 16u), 16u),
                16u,
                256u);
        }
        if (!operation.fused_operation_names.empty()) {
            config.tile_x = std::clamp(config.tile_x + 16u, 32u, 256u);
        }
        break;
    case OperationClass::convolution_2d:
    case OperationClass::resample_2d:
        config.tile_x = std::clamp(round_down_to_multiple(lanes * 4u, 8u), 16u, 64u);
        config.tile_y = std::clamp(round_down_to_multiple(lanes * 2u, 8u), 8u, 32u);
        config.tile_k = vector_width * 8u;
        config.tile_x = choose_spatial_tile_from_local_memory(
            config.tile_x,
            summary.local_scratch_bytes,
            16u,
            128u);
        config.tile_y = choose_spatial_tile_from_local_memory(
            config.tile_y,
            summary.local_scratch_bytes,
            8u,
            96u);
        if (tensorized) {
            config.tile_x = std::clamp(config.tile_x + 16u, 16u, 128u);
            config.tile_y = std::clamp(config.tile_y + 8u, 8u, 96u);
        }
        break;
    case OperationClass::reduction:
    case OperationClass::elementwise_map:
    default:
        config.tile_x = std::clamp(round_up_to_multiple(lanes * 32u, 32u), 128u, 4096u);
        config.tile_y = 1;
        config.tile_k = vector_width * 4u;
        if (summary.cache_bytes > 0u) {
            const std::uint64_t cache_lines = std::max<std::uint64_t>(summary.cache_bytes / (64u * 1024u), 1u);
            config.tile_x = std::clamp(
                round_up_to_multiple(static_cast<std::uint32_t>(cache_lines * 128u), 64u),
                128u,
                4096u);
        }
        if (vectorized || tensorized) {
            config.tile_x = std::clamp(config.tile_x + 128u, 128u, 4096u);
        }
        break;
    }

    if (micro_kernel_unroll > 1u) {
        config.tile_k = std::clamp(
            round_up_to_multiple(config.tile_k * micro_kernel_unroll, 8u),
            4u,
            1024u);
    }
}

bool supports_low_precision(const HardwareGraphSummary& summary) {
    return summary.supports_fp16 || summary.supports_bf16 || summary.supports_int8;
}

bool is_host_graph(const HardwareGraph* graph) {
    return graph != nullptr && graph->probe == "host";
}

bool is_host_device_uid(const std::string_view uid) {
    return uid == "host" || uid.rfind("host:", 0) == 0;
}

bool is_llm_cpu_exploration_workload(const WorkloadSpec& workload) {
    return workload.dataset_tag.rfind("llm-", 0) == 0;
}

bool is_small_matmul(const OperationSpec& operation) {
    if (operation.op_class != OperationClass::matmul || operation.extents.size() < 3) {
        return false;
    }
    const auto m = operation.extents[0];
    const auto n = operation.extents[1];
    const auto k = operation.extents[2];
    return (m * n * k) <= 2'500'000ull;
}

bool should_prefer_host_for_llm_operation(const WorkloadSpec& workload, const OperationSpec& operation) {
    if (!is_llm_cpu_exploration_workload(workload)) {
        return false;
    }

    if (workload.dataset_tag == "llm-prefill-context-lite") {
        return operation.op_class == OperationClass::elementwise_map ||
               operation.op_class == OperationClass::reduction;
    }

    if (workload.dataset_tag == "llm-decode-token-lite") {
        return operation.op_class == OperationClass::elementwise_map ||
               operation.op_class == OperationClass::reduction ||
               is_small_matmul(operation);
    }

    if (workload.dataset_tag == "llm-kv-cache-update-lite") {
        return true;
    }

    if (workload.dataset_tag == "llm-int4-dequant-lite") {
        return operation.op_class == OperationClass::elementwise_map ||
               operation.op_class == OperationClass::reduction;
    }

    return false;
}

bool should_prefer_accelerator_for_llm_operation(const WorkloadSpec& workload, const OperationSpec& operation) {
    if (!is_llm_cpu_exploration_workload(workload)) {
        return false;
    }

    if (workload.dataset_tag == "llm-prefill-context-lite") {
        return operation.op_class == OperationClass::matmul;
    }

    if (workload.dataset_tag == "llm-decode-token-lite") {
        return operation.op_class == OperationClass::matmul && !is_small_matmul(operation);
    }

    if (workload.dataset_tag == "llm-int4-dequant-lite") {
        return operation.op_class == OperationClass::matmul;
    }

    return false;
}

bool should_prefer_host_for_phase_operation(const WorkloadSpec& workload, const OperationSpec& operation) {
    const std::string_view name = operation.name;
    switch (canonical_workload_phase(workload)) {
    case WorkloadPhase::decode:
        return operation.op_class == OperationClass::elementwise_map ||
               operation.op_class == OperationClass::reduction ||
               name.find("sample") != std::string_view::npos ||
               name.find("kv") != std::string_view::npos ||
               name.find("cache") != std::string_view::npos ||
               (operation.op_class == OperationClass::matmul && is_small_matmul(operation));
    case WorkloadPhase::cache_update:
    case WorkloadPhase::dequantize:
        return operation.op_class != OperationClass::matmul &&
               operation.op_class != OperationClass::convolution_2d;
    case WorkloadPhase::prefill:
        return (operation.op_class == OperationClass::elementwise_map ||
                operation.op_class == OperationClass::reduction) &&
               operation_working_set_bytes(operation) <= static_cast<double>(16ull * kMiB);
    case WorkloadPhase::training_step:
    case WorkloadPhase::unknown:
    default:
        return false;
    }
}

bool should_prefer_accelerator_for_phase_operation(const WorkloadSpec& workload, const OperationSpec& operation) {
    switch (canonical_workload_phase(workload)) {
    case WorkloadPhase::prefill:
        return operation.op_class == OperationClass::matmul ||
               operation.op_class == OperationClass::convolution_2d ||
               (operation.op_class == OperationClass::resample_2d &&
                operation.output_bytes >= (4ull * kMiB));
    case WorkloadPhase::decode:
        return (operation.op_class == OperationClass::matmul && !is_small_matmul(operation)) ||
               operation.op_class == OperationClass::convolution_2d;
    case WorkloadPhase::training_step:
        return operation.matrix_friendly || operation.op_class == OperationClass::convolution_2d;
    case WorkloadPhase::cache_update:
    case WorkloadPhase::dequantize:
    case WorkloadPhase::unknown:
    default:
        return false;
    }
}

enum class StructuralPlacementPreference {
    neutral,
    host,
    accelerator
};

struct StructuralPlacementAffinity {
    StructuralPlacementPreference preferred = StructuralPlacementPreference::neutral;
    bool avoid_sharding = false;
    bool prefer_low_overhead = false;
    double bias_scale = 0.0;
};

bool is_dispatch_bound_operation(const OperationSpec& operation) {
    if (operation.op_class == OperationClass::elementwise_map ||
        operation.op_class == OperationClass::reduction) {
        return true;
    }
    if (operation.op_class == OperationClass::matmul) {
        return is_small_matmul(operation);
    }

    const double working_set_bytes = operation_working_set_bytes(operation);
    const double compute_intensity = operation_compute_intensity(operation);
    return working_set_bytes <= static_cast<double>(8ull * kMiB) && compute_intensity < 24.0;
}

bool is_large_accelerator_operation(const OperationSpec& operation) {
    const double working_set_bytes = operation_working_set_bytes(operation);
    const double compute_intensity = operation_compute_intensity(operation);
    switch (operation.op_class) {
    case OperationClass::matmul:
        return !is_small_matmul(operation) && compute_intensity >= 24.0;
    case OperationClass::convolution_2d:
        return working_set_bytes >= static_cast<double>(12ull * kMiB);
    case OperationClass::resample_2d:
        return operation.output_bytes >= (8ull * kMiB) || operation.streaming_friendly;
    case OperationClass::elementwise_map:
    case OperationClass::reduction:
    default:
        return false;
    }
}

StructuralPlacementAffinity structural_placement_affinity(
    const WorkloadSpec& workload,
    const OperationSpec& operation) {
    if (is_llm_cpu_exploration_workload(workload)) {
        return {};
    }

    if (workload.kind == WorkloadKind::tensor &&
        operation.op_class == OperationClass::matmul &&
        !workload.latency_sensitive &&
        workload.batch_size <= 4u &&
        workload_host_exchange_ratio(workload) >= 0.20) {
        return {
            StructuralPlacementPreference::host,
            true,
            true,
            0.90};
    }

    const bool phase_host_preferred = should_prefer_host_for_phase_operation(workload, operation);
    const bool phase_accelerator_preferred =
        should_prefer_accelerator_for_phase_operation(workload, operation);

    const double exchange_ratio = workload_host_exchange_ratio(workload);
    const bool dispatch_bound = is_dispatch_bound_operation(operation);
    const bool large_accelerator_op = is_large_accelerator_operation(operation);
    const double latency_pressure =
        (workload.latency_sensitive ? (workload.batch_size <= 1u ? 0.55 : 0.25) : 0.0) +
        std::clamp(exchange_ratio * 0.40, 0.0, 0.30);

    if (phase_host_preferred && !phase_accelerator_preferred) {
        return {
            StructuralPlacementPreference::host,
            true,
            true,
            std::clamp(0.55 + latency_pressure + (dispatch_bound ? 0.15 : 0.0), 0.0, 1.0)};
    }

    if (phase_accelerator_preferred && !phase_host_preferred) {
        return {
            StructuralPlacementPreference::accelerator,
            canonical_workload_phase(workload) == WorkloadPhase::decode,
            workload.latency_sensitive,
            std::clamp(
                0.60 + (operation.matrix_friendly ? 0.15 : 0.0) +
                    (canonical_workload_phase(workload) == WorkloadPhase::prefill ? 0.10 : 0.0),
                0.0,
                1.0)};
    }

    if ((dispatch_bound || operation.op_class == OperationClass::reduction) &&
        (latency_pressure > 0.0 || !workload.matrix_friendly || exchange_ratio > 0.10)) {
        return {
            StructuralPlacementPreference::host,
            true,
            true,
            std::clamp(0.45 + latency_pressure + (workload.matrix_friendly ? 0.0 : 0.15), 0.0, 1.0)};
    }

    if (large_accelerator_op && !(dispatch_bound && latency_pressure > 0.40)) {
        return {
            StructuralPlacementPreference::accelerator,
            workload.latency_sensitive && workload.batch_size <= 1u,
            workload.latency_sensitive,
            std::clamp(
                (operation.matrix_friendly ? 0.60 : 0.35) +
                    (operation_compute_intensity(operation) >= 64.0 ? 0.15 : 0.0) +
                    (workload.matrix_friendly ? 0.10 : 0.0),
                0.0,
                1.0)};
    }

    return {
        StructuralPlacementPreference::neutral,
        dispatch_bound || latency_pressure > 0.35,
        dispatch_bound || (workload.latency_sensitive && workload.batch_size <= 1u),
        dispatch_bound ? 0.30 : 0.0};
}

struct OperationPlacementProfile {
    double host_device_bias = 0.0;
    double accelerator_device_bias = 0.0;
    double host_execution_bias_us = 0.0;
    double accelerator_execution_bias_us = 0.0;
    double mixed_execution_bias_us = 0.0;
    std::size_t preferred_max_devices = 2u;
    bool avoid_mixed = false;
};

OperationPlacementProfile score_operation_placement(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const StructuralPlacementAffinity& affinity) {
    OperationPlacementProfile profile;
    const auto phase = canonical_workload_phase(workload);
    const bool host_phase_preferred =
        should_prefer_host_for_llm_operation(workload, operation) ||
        should_prefer_host_for_phase_operation(workload, operation);
    const bool accelerator_phase_preferred =
        should_prefer_accelerator_for_llm_operation(workload, operation) ||
        should_prefer_accelerator_for_phase_operation(workload, operation);

    if (host_phase_preferred && !accelerator_phase_preferred) {
        profile.host_device_bias += phase == WorkloadPhase::decode ? 0.55 : 0.35;
        profile.host_execution_bias_us -= operation.op_class == OperationClass::matmul ? 420.0 : 560.0;
        profile.accelerator_execution_bias_us += operation.op_class == OperationClass::matmul ? 620.0 : 760.0;
        profile.mixed_execution_bias_us += phase == WorkloadPhase::decode ? 420.0 : 260.0;
        profile.preferred_max_devices = 1u;
        profile.avoid_mixed = true;
    } else if (accelerator_phase_preferred && !host_phase_preferred) {
        profile.accelerator_device_bias += phase == WorkloadPhase::prefill ? 0.45 : 0.35;
        profile.host_execution_bias_us += operation.op_class == OperationClass::matmul ? 760.0 : 320.0;
        profile.accelerator_execution_bias_us -= operation.op_class == OperationClass::matmul ? 420.0 : 120.0;
        profile.mixed_execution_bias_us += canonical_workload_phase(workload) == WorkloadPhase::decode ? 120.0 : 40.0;
        profile.preferred_max_devices =
            phase == WorkloadPhase::prefill && operation.parallelizable && operation.matrix_friendly ? 2u : 1u;
    }

    if (affinity.preferred == StructuralPlacementPreference::host) {
        profile.host_device_bias += 0.20 * (1.0 + affinity.bias_scale);
        profile.host_execution_bias_us -= 90.0 * (1.0 + affinity.bias_scale);
        profile.accelerator_execution_bias_us += 120.0 * (1.0 + affinity.bias_scale);
        if (affinity.prefer_low_overhead) {
            profile.mixed_execution_bias_us += 80.0 * (1.0 + affinity.bias_scale);
        }
        if (affinity.avoid_sharding) {
            profile.preferred_max_devices = 1u;
            profile.avoid_mixed = true;
        }
    } else if (affinity.preferred == StructuralPlacementPreference::accelerator) {
        profile.accelerator_device_bias += 0.20 * (1.0 + affinity.bias_scale);
        profile.host_execution_bias_us += 130.0 * (1.0 + affinity.bias_scale);
        profile.accelerator_execution_bias_us -= 55.0 * (1.0 + affinity.bias_scale);
        if (affinity.avoid_sharding) {
            profile.preferred_max_devices = std::min<std::size_t>(profile.preferred_max_devices, 1u);
        }
    }

    if (phase == WorkloadPhase::decode && workload.latency_sensitive) {
        profile.preferred_max_devices = 1u;
        profile.avoid_mixed = true;
        profile.mixed_execution_bias_us += 120.0;
    } else if (phase == WorkloadPhase::prefill && operation.matrix_friendly && !is_small_matmul(operation)) {
        profile.preferred_max_devices = std::max<std::size_t>(profile.preferred_max_devices, 2u);
        profile.mixed_execution_bias_us -= 20.0;
    }

    if (operation.op_class == OperationClass::reduction || operation.op_class == OperationClass::elementwise_map) {
        profile.preferred_max_devices = 1u;
    }

    if (operation.preferred_kv_residency == "host") {
        profile.host_device_bias += 0.18;
        profile.host_execution_bias_us -= 140.0;
        profile.accelerator_execution_bias_us += 180.0;
        profile.preferred_max_devices = 1u;
    } else if (operation.preferred_kv_residency == "accelerator") {
        profile.accelerator_device_bias += 0.12;
        profile.host_execution_bias_us += 120.0;
        profile.accelerator_execution_bias_us -= 60.0;
    } else if (operation.preferred_kv_residency == "shared" && phase == WorkloadPhase::decode) {
        profile.mixed_execution_bias_us -= 45.0;
        profile.preferred_max_devices = std::max<std::size_t>(profile.preferred_max_devices, 2u);
    }

    if (operation.attention_head_count >= 16u) {
        if (phase == WorkloadPhase::decode) {
            profile.host_device_bias += operation.attention_head_group_size <= 4u ? 0.10 : 0.04;
            profile.accelerator_device_bias += 0.08;
        } else if (phase == WorkloadPhase::prefill) {
            profile.accelerator_device_bias += 0.14;
            profile.mixed_execution_bias_us -= operation.preferred_token_block >= 64u ? 30.0 : 10.0;
        }
    }

    return profile;
}

double graph_operation_device_bias(
    const OperationPlacementProfile& profile,
    const HardwareGraph* graph) {
    return is_host_graph(graph) ? profile.host_device_bias : profile.accelerator_device_bias;
}

double operation_placement_bias_us(
    const OperationPlacementProfile& profile,
    const ExecutionConfig& config,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    const auto* primary_graph = find_graph(graph_lookup, config.primary_device_uid);
    const bool host_only = config.participating_devices.size() == 1u && is_host_graph(primary_graph);
    const bool accelerator_only =
        config.participating_devices.size() == 1u && primary_graph != nullptr && !is_host_graph(primary_graph);
    const bool mixed_devices = config.participating_devices.size() > 1u;

    if (host_only) {
        return profile.host_execution_bias_us;
    }
    if (accelerator_only) {
        return profile.accelerator_execution_bias_us;
    }
    if (mixed_devices) {
        return profile.mixed_execution_bias_us;
    }
    return 0.0;
}

std::vector<std::string> accelerator_device_uids(
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    std::vector<std::string> uids;
    for (const auto& allocation : placement.allocations) {
        const auto* graph = find_graph(graph_lookup, allocation.device.uid);
        if (!is_host_graph(graph)) {
            uids.push_back(allocation.device.uid);
        }
    }
    return uids;
}

std::vector<std::size_t> active_continuous_dimensions(
    const WorkloadSpec& workload,
    const std::vector<OperationSpec>& operations,
    const ExecutionPlan& placement,
    const SystemProfile& system) {
    bool has_parallelizable = false;
    bool has_streaming = false;
    bool has_matrix = false;
    bool tolerates_low_precision = false;
    for (const auto& operation : operations) {
        has_parallelizable = has_parallelizable || operation.parallelizable;
        has_streaming = has_streaming || operation.streaming_friendly;
        has_matrix = has_matrix || operation.matrix_friendly;
        tolerates_low_precision = tolerates_low_precision || operation.max_relative_error >= 4.0e-4;
    }

    std::vector<std::size_t> dims{0u, 1u, 2u, 3u, 6u, 9u};
    if (has_parallelizable) {
        dims.push_back(4u);
    }
    if (tolerates_low_precision) {
        dims.push_back(5u);
    }
    if (placement.allocations.size() > 1u && !workload.latency_sensitive) {
        dims.push_back(7u);
    }
    if (has_streaming && (!system.low_spec_mode || workload.kind == WorkloadKind::image)) {
        dims.push_back(8u);
    }
    if (!has_matrix) {
        dims.erase(std::remove(dims.begin(), dims.end(), 2u), dims.end());
    }
    return dims;
}

std::uint32_t choose_graph_optimization_passes(
    const WorkloadSpec& workload,
    const std::vector<OperationSpec>& operations,
    const ExecutionPlan& placement,
    const SystemProfile& system) {
    if (operations.size() <= 2u) {
        return 1u;
    }
    if (system.low_spec_mode || workload.latency_sensitive) {
        return 2u;
    }
    if (placement.allocations.size() <= 1u && operations.size() <= 4u) {
        return 2u;
    }
    if (operations.size() <= 4u) {
        return 3u;
    }
    return kGraphOptimizationPasses;
}

bool should_auto_accelerator_data_parallel(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    const auto accelerators = accelerator_device_uids(placement, graph_lookup);
    if (accelerators.size() < 2u) {
        return false;
    }
    if (!operation.parallelizable) {
        return false;
    }
    if (workload.latency_sensitive && workload.batch_size <= 1u) {
        return false;
    }
    if (operation.op_class == OperationClass::matmul) {
        return true;
    }
    if (operation.op_class == OperationClass::convolution_2d) {
        return workload.batch_size > 1u || workload.kind == WorkloadKind::training;
    }
    return false;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool operation_name_contains_any(const std::string& name, std::initializer_list<const char*> needles) {
    return std::any_of(needles.begin(), needles.end(), [&](const char* needle) {
        return name.find(needle) != std::string::npos;
    });
}

enum class PartitionMicroStage {
    norm_gate,
    kv_cache,
    reduction,
    projection,
    sampling
};

enum class PartitionPlacementMode {
    host_only,
    accelerator_only,
    sharded
};

struct PartitionPlacement {
    PartitionPlacementMode mode = PartitionPlacementMode::host_only;
    ExecutionStrategy strategy = ExecutionStrategy::single_device;
    std::uint32_t logical_partitions = 1u;
    bool overlap_transfers = false;
    std::uint32_t min_queue_depth = 1u;
    std::uint32_t min_stages = 1u;
};

struct PartitionRuntimeStrategy {
    PartitionPlacement norm_gate;
    PartitionPlacement kv_cache;
    PartitionPlacement reduction;
    PartitionPlacement projection;
    PartitionPlacement sampling;
};

PartitionMicroStage classify_partition_micro_stage(const OperationSpec& operation) {
    const auto name = lowercase(operation.name);
    if (operation_name_contains_any(name, {"sample"})) {
        return PartitionMicroStage::sampling;
    }
    if (operation_name_contains_any(name, {"qkv", "context", "proj", "mlp-up", "mlp-down", "logits"}) ||
        operation.op_class == OperationClass::matmul ||
        operation.op_class == OperationClass::convolution_2d) {
        return PartitionMicroStage::projection;
    }
    if (operation_name_contains_any(name, {"kv", "cache", "append", "scan", "evict", "writeback"})) {
        return PartitionMicroStage::kv_cache;
    }
    if (operation_name_contains_any(name, {"reduce", "pool"}) || operation.op_class == OperationClass::reduction) {
        return PartitionMicroStage::reduction;
    }
    return PartitionMicroStage::norm_gate;
}

const PartitionRuntimeStrategy* runtime_partition_strategy(const PartitionStrategy strategy) {
    static const PartitionRuntimeStrategy kBlindSharded{
        {PartitionPlacementMode::sharded, ExecutionStrategy::sharded, 2u, true, 2u, 2u},
        {PartitionPlacementMode::sharded, ExecutionStrategy::sharded, 2u, true, 2u, 2u},
        {PartitionPlacementMode::sharded, ExecutionStrategy::sharded, 2u, true, 2u, 2u},
        {PartitionPlacementMode::sharded, ExecutionStrategy::sharded, 2u, true, 2u, 2u},
        {PartitionPlacementMode::sharded, ExecutionStrategy::sharded, 2u, true, 2u, 2u}};
    static const PartitionRuntimeStrategy kRoleSplit{
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u},
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u},
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u},
        {PartitionPlacementMode::accelerator_only, ExecutionStrategy::single_device, 2u, false, 2u, 2u},
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u}};
    static const PartitionRuntimeStrategy kReduceOnGpu{
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u},
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u},
        {PartitionPlacementMode::accelerator_only, ExecutionStrategy::single_device, 2u, false, 2u, 2u},
        {PartitionPlacementMode::accelerator_only, ExecutionStrategy::single_device, 2u, false, 2u, 2u},
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u}};
    static const PartitionRuntimeStrategy kProjectionSharded{
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u},
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u},
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u},
        {PartitionPlacementMode::sharded, ExecutionStrategy::sharded, 4u, true, 2u, 2u},
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u}};
    static const PartitionRuntimeStrategy kTpuLike{
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u},
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u},
        {PartitionPlacementMode::accelerator_only, ExecutionStrategy::overlapped, 2u, true, 2u, 2u},
        {PartitionPlacementMode::accelerator_only, ExecutionStrategy::overlapped, 4u, true, 3u, 3u},
        {PartitionPlacementMode::host_only, ExecutionStrategy::single_device, 1u, false, 1u, 1u}};

    switch (strategy) {
    case PartitionStrategy::blind_sharded:
        return &kBlindSharded;
    case PartitionStrategy::role_split:
        return &kRoleSplit;
    case PartitionStrategy::reduce_on_gpu:
        return &kReduceOnGpu;
    case PartitionStrategy::projection_sharded:
        return &kProjectionSharded;
    case PartitionStrategy::tpu_like:
        return &kTpuLike;
    case PartitionStrategy::auto_balanced:
    default:
        return nullptr;
    }
}

const PartitionPlacement& placement_for_micro_stage(
    const PartitionRuntimeStrategy& strategy,
    const PartitionMicroStage stage) {
    switch (stage) {
    case PartitionMicroStage::kv_cache:
        return strategy.kv_cache;
    case PartitionMicroStage::reduction:
        return strategy.reduction;
    case PartitionMicroStage::projection:
        return strategy.projection;
    case PartitionMicroStage::sampling:
        return strategy.sampling;
    case PartitionMicroStage::norm_gate:
    default:
        return strategy.norm_gate;
    }
}

bool apply_partition_strategy_execution_policy(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    ExecutionConfig& config) {
    const auto requested_strategy =
        placement.strategy_source == PlanStrategySource::explicit_request
            ? placement.resolved_partition_strategy
            : ((workload.heuristic_partition_hint_confidence >= 0.35 &&
                workload.heuristic_partition_hint.has_value())
                   ? *workload.heuristic_partition_hint
                   : PartitionStrategy::auto_balanced);
    const auto* strategy = runtime_partition_strategy(requested_strategy);
    if (strategy == nullptr) {
        return false;
    }

    const auto host_it = std::find_if(placement.allocations.begin(), placement.allocations.end(), [&](const PlanAllocation& allocation) {
        const auto* graph = find_graph(graph_lookup, allocation.device.uid);
        return is_host_graph(graph);
    });
    const auto accelerator_it =
        std::find_if(placement.allocations.begin(), placement.allocations.end(), [&](const PlanAllocation& allocation) {
            const auto* graph = find_graph(graph_lookup, allocation.device.uid);
            return graph != nullptr && !is_host_graph(graph);
        });
    if (host_it == placement.allocations.end() || accelerator_it == placement.allocations.end()) {
        return false;
    }

    const auto stage = classify_partition_micro_stage(operation);
    const auto& forced = placement_for_micro_stage(*strategy, stage);
    const bool hard_force = placement.strategy_source == PlanStrategySource::explicit_request;
    const auto force_single_device = [&](const std::string& device_uid) {
        config.primary_device_uid = device_uid;
        config.participating_devices = {device_uid};
        config.strategy = forced.strategy == ExecutionStrategy::sharded ? ExecutionStrategy::single_device : forced.strategy;
        config.logical_partitions = 1u;
    };

    if (!hard_force) {
        config.queue_depth = std::max(config.queue_depth, forced.min_queue_depth);
        config.stages = std::max(config.stages, forced.min_stages);
        if (forced.mode == PartitionPlacementMode::sharded &&
            operation.parallelizable &&
            config.participating_devices.size() > 1u) {
            config.strategy = forced.strategy;
            config.logical_partitions = std::max(config.logical_partitions, forced.logical_partitions);
            config.overlap_transfers = config.overlap_transfers || forced.overlap_transfers;
        } else if (config.participating_devices.size() <= 1u) {
            config.strategy = ExecutionStrategy::single_device;
            config.logical_partitions = 1u;
            if (forced.mode == PartitionPlacementMode::host_only) {
                config.overlap_transfers = false;
            } else if (forced.mode == PartitionPlacementMode::accelerator_only) {
                config.overlap_transfers = config.overlap_transfers || forced.overlap_transfers;
            }
        }
        return true;
    }

    if (forced.mode == PartitionPlacementMode::host_only) {
        force_single_device(host_it->device.uid);
    } else if (forced.mode == PartitionPlacementMode::accelerator_only) {
        force_single_device(accelerator_it->device.uid);
        config.logical_partitions = operation.parallelizable ? std::max(forced.logical_partitions, 1u) : 1u;
    } else if (operation.parallelizable) {
        config.primary_device_uid = accelerator_it->device.uid;
        config.participating_devices = {host_it->device.uid, accelerator_it->device.uid};
        config.strategy = forced.strategy;
        config.logical_partitions = std::max(forced.logical_partitions, 1u);
    } else {
        force_single_device(accelerator_it->device.uid);
    }

    const auto* primary_graph = find_graph(graph_lookup, config.primary_device_uid);
    config.overlap_transfers = forced.overlap_transfers;
    if (forced.strategy == ExecutionStrategy::overlapped &&
        primary_graph != nullptr &&
        !summarize_graph(*primary_graph).supports_asynchronous_dispatch) {
        config.strategy = ExecutionStrategy::single_device;
        config.overlap_transfers = false;
    }

    config.queue_depth = std::max(config.queue_depth, forced.min_queue_depth);
    config.stages = std::max(config.stages, forced.min_stages);
    if (config.strategy == ExecutionStrategy::single_device && config.participating_devices.size() == 1u) {
        config.logical_partitions = std::max(config.logical_partitions, 1u);
    }
    return true;
}

void apply_llm_cpu_execution_policy(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const HardwareGraph* graph,
    ExecutionConfig& config) {
    if (!is_llm_cpu_exploration_workload(workload)) {
        return;
    }

    const bool host_preferred = should_prefer_host_for_llm_operation(workload, operation);
    const bool accelerator_preferred = should_prefer_accelerator_for_llm_operation(workload, operation);
    const bool host_graph = is_host_graph(graph);

    if (host_preferred && host_graph) {
        config.participating_devices = {config.primary_device_uid};
        config.strategy =
            operation.op_class == OperationClass::elementwise_map || operation.op_class == OperationClass::reduction
                ? ExecutionStrategy::overlapped
                : ExecutionStrategy::single_device;
        config.overlap_transfers = config.strategy == ExecutionStrategy::overlapped;
        config.queue_depth = std::min(config.queue_depth, 2u);
        config.stages = 1u;
        config.logical_partitions = 1u;
        config.tile_y = std::max(config.tile_y, 1u);
        config.tile_k = std::max(config.tile_k, operation.op_class == OperationClass::matmul ? 16u : 4u);
        return;
    }

    if (accelerator_preferred && host_graph) {
        config.strategy = ExecutionStrategy::single_device;
        config.overlap_transfers = false;
        config.logical_partitions = 1u;
        config.queue_depth = 1u;
        config.stages = 1u;
        return;
    }

    if (host_preferred && !host_graph) {
        config.logical_partitions = 1u;
        if (workload.latency_sensitive) {
            config.queue_depth = 1u;
            config.stages = 1u;
        }
        return;
    }

    if (accelerator_preferred && !host_graph) {
        config.logical_partitions = std::max(config.logical_partitions, 1u);
        if (operation.op_class == OperationClass::matmul) {
            config.queue_depth = std::max(config.queue_depth, 2u);
        }
    }
}

void apply_device_agnostic_execution_policy(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const HardwareGraph* graph,
    ExecutionConfig& config) {
    const auto affinity = structural_placement_affinity(workload, operation);
    const bool host_graph = is_host_graph(graph);

    if (affinity.preferred == StructuralPlacementPreference::host) {
        config.participating_devices = {config.primary_device_uid};
        config.strategy = ExecutionStrategy::single_device;
        config.overlap_transfers = false;
        config.logical_partitions = 1u;
        config.queue_depth = std::min(config.queue_depth, workload.latency_sensitive ? 1u : 2u);
        config.stages = 1u;
        if (host_graph) {
            config.tile_y = std::max(config.tile_y, 1u);
            config.tile_k = std::max(config.tile_k, operation.op_class == OperationClass::matmul ? 16u : 4u);
        }
        return;
    }

    if (affinity.preferred == StructuralPlacementPreference::accelerator) {
        if (host_graph) {
            config.strategy = ExecutionStrategy::single_device;
            config.overlap_transfers = false;
            config.logical_partitions = 1u;
            config.queue_depth = 1u;
            config.stages = 1u;
            return;
        }

        if (operation.op_class == OperationClass::matmul || operation.op_class == OperationClass::convolution_2d) {
            config.queue_depth = std::max(config.queue_depth, 2u);
        }
        if (affinity.avoid_sharding && config.strategy == ExecutionStrategy::sharded) {
            config.strategy = ExecutionStrategy::single_device;
            config.participating_devices = {config.primary_device_uid};
            config.logical_partitions = 1u;
            config.overlap_transfers = false;
        }
    }

    if (affinity.avoid_sharding && config.strategy == ExecutionStrategy::sharded) {
        config.strategy = ExecutionStrategy::single_device;
        config.participating_devices = {config.primary_device_uid};
        config.logical_partitions = 1u;
        config.overlap_transfers = false;
    }

    if (affinity.prefer_low_overhead) {
        config.queue_depth = std::min(config.queue_depth, workload.latency_sensitive ? 1u : 2u);
        config.stages = std::min(config.stages, host_graph ? 1u : 2u);
        if (workload.latency_sensitive || affinity.avoid_sharding) {
            config.logical_partitions = 1u;
        }
    }
}

void apply_phase_aware_execution_policy(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const OperationPlacementProfile& profile,
    const HardwareGraph* graph,
    ExecutionConfig& config) {
    const bool host_graph = is_host_graph(graph);
    const auto phase = canonical_workload_phase(workload);
    const bool host_phase_preferred =
        should_prefer_host_for_llm_operation(workload, operation) ||
        should_prefer_host_for_phase_operation(workload, operation);
    const bool accelerator_phase_preferred =
        should_prefer_accelerator_for_llm_operation(workload, operation) ||
        should_prefer_accelerator_for_phase_operation(workload, operation);

    if (profile.avoid_mixed && config.participating_devices.size() > 1u) {
        config.participating_devices = {config.primary_device_uid};
        config.strategy = ExecutionStrategy::single_device;
        config.logical_partitions = 1u;
        config.overlap_transfers = false;
    }

    if (phase == WorkloadPhase::decode) {
        config.queue_depth = std::min(config.queue_depth, 2u);
        config.stages = 1u;
        if (host_graph) {
            config.logical_partitions = 1u;
            config.overlap_transfers =
                host_phase_preferred &&
                (operation.op_class == OperationClass::elementwise_map ||
                 operation.op_class == OperationClass::reduction ||
                 operation.name.find("sample") != std::string::npos ||
                 operation.name.find("kv") != std::string::npos ||
                 operation.name.find("cache") != std::string::npos);
            if (config.overlap_transfers) {
                config.queue_depth = std::max(config.queue_depth, 2u);
            }
        } else if (accelerator_phase_preferred &&
                   (operation.op_class == OperationClass::matmul ||
                    operation.op_class == OperationClass::convolution_2d)) {
            config.queue_depth = std::max(config.queue_depth, 3u);
            config.stages = std::max(config.stages, 2u);
            config.overlap_transfers = true;
        }
        return;
    }

    if (phase == WorkloadPhase::prefill && !host_graph &&
        (operation.op_class == OperationClass::matmul || operation.op_class == OperationClass::convolution_2d)) {
        config.queue_depth = std::max(config.queue_depth, 2u);
        config.stages = std::max(config.stages, 2u);
        if (profile.preferred_max_devices > 1u && operation.parallelizable && config.participating_devices.size() > 1u) {
            config.strategy = ExecutionStrategy::sharded;
            config.overlap_transfers = true;
            config.logical_partitions = std::max(config.logical_partitions, 1u);
        }
        return;
    }

    if (phase == WorkloadPhase::cache_update || phase == WorkloadPhase::dequantize) {
        config.participating_devices = {config.primary_device_uid};
        config.strategy = ExecutionStrategy::single_device;
        config.logical_partitions = 1u;
        config.overlap_transfers = false;
    }
}

void apply_auto_accelerator_data_parallel_policy(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    ExecutionConfig& config) {
    if (!should_auto_accelerator_data_parallel(workload, operation, placement, graph_lookup)) {
        return;
    }

    const auto accelerators = accelerator_device_uids(placement, graph_lookup);
    if (accelerators.size() < 2u) {
        return;
    }

    if (config.strategy == ExecutionStrategy::sharded || config.participating_devices.size() > 1u) {
        config.strategy = ExecutionStrategy::sharded;
        config.participating_devices = accelerators;
        config.primary_device_uid = accelerators.front();
        config.overlap_transfers = true;
        config.logical_partitions = 1u;
        config.queue_depth = std::max(config.queue_depth, 2u);
        config.stages = std::max(config.stages, 2u);
    }
}

void stabilize_level_zero_host_mix_policy(
    const OperationSpec& operation,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    ExecutionConfig& config) {
    if (config.participating_devices.size() < 2u) {
        return;
    }

    const bool host_present = std::any_of(
        config.participating_devices.begin(),
        config.participating_devices.end(),
        [&](const std::string& uid) {
            return is_host_graph(find_graph(graph_lookup, uid));
        });
    if (!host_present) {
        return;
    }

    std::string level_zero_uid;
    for (const auto& uid : config.participating_devices) {
        const auto* graph = find_graph(graph_lookup, uid);
        if (graph != nullptr && graph->probe == "level-zero") {
            level_zero_uid = uid;
            break;
        }
    }
    if (level_zero_uid.empty()) {
        return;
    }

    if (operation.op_class != OperationClass::matmul &&
        operation.op_class != OperationClass::convolution_2d) {
        return;
    }

    config.primary_device_uid = level_zero_uid;
    config.participating_devices = {level_zero_uid};
    config.strategy = ExecutionStrategy::single_device;
    config.logical_partitions = 1u;
    config.overlap_transfers = false;
}

double llm_cpu_policy_bias_us(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const ExecutionConfig& config,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    if (!is_llm_cpu_exploration_workload(workload)) {
        return 0.0;
    }

    const auto* primary_graph = find_graph(graph_lookup, config.primary_device_uid);
    const bool host_only = config.participating_devices.size() == 1 && is_host_graph(primary_graph);
    const bool accelerator_only = config.participating_devices.size() == 1 && !host_only;
    const bool host_preferred = should_prefer_host_for_llm_operation(workload, operation);
    const bool accelerator_preferred = should_prefer_accelerator_for_llm_operation(workload, operation);

    double bias_us = 0.0;
    if (host_preferred) {
        if (host_only) {
            bias_us -= operation.op_class == OperationClass::matmul ? 70.0 : 140.0;
        }
        if (accelerator_only) {
            bias_us += operation.op_class == OperationClass::matmul ? 80.0 : 180.0;
        }
        if (config.strategy == ExecutionStrategy::sharded) {
            bias_us += 140.0;
        }
        if (config.logical_partitions > 1u) {
            bias_us += static_cast<double>(config.logical_partitions - 1u) * 60.0;
        }
    }

    if (accelerator_preferred) {
        if (host_only) {
            bias_us += 120.0;
        }
        if (accelerator_only) {
            bias_us -= 35.0;
        }
    }

    return bias_us;
}

double device_agnostic_execution_bias_us(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const ExecutionConfig& config,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    const auto affinity = structural_placement_affinity(workload, operation);
    if (affinity.preferred == StructuralPlacementPreference::neutral &&
        !affinity.avoid_sharding &&
        !affinity.prefer_low_overhead) {
        return 0.0;
    }

    const auto* primary_graph = find_graph(graph_lookup, config.primary_device_uid);
    const bool host_only = config.participating_devices.size() == 1u && is_host_graph(primary_graph);
    const bool accelerator_only =
        config.participating_devices.size() == 1u && primary_graph != nullptr && !is_host_graph(primary_graph);
    const bool mixed_devices = config.participating_devices.size() > 1u;
    const double scale = std::max(0.15, affinity.bias_scale);

    double bias_us = 0.0;
    if (affinity.preferred == StructuralPlacementPreference::host) {
        if (host_only) {
            bias_us -= 140.0 * (1.0 + scale);
        }
        if (accelerator_only) {
            bias_us += 180.0 * (1.0 + scale);
        }
        if (mixed_devices) {
            bias_us += 150.0 * (1.0 + scale);
        }
    } else if (affinity.preferred == StructuralPlacementPreference::accelerator) {
        if (host_only) {
            bias_us +=
                (operation.op_class == OperationClass::matmul ? 420.0 : 150.0) * (1.0 + scale);
        }
        if (accelerator_only) {
            bias_us -= operation.op_class == OperationClass::matmul ? 220.0 : 45.0;
        }
    }

    if (affinity.avoid_sharding && config.strategy == ExecutionStrategy::sharded) {
        bias_us += 160.0 * (1.0 + scale);
    }
    if (affinity.prefer_low_overhead) {
        if (config.overlap_transfers) {
            bias_us += 35.0;
        }
        if (config.queue_depth > 2u) {
            bias_us += static_cast<double>(config.queue_depth - 2u) * 24.0 * (1.0 + scale);
        }
        if (config.stages > 2u) {
            bias_us += static_cast<double>(config.stages - 2u) * 28.0 * (1.0 + scale);
        }
        if (config.logical_partitions > 1u) {
            bias_us += static_cast<double>(config.logical_partitions - 1u) * 44.0 * (1.0 + scale);
        }
    }

    if (primary_graph != nullptr &&
        operation.op_class == OperationClass::matmul &&
        workload.kind == WorkloadKind::tensor &&
        !workload.latency_sensitive &&
        primary_graph->probe == "level-zero") {
        bias_us += 2600.0;
    }

    return bias_us;
}

double auto_accelerator_data_parallel_bias_us(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const ExecutionPlan& placement,
    const ExecutionConfig& config,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    if (!should_auto_accelerator_data_parallel(workload, operation, placement, graph_lookup)) {
        return 0.0;
    }

    const auto accelerators = accelerator_device_uids(placement, graph_lookup);
    const bool accelerator_only =
        !config.participating_devices.empty() &&
        std::all_of(
            config.participating_devices.begin(),
            config.participating_devices.end(),
            [&](const std::string& uid) {
                const auto* graph = find_graph(graph_lookup, uid);
                return !is_host_graph(graph);
            });
    const bool full_accelerator_set = accelerator_only && config.participating_devices.size() == accelerators.size();

    double bias_us = 0.0;
    if (config.strategy == ExecutionStrategy::sharded && full_accelerator_set) {
        bias_us -= operation.op_class == OperationClass::matmul ? 1800.0 : 1200.0;
    } else if (accelerator_only && config.strategy != ExecutionStrategy::sharded) {
        bias_us += operation.op_class == OperationClass::matmul ? 900.0 : 500.0;
    }

    if (config.variant_id.find("accelerator-ddp") != std::string::npos && full_accelerator_set) {
        bias_us -= operation.op_class == OperationClass::matmul ? 2200.0 : 1400.0;
    }

    if (!accelerator_only) {
        bias_us += operation.op_class == OperationClass::matmul ? 2200.0 : 1400.0;
    }
    if (config.logical_partitions > 1u) {
        bias_us += static_cast<double>(config.logical_partitions - 1u) * 120.0;
    }
    return bias_us;
}

double hybrid_multi_accelerator_host_bias_us(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const ExecutionPlan& placement,
    const ExecutionConfig& config,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    const auto accelerators = accelerator_device_uids(placement, graph_lookup);
    const bool has_host = std::any_of(
        placement.allocations.begin(),
        placement.allocations.end(),
        [&](const PlanAllocation& allocation) {
            return is_host_graph(find_graph(graph_lookup, allocation.device.uid));
        });
    if (!has_host || accelerators.size() < 2u) {
        return 0.0;
    }

    const auto phase = canonical_workload_phase(workload);
    const auto* primary_graph = find_graph(graph_lookup, config.primary_device_uid);
    const bool host_only = config.participating_devices.size() == 1u && is_host_graph(primary_graph);
    const bool accelerator_only =
        !config.participating_devices.empty() &&
        std::all_of(
            config.participating_devices.begin(),
            config.participating_devices.end(),
            [&](const std::string& uid) {
                const auto* graph = find_graph(graph_lookup, uid);
                return graph != nullptr && !is_host_graph(graph);
            });
    const bool full_accelerator_set = accelerator_only && config.participating_devices.size() == accelerators.size();
    const bool mixed_same_op = config.participating_devices.size() > accelerators.size();

    double bias_us = 0.0;
    if (host_only &&
        (operation.op_class == OperationClass::elementwise_map ||
         operation.op_class == OperationClass::reduction ||
         operation_mentions_kv_cache(operation) ||
         string_contains_any(operation.name, {"sample"}))) {
        bias_us -= phase == WorkloadPhase::decode ? 220.0 : 120.0;
    }
    if (full_accelerator_set &&
        (operation.op_class == OperationClass::matmul || operation.op_class == OperationClass::convolution_2d)) {
        bias_us -= phase == WorkloadPhase::prefill ? 260.0 : 150.0;
        if (operation.attention_head_count > 0u) {
            bias_us -= 40.0;
        }
    }
    if (mixed_same_op) {
        bias_us += 180.0;
    }
    return bias_us;
}

struct WorkloadCandidatePolicy {
    std::size_t max_ranked_strategies = 2u;
    std::size_t max_devices = 2u;
    std::size_t max_candidates = 8u;
    std::size_t validation_shortlist = 2u;
};

WorkloadCandidatePolicy candidate_policy_for_workload(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const SystemProfile& system,
    const bool activation_driven,
    const StructuralPlacementAffinity& affinity) {
    WorkloadCandidatePolicy policy;
    if (workload.kind == WorkloadKind::training || workload.kind == WorkloadKind::custom) {
        policy.max_ranked_strategies = 3u;
        policy.max_devices = 3u;
        policy.max_candidates = 12u;
        policy.validation_shortlist = 3u;
        return policy;
    }
    if (workload.kind == WorkloadKind::inference &&
        workload.latency_sensitive &&
        canonical_workload_phase(workload) == WorkloadPhase::decode) {
        policy.max_ranked_strategies = 1u;
        policy.max_devices = 1u;
        policy.max_candidates = operation.matrix_friendly ? 6u : 4u;
        policy.validation_shortlist = 1u;
        return policy;
    }
    if (workload.kind == WorkloadKind::inference &&
        canonical_workload_phase(workload) == WorkloadPhase::prefill) {
        policy.max_ranked_strategies = operation.matrix_friendly ? 2u : 1u;
        policy.max_devices = operation.matrix_friendly ? 2u : 1u;
        policy.max_candidates = operation.matrix_friendly ? 8u : 5u;
        policy.validation_shortlist = operation.matrix_friendly ? 2u : 1u;
        return policy;
    }
    if (workload.kind == WorkloadKind::gaming || workload.kind == WorkloadKind::image) {
        policy.max_ranked_strategies = 1u;
        policy.max_devices = 1u;
        policy.max_candidates = 5u;
        policy.validation_shortlist = 1u;
        return policy;
    }
    if (is_small_dispatch_bound_operation(operation)) {
        policy.max_ranked_strategies = 1u;
        policy.max_devices = 1u;
        policy.max_candidates = operation.fused_operation_names.empty() ? 4u : 3u;
        policy.validation_shortlist = 1u;
        return policy;
    }
    if (system.low_spec_mode || affinity.prefer_low_overhead || (!operation.matrix_friendly && activation_driven)) {
        policy.max_ranked_strategies = 1u;
        policy.max_devices = 1u;
        policy.max_candidates = 5u;
        policy.validation_shortlist = 1u;
        return policy;
    }
    if (operation.matrix_friendly) {
        policy.max_ranked_strategies = 2u;
        policy.max_devices = 2u;
        policy.max_candidates = 8u;
        policy.validation_shortlist = 2u;
    } else {
        policy.max_ranked_strategies = 1u;
        policy.max_devices = 2u;
        policy.max_candidates = 6u;
        policy.validation_shortlist = 1u;
    }
    return policy;
}

CandidatePolicySummary summarize_candidate_policy(const WorkloadCandidatePolicy& policy) {
    return CandidatePolicySummary{
        static_cast<std::uint32_t>(policy.max_ranked_strategies),
        static_cast<std::uint32_t>(policy.max_devices),
        static_cast<std::uint32_t>(policy.max_candidates),
        static_cast<std::uint32_t>(policy.validation_shortlist)};
}

std::vector<ExecutionConfig> build_candidate_configs(
    const OperationSpec& operation,
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const SystemProfile& system,
    const ContinuousExecutionState* continuous_state,
    const std::string& report_signature) {
    struct WeightedStrategy {
        ExecutionStrategy strategy = ExecutionStrategy::single_device;
        double weight = -1.0;
    };

    auto rank_activation_strategies = [&](const bool allow_sharded) {
        std::vector<WeightedStrategy> ranked;
        if (continuous_state == nullptr) {
            return ranked;
        }

        const auto strategy_weights = strategy_soft_weights(*continuous_state);
        ranked = {
            {ExecutionStrategy::single_device, strategy_weights[0]},
            {ExecutionStrategy::sharded, allow_sharded ? strategy_weights[1] : -1.0},
            {ExecutionStrategy::streaming, operation.streaming_friendly ? strategy_weights[2] : -1.0},
            {ExecutionStrategy::overlapped, strategy_weights[3]}};
        std::sort(ranked.begin(), ranked.end(), [](const WeightedStrategy& left, const WeightedStrategy& right) {
            return left.weight > right.weight;
        });
        return ranked;
    };

    const auto placement_affinity = structural_placement_affinity(workload, operation);
    const auto placement_profile = score_operation_placement(workload, operation, placement_affinity);
    const auto candidate_policy =
        candidate_policy_for_workload(workload, operation, system, continuous_state != nullptr, placement_affinity);
    auto ordered_allocations_for_operation = [&]() {
        std::vector<std::reference_wrapper<const PlanAllocation>> ordered_allocations;
        ordered_allocations.reserve(placement.allocations.size());
        const bool phase_host_preferred =
            should_prefer_host_for_llm_operation(workload, operation) ||
            should_prefer_host_for_phase_operation(workload, operation);
        const bool phase_accelerator_preferred =
            should_prefer_accelerator_for_llm_operation(workload, operation) ||
            should_prefer_accelerator_for_phase_operation(workload, operation);
        for (const auto& allocation : placement.allocations) {
            ordered_allocations.emplace_back(allocation);
        }
        std::stable_sort(
            ordered_allocations.begin(),
            ordered_allocations.end(),
            [&](const std::reference_wrapper<const PlanAllocation>& left,
                const std::reference_wrapper<const PlanAllocation>& right) {
                const auto* left_graph = find_graph(graph_lookup, left.get().device.uid);
                const auto* right_graph = find_graph(graph_lookup, right.get().device.uid);
                if (phase_host_preferred != phase_accelerator_preferred &&
                    left_graph != nullptr &&
                    right_graph != nullptr) {
                    if (phase_host_preferred && is_host_graph(left_graph) != is_host_graph(right_graph)) {
                        return is_host_graph(left_graph);
                    }
                    if (phase_accelerator_preferred && is_host_graph(left_graph) != is_host_graph(right_graph)) {
                        return !is_host_graph(left_graph);
                    }
                }
                const double left_score =
                    left.get().score * (1.0 + graph_operation_device_bias(placement_profile, left_graph));
                const double right_score =
                    right.get().score * (1.0 + graph_operation_device_bias(placement_profile, right_graph));
                return left_score > right_score;
            });
        return ordered_allocations;
    };

    const bool activation_driven = continuous_state != nullptr;
    const auto ordered_allocations = ordered_allocations_for_operation();
    const auto accelerators = accelerator_device_uids(placement, graph_lookup);
    const bool placement_has_host = std::any_of(
        placement.allocations.begin(),
        placement.allocations.end(),
        [&](const PlanAllocation& allocation) {
            return is_host_graph(find_graph(graph_lookup, allocation.device.uid));
        });
    const bool placement_has_accelerator = !accelerators.empty();
    const bool auto_accelerator_ddp =
        should_auto_accelerator_data_parallel(workload, operation, placement, graph_lookup);
    const bool allow_placement_sharding =
        operation.parallelizable &&
        placement.allocations.size() > 1u &&
        !workload.latency_sensitive &&
        !placement_affinity.avoid_sharding &&
        !(system.low_spec_mode && system.paging_risk > 0.25) &&
        !auto_accelerator_ddp;

    auto variant_allowed_by_activation = [&](const OperationVariantSpec& spec, const std::vector<WeightedStrategy>& ranked) {
        if (spec.scope == OperationVariantScope::accelerator_sharded && auto_accelerator_ddp) {
            return true;
        }
        if (continuous_state == nullptr) {
            return true;
        }

        const std::size_t max_ranked_strategies = candidate_policy.max_ranked_strategies;
        std::size_t accepted = 0;
        for (const auto& entry : ranked) {
            if (entry.weight < 0.0) {
                continue;
            }
            if (entry.strategy == spec.strategy) {
                return accepted < max_ranked_strategies;
            }
            ++accepted;
            if (accepted >= max_ranked_strategies) {
                break;
            }
        }
        return false;
    };

    auto supports_variant_on_graph = [&](const OperationVariantSpec& spec, const HardwareGraphSummary& summary) {
        if (spec.use_low_precision && !supports_low_precision(summary)) {
            return false;
        }
        if (spec.strategy == ExecutionStrategy::streaming &&
            !(summary.supports_asynchronous_dispatch || system.low_spec_mode)) {
            return false;
        }
        return true;
    };

    auto apply_variant_to_config =
        [&](const OperationVariantSpec& spec, const HardwareGraphSummary& summary, ExecutionConfig& config) {
            config.variant_id = spec.id;
            config.strategy = spec.strategy;
            config.overlap_transfers = spec.overlap_transfers;
            config.use_low_precision = spec.use_low_precision;

            if (spec.forced_queue_depth.has_value()) {
                config.queue_depth = *spec.forced_queue_depth;
            } else {
                config.queue_depth = std::max(config.queue_depth, spec.min_queue_depth);
            }

            if (spec.forced_stages.has_value()) {
                config.stages = *spec.forced_stages;
            } else {
                config.stages = std::max(config.stages, spec.min_stages);
            }

            if (spec.forced_logical_partitions.has_value()) {
                config.logical_partitions = *spec.forced_logical_partitions;
            } else {
                config.logical_partitions = std::max(config.logical_partitions, 1u);
            }

            if (config.strategy == ExecutionStrategy::overlapped && !summary.supports_asynchronous_dispatch) {
                config.strategy = ExecutionStrategy::single_device;
                config.overlap_transfers = false;
            }

            if (config.strategy != ExecutionStrategy::sharded) {
                config.participating_devices = {config.primary_device_uid};
            }
        };

    std::vector<ExecutionConfig> candidates;
    if (placement.allocations.empty()) {
        return candidates;
    }
    const auto ranked_strategies = rank_activation_strategies(allow_placement_sharding || auto_accelerator_ddp);
    const auto append_variant_suffix = [](std::string& variant_id, const std::string& suffix) {
        if (variant_id.find(suffix) == std::string::npos) {
            variant_id += suffix;
        }
    };

    auto add_candidate = [&](ExecutionConfig config) {
        config.operation_name = operation.name;
        config.target_error_tolerance = operation.max_relative_error;
        if (config.variant_id.empty()) {
            config.variant_id = "legacy";
        }
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
            apply_device_agnostic_execution_policy(workload, operation, graph, config);
            if (placement.strategy_source != PlanStrategySource::explicit_request) {
                apply_phase_aware_execution_policy(workload, operation, placement_profile, graph, config);
            }
            apply_llm_cpu_execution_policy(workload, operation, graph, config);
            apply_auto_accelerator_data_parallel_policy(workload, operation, placement, graph_lookup, config);
            apply_partition_strategy_execution_policy(workload, operation, placement, graph_lookup, config);
            stabilize_level_zero_host_mix_policy(operation, graph_lookup, config);
            const auto* resolved_primary_graph = find_graph(graph_lookup, config.primary_device_uid);
            if (placement.strategy_source != PlanStrategySource::explicit_request &&
                resolved_primary_graph != nullptr) {
                const auto affinity = structural_placement_affinity(workload, operation);
                const bool phase_host_preferred =
                    should_prefer_host_for_llm_operation(workload, operation) ||
                    should_prefer_host_for_phase_operation(workload, operation);
                const bool phase_accelerator_preferred =
                    should_prefer_accelerator_for_llm_operation(workload, operation) ||
                    should_prefer_accelerator_for_phase_operation(workload, operation);
                if (placement_has_host &&
                    phase_host_preferred &&
                    !phase_accelerator_preferred &&
                    !is_host_graph(resolved_primary_graph)) {
                    return;
                }
                if (placement_has_accelerator &&
                    phase_accelerator_preferred &&
                    !phase_host_preferred &&
                    is_host_graph(resolved_primary_graph)) {
                    return;
                }
                if (placement_has_accelerator &&
                    affinity.preferred == StructuralPlacementPreference::accelerator &&
                    (operation.op_class == OperationClass::matmul ||
                     operation.op_class == OperationClass::convolution_2d ||
                     operation.matrix_friendly) &&
                    is_host_graph(resolved_primary_graph)) {
                    return;
                }
                if (workload.kind == WorkloadKind::tensor &&
                    operation.op_class == OperationClass::matmul &&
                    !workload.latency_sensitive &&
                    workload.batch_size <= 4u &&
                    workload_host_exchange_ratio(workload) >= 0.20 &&
                    !is_host_graph(resolved_primary_graph)) {
                    return;
                }
            }
            if (config.participating_devices.size() <= 1u &&
                config.strategy == ExecutionStrategy::sharded) {
                config.strategy = ExecutionStrategy::single_device;
                config.logical_partitions = 1u;
                config.overlap_transfers = false;
            }
            if (is_host_graph(resolved_primary_graph) &&
                config.participating_devices.size() <= 1u &&
                operation.op_class == OperationClass::matmul) {
                config.logical_partitions = 1u;
                config.queue_depth = std::min(config.queue_depth, 2u);
                config.stages = 1u;
            }
            if (operation.attention_head_count > 0u) {
                const auto phase = canonical_workload_phase(workload);
                config.semantic_head_partitioning =
                    config.participating_devices.size() > 1u &&
                    (phase == WorkloadPhase::decode ||
                     attention_head_group_count(operation) >=
                         static_cast<std::uint32_t>(config.participating_devices.size()));
                config.semantic_token_partitioning =
                    config.participating_devices.size() > 1u &&
                    operation.preferred_token_block > 0u &&
                    phase == WorkloadPhase::prefill;
                config.logical_partitions = std::max(
                    config.logical_partitions,
                    desired_semantic_partition_count(workload, operation, config));
                if (operation.preferred_token_block > 0u &&
                    phase == WorkloadPhase::prefill &&
                    config.participating_devices.size() > 1u) {
                    config.strategy =
                        config.strategy == ExecutionStrategy::single_device ? ExecutionStrategy::sharded : config.strategy;
                    config.overlap_transfers = config.overlap_transfers || operation.preferred_kv_residency == "shared";
                }
                if (phase == WorkloadPhase::decode &&
                    config.participating_devices.size() > 1u &&
                    config.strategy == ExecutionStrategy::single_device) {
                    config.strategy = ExecutionStrategy::sharded;
                    config.overlap_transfers =
                        config.overlap_transfers || operation.preferred_kv_residency == "shared";
                }
            }
            if (is_host_graph(resolved_primary_graph) && cpu_materialized_lowering_active(operation)) {
                append_variant_suffix(config.variant_id, "+cpu-lowered");
            } else if (!is_host_graph(resolved_primary_graph) && gpu_materialized_lowering_active(operation)) {
                append_variant_suffix(config.variant_id, "+gpu-lowered");
            }
            if (!operation.fused_operation_names.empty()) {
                append_variant_suffix(config.variant_id, "+fused");
            }
        }
        if (placement.strategy_source != PlanStrategySource::explicit_request &&
            placement_profile.avoid_mixed &&
            config.participating_devices.size() > 1u) {
            config.participating_devices = {config.primary_device_uid};
            config.strategy = ExecutionStrategy::single_device;
            config.logical_partitions = 1u;
            config.overlap_transfers = false;
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

    const OperationVariantRequest request{
        workload,
        operation,
        placement.allocations.size(),
        accelerators.size(),
        system.low_spec_mode,
        activation_driven,
        operation.max_relative_error >= 4.0e-4,
        allow_placement_sharding,
        auto_accelerator_ddp};

    for (const auto& variant : OperationVariantRegistry::builtin().resolve(request)) {
        if (!variant_allowed_by_activation(variant, ranked_strategies)) {
            continue;
        }

        if (variant.scope == OperationVariantScope::per_allocation) {
            const std::size_t max_devices =
                activation_driven
                    ? std::min(candidate_policy.max_devices, placement_profile.preferred_max_devices)
                    : std::min<std::size_t>(
                          ordered_allocations.size(),
                          std::min(candidate_policy.max_devices, placement_profile.preferred_max_devices));
            std::size_t added_devices = 0;
            for (const auto& allocation_ref : ordered_allocations) {
                const auto& allocation = allocation_ref.get();
                const auto* graph = find_graph(graph_lookup, allocation.device.uid);
                if (graph == nullptr) {
                    continue;
                }

                const auto summary = summarize_graph(*graph);
                if (!supports_variant_on_graph(variant, summary)) {
                    continue;
                }

                ExecutionConfig config;
                config.primary_device_uid = allocation.device.uid;
                config.participating_devices = {allocation.device.uid};
                config.queue_depth = choose_queue_depth(summary);
                config.stages = summary.supports_asynchronous_dispatch ? 2u : 1u;
                config.logical_partitions = 1u;
                if (system.low_spec_mode) {
                    config.queue_depth = std::min(config.queue_depth, 2u);
                    config.stages = 1u;
                }

                apply_variant_to_config(variant, summary, config);
                add_candidate(config);
                ++added_devices;
                if (added_devices >= max_devices) {
                    break;
                }
                if (candidates.size() >= candidate_policy.max_candidates) {
                    break;
                }
            }
            continue;
        }

        std::vector<std::string> participating_devices;
        if (variant.scope == OperationVariantScope::accelerator_sharded) {
            participating_devices = accelerators;
        } else {
            participating_devices.reserve(placement.allocations.size());
            for (const auto& allocation_ref : ordered_allocations) {
                participating_devices.push_back(allocation_ref.get().device.uid);
            }
        }
        if (participating_devices.empty()) {
            continue;
        }
        const auto max_devices = std::min(candidate_policy.max_devices, placement_profile.preferred_max_devices);
        if (participating_devices.size() > max_devices) {
            participating_devices.resize(max_devices);
        }
        if (placement_profile.avoid_mixed && participating_devices.size() > 1u) {
            participating_devices.resize(1u);
        }

        const auto* primary_graph = find_graph(graph_lookup, participating_devices.front());
        if (primary_graph == nullptr) {
            continue;
        }
        const auto primary_summary = summarize_graph(*primary_graph);

        ExecutionConfig config;
        config.primary_device_uid = participating_devices.front();
        config.participating_devices = std::move(participating_devices);
        config.queue_depth = choose_queue_depth(primary_summary);
        config.stages = primary_summary.supports_asynchronous_dispatch ? 2u : 1u;
        config.logical_partitions = 1u;
        if (system.low_spec_mode) {
            config.queue_depth = std::min(config.queue_depth, 2u);
            config.stages = 1u;
        }

        apply_variant_to_config(variant, primary_summary, config);
        add_candidate(config);
        if (candidates.size() >= candidate_policy.max_candidates) {
            break;
        }
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
    if (graph.nodes.empty()) {
        graph.predicted_latency_us = 0.0;
        return 0.0;
    }

    std::vector<double> node_latency(graph.nodes.size(), 0.0);
    std::vector<int> indegree(graph.nodes.size(), 0);
    std::vector<std::vector<const ExecutionEdge*>> outgoing(graph.nodes.size());
    std::unordered_map<std::string, std::uint32_t> node_indices;
    node_indices.reserve(graph.nodes.size());

    for (std::uint32_t index = 0; index < graph.nodes.size(); ++index) {
        const auto& node = graph.nodes[index];
        node_latency[index] = node.predicted_latency_us;
        node_indices.emplace(node.id, index);
    }

    for (auto& edge : graph.edges) {
        if (edge.source_node_index == kInvalidExecutionIndex) {
            if (const auto it = node_indices.find(edge.source_id); it != node_indices.end()) {
                edge.source_node_index = it->second;
            }
        }
        if (edge.target_node_index == kInvalidExecutionIndex) {
            if (const auto it = node_indices.find(edge.target_id); it != node_indices.end()) {
                edge.target_node_index = it->second;
            }
        }
        if (edge.source_node_index == kInvalidExecutionIndex || edge.target_node_index == kInvalidExecutionIndex) {
            continue;
        }
        ++indegree[edge.target_node_index];
        outgoing[edge.source_node_index].push_back(&edge);
    }

    std::vector<std::uint32_t> frontier;
    frontier.reserve(graph.nodes.size());
    std::vector<double> longest(graph.nodes.size(), 0.0);
    std::vector<bool> seen(graph.nodes.size(), false);
    for (std::uint32_t index = 0; index < graph.nodes.size(); ++index) {
        if (indegree[index] == 0) {
            frontier.push_back(index);
            longest[index] = node_latency[index];
            seen[index] = true;
        }
    }

    for (std::size_t frontier_index = 0; frontier_index < frontier.size(); ++frontier_index) {
        const auto current = frontier[frontier_index];
        const double base = longest[current];
        for (const auto* edge : outgoing[current]) {
            const auto target = edge->target_node_index;
            const double candidate = base + effective_edge_latency(*edge) + node_latency[target];
            longest[target] = std::max(longest[target], candidate);
            seen[target] = true;
            if (--indegree[target] == 0) {
                frontier.push_back(target);
            }
        }
    }

    double total = 0.0;
    for (std::uint32_t index = 0; index < longest.size(); ++index) {
        if (seen[index]) {
            total = std::max(total, longest[index]);
        }
    }
    graph.predicted_latency_us = total;
    return total;
}

ExecutionGraph build_execution_graph(
    const std::string& report_signature,
    const WorkloadSpec& workload,
    const WorkloadGraph& workload_graph,
    const CompiledWorkloadGraph& compiled_workload,
    const OperationSpec& operation,
    ExecutionConfig& config,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::unordered_map<std::string, ExecutionConfig>& selected_configs) {
    ExecutionGraph graph;
    graph.workload_signature = report_signature;
    graph.operation = operation;
    graph.participating_devices = config.participating_devices;
    graph.indexed_devices = config.participating_devices;
    graph.indexed_operations.reserve(compiled_workload.operations.size());
    for (const auto& compiled_operation : compiled_workload.operations) {
        graph.indexed_operations.push_back(compiled_operation.name);
    }
    config.mapped_structural_nodes.clear();

    const auto ratios = normalized_ratios(placement, config);
    const auto operation_index = operation_index_or_default(compiled_workload.operation_indices, operation.name, 0u);
    const auto phase = canonical_workload_phase(workload);
    std::unordered_map<std::string, std::uint64_t> resident_bytes_by_device;
    std::unordered_map<std::string, double> resident_pressure_by_device;
    std::unordered_map<std::string, std::uint32_t> node_indices;
    node_indices.reserve(32u);
    std::unordered_map<std::string, std::uint32_t> device_indices;
    device_indices.reserve(graph.indexed_devices.size() + 1u);
    for (std::uint32_t index = 0; index < graph.indexed_devices.size(); ++index) {
        device_indices.emplace(graph.indexed_devices[index], index);
    }

    auto device_index_for = [&](const std::string& device_uid) {
        if (const auto it = device_indices.find(device_uid); it != device_indices.end()) {
            return it->second;
        }
        const auto index = static_cast<std::uint32_t>(graph.indexed_devices.size());
        graph.indexed_devices.push_back(device_uid);
        device_indices.emplace(device_uid, index);
        return index;
    };
    auto structural_index_for = [&](const std::string& structural_node_id) {
        if (structural_node_id.empty()) {
            return kInvalidExecutionIndex;
        }
        const auto it = std::find(graph.indexed_structural_nodes.begin(), graph.indexed_structural_nodes.end(), structural_node_id);
        if (it != graph.indexed_structural_nodes.end()) {
            return static_cast<std::uint32_t>(std::distance(graph.indexed_structural_nodes.begin(), it));
        }
        graph.indexed_structural_nodes.push_back(structural_node_id);
        return static_cast<std::uint32_t>(graph.indexed_structural_nodes.size() - 1u);
    };
    auto add_node = [&](ExecutionNode node) {
        node.node_index = static_cast<std::uint32_t>(graph.nodes.size());
        node.device_index = device_index_for(node.device_uid);
        node.structural_node_index = structural_index_for(node.structural_node_id);
        node.operation_index = operation_index;
        node_indices.emplace(node.id, node.node_index);
        graph.nodes.push_back(std::move(node));
    };

    auto add_edge = [&](ExecutionEdge edge) {
        if (const auto it = node_indices.find(edge.source_id); it != node_indices.end()) {
            edge.source_node_index = it->second;
        }
        if (const auto it = node_indices.find(edge.target_id); it != node_indices.end()) {
            edge.target_node_index = it->second;
        }
        graph.edges.push_back(std::move(edge));
    };

    const std::string global_sink_id = operation.name + ".sink";
    const std::string global_sync_id = operation.name + ".sync";
    const bool needs_sync = config.participating_devices.size() > 1;
    const bool uses_partitioned_execution =
        config.logical_partitions > 1 || config.strategy != ExecutionStrategy::single_device;
    const bool decode_pipeline = phase == WorkloadPhase::decode;
    const bool shared_staging_enabled =
        config.overlap_transfers && (decode_pipeline || needs_sync || uses_partitioned_execution);

    if (needs_sync) {
        add_node(ExecutionNode{
            global_sync_id,
            operation.name + "-sync",
            config.primary_device_uid,
            "",
            kInvalidExecutionIndex,
            kInvalidExecutionIndex,
            kInvalidExecutionIndex,
            operation_index,
            kInvalidExecutionIndex,
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
        kInvalidExecutionIndex,
        kInvalidExecutionIndex,
        kInvalidExecutionIndex,
        operation_index,
        kInvalidExecutionIndex,
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
                append_unique(config.mapped_structural_nodes, control_node->id + "#p" + std::to_string(partition));
            }
            if (compute_node != nullptr) {
                append_unique(config.mapped_structural_nodes, compute_node->id + "#p" + std::to_string(partition));
            }
            if (storage_node != nullptr) {
                append_unique(config.mapped_structural_nodes, storage_node->id + "#p" + std::to_string(partition));
            }

            const std::string partition_suffix =
                config.logical_partitions > 1 ? (".part" + std::to_string(partition)) : std::string();
            const std::string prefix = operation.name + "." + device_uid + partition_suffix;
            const std::string source_id = prefix + ".source";
            const std::string staging_id = prefix + ".staging";
            const std::string dispatch_id = prefix + ".dispatch";
            const std::string compute_id = prefix + ".compute";
            const std::string aggregate_id = prefix + ".aggregate";
            std::uint64_t partition_input_bytes = 0u;
            std::uint64_t partition_output_bytes = 0u;
            std::uint64_t partition_temporary_bytes = 0u;
            double input_transfer_latency_us = 0.0;
            double output_transfer_latency_us = 0.0;
            std::uint32_t cross_device_input_count = 0u;
            bool has_cross_device_input = false;

            for (const auto& tensor_id : operation.input_tensor_ids) {
                const auto* tensor = find_workload_tensor(workload_graph, compiled_workload, tensor_id);
                if (tensor == nullptr) {
                    continue;
                }

                const auto* lifetime = find_tensor_lifetime(workload_graph, compiled_workload, tensor_id);
                const auto mapped_bytes =
                    bytes_for_tensor_mapping(*tensor, operation, config, partition_ratio, partitions);
                partition_input_bytes += mapped_bytes;

                const auto source_devices =
                    resident_devices_for_tensor(workload, *tensor, operation, config, selected_configs);
                const bool resident_on_target = std::find(source_devices.begin(), source_devices.end(), device_uid) != source_devices.end();
                const std::string source_device_uid =
                    resident_on_target ? device_uid : (source_devices.empty() ? std::string("host") : source_devices.front());
                const auto* source_graph =
                    source_device_uid == "host" ? nullptr : find_graph(graph_lookup, source_device_uid);
                const auto source_summary = source_graph == nullptr ? HardwareGraphSummary{} : summarize_graph(*source_graph);
                const auto* source_summary_ptr = source_graph == nullptr ? nullptr : &source_summary;
                const bool cross_device = source_device_uid != device_uid;
                has_cross_device_input = has_cross_device_input || cross_device || source_device_uid == "host";
                if (cross_device || source_device_uid == "host") {
                    ++cross_device_input_count;
                }
                const double transfer_latency =
                    scheduled_transfer_latency_us(mapped_bytes, source_summary_ptr, summary, cross_device);
                input_transfer_latency_us = std::max(input_transfer_latency_us, transfer_latency);
                graph.predicted_transfer_latency_us += transfer_latency;
                graph.predicted_overlap_gain_us +=
                    transfer_latency * std::max(config.overlap_ratio, config.overlap_transfers ? 0.35 : 0.0);

                if ((cross_device || source_device_uid == "host") &&
                    std::none_of(
                        graph.residency_plan.begin(),
                        graph.residency_plan.end(),
                        [&](const TensorResidencyPlanEntry& entry) {
                            return entry.tensor_id == tensor->id && entry.device_uid == source_device_uid;
                        })) {
                    graph.residency_plan.push_back(TensorResidencyPlanEntry{
                        tensor->id,
                        source_device_uid,
                        std::string(),
                        compiled_workload.tensor_indices.contains(tensor->id) ? compiled_workload.tensor_indices.at(tensor->id)
                                                                             : kInvalidExecutionIndex,
                        device_index_for(source_device_uid),
                        kInvalidExecutionIndex,
                        mapped_bytes,
                        tensor->persistent,
                        true,
                        true,
                        false,
                        0.0});
                }

                graph.residency_plan.push_back(TensorResidencyPlanEntry{
                    tensor->id,
                    device_uid,
                    storage_node == nullptr ? std::string() : storage_node->id,
                    compiled_workload.tensor_indices.contains(tensor->id) ? compiled_workload.tensor_indices.at(tensor->id)
                                                                         : kInvalidExecutionIndex,
                    device_index_for(device_uid),
                    structural_index_for(storage_node == nullptr ? std::string() : storage_node->id),
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
                        compiled_workload.tensor_indices.contains(tensor->id) ? compiled_workload.tensor_indices.at(tensor->id)
                                                                             : kInvalidExecutionIndex,
                        device_index_for(source_device_uid),
                        device_index_for(device_uid),
                        tensor->producer_operation.empty()
                            ? kInvalidExecutionIndex
                            : operation_index_or_default(compiled_workload.operation_indices, tensor->producer_operation, kInvalidExecutionIndex),
                        operation_index,
                        mapped_bytes,
                        transfer_latency,
                        std::max(config.overlap_ratio, config.overlap_transfers ? 0.35 : 0.0),
                        cross_device});
                }
            }

            for (const auto& tensor_id : operation.output_tensor_ids) {
                const auto* tensor = find_workload_tensor(workload_graph, compiled_workload, tensor_id);
                if (tensor == nullptr) {
                    continue;
                }

                const auto* lifetime = find_tensor_lifetime(workload_graph, compiled_workload, tensor_id);
                const auto mapped_bytes =
                    bytes_for_tensor_mapping(*tensor, operation, config, partition_ratio, partitions);
                partition_output_bytes += mapped_bytes;
                graph.residency_plan.push_back(TensorResidencyPlanEntry{
                    tensor->id,
                    device_uid,
                    storage_node == nullptr ? std::string() : storage_node->id,
                    compiled_workload.tensor_indices.contains(tensor->id) ? compiled_workload.tensor_indices.at(tensor->id)
                                                                         : kInvalidExecutionIndex,
                    device_index_for(device_uid),
                    structural_index_for(storage_node == nullptr ? std::string() : storage_node->id),
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
                    graph.predicted_overlap_gain_us += transfer_latency * (config.overlap_ratio * 0.7);
                    graph.transfer_schedule.push_back(TransferScheduleEntry{
                        tensor->id,
                        device_uid,
                        config.primary_device_uid,
                        operation.name,
                        operation.name,
                        compiled_workload.tensor_indices.contains(tensor->id) ? compiled_workload.tensor_indices.at(tensor->id)
                                                                             : kInvalidExecutionIndex,
                        device_index_for(device_uid),
                        device_index_for(config.primary_device_uid),
                        operation_index,
                        operation_index,
                        mapped_bytes,
                        transfer_latency,
                        config.overlap_ratio * 0.7,
                        true});
                }
            }

            for (const auto& tensor_id : operation.temporary_tensor_ids) {
                const auto* tensor = find_workload_tensor(workload_graph, compiled_workload, tensor_id);
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
                    compiled_workload.tensor_indices.contains(tensor->id) ? compiled_workload.tensor_indices.at(tensor->id)
                                                                         : kInvalidExecutionIndex,
                    device_index_for(device_uid),
                    structural_index_for(storage_node == nullptr ? std::string() : storage_node->id),
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
                kInvalidExecutionIndex,
                kInvalidExecutionIndex,
                kInvalidExecutionIndex,
                operation_index,
                kInvalidExecutionIndex,
                ExecutionNodeKind::source,
                partition_input_bytes == 0u ? input_share : partition_input_bytes,
                0.0,
                0.0});

            const bool needs_input_staging =
                shared_staging_enabled && has_cross_device_input &&
                (cross_device_input_count > 1u || decode_pipeline || needs_sync);
            if (needs_input_staging) {
                add_node(ExecutionNode{
                    staging_id,
                    hardware->presentation_name + partition_suffix + "-staging",
                    device_uid,
                    storage_node == nullptr ? std::string() : storage_node->id,
                    kInvalidExecutionIndex,
                    kInvalidExecutionIndex,
                    kInvalidExecutionIndex,
                    operation_index,
                    kInvalidExecutionIndex,
                    ExecutionNodeKind::aggregate,
                    partition_input_bytes == 0u ? input_share : partition_input_bytes,
                    0.0,
                    std::max(0.05, input_transfer_latency_us * (decode_pipeline ? 0.20 : 0.10))});
            }

            add_node(ExecutionNode{
                dispatch_id,
                hardware->presentation_name + partition_suffix + "-dispatch",
                device_uid,
                control_node == nullptr ? std::string() : control_node->id,
                kInvalidExecutionIndex,
                kInvalidExecutionIndex,
                kInvalidExecutionIndex,
                operation_index,
                kInvalidExecutionIndex,
                ExecutionNodeKind::dispatch,
                0,
                0.0,
                dispatch_cost_us(summary)});

            add_node(ExecutionNode{
                compute_id,
                hardware->presentation_name + partition_suffix + "-compute",
                device_uid,
                compute_node == nullptr ? std::string() : compute_node->id,
                kInvalidExecutionIndex,
                kInvalidExecutionIndex,
                kInvalidExecutionIndex,
                operation_index,
                kInvalidExecutionIndex,
                ExecutionNodeKind::compute,
                partition_output_bytes == 0u ? output_share : partition_output_bytes,
                flops_share,
                std::max(0.01, compute_latency_us)});

            const double input_overlap_ratio = std::max(
                config.overlap_ratio,
                decode_pipeline && has_cross_device_input
                    ? 0.55
                    : (config.overlap_transfers ? 0.35 : 0.0));
            add_edge(ExecutionEdge{
                source_id,
                needs_input_staging ? staging_id : dispatch_id,
                kInvalidExecutionIndex,
                kInvalidExecutionIndex,
                ExecutionEdgeKind::dataflow,
                true,
                partition_input_bytes == 0u ? input_share : partition_input_bytes,
                std::max(
                    transfer_cost_us(partition_input_bytes == 0u ? input_share : partition_input_bytes, summary),
                    input_transfer_latency_us),
                input_overlap_ratio});
            if (needs_input_staging) {
                add_edge(ExecutionEdge{
                    staging_id,
                    dispatch_id,
                    kInvalidExecutionIndex,
                    kInvalidExecutionIndex,
                    ExecutionEdgeKind::dependency,
                    true,
                    0u,
                    std::max(0.05, dispatch_cost_us(summary) * (decode_pipeline ? 0.50 : 0.25)),
                    decode_pipeline ? 0.15 : 0.0});
            }

            add_edge(ExecutionEdge{
                dispatch_id,
                compute_id,
                kInvalidExecutionIndex,
                kInvalidExecutionIndex,
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
                    kInvalidExecutionIndex,
                    kInvalidExecutionIndex,
                    kInvalidExecutionIndex,
                    operation_index,
                    kInvalidExecutionIndex,
                    ExecutionNodeKind::aggregate,
                    operation.output_bytes,
                    static_cast<double>(operation.output_bytes),
                    sync_cost_us(summary) * 0.5});

                add_edge(ExecutionEdge{
                    compute_id,
                    aggregate_id,
                    kInvalidExecutionIndex,
                    kInvalidExecutionIndex,
                    ExecutionEdgeKind::aggregation,
                    true,
                    output_share,
                    transfer_cost_us(output_share, summary) * 0.5,
                    config.overlap_ratio * 0.5});

                if (needs_sync) {
                    add_edge(ExecutionEdge{
                        aggregate_id,
                        global_sync_id,
                        kInvalidExecutionIndex,
                        kInvalidExecutionIndex,
                        ExecutionEdgeKind::dependency,
                        true,
                        operation.output_bytes,
                        std::max(sync_cost_us(summary), output_transfer_latency_us),
                        0.0});
                } else {
                    add_edge(ExecutionEdge{
                        aggregate_id,
                        global_sink_id,
                        kInvalidExecutionIndex,
                        kInvalidExecutionIndex,
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
                    kInvalidExecutionIndex,
                    kInvalidExecutionIndex,
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
                    kInvalidExecutionIndex,
                    kInvalidExecutionIndex,
                    ExecutionEdgeKind::dataflow,
                    true,
                    output_share,
                    transfer_cost_us(output_share, summary),
                    uses_partitioned_execution ? std::max(config.overlap_ratio, 0.20) : (config.overlap_ratio * 0.7)});
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
            kInvalidExecutionIndex,
            kInvalidExecutionIndex,
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

ValidationResult validate_elementwise(
    const OperationSpec& operation,
    const ExecutionConfig& config,
    const std::uint32_t rounds) {
    const auto count = static_cast<std::size_t>(operation.extents.at(0));
    const auto a = make_pattern(count, 1.0f);
    const auto b = make_pattern(count, 2.0f);
    std::vector<float> reference(count, 0.0f);
    std::vector<float> candidate(count, 0.0f);

    const auto reference_profile = profile_measurements(rounds, [&]() {
        for (std::size_t index = 0; index < count; ++index) {
            reference[index] = (a[index] * 1.125f) + (b[index] * 0.25f) - 0.03125f;
        }
    });

    const std::size_t tile = std::max<std::size_t>(config.tile_x, 128u);
    const auto candidate_profile = profile_measurements(rounds, [&]() {
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

ValidationResult validate_reduction(
    const OperationSpec& operation,
    const ExecutionConfig& config,
    const std::uint32_t rounds) {
    const auto count = static_cast<std::size_t>(operation.extents.at(0));
    const auto data = make_pattern(count, 3.0f);
    double reference_value = 0.0;
    double candidate_value = 0.0;

    const auto reference_profile = profile_measurements(rounds, [&]() {
        double total = 0.0;
        for (const auto value : data) {
            total += value;
        }
        reference_value = total;
    });

    const std::size_t tile = std::max<std::size_t>(config.tile_x, 256u);
    const auto candidate_profile = profile_measurements(rounds, [&]() {
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

ValidationResult validate_matmul(
    const OperationSpec& operation,
    const ExecutionConfig& config,
    const std::uint32_t rounds) {
    const auto m = static_cast<std::size_t>(operation.extents.at(0));
    const auto n = static_cast<std::size_t>(operation.extents.at(1));
    const auto k = static_cast<std::size_t>(operation.extents.at(2));
    const auto a = make_pattern(m * k, 4.0f);
    const auto b = make_pattern(k * n, 5.0f);
    std::vector<float> reference(m * n, 0.0f);
    std::vector<float> candidate(m * n, 0.0f);

    const auto reference_profile = profile_measurements(rounds, [&]() {
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
    const auto candidate_profile = profile_measurements(rounds, [&]() {
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

ValidationResult validate_convolution(
    const OperationSpec& operation,
    const ExecutionConfig& config,
    const std::uint32_t rounds) {
    const auto height = static_cast<std::size_t>(operation.extents.at(0));
    const auto width = static_cast<std::size_t>(operation.extents.at(1));
    const auto input = make_pattern(height * width, 6.0f);
    const std::array<float, 9> kernel{0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.125f, 0.0625f, 0.125f, 0.0625f};
    std::vector<float> reference((height - 2) * (width - 2), 0.0f);
    std::vector<float> candidate((height - 2) * (width - 2), 0.0f);

    const auto reference_profile = profile_measurements(rounds, [&]() {
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
    const auto candidate_profile = profile_measurements(rounds, [&]() {
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

ValidationResult validate_resample(
    const OperationSpec& operation,
    const ExecutionConfig& config,
    const std::uint32_t rounds) {
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

    const auto reference_profile = profile_measurements(rounds, [&]() {
        bilinear(reference, false);
    });
    const auto candidate_profile = profile_measurements(rounds, [&]() {
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
    const double system_penalty_us,
    const std::uint32_t validation_rounds) {
    if (validation_rounds == 0u) {
        BenchmarkRecord record;
        record.operation_name = operation.name;
        record.config_signature = config.signature;
        record.shape_bucket = shape_bucket;
        record.reference_latency_us = 0.0;
        record.validation_latency_us = 0.0;
        record.predicted_latency_us = graph.predicted_latency_us;
        record.surrogate_latency_us = surrogate_latency_us;
        record.calibrated_prediction_us = std::max(0.01, surrogate_latency_us + system_penalty_us);
        record.calibration_ratio = 1.0;
        record.calibration_confidence = 0.0;
        record.reference_spread_us = 0.0;
        record.candidate_spread_us = 0.0;
        record.validation_samples = 0u;
        record.system_penalty_us = system_penalty_us;
        record.effective_latency_us = std::max(0.01, surrogate_latency_us + system_penalty_us);
        record.speedup_vs_reference = 1.0;
        record.relative_error = graph.expected_relative_error;
        record.accuracy_within_tolerance = graph.expected_relative_error <= (config.target_error_tolerance + 1.0e-12);
        record.simulated = true;
        return record;
    }

    ValidationResult validation;
    switch (operation.op_class) {
    case OperationClass::elementwise_map:
        validation = validate_elementwise(operation, config, validation_rounds);
        break;
    case OperationClass::reduction:
        validation = validate_reduction(operation, config, validation_rounds);
        break;
    case OperationClass::matmul:
        validation = validate_matmul(operation, config, validation_rounds);
        break;
    case OperationClass::convolution_2d:
        validation = validate_convolution(operation, config, validation_rounds);
        break;
    case OperationClass::resample_2d:
    default:
        validation = validate_resample(operation, config, validation_rounds);
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
        (static_cast<double>(std::max(validation.samples, 1u)) /
         static_cast<double>(std::max(validation_rounds, 1u))) *
            (1.0 / (1.0 + (spread_ratio * 6.0))),
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
    double historical_latency_us = 0.0;
    double average_reward = 0.0;
    std::uint32_t observations = 0;
};

OperationOptimizationResult optimize_operation(
    const std::string& report_signature,
    const WorkloadGraph& workload_graph,
    const CompiledWorkloadGraph& compiled_workload,
    const OperationSpec& operation,
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::unordered_map<std::string, ExecutionConfig>& selected_configs,
    const SystemProfile& system,
    const std::unordered_map<std::string, PerformanceSummary>& performance_cache,
    const std::unordered_map<std::string, PerformanceSummary>& graph_family_performance_cache,
    const std::unordered_map<std::string, double>& backend_penalty_cache,
    const std::unordered_map<std::string, bool>& warmed_devices,
    const std::string& graph_set_signature,
    const ExecutionConfig* cached_config,
    const ContinuousExecutionState* continuous_state,
    const bool allow_validation,
    const ValidationTier validation_tier,
    const OptimizationDeadline& deadline);

std::optional<ExecutionConfig> make_spsa_candidate(
    const ExecutionConfig& seed,
    const std::string& report_signature,
    const WorkloadSpec& workload,
    const WorkloadGraph& workload_graph,
    const CompiledWorkloadGraph& compiled_workload,
    const OperationSpec& operation,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::unordered_map<std::string, ExecutionConfig>& selected_configs,
    const SystemProfile& system,
    const std::unordered_map<std::string, bool>& warmed_devices) {
    auto score_of = [&](ExecutionConfig candidate) {
        auto graph = build_execution_graph(
            report_signature,
            workload,
            workload_graph,
            compiled_workload,
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

void refresh_candidate_objectives(
    CandidateEvaluation& evaluation,
    const double trace_weight,
    const double historical_latency_us,
    const double average_reward,
    const double log_observation_budget) {
    auto& benchmark = evaluation.benchmark;
    double heuristic_objective = benchmark.calibrated_prediction_us > 0.0
                                     ? benchmark.calibrated_prediction_us
                                     : benchmark.surrogate_latency_us;
    if (!benchmark.simulated) {
        heuristic_objective = std::min(heuristic_objective, benchmark.validation_latency_us);
    }
    if (!benchmark.accuracy_within_tolerance) {
        const double tolerance = std::max(evaluation.config.target_error_tolerance, 1.0e-9);
        heuristic_objective *= 10.0 + (benchmark.relative_error / tolerance);
    }

    evaluation.heuristic_objective = heuristic_objective;
    evaluation.learned_objective =
        (heuristic_objective * 0.45) + (benchmark.effective_latency_us * 0.45) +
        (evaluation.graph.predicted_transfer_latency_us * 0.05) +
        (evaluation.graph.predicted_memory_pressure * benchmark.effective_latency_us * 0.05);
    const double trace_bias =
        evaluation.observations >= kLearningWarmupSamples ? (trace_weight + 2.0) : trace_weight;
    evaluation.trace_objective =
        evaluation.observations == 0
            ? evaluation.learned_objective
            : ((historical_latency_us * trace_bias) + evaluation.learned_objective) / (trace_bias + 1.0);
    const double exploration_bonus =
        evaluation.learned_objective * 0.35 *
        std::sqrt(std::log(log_observation_budget + 1.0) / static_cast<double>(evaluation.observations + 1u));
    evaluation.explore_objective = std::max(0.01, evaluation.learned_objective - exploration_bonus);
    const double preference_scale = std::exp(std::clamp(average_reward, -1.5, 1.5));
    evaluation.reinforce_objective = evaluation.learned_objective / std::max(0.20, preference_scale);
    benchmark.trace_weight = trace_weight;
}

std::size_t choose_validation_shortlist_size(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const SystemProfile& system,
    const std::vector<CandidateEvaluation>& evaluations,
    const OptimizationPolicy policy) {
    if (evaluations.size() <= 1u) {
        return evaluations.size();
    }

    const auto policy_limits = candidate_policy_for_workload(
        workload,
        operation,
        system,
        true,
        structural_placement_affinity(workload, operation));

    std::vector<double> objectives;
    objectives.reserve(evaluations.size());
    for (const auto& evaluation : evaluations) {
        objectives.push_back(objective_for_policy(evaluation, policy));
    }
    std::sort(objectives.begin(), objectives.end());
    const double best = objectives[0];
    const double second = objectives[1];
    const double gap_ratio = (second - best) / std::max(best, 1.0);
    const auto max_observations = std::max_element(
        evaluations.begin(),
        evaluations.end(),
        [](const CandidateEvaluation& left, const CandidateEvaluation& right) {
            return left.observations < right.observations;
        })->observations;
    if (gap_ratio > 0.20 && max_observations >= kLearningWarmupSamples) {
        return std::min<std::size_t>(1u, policy_limits.validation_shortlist);
    }
    return std::min<std::size_t>(policy_limits.validation_shortlist, evaluations.size());
}

struct GraphStateEvaluation {
    ContinuousExecutionState state;
    double total_objective_us = 0.0;
    std::uint32_t logical_partitions = 0;
    std::vector<OperationOptimizationResult> operations;
};

ContinuousExecutionState default_continuous_state(const WorkloadSpec& workload, const SystemProfile& system) {
    const double exchange_ratio = workload_host_exchange_ratio(workload);
    const double dispatch_pressure = std::clamp(
        (workload.latency_sensitive ? (workload.batch_size <= 1u ? 0.55 : 0.20) : 0.0) +
            std::clamp(exchange_ratio * 0.35, 0.0, 0.30) +
            (workload.matrix_friendly ? 0.0 : 0.10),
        0.0,
        1.0);

    ContinuousExecutionState state;
    state.queue_depth_raw = system.low_spec_mode ? -0.4 : 0.2;
    state.stage_raw = system.low_spec_mode ? -0.2 : 0.3;
    state.tile_raw = workload.matrix_friendly ? 0.6 : 0.0;
    state.overlap_raw = (workload.latency_sensitive ? -0.8 : 0.2) - (dispatch_pressure * 0.35);
    state.partition_raw = (system.low_spec_mode ? -0.9 : -0.3) - (dispatch_pressure * 0.60);
    state.precision_raw = workload.matrix_friendly ? 0.5 : -0.5;
    state.single_device_logit = (system.low_spec_mode ? 1.4 : 0.7) + (dispatch_pressure * 0.55);
    state.sharded_logit = (workload.latency_sensitive ? -1.2 : 0.1) - (dispatch_pressure * 0.85);
    state.streaming_logit =
        ((workload.kind == WorkloadKind::gaming || workload.kind == WorkloadKind::image) ? 0.6 : 0.1) -
        (dispatch_pressure * 0.20);
    state.overlapped_logit = 0.6 - (dispatch_pressure * 0.25);
    return clamp_state(state);
}

GraphStateEvaluation evaluate_global_state(
    const ContinuousExecutionState& state,
    const std::string& report_signature,
    const WorkloadSpec& workload,
    const WorkloadGraph& workload_graph,
    const CompiledWorkloadGraph& compiled_workload,
    const ExecutionPlan& placement,
    const std::vector<OperationSpec>& operations,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const SystemProfile& system,
    const std::unordered_map<std::string, PerformanceSummary>& performance_cache,
    const std::unordered_map<std::string, PerformanceSummary>& graph_family_performance_cache,
    const std::unordered_map<std::string, double>& backend_penalty_cache,
    const std::unordered_map<std::string, bool>& warmed_devices,
    const std::string& graph_set_signature,
    const std::unordered_map<std::string, ExecutionConfig>* cached_configs,
    const bool keep_operations,
    const ValidationTier validation_tier,
    const OptimizationDeadline& deadline) {
    GraphStateEvaluation evaluation;
    evaluation.state = clamp_state(state);
    if (keep_operations) {
        evaluation.operations.reserve(operations.size());
    }
    std::unordered_map<std::string, ExecutionConfig> selected_configs;

    for (const auto& operation : operations) {
        if (deadline_reached(deadline)) {
            break;
        }
        const ExecutionConfig* cached_config = nullptr;
        if (cached_configs != nullptr) {
            if (const auto it = cached_configs->find(operation.name); it != cached_configs->end()) {
                cached_config = &it->second;
            }
        }
        auto optimized = optimize_operation(
            report_signature,
            workload_graph,
            compiled_workload,
            operation,
            workload,
            placement,
            graph_lookup,
            selected_configs,
            system,
            performance_cache,
            graph_family_performance_cache,
            backend_penalty_cache,
            warmed_devices,
            graph_set_signature,
            cached_config,
            &evaluation.state,
            false,
            validation_tier,
            deadline);
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
    const CompiledWorkloadGraph& compiled_workload,
    const ExecutionPlan& placement,
    const std::vector<OperationSpec>& operations,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const SystemProfile& system,
    const std::unordered_map<std::string, PerformanceSummary>& performance_cache,
    const std::unordered_map<std::string, PerformanceSummary>& graph_family_performance_cache,
    const std::unordered_map<std::string, double>& backend_penalty_cache,
    const std::unordered_map<std::string, bool>& warmed_devices,
    const std::string& graph_set_signature,
    const std::unordered_map<std::string, ExecutionConfig>* cached_configs,
    const bool lightweight_path,
    const ExecutionTuningOverrides* tuning_overrides,
    const ValidationTier validation_tier,
    const OptimizationDeadline& deadline) {
    constexpr std::array<double, 10> kDeltas{0.35, 0.35, 0.30, 0.25, 0.25, 0.25, 0.30, 0.30, 0.30, 0.30};
    constexpr double kBeta1 = 0.9;
    constexpr double kBeta2 = 0.999;
    constexpr double kEpsilon = 1.0e-8;

    GraphOptimizationSummary summary;
    summary.optimizer_name = "adam_surrogate";
    summary.time_budget_ms =
        tuning_overrides != nullptr && tuning_overrides->optimizer_wall_time_budget_ms.has_value()
            ? *tuning_overrides->optimizer_wall_time_budget_ms
            : 0u;
    auto state = tuning_overrides != nullptr && tuning_overrides->initial_state_override.has_value()
                     ? clamp_state(*tuning_overrides->initial_state_override)
                     : default_continuous_state(workload, system);
    summary.initial_state = state;
    const auto active_dims = active_continuous_dimensions(workload, operations, placement, system);
    const auto graph_passes =
        lightweight_path
            ? 0u
            : tuning_overrides != nullptr && tuning_overrides->graph_optimization_passes_override.has_value()
                  ? *tuning_overrides->graph_optimization_passes_override
                  : choose_graph_optimization_passes(workload, operations, placement, system);

    const auto initial = evaluate_global_state(
        state,
        report_signature,
        workload,
        workload_graph,
        compiled_workload,
        placement,
        operations,
        graph_lookup,
        system,
        performance_cache,
        graph_family_performance_cache,
        backend_penalty_cache,
        warmed_devices,
        graph_set_signature,
        cached_configs,
        false,
        validation_tier,
        deadline);
    summary.initial_objective_us = initial.total_objective_us;

    std::array<double, 10> m{};
    std::array<double, 10> v{};
    double learning_rate = system.low_spec_mode ? 0.06 : 0.08;
    double best_objective = initial.total_objective_us;
    auto best_state = state;
    if (active_dims.empty() || graph_passes == 0u) {
        summary.final_state = best_state;
        summary.final_objective_us = best_objective;
        summary.budget_exhausted = deadline_reached(deadline);
        summary.converged = true;
        return summary;
    }

    for (std::uint32_t pass_index = 1; pass_index <= graph_passes; ++pass_index) {
        if (deadline_reached(deadline)) {
            summary.budget_exhausted = true;
            break;
        }
        std::array<double, 10> gradient{};
        bool pass_budget_exhausted = false;

        for (const std::size_t dim : active_dims) {
            if (deadline_reached(deadline)) {
                pass_budget_exhausted = true;
                break;
            }
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
                compiled_workload,
                placement,
                operations,
                graph_lookup,
                system,
                performance_cache,
                graph_family_performance_cache,
                backend_penalty_cache,
                warmed_devices,
                graph_set_signature,
                cached_configs,
                false,
                validation_tier,
                deadline);
            if (deadline_reached(deadline)) {
                pass_budget_exhausted = true;
                break;
            }
            const auto minus_eval = evaluate_global_state(
                clamp_state(minus),
                report_signature,
                workload,
                workload_graph,
                compiled_workload,
                placement,
                operations,
                graph_lookup,
                system,
                performance_cache,
                graph_family_performance_cache,
                backend_penalty_cache,
                warmed_devices,
                graph_set_signature,
                cached_configs,
                false,
                validation_tier,
                deadline);
            if (deadline_reached(deadline)) {
                pass_budget_exhausted = true;
                break;
            }
            gradient[dim] = (plus_eval.total_objective_us - minus_eval.total_objective_us) / (2.0 * kDeltas[dim]);
        }
        if (pass_budget_exhausted) {
            summary.budget_exhausted = true;
            break;
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
            compiled_workload,
            placement,
            operations,
            graph_lookup,
            system,
            performance_cache,
            graph_family_performance_cache,
            backend_penalty_cache,
            warmed_devices,
            graph_set_signature,
            cached_configs,
            false,
            validation_tier,
            deadline);
        if (deadline_reached(deadline)) {
            summary.budget_exhausted = true;
            break;
        }
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
    summary.converged = !summary.budget_exhausted && best_objective <= (summary.initial_objective_us * 0.995);
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
    const CompiledWorkloadGraph& compiled_workload,
    const OperationSpec& operation,
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::unordered_map<std::string, ExecutionConfig>& selected_configs,
    const SystemProfile& system,
    const std::unordered_map<std::string, PerformanceSummary>& performance_cache,
    const std::unordered_map<std::string, PerformanceSummary>& graph_family_performance_cache,
    const std::unordered_map<std::string, double>& backend_penalty_cache,
    const std::unordered_map<std::string, bool>& warmed_devices,
    const std::string& graph_set_signature,
    const ExecutionConfig* cached_config,
    const ContinuousExecutionState* continuous_state,
    const bool allow_validation,
    const ValidationTier validation_tier,
    const OptimizationDeadline& deadline) {
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

    if (should_auto_accelerator_data_parallel(workload, operation, placement, graph_lookup) &&
        operation.op_class == OperationClass::matmul) {
        const auto accelerators = accelerator_device_uids(placement, graph_lookup);
        const auto existing_ddp = std::find_if(
            candidates.begin(),
            candidates.end(),
            [&](const ExecutionConfig& candidate) {
                return candidate.strategy == ExecutionStrategy::sharded &&
                       candidate.participating_devices.size() == accelerators.size() &&
                       std::is_permutation(
                           candidate.participating_devices.begin(),
                           candidate.participating_devices.end(),
                        accelerators.begin(),
                        accelerators.end());
            });
        if (existing_ddp == candidates.end() && !accelerators.empty()) {
            auto ddp = candidates.front();
            ddp.variant_id = "auto-accelerator-ddp";
            ddp.strategy = ExecutionStrategy::sharded;
            ddp.primary_device_uid = accelerators.front();
            ddp.participating_devices = accelerators;
            ddp.logical_partitions = 1u;
            ddp.overlap_transfers = true;
            ddp.queue_depth = std::max(ddp.queue_depth, 2u);
            ddp.stages = std::max(ddp.stages, 2u);
            ddp.signature = build_config_signature(report_signature, ddp);
            candidates.push_back(std::move(ddp));
        }
    }

    std::string spsa_signature;
    if (allow_validation || continuous_state == nullptr) {
            if (const auto spsa_candidate = make_spsa_candidate(
                candidates.front(),
                report_signature,
                workload,
                workload_graph,
                compiled_workload,
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
    }

    if (continuous_state != nullptr && !allow_validation && candidates.size() > 4u) {
        const auto ddp_policy = candidate_policy_for_workload(
            workload,
            operation,
            system,
            true,
            structural_placement_affinity(workload, operation));
        const auto accelerators = accelerator_device_uids(placement, graph_lookup);
        const auto ddp_candidate = std::find_if(
            candidates.begin(),
            candidates.end(),
            [&](const ExecutionConfig& candidate) {
                return ddp_policy.max_devices > 1u &&
                       should_auto_accelerator_data_parallel(workload, operation, placement, graph_lookup) &&
                       candidate.strategy == ExecutionStrategy::sharded &&
                       candidate.participating_devices.size() == accelerators.size() &&
                       std::is_permutation(
                           candidate.participating_devices.begin(),
                           candidate.participating_devices.end(),
                           accelerators.begin(),
                           accelerators.end());
            });
        std::vector<ExecutionConfig> limited;
        limited.reserve(4u);
        for (const auto& candidate : candidates) {
            if (limited.size() >= 4u) {
                break;
            }
            limited.push_back(candidate);
        }
        if (ddp_candidate != candidates.end()) {
            const auto present = std::find_if(
                limited.begin(),
                limited.end(),
                [&](const ExecutionConfig& candidate) {
                    return candidate.signature == ddp_candidate->signature;
                });
            if (present == limited.end()) {
                limited.back() = *ddp_candidate;
            }
        }
        candidates = std::move(limited);
    }

    const auto shape_bucket = shape_bucket_for(operation);
    const double trace_weight = preset_trace_weight(workload);
    std::vector<CandidateEvaluation> evaluations;
    evaluations.reserve(candidates.size());
    double log_observation_budget = 1.0;
    for (auto candidate : candidates) {
        if (deadline_reached(deadline) && !evaluations.empty()) {
            break;
        }
        auto graph = build_execution_graph(
            report_signature,
            workload,
            workload_graph,
            compiled_workload,
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
        double average_relative_error = graph.expected_relative_error;
        double backend_penalty_us = 0.0;
        double calibration_ratio = 1.0;
        double validation_spread_us = 0.0;
        double overlap_penalty_us = 0.0;
        double telemetry_staging_hit_rate = 0.0;
        double telemetry_cross_device_sync_cost_us = 0.0;
        double telemetry_residency_pressure = graph.predicted_memory_pressure;
        const auto candidate_performance_key =
            performance_key(report_signature, graph_set_signature, workload, system, shape_bucket, candidate);
        if (const auto it = performance_cache.find(candidate_performance_key); it != performance_cache.end()) {
            observations = it->second.observations;
            historical_latency_us = it->second.average_effective_latency_us;
            average_reward = it->second.average_reward;
            average_relative_error = it->second.average_relative_error;
            telemetry_staging_hit_rate = it->second.average_staging_hit_rate;
            telemetry_cross_device_sync_cost_us = it->second.average_cross_device_sync_cost_us;
            telemetry_residency_pressure = std::max(
                telemetry_residency_pressure,
                it->second.average_residency_pressure);
            learning_scale = std::clamp(it->second.average_prediction_scale, 0.50, 2.50);
            calibration_ratio = std::clamp(it->second.average_calibration_ratio, 0.50, 2.50);
            validation_spread_us = it->second.average_validation_spread_us;
            overlap_penalty_us =
                std::max(0.0, (it->second.average_copy_share - it->second.average_transfer_overlap_ratio) * 80.0) +
                (it->second.average_budget_pressure * 60.0) -
                (it->second.average_queue_separation_ratio * 35.0) +
                ((1.0 - it->second.average_staging_hit_rate) * 55.0) +
                (it->second.average_cross_device_sync_cost_us * 0.12) +
                (it->second.average_residency_pressure * 42.0);
            if (it->second.observations < kLearningWarmupSamples) {
                learning_scale *= 0.97;
            }
            if (it->second.average_relative_error > candidate.target_error_tolerance) {
                learning_scale *= 1.5;
            }
        } else if (const auto family_it =
                       graph_family_performance_cache.find(
                           performance_family_key(report_signature, shape_bucket, candidate));
                   family_it != graph_family_performance_cache.end()) {
            observations = std::max<std::uint32_t>(1u, family_it->second.observations / 2u);
            historical_latency_us = family_it->second.average_effective_latency_us;
            average_reward = family_it->second.average_reward * 0.85;
            average_relative_error = family_it->second.average_relative_error;
            telemetry_staging_hit_rate = family_it->second.average_staging_hit_rate;
            telemetry_cross_device_sync_cost_us = family_it->second.average_cross_device_sync_cost_us;
            telemetry_residency_pressure = std::max(
                telemetry_residency_pressure,
                family_it->second.average_residency_pressure);
            learning_scale = std::clamp(family_it->second.average_prediction_scale, 0.60, 2.25);
            calibration_ratio = std::clamp(family_it->second.average_calibration_ratio, 0.60, 2.25);
            validation_spread_us = family_it->second.average_validation_spread_us;
            overlap_penalty_us =
                std::max(0.0, (family_it->second.average_copy_share - family_it->second.average_transfer_overlap_ratio) * 64.0) +
                (family_it->second.average_budget_pressure * 48.0) -
                (family_it->second.average_queue_separation_ratio * 24.0) +
                ((1.0 - family_it->second.average_staging_hit_rate) * 42.0) +
                (family_it->second.average_cross_device_sync_cost_us * 0.10) +
                (family_it->second.average_residency_pressure * 32.0);
        } else if (const auto* family_summary = best_graph_family_summary(
                       graph_family_performance_cache,
                       report_signature,
                       shape_bucket,
                       candidate,
                       graph_lookup);
                   family_summary != nullptr) {
            observations = std::max<std::uint32_t>(1u, family_summary->observations / 2u);
            historical_latency_us = family_summary->average_effective_latency_us;
            average_reward = family_summary->average_reward * 0.82;
            average_relative_error = family_summary->average_relative_error;
            telemetry_staging_hit_rate = family_summary->average_staging_hit_rate;
            telemetry_cross_device_sync_cost_us = family_summary->average_cross_device_sync_cost_us;
            telemetry_residency_pressure = std::max(
                telemetry_residency_pressure,
                family_summary->average_residency_pressure);
            learning_scale = std::clamp(family_summary->average_prediction_scale, 0.65, 2.10);
            calibration_ratio = std::clamp(family_summary->average_calibration_ratio, 0.65, 2.10);
            validation_spread_us = family_summary->average_validation_spread_us;
            overlap_penalty_us =
                std::max(0.0, (family_summary->average_copy_share - family_summary->average_transfer_overlap_ratio) * 56.0) +
                (family_summary->average_budget_pressure * 42.0) -
                (family_summary->average_queue_separation_ratio * 18.0) +
                ((1.0 - family_summary->average_staging_hit_rate) * 36.0) +
                (family_summary->average_cross_device_sync_cost_us * 0.08) +
                (family_summary->average_residency_pressure * 24.0);
        }
        if (const auto penalty_it =
                backend_penalty_cache.find(
                    backend_penalty_key(report_signature, graph_set_signature, workload, operation.name, candidate));
            penalty_it != backend_penalty_cache.end()) {
            backend_penalty_us = penalty_it->second;
        }
        candidate.telemetry_staging_hit_rate = telemetry_staging_hit_rate;
        candidate.telemetry_cross_device_sync_cost_us = telemetry_cross_device_sync_cost_us;
        candidate.telemetry_residency_pressure = telemetry_residency_pressure;

        const double placement_bias_us =
            placement.strategy_source == PlanStrategySource::explicit_request
                ? 0.0
                : operation_placement_bias_us(
                      score_operation_placement(workload, operation, structural_placement_affinity(workload, operation)),
                      candidate,
                      graph_lookup);
        const double surrogate_latency_us =
            (graph.predicted_latency_us * system.sustained_slowdown * learning_scale * calibration_ratio) + system_penalty_us +
            backend_penalty_us +
            overlap_penalty_us +
            placement_bias_us +
            llm_cpu_policy_bias_us(workload, operation, candidate, graph_lookup) +
            device_agnostic_execution_bias_us(workload, operation, candidate, graph_lookup) +
            auto_accelerator_data_parallel_bias_us(workload, operation, placement, candidate, graph_lookup) +
            hybrid_multi_accelerator_host_bias_us(workload, operation, placement, candidate, graph_lookup);
        BenchmarkRecord benchmark;
        const auto validation_rounds =
            choose_validation_rounds(
                observations,
                average_relative_error,
                validation_spread_us,
                calibration_ratio,
                cached_config != nullptr,
                validation_tier);
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
        benchmark.validation_samples = validation_rounds;
        benchmark.effective_latency_us = std::max(0.01, surrogate_latency_us);
        benchmark.system_penalty_us = system_penalty_us;
        benchmark.relative_error = graph.expected_relative_error;
        benchmark.accuracy_within_tolerance =
            graph.expected_relative_error <= (candidate.target_error_tolerance + 1.0e-12);
        benchmark.simulated = true;
        benchmark.speedup_vs_reference = 1.0;
        benchmark.reference_latency_us = historical_latency_us;
        log_observation_budget += static_cast<double>(observations + 1u);

        evaluations.push_back(CandidateEvaluation{
            std::move(candidate),
            std::move(graph),
            std::move(benchmark),
            OptimizationPolicy::heuristic_greedy,
            !spsa_signature.empty() && candidate.signature == spsa_signature,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            historical_latency_us,
            average_reward,
            observations});
        refresh_candidate_objectives(
            evaluations.back(),
            trace_weight,
            evaluations.back().historical_latency_us,
            evaluations.back().average_reward,
            log_observation_budget);
        if (deadline_reached(deadline) && !evaluations.empty()) {
            break;
        }
    }

    if (evaluations.empty()) {
        return {};
    }

    if (should_auto_accelerator_data_parallel(workload, operation, placement, graph_lookup) &&
        operation.op_class == OperationClass::matmul) {
        const auto accelerators = accelerator_device_uids(placement, graph_lookup);
        const auto forced = std::find_if(
            evaluations.begin(),
            evaluations.end(),
            [&](const CandidateEvaluation& evaluation) {
                return evaluation.config.strategy == ExecutionStrategy::sharded &&
                       evaluation.config.participating_devices.size() == accelerators.size() &&
                       std::is_permutation(
                           evaluation.config.participating_devices.begin(),
                           evaluation.config.participating_devices.end(),
                           accelerators.begin(),
                           accelerators.end());
            });
        if (forced != evaluations.end()) {
            if (allow_validation && validation_tier != ValidationTier::disabled && !deadline_reached(deadline)) {
                forced->benchmark = benchmark_operation(
                    operation,
                    forced->config,
                    forced->graph,
                    graph_lookup,
                    shape_bucket,
                    forced->benchmark.surrogate_latency_us,
                    forced->benchmark.system_penalty_us,
                    choose_validation_rounds(
                        forced->observations,
                        forced->benchmark.relative_error,
                        forced->benchmark.candidate_spread_us,
                        forced->benchmark.calibration_ratio,
                        cached_config != nullptr,
                        validation_tier));
                refresh_candidate_objectives(
                    *forced,
                    trace_weight,
                    forced->historical_latency_us,
                    forced->average_reward,
                    static_cast<double>(forced->observations + 1u));
            }
            OperationOptimizationResult best;
            best.operation = operation;
            best.config = forced->config;
            best.graph = std::move(forced->graph);
            best.benchmark = std::move(forced->benchmark);
            best.benchmark.optimizer_name = "auto_accelerator_data_parallel";
            best.benchmark.objective_score = forced->learned_objective;
            return best;
        }
    }

    const auto selected_policy = select_policy_for_workload(workload, evaluations);
    if (allow_validation && validation_tier != ValidationTier::disabled) {
        std::vector<std::size_t> ranked_indices(evaluations.size());
        for (std::size_t index = 0; index < ranked_indices.size(); ++index) {
            ranked_indices[index] = index;
        }
        std::sort(
            ranked_indices.begin(),
            ranked_indices.end(),
            [&](const std::size_t left, const std::size_t right) {
                return objective_for_policy(evaluations[left], selected_policy) <
                       objective_for_policy(evaluations[right], selected_policy);
            });
        const auto shortlist =
            choose_validation_shortlist_size(workload, operation, system, evaluations, selected_policy);
        for (std::size_t rank = 0; rank < shortlist; ++rank) {
            if (deadline_reached(deadline)) {
                break;
            }
            auto& candidate = evaluations[ranked_indices[rank]];
            candidate.benchmark = benchmark_operation(
                operation,
                candidate.config,
                candidate.graph,
                graph_lookup,
                shape_bucket,
                candidate.benchmark.surrogate_latency_us,
                candidate.benchmark.system_penalty_us,
                choose_validation_rounds(
                    candidate.observations,
                    candidate.benchmark.relative_error,
                    candidate.benchmark.candidate_spread_us,
                    candidate.benchmark.calibration_ratio,
                    cached_config != nullptr,
                    validation_tier));
            refresh_candidate_objectives(
                candidate,
                trace_weight,
                candidate.historical_latency_us,
                candidate.average_reward,
                static_cast<double>(candidate.observations + 1u));
        }
    }
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
    std::unordered_map<std::string, PerformanceSummary>& performance_cache,
    const std::string& key,
    const std::string& shape_bucket,
    const std::string& device_family_signature,
    const ExecutionConfig& config,
    const double effective_latency_us,
    const double relative_error,
    const double predicted_latency_us,
    const double calibration_ratio,
    const double system_penalty_us,
    const double validation_spread_us,
    const double reward,
    const double copy_share,
    const double transfer_overlap_ratio,
    const double budget_pressure,
    const double queue_separation_ratio,
    const double staging_hit_rate,
    const double cross_device_sync_cost_us,
    const double residency_pressure,
    const double kv_host_residency_ratio) {
    auto& summary = performance_cache[key];
    summary.shape_bucket = shape_bucket;
    summary.device_family_signature = device_family_signature;
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
    summary.average_copy_share +=
        (copy_share - summary.average_copy_share) / sample_count;
    summary.average_transfer_overlap_ratio +=
        (transfer_overlap_ratio - summary.average_transfer_overlap_ratio) / sample_count;
    summary.average_budget_pressure +=
        (budget_pressure - summary.average_budget_pressure) / sample_count;
    summary.average_queue_separation_ratio +=
        (queue_separation_ratio - summary.average_queue_separation_ratio) / sample_count;
    summary.average_staging_hit_rate +=
        (staging_hit_rate - summary.average_staging_hit_rate) / sample_count;
    summary.average_cross_device_sync_cost_us +=
        (cross_device_sync_cost_us - summary.average_cross_device_sync_cost_us) / sample_count;
    summary.average_residency_pressure +=
        (residency_pressure - summary.average_residency_pressure) / sample_count;
    summary.average_kv_host_residency_ratio +=
        (kv_host_residency_ratio - summary.average_kv_host_residency_ratio) / sample_count;
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

std::string to_string(const ValidationTier tier) {
    switch (tier) {
    case ValidationTier::full:
        return "full";
    case ValidationTier::adaptive:
        return "adaptive";
    case ValidationTier::minimal:
        return "minimal";
    case ValidationTier::disabled:
    default:
        return "disabled";
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

BootstrapExecutionOptimizer::BootstrapExecutionOptimizer(std::filesystem::path cache_path)
    : cache_path_(std::move(cache_path)) {}

AdaptiveExecutionOptimizer::AdaptiveExecutionOptimizer(std::filesystem::path cache_path)
    : performance_cache_path_(performance_cache_path_for(cache_path)),
      graph_family_cache_path_(graph_family_cache_path_for(cache_path)),
      cpu_runtime_hint_cache_path_(cpu_runtime_hint_cache_path_for(cache_path)) {}

ExecutionOptimizer::ExecutionOptimizer(std::filesystem::path cache_path)
    : bootstrap_optimizer_(cache_path),
      adaptive_optimizer_(std::move(cache_path)) {}

std::filesystem::path ExecutionOptimizer::default_cache_path() {
    try {
        return std::filesystem::temp_directory_path() / "jakal_core_execution_cache.tsv";
    } catch (const std::exception&) {
        return std::filesystem::path("jakal_core_execution_cache.tsv");
    }
}

OptimizationReport ExecutionOptimizer::optimize(
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::vector<HardwareGraph>& graphs,
    const WorkloadGraph* workload_graph_override,
    const ExecutionTuningOverrides* tuning_overrides) {
    bootstrap_optimizer_.load_cache();
    adaptive_optimizer_.load_cache();
    const auto placement_pass = run_placement_policy_pass(workload, placement, graphs);
    const auto& effective_workload = placement_pass.effective_workload;
    const auto& graph_lookup = placement_pass.graph_lookup;

    OptimizationReport report;
    report.workload_kind = effective_workload.kind;
    report.workload_phase = canonical_workload_phase(effective_workload);
    report.workload_shape_bucket = canonical_workload_shape_bucket(effective_workload);
    report.dataset_tag = effective_workload.dataset_tag;
    report.workload_working_set_bytes = effective_workload.working_set_bytes;
    report.workload_host_exchange_bytes = effective_workload.host_exchange_bytes;
    report.partition_strategy = effective_workload.partition_strategy;
    report.placement = placement;
    report.system_profile = capture_system_profile(effective_workload, graphs);
    adaptive_optimizer_.apply_runtime_state(report.system_profile, graphs);

    report.workload_graph =
        workload_graph_override == nullptr ? default_workload_graph(effective_workload) : *workload_graph_override;
    optimize_workload_operations_for_targets(
        effective_workload,
        placement,
        graph_lookup,
        report.workload_graph,
        tuning_overrides);
    adaptive_optimizer_.apply_feedback_tuning_hints(report.workload_graph);
    const auto config_search_pass = prepare_execution_config_search_pass(
        adaptive_optimizer_,
        bootstrap_optimizer_,
        effective_workload,
        placement,
        report.workload_graph,
        graph_lookup,
        tuning_overrides);
    const auto& compiled_workload = config_search_pass.compiled_workload;
    const auto& operations = report.workload_graph.operations;
    report.signature = build_report_signature(
        effective_workload,
        placement,
        compiled_workload.signature,
        operations,
        tuning_overrides);

    std::vector<CachedExecutionConfig> persisted_configs;
    persisted_configs.reserve(operations.size());

    std::unordered_map<std::string, ExecutionConfig> cache_input = config_search_pass.cached_by_operation;
    report.graph_optimization = optimize_graph_continuous_state(
        report.signature,
        effective_workload,
        report.workload_graph,
        compiled_workload,
        placement,
        operations,
        graph_lookup,
        report.system_profile,
        adaptive_optimizer_.performance_cache(),
        adaptive_optimizer_.graph_family_performance_cache(),
        adaptive_optimizer_.backend_penalty_cache(),
        adaptive_optimizer_.warmed_devices(),
        placement_pass.graph_set_signature,
        (config_search_pass.fully_cached && !config_search_pass.runtime_sensitive_path) ? &cache_input : nullptr,
        config_search_pass.lightweight_path || config_search_pass.runtime_sensitive_path,
        tuning_overrides,
        config_search_pass.validation_tier,
        config_search_pass.optimization_deadline);
    report.graph_optimization.optimizer_name =
        config_search_pass.optimizer_route + ":" + report.graph_optimization.optimizer_name;

    std::unordered_map<std::string, ExecutionConfig> selected_configs;
    for (const auto& operation : operations) {
        const auto cached_it = config_search_pass.cached_by_operation.find(operation.name);
        const ExecutionConfig* cached_config =
            cached_it == config_search_pass.cached_by_operation.end() ? nullptr : &cached_it->second;
        const bool operation_needs_fast_recovery =
            config_search_pass.runtime_sensitive_path &&
            adaptive_optimizer_.should_reoptimize_operation(report.signature, operation.name);
        auto result = optimize_operation(
            report.signature,
            report.workload_graph,
            compiled_workload,
            operation,
            effective_workload,
            placement,
            graph_lookup,
            selected_configs,
            report.system_profile,
            adaptive_optimizer_.performance_cache(),
            adaptive_optimizer_.graph_family_performance_cache(),
            adaptive_optimizer_.backend_penalty_cache(),
            adaptive_optimizer_.warmed_devices(),
            placement_pass.graph_set_signature,
            cached_config,
            &report.graph_optimization.final_state,
            !config_search_pass.lightweight_path || operation_needs_fast_recovery,
            config_search_pass.validation_tier,
            config_search_pass.optimization_deadline);
        if (result.config.signature.empty()) {
            continue;
        }

        result.benchmark.optimizer_name =
            config_search_pass.optimizer_route + ":" + result.benchmark.optimizer_name;
        persisted_configs.push_back(CachedExecutionConfig{
            operation.name,
            result.config});
        report.graph_optimization.total_logical_partitions += result.config.logical_partitions;
        selected_configs[result.operation.name] = result.config;
        report.operations.push_back(std::move(result));
    }

    bootstrap_optimizer_.store_configs(report.signature, std::move(persisted_configs));
    report.loaded_from_cache =
        config_search_pass.fully_cached && !config_search_pass.runtime_sensitive_path;
    return report;
}

CandidatePolicySummary describe_candidate_policy(
    const WorkloadSpec& workload,
    const OperationSpec& operation,
    const SystemProfile& system,
    const bool continuous_state_active,
    const double placement_affinity) {
    StructuralPlacementAffinity affinity;
    affinity.bias_scale = placement_affinity;
    return summarize_candidate_policy(
        candidate_policy_for_workload(workload, operation, system, continuous_state_active, affinity));
}

void ExecutionOptimizer::ingest_execution_feedback(
    const OptimizationReport& report,
    const std::vector<ExecutionFeedbackRecord>& feedback,
    const std::vector<HardwareGraph>& graphs) {
    run_feedback_retuning_pass(adaptive_optimizer_, report, feedback, graphs);
}

void BootstrapExecutionOptimizer::load_cache() {
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
        if (fields.size() != 14 && fields.size() != 21 && fields.size() != 22) {
            continue;
        }

        try {
            ExecutionConfig config;
            const bool has_variant = fields.size() == 22;
            const bool has_extended = fields.size() == 21 || fields.size() == 22;
            std::size_t next = 1;
            config.operation_name = fields[next++];
            config.variant_id = has_variant ? fields[next++] : "legacy";
            config.strategy = parse_strategy(fields[next++]);
            config.primary_device_uid = fields[next++];
            config.participating_devices = split_csv(fields[next++]);
            config.mapped_structural_nodes = split_csv(fields[next++]);
            config.queue_depth = static_cast<std::uint32_t>(std::stoul(fields[next++]));
            config.stages = static_cast<std::uint32_t>(std::stoul(fields[next++]));
            config.tile_x = static_cast<std::uint32_t>(std::stoul(fields[next++]));
            config.tile_y = static_cast<std::uint32_t>(std::stoul(fields[next++]));
            config.tile_k = static_cast<std::uint32_t>(std::stoul(fields[next++]));
            if (has_extended) {
                config.logical_partitions = static_cast<std::uint32_t>(std::stoul(fields[next++]));
            }
            config.overlap_transfers = std::stoi(fields[next++]) != 0;
            config.use_low_precision = std::stoi(fields[next++]) != 0;
            config.target_error_tolerance = std::stod(fields[next++]);
            if (has_extended) {
                config.queue_depth_scale = std::stod(fields[next++]);
                config.stage_scale = std::stod(fields[next++]);
                config.tile_scale = std::stod(fields[next++]);
                config.overlap_ratio = std::stod(fields[next++]);
                config.partition_intensity = std::stod(fields[next++]);
                config.precision_mix = std::stod(fields[next++]);
            }
            cache_[fields[0]].push_back(CachedExecutionConfig{config.operation_name, std::move(config)});
        } catch (const std::exception&) {
            continue;
        }
    }
}

bool BootstrapExecutionOptimizer::has_full_cache(
    const std::string& report_signature,
    const std::vector<OperationSpec>& operations,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    std::unordered_map<std::string, ExecutionConfig>* cached_by_operation) {
    if (cached_by_operation == nullptr) {
        return false;
    }

    load_cache();
    cached_by_operation->clear();
    const auto cache_it = cache_.find(report_signature);
    if (cache_it == cache_.end()) {
        return false;
    }

    for (const auto& cached : cache_it->second) {
        if (!graph_lookup.contains(cached.config.primary_device_uid) ||
            std::any_of(
                cached.config.participating_devices.begin(),
                cached.config.participating_devices.end(),
                [&](const std::string& device_uid) { return !graph_lookup.contains(device_uid); })) {
            cached_by_operation->clear();
            return false;
        }
        cached_by_operation->emplace(cached.operation_name, cached.config);
    }
    for (const auto& operation : operations) {
        if (!cached_by_operation->contains(operation.name)) {
            cached_by_operation->clear();
            return false;
        }
    }
    return true;
}

void BootstrapExecutionOptimizer::store_configs(
    const std::string& report_signature,
    std::vector<CachedExecutionConfig> configs) {
    load_cache();
    cache_[report_signature] = std::move(configs);
    persist_cache();
}

void AdaptiveExecutionOptimizer::load_cache() {
    if (cache_loaded_) {
        return;
    }
    cache_loaded_ = true;

    const auto load_summary_map =
        [&](const std::filesystem::path& path, std::unordered_map<std::string, PerformanceSummary>& target) {
            std::ifstream input(path);
            if (!input.is_open()) {
                return;
            }

            std::string line;
            while (std::getline(input, line)) {
                if (line.empty() || line[0] == '#') {
                    continue;
                }

                const auto fields = split_tab(line);
                if (fields.size() != 19 && fields.size() != 20 && fields.size() != 26 && fields.size() != 27 &&
                    fields.size() != 28 && fields.size() != 29 && fields.size() != 30 &&
                    fields.size() != 31 && fields.size() != 34 && fields.size() != 38) {
                    continue;
                }

                try {
                    PerformanceSummary summary;
                    const bool has_variant = fields.size() == 29 || fields.size() == 30 || fields.size() == 31 || fields.size() == 34;
                    const bool has_extended =
                        fields.size() == 26 || fields.size() == 27 || fields.size() == 28 ||
                        fields.size() == 29 || fields.size() == 30 || fields.size() == 31 || fields.size() == 34;
                    const bool has_calibration =
                        fields.size() == 28 || fields.size() == 29 || fields.size() == 30 || fields.size() == 31 || fields.size() == 34;
                    const bool has_family =
                        fields.size() == 20 || fields.size() == 27 || fields.size() == 30 || fields.size() == 31 || fields.size() == 34;
                    const bool has_transfer_metrics = fields.size() == 31 || fields.size() == 34 || fields.size() == 38;
                    const bool has_residency_metrics = fields.size() == 38;
                    std::size_t next = 1;
                    summary.shape_bucket = fields[next++];
                    summary.device_family_signature = has_family ? fields[next++] : "";
                    summary.config.operation_name = fields[next++];
                    summary.config.variant_id = has_variant ? fields[next++] : "legacy";
                    summary.config.strategy = parse_strategy(fields[next++]);
                    summary.config.primary_device_uid = fields[next++];
                    summary.config.participating_devices = split_csv(fields[next++]);
                    summary.config.queue_depth = static_cast<std::uint32_t>(std::stoul(fields[next++]));
                    summary.config.stages = static_cast<std::uint32_t>(std::stoul(fields[next++]));
                    summary.config.tile_x = static_cast<std::uint32_t>(std::stoul(fields[next++]));
                    summary.config.tile_y = static_cast<std::uint32_t>(std::stoul(fields[next++]));
                    summary.config.tile_k = static_cast<std::uint32_t>(std::stoul(fields[next++]));
                    if (has_extended) {
                        summary.config.logical_partitions = static_cast<std::uint32_t>(std::stoul(fields[next++]));
                    }
                    summary.config.overlap_transfers = std::stoi(fields[next++]) != 0;
                    summary.config.use_low_precision = std::stoi(fields[next++]) != 0;
                    if (has_extended) {
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
                    if (has_calibration) {
                        summary.average_calibration_ratio = std::stod(fields[next++]);
                    }
                    summary.average_system_penalty_us = std::stod(fields[next++]);
                    if (has_calibration) {
                        summary.average_validation_spread_us = std::stod(fields[next++]);
                    }
                    summary.average_reward = std::stod(fields[next++]);
                    if (has_transfer_metrics) {
                        summary.average_copy_share = std::stod(fields[next++]);
                        summary.average_transfer_overlap_ratio = std::stod(fields[next++]);
                        summary.average_budget_pressure = std::stod(fields[next++]);
                        summary.average_queue_separation_ratio = std::stod(fields[next++]);
                    }
                    if (has_residency_metrics) {
                        summary.average_staging_hit_rate = std::stod(fields[next++]);
                        summary.average_cross_device_sync_cost_us = std::stod(fields[next++]);
                        summary.average_residency_pressure = std::stod(fields[next++]);
                        summary.average_kv_host_residency_ratio = std::stod(fields[next++]);
                    }
                    target[fields[0]] = std::move(summary);
                } catch (const std::exception&) {
                    continue;
                }
            }
        };

    load_summary_map(performance_cache_path_, performance_cache_);
    load_summary_map(graph_family_cache_path_, graph_family_performance_cache_);

    std::ifstream hint_input(cpu_runtime_hint_cache_path_);
    if (!hint_input.is_open()) {
        return;
    }

    std::string hint_line;
    while (std::getline(hint_input, hint_line)) {
        if (hint_line.empty() || hint_line[0] == '#') {
            continue;
        }
        const auto fields = split_tab(hint_line);
        if (fields.size() != 5u && fields.size() != 8u && fields.size() != 9u &&
            fields.size() != 10u && fields.size() != 11u) {
            continue;
        }
        try {
            CpuRuntimeHintSummary summary;
            summary.shape_bucket = fields[1];
            summary.preferred_parallel_chunk = static_cast<std::uint32_t>(std::stoul(fields[2]));
            if (fields.size() == 11u) {
                summary.preferred_tile_m = static_cast<std::uint32_t>(std::stoul(fields[3]));
                summary.preferred_tile_n = static_cast<std::uint32_t>(std::stoul(fields[4]));
                summary.preferred_tile_k = static_cast<std::uint32_t>(std::stoul(fields[5]));
                summary.preferred_single_thread_cutoff = static_cast<std::uint32_t>(std::stoul(fields[6]));
                summary.preferred_kernel_family = fields[7];
                summary.preferred_use_avx512 = std::stoi(fields[8]) != 0;
                summary.observations = static_cast<std::uint32_t>(std::stoul(fields[9]));
                summary.average_effective_latency_us = std::stod(fields[10]);
            } else if (fields.size() == 10u) {
                summary.preferred_tile_m = static_cast<std::uint32_t>(std::stoul(fields[3]));
                summary.preferred_tile_n = static_cast<std::uint32_t>(std::stoul(fields[4]));
                summary.preferred_tile_k = static_cast<std::uint32_t>(std::stoul(fields[5]));
                summary.preferred_single_thread_cutoff = static_cast<std::uint32_t>(std::stoul(fields[6]));
                summary.preferred_kernel_family = fields[7];
                summary.observations = static_cast<std::uint32_t>(std::stoul(fields[8]));
                summary.average_effective_latency_us = std::stod(fields[9]);
            } else if (fields.size() == 9u) {
                summary.preferred_tile_m = static_cast<std::uint32_t>(std::stoul(fields[3]));
                summary.preferred_tile_n = static_cast<std::uint32_t>(std::stoul(fields[4]));
                summary.preferred_tile_k = static_cast<std::uint32_t>(std::stoul(fields[5]));
                summary.preferred_use_avx512 = std::stoi(fields[6]) != 0;
                summary.observations = static_cast<std::uint32_t>(std::stoul(fields[7]));
                summary.average_effective_latency_us = std::stod(fields[8]);
            } else if (fields.size() == 8u) {
                summary.preferred_tile_m = static_cast<std::uint32_t>(std::stoul(fields[3]));
                summary.preferred_tile_n = static_cast<std::uint32_t>(std::stoul(fields[4]));
                summary.preferred_tile_k = static_cast<std::uint32_t>(std::stoul(fields[5]));
                summary.observations = static_cast<std::uint32_t>(std::stoul(fields[6]));
                summary.average_effective_latency_us = std::stod(fields[7]);
            } else {
                summary.observations = static_cast<std::uint32_t>(std::stoul(fields[3]));
                summary.average_effective_latency_us = std::stod(fields[4]);
            }
            cpu_runtime_hint_cache_[fields[0]] = std::move(summary);
        } catch (const std::exception&) {
            continue;
        }
    }
}

void AdaptiveExecutionOptimizer::apply_runtime_state(
    SystemProfile& profile,
    const std::vector<HardwareGraph>& graphs) const {
    if (!device_sustained_slowdown_.empty()) {
        double total_slowdown = 0.0;
        for (const auto& [uid, slowdown] : device_sustained_slowdown_) {
            (void)uid;
            total_slowdown += slowdown;
        }
        profile.sustained_slowdown *=
            std::max(1.0, total_slowdown / static_cast<double>(device_sustained_slowdown_.size()));
    }

    const double warmed_fraction =
        graphs.empty() ? 0.0 : (static_cast<double>(warmed_devices_.size()) / static_cast<double>(graphs.size()));
    profile.readiness_score = clamp_unit(
        (0.12 + (0.88 * warmed_fraction * profile.amortization_gain)) /
        std::max(profile.sustained_slowdown, 1.0));
    profile.stability_score = clamp_unit(profile.stability_score * (0.85 + (0.15 * warmed_fraction)));
    if (!reoptimization_pressure_.empty()) {
        double pressure_sum = 0.0;
        for (const auto& [signature, pressure] : reoptimization_pressure_) {
            (void)signature;
            pressure_sum += static_cast<double>(pressure);
        }
        const double average_pressure = pressure_sum / static_cast<double>(reoptimization_pressure_.size());
        profile.stability_score = clamp_unit(profile.stability_score / (1.0 + (0.08 * average_pressure)));
    }
}

bool AdaptiveExecutionOptimizer::should_use_lightweight_path(
    const std::string& report_signature,
    const bool fully_cached) const {
    return fully_cached && !should_reoptimize(report_signature);
}

bool AdaptiveExecutionOptimizer::should_reoptimize(const std::string& report_signature) const {
    const auto it = reoptimization_pressure_.find(report_signature);
    return it != reoptimization_pressure_.end() && it->second > 0u;
}

bool AdaptiveExecutionOptimizer::should_reoptimize_operation(
    const std::string& report_signature,
    const std::string& operation_name) const {
    const auto key = report_signature + "|" + operation_name;
    const auto it = operation_reoptimization_pressure_.find(key);
    return it != operation_reoptimization_pressure_.end() && it->second > 0u;
}

const std::unordered_map<std::string, PerformanceSummary>& AdaptiveExecutionOptimizer::performance_cache() const {
    return performance_cache_;
}

const std::unordered_map<std::string, PerformanceSummary>& AdaptiveExecutionOptimizer::graph_family_performance_cache() const {
    return graph_family_performance_cache_;
}

const std::unordered_map<std::string, CpuRuntimeHintSummary>& AdaptiveExecutionOptimizer::cpu_runtime_hint_cache() const {
    return cpu_runtime_hint_cache_;
}

const std::unordered_map<std::string, double>& AdaptiveExecutionOptimizer::backend_penalty_cache() const {
    return backend_penalty_cache_;
}

const std::unordered_map<std::string, bool>& AdaptiveExecutionOptimizer::warmed_devices() const {
    return warmed_devices_;
}

void AdaptiveExecutionOptimizer::apply_feedback_tuning_hints(WorkloadGraph& graph) const {
    for (auto& operation : graph.operations) {
        if (!supports_feedback_tuned_cpu_chunk(operation)) {
            continue;
        }
        const auto key = cpu_runtime_hint_bucket_for(operation);
        const auto it = cpu_runtime_hint_cache_.find(key);
        if (it == cpu_runtime_hint_cache_.end()) {
            continue;
        }
        if (it->second.preferred_parallel_chunk > 0u) {
            operation.cpu_parallel_chunk =
                clamp_cpu_parallel_chunk_hint(operation, it->second.preferred_parallel_chunk);
        }
        if (it->second.preferred_single_thread_cutoff > 0u) {
            operation.cpu_single_thread_cutoff =
                clamp_cpu_single_thread_cutoff_hint(operation, it->second.preferred_single_thread_cutoff);
        }
        if (!it->second.preferred_kernel_family.empty() &&
            it->second.preferred_kernel_family != "auto") {
            operation.cpu_kernel_family = it->second.preferred_kernel_family;
        }
        if (supports_feedback_tuned_cpu_tiles(operation)) {
            if (it->second.preferred_tile_m > 0u) {
                operation.cpu_tile_m =
                    clamp_cpu_tile_hint(it->second.preferred_tile_m, static_cast<std::uint32_t>(operation.extents[0]), 4u, 4u);
            }
            if (it->second.preferred_tile_n > 0u && operation.extents.size() > 1u) {
                operation.cpu_tile_n =
                    clamp_cpu_tile_hint(it->second.preferred_tile_n, static_cast<std::uint32_t>(operation.extents[1]), 8u, 8u);
            }
            if (it->second.preferred_tile_k > 0u && operation.extents.size() > 2u) {
                operation.cpu_tile_k =
                    clamp_cpu_tile_hint(it->second.preferred_tile_k, static_cast<std::uint32_t>(operation.extents[2]), 8u, 8u);
            }
        }
        if (supports_feedback_tuned_cpu_avx512(operation) &&
            executors::host_native_kernels_compiled_with_avx512()) {
            operation.cpu_use_avx512 = it->second.preferred_use_avx512;
        }
    }
}

void AdaptiveExecutionOptimizer::ingest_execution_feedback(
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
    std::unordered_map<std::string, const HardwareGraph*> graph_lookup;
    graph_lookup.reserve(graphs.size());
    for (const auto& graph : graphs) {
        graph_lookup.emplace(graph.uid, &graph);
    }
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
        false,
        report.partition_strategy};

    std::uint32_t report_pressure = 0u;
    for (const auto& optimized : report.operations) {
        const auto feedback_it = feedback_by_operation.find(optimized.operation.name);
        if (feedback_it == feedback_by_operation.end()) {
            continue;
        }

        const auto& record = *feedback_it->second;
        const double effective_latency_us = std::max(0.01, record.runtime_us);
        const double observed_penalty_us =
            std::max(0.0, effective_latency_us - std::max(optimized.graph.predicted_latency_us, 0.0));
        const auto family_signature = device_family_signature_for_config(optimized.config, graph_lookup);
        const auto perf_key = performance_key(
            report.signature,
            graph_set_signature,
            feedback_workload,
            report.system_profile,
            optimized.benchmark.shape_bucket,
            optimized.config);
        update_performance_summary(
            performance_cache_,
            perf_key,
            optimized.benchmark.shape_bucket,
            family_signature,
            optimized.config,
            effective_latency_us,
            record.relative_error,
            optimized.graph.predicted_latency_us,
            optimized.benchmark.calibration_ratio,
            observed_penalty_us,
            optimized.benchmark.candidate_spread_us,
            std::log(std::max(record.reference_runtime_us / effective_latency_us, 1.0e-6)),
            record.copy_share,
            record.transfer_overlap_ratio,
            record.budget_pressure,
            record.queue_separation_ratio,
            record.staging_hit_rate,
            record.cross_device_sync_cost_us,
            record.residency_pressure,
            record.kv_host_residency_ratio);
        update_performance_summary(
            graph_family_performance_cache_,
            performance_family_key(report.signature, optimized.benchmark.shape_bucket, optimized.config),
            optimized.benchmark.shape_bucket,
            family_signature,
            optimized.config,
            effective_latency_us,
            record.relative_error,
            optimized.graph.predicted_latency_us,
            optimized.benchmark.calibration_ratio,
            observed_penalty_us,
            optimized.benchmark.candidate_spread_us,
            std::log(std::max(record.reference_runtime_us / effective_latency_us, 1.0e-6)),
            record.copy_share,
            record.transfer_overlap_ratio,
            record.budget_pressure,
            record.queue_separation_ratio,
            record.staging_hit_rate,
            record.cross_device_sync_cost_us,
            record.residency_pressure,
            record.kv_host_residency_ratio);
        update_device_learning_state(
            warmed_devices_,
            device_sustained_slowdown_,
            optimized.config,
            effective_latency_us,
            optimized.graph.predicted_latency_us);

        const double slowdown_vs_reference =
            record.reference_runtime_us > 0.0 ? (effective_latency_us / record.reference_runtime_us) : 1.0;
        const auto penalty_key =
            backend_penalty_key(
                report.signature,
                graph_set_signature,
                feedback_workload,
                optimized.operation.name,
                optimized.config);
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

        if (record.used_host &&
            !record.used_opencl &&
            supports_feedback_tuned_cpu_chunk(optimized.operation)) {
            const auto hint_key = cpu_runtime_hint_bucket_for(optimized.operation);
            const auto tuned_chunk = feedback_adjusted_cpu_parallel_chunk(
                optimized.operation,
                graphs,
                effective_latency_us,
                optimized.graph.predicted_latency_us);
            const auto tuned_single_thread_cutoff = clamp_cpu_single_thread_cutoff_hint(
                optimized.operation,
                std::max(
                    optimized.operation.cpu_single_thread_cutoff,
                    tuned_chunk * std::max<std::uint32_t>(1u, std::min<std::uint32_t>(8u, host_parallelism_hint(graphs)))));
            auto& hint_summary = cpu_runtime_hint_cache_[hint_key];
            hint_summary.shape_bucket = hint_key;
            const double observation_count = static_cast<double>(hint_summary.observations);
            const double blended_chunk =
                ((static_cast<double>(std::max(hint_summary.preferred_parallel_chunk, 1u)) * observation_count) +
                 static_cast<double>(tuned_chunk)) /
                (observation_count + 1.0);
            hint_summary.preferred_parallel_chunk = clamp_cpu_parallel_chunk_hint(
                optimized.operation,
                static_cast<std::uint32_t>(std::lround(blended_chunk)));
            hint_summary.preferred_single_thread_cutoff = clamp_cpu_single_thread_cutoff_hint(
                optimized.operation,
                static_cast<std::uint32_t>(std::lround(
                    ((static_cast<double>(std::max(hint_summary.preferred_single_thread_cutoff, 1u)) * observation_count) +
                     static_cast<double>(tuned_single_thread_cutoff)) /
                    (observation_count + 1.0))));
            hint_summary.preferred_kernel_family = optimized.operation.cpu_kernel_family;
            if (supports_feedback_tuned_cpu_tiles(optimized.operation)) {
                std::uint32_t tuned_tile_m =
                    std::max(optimized.operation.cpu_tile_m, std::uint32_t{4u});
                std::uint32_t tuned_tile_n =
                    std::max(optimized.operation.cpu_tile_n, std::uint32_t{8u});
                std::uint32_t tuned_tile_k =
                    std::max(optimized.operation.cpu_tile_k, std::uint32_t{8u});
                feedback_adjusted_cpu_tiles(
                    optimized.operation,
                    graphs,
                    effective_latency_us,
                    optimized.graph.predicted_latency_us,
                    tuned_tile_m,
                    tuned_tile_n,
                    tuned_tile_k);
                const double tile_observation_count =
                    static_cast<double>(hint_summary.observations);
                const auto blend_tile = [&](const std::uint32_t previous, const std::uint32_t current) {
                    return static_cast<std::uint32_t>(std::lround(
                        ((static_cast<double>(std::max(previous, 1u)) * tile_observation_count) +
                         static_cast<double>(current)) /
                        (tile_observation_count + 1.0)));
                };
                hint_summary.preferred_tile_m = clamp_cpu_tile_hint(
                    blend_tile(hint_summary.preferred_tile_m, tuned_tile_m),
                    static_cast<std::uint32_t>(optimized.operation.extents[0]),
                    4u,
                    4u);
                hint_summary.preferred_tile_n = clamp_cpu_tile_hint(
                    blend_tile(hint_summary.preferred_tile_n, tuned_tile_n),
                    static_cast<std::uint32_t>(optimized.operation.extents[1]),
                    8u,
                    8u);
                hint_summary.preferred_tile_k = clamp_cpu_tile_hint(
                    blend_tile(hint_summary.preferred_tile_k, tuned_tile_k),
                    static_cast<std::uint32_t>(optimized.operation.extents[2]),
                    8u,
                    8u);
            }
            if (supports_feedback_tuned_cpu_avx512(optimized.operation) &&
                !optimized.config.use_low_precision) {
                const bool tuned_use_avx512 = feedback_adjusted_cpu_avx512_choice(
                    optimized.operation,
                    graphs,
                    effective_latency_us,
                    optimized.graph.predicted_latency_us);
                const auto prior_observations = hint_summary.observations;
                const auto avx512_votes =
                    (hint_summary.preferred_use_avx512 ? prior_observations : 0u) +
                    (tuned_use_avx512 ? 1u : 0u);
                hint_summary.preferred_use_avx512 =
                    avx512_votes * 2u >= (prior_observations + 1u);
            }
            ++hint_summary.observations;
            hint_summary.average_effective_latency_us +=
                (effective_latency_us - hint_summary.average_effective_latency_us) /
                static_cast<double>(hint_summary.observations);
        }

        std::uint32_t operation_pressure = 0u;
        if (!record.verified || record.relative_error > (optimized.config.target_error_tolerance + 1.0e-12)) {
            operation_pressure = std::max(operation_pressure, 3u);
        }
        if (!record.used_host && slowdown_vs_reference > 1.10) {
            operation_pressure = std::max(operation_pressure, slowdown_vs_reference > 1.50 ? 3u : 2u);
        }
        if (!record.used_host && optimized.graph.predicted_latency_us > 0.0 &&
            effective_latency_us > (optimized.graph.predicted_latency_us * 1.20)) {
            operation_pressure = std::max(operation_pressure, 2u);
        }
        if (record.used_host &&
            !record.used_opencl &&
            supports_feedback_tuned_cpu_chunk(optimized.operation) &&
            optimized.graph.predicted_latency_us > 0.0 &&
            effective_latency_us > (optimized.graph.predicted_latency_us * 1.08)) {
            operation_pressure = std::max(operation_pressure, 1u);
        }
        if (record.staging_hit_rate < 0.35 ||
            record.cross_device_sync_cost_us > 120.0 ||
            record.residency_pressure > 0.85) {
            operation_pressure = std::max(operation_pressure, 2u);
        }

        const auto operation_key = report.signature + "|" + optimized.operation.name;
        if (operation_pressure == 0u) {
            operation_reoptimization_pressure_.erase(operation_key);
        } else {
            operation_reoptimization_pressure_[operation_key] = operation_pressure;
        }
        report_pressure = std::max(report_pressure, operation_pressure);
    }

    if (report_pressure == 0u) {
        reoptimization_pressure_.erase(report.signature);
    } else {
        reoptimization_pressure_[report.signature] = report_pressure;
    }

    persist_cache();
}

void BootstrapExecutionOptimizer::persist_cache() const {
    const auto parent = cache_path_.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }

    std::ofstream output(cache_path_, std::ios::trunc);
    if (!output.is_open()) {
        return;
    }

    output << "# signature\toperation\tvariant\tstrategy\tprimary_device\tparticipating_devices\tmapped_nodes\tqueue_depth\tstages\ttile_x\ttile_y\ttile_k\tlogical_partitions\toverlap\tlow_precision\ttolerance\tqueue_scale\tstage_scale\ttile_scale\toverlap_ratio\tpartition_intensity\tprecision_mix\n";
    for (const auto& [signature, configs] : cache_) {
        for (const auto& cached : configs) {
            output << signature << '\t'
                   << cached.operation_name << '\t'
                   << cached.config.variant_id << '\t'
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
}

void AdaptiveExecutionOptimizer::persist_cache() const {
    const auto parent = performance_cache_path_.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }

    const auto persist_summary_map =
        [](const std::filesystem::path& path, const std::unordered_map<std::string, PerformanceSummary>& source) {
            std::ofstream output(path, std::ios::trunc);
            if (!output.is_open()) {
                return;
            }

            output
                << "# key\tshape_bucket\tdevice_family_signature\toperation\tvariant\tstrategy\tprimary_device\tparticipating_devices\tqueue_depth\tstages\ttile_x\ttile_y\ttile_k\tlogical_partitions\toverlap\tlow_precision\tqueue_scale\tstage_scale\ttile_scale\toverlap_ratio\tpartition_intensity\tprecision_mix\tobservations\tavg_latency\tavg_error\tavg_scale\tavg_calibration_ratio\tavg_system_penalty\tavg_validation_spread\tavg_reward\tavg_copy_share\tavg_transfer_overlap_ratio\tavg_budget_pressure\tavg_queue_separation_ratio\tavg_staging_hit_rate\tavg_cross_device_sync_cost_us\tavg_residency_pressure\tavg_kv_host_residency_ratio\n";
            for (const auto& [key, summary] : source) {
                output << key << '\t'
                       << summary.shape_bucket << '\t'
                       << summary.device_family_signature << '\t'
                       << summary.config.operation_name << '\t'
                       << summary.config.variant_id << '\t'
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
                       << summary.average_reward << '\t'
                       << summary.average_copy_share << '\t'
                       << summary.average_transfer_overlap_ratio << '\t'
                       << summary.average_budget_pressure << '\t'
                       << summary.average_queue_separation_ratio << '\t'
                       << summary.average_staging_hit_rate << '\t'
                       << summary.average_cross_device_sync_cost_us << '\t'
                       << summary.average_residency_pressure << '\t'
                       << summary.average_kv_host_residency_ratio << '\n';
            }
        };
    persist_summary_map(performance_cache_path_, performance_cache_);
    persist_summary_map(graph_family_cache_path_, graph_family_performance_cache_);

    std::ofstream hint_output(cpu_runtime_hint_cache_path_, std::ios::trunc);
    if (!hint_output.is_open()) {
        return;
    }
    hint_output << "# key\tshape_bucket\tpreferred_parallel_chunk\tpreferred_tile_m\tpreferred_tile_n\tpreferred_tile_k\tpreferred_single_thread_cutoff\tpreferred_kernel_family\tpreferred_use_avx512\tobservations\tavg_latency\n";
    for (const auto& [key, summary] : cpu_runtime_hint_cache_) {
        hint_output << key << '\t'
                    << summary.shape_bucket << '\t'
                    << summary.preferred_parallel_chunk << '\t'
                    << summary.preferred_tile_m << '\t'
                    << summary.preferred_tile_n << '\t'
                    << summary.preferred_tile_k << '\t'
                    << summary.preferred_single_thread_cutoff << '\t'
                    << summary.preferred_kernel_family << '\t'
                    << (summary.preferred_use_avx512 ? 1 : 0) << '\t'
                    << summary.observations << '\t'
                    << summary.average_effective_latency_us << '\n';
    }
}

}  // namespace jakal

