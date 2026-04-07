#include "jakal/execution.hpp"
#include "jakal/workloads.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>

namespace jakal {
namespace {

constexpr std::uint64_t kKiB = 1024ull;
constexpr std::uint64_t kMiB = 1024ull * 1024ull;

std::uint64_t clamp_u64(const std::uint64_t value, const std::uint64_t min_value, const std::uint64_t max_value) {
    return std::min(std::max(value, min_value), max_value);
}

std::uint32_t round_down_to_multiple(std::uint32_t value, const std::uint32_t multiple) {
    if (multiple == 0u) {
        return value;
    }
    value = std::max(value, multiple);
    return value - (value % multiple);
}

std::unordered_map<std::string, std::uint32_t> operation_indices(const WorkloadGraph& graph) {
    std::unordered_map<std::string, std::uint32_t> indices;
    indices.reserve(graph.operations.size());
    for (std::uint32_t index = 0; index < graph.operations.size(); ++index) {
        indices.emplace(graph.operations[index].name, index);
    }
    return indices;
}

std::uint32_t lookup_index(
    const std::unordered_map<std::string, std::uint32_t>& indices,
    const std::string& operation_name,
    const std::uint32_t fallback) {
    const auto it = indices.find(operation_name);
    return it == indices.end() ? fallback : it->second;
}

OperationSpec make_elementwise(
    const std::string& name,
    const std::uint64_t elements,
    std::vector<std::string> inputs,
    std::vector<std::string> outputs,
    std::vector<std::string> temporaries = {},
    const double tolerance = 5.0e-4) {
    OperationSpec operation;
    operation.name = name;
    operation.op_class = OperationClass::elementwise_map;
    operation.extents = {elements};
    operation.input_bytes = elements * sizeof(float) * 2ull;
    operation.output_bytes = elements * sizeof(float);
    operation.estimated_flops = static_cast<double>(elements) * 3.0;
    operation.max_relative_error = tolerance;
    operation.parallelizable = true;
    operation.streaming_friendly = true;
    operation.input_tensor_ids = std::move(inputs);
    operation.output_tensor_ids = std::move(outputs);
    operation.temporary_tensor_ids = std::move(temporaries);
    return operation;
}

OperationSpec make_reduction(
    const std::string& name,
    const std::uint64_t elements,
    std::vector<std::string> inputs,
    std::vector<std::string> outputs,
    std::vector<std::string> temporaries = {},
    const double tolerance = 1.0e-3) {
    OperationSpec operation;
    operation.name = name;
    operation.op_class = OperationClass::reduction;
    operation.extents = {elements};
    operation.input_bytes = elements * sizeof(float);
    operation.output_bytes = sizeof(float);
    operation.temporary_bytes = 32ull * kKiB;
    operation.estimated_flops = static_cast<double>(elements);
    operation.max_relative_error = tolerance;
    operation.parallelizable = true;
    operation.reduction_like = true;
    operation.input_tensor_ids = std::move(inputs);
    operation.output_tensor_ids = std::move(outputs);
    operation.temporary_tensor_ids = std::move(temporaries);
    return operation;
}

OperationSpec make_matmul(
    const std::string& name,
    const std::uint32_t m,
    const std::uint32_t n,
    const std::uint32_t k,
    std::vector<std::string> inputs,
    std::vector<std::string> outputs,
    std::vector<std::string> temporaries = {},
    const double tolerance = 2.0e-3) {
    OperationSpec operation;
    operation.name = name;
    operation.op_class = OperationClass::matmul;
    operation.extents = {m, n, k};
    operation.input_bytes = 2ull * m * k * sizeof(float);
    operation.output_bytes = 1ull * m * n * sizeof(float);
    operation.estimated_flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    operation.max_relative_error = tolerance;
    operation.parallelizable = true;
    operation.matrix_friendly = true;
    operation.input_tensor_ids = std::move(inputs);
    operation.output_tensor_ids = std::move(outputs);
    operation.temporary_tensor_ids = std::move(temporaries);
    return operation;
}

OperationSpec make_convolution(
    const std::string& name,
    const std::uint32_t height,
    const std::uint32_t width,
    std::vector<std::string> inputs,
    std::vector<std::string> outputs,
    std::vector<std::string> temporaries = {},
    const double tolerance = 2.0e-3) {
    OperationSpec operation;
    operation.name = name;
    operation.op_class = OperationClass::convolution_2d;
    operation.extents = {height, width};
    operation.input_bytes = 1ull * height * width * sizeof(float);
    operation.output_bytes = 1ull * (height - 2u) * (width - 2u) * sizeof(float);
    operation.temporary_bytes = 9ull * sizeof(float);
    operation.estimated_flops = 18.0 * static_cast<double>(height - 2u) * static_cast<double>(width - 2u);
    operation.max_relative_error = tolerance;
    operation.parallelizable = true;
    operation.streaming_friendly = true;
    operation.input_tensor_ids = std::move(inputs);
    operation.output_tensor_ids = std::move(outputs);
    operation.temporary_tensor_ids = std::move(temporaries);
    return operation;
}

OperationSpec make_resample(
    const std::string& name,
    const std::uint32_t src_h,
    const std::uint32_t src_w,
    const std::uint32_t dst_h,
    const std::uint32_t dst_w,
    std::vector<std::string> inputs,
    std::vector<std::string> outputs,
    std::vector<std::string> temporaries = {},
    const double tolerance = 1.5e-3) {
    OperationSpec operation;
    operation.name = name;
    operation.op_class = OperationClass::resample_2d;
    operation.extents = {src_h, src_w, dst_h, dst_w};
    operation.input_bytes = 1ull * src_h * src_w * sizeof(float);
    operation.output_bytes = 1ull * dst_h * dst_w * sizeof(float);
    operation.estimated_flops = 8.0 * static_cast<double>(dst_h) * static_cast<double>(dst_w);
    operation.max_relative_error = tolerance;
    operation.parallelizable = true;
    operation.streaming_friendly = true;
    operation.input_tensor_ids = std::move(inputs);
    operation.output_tensor_ids = std::move(outputs);
    operation.temporary_tensor_ids = std::move(temporaries);
    return operation;
}

void add_tensor(
    WorkloadGraph& graph,
    const std::string& id,
    const std::uint64_t bytes,
    const std::string& producer,
    std::vector<std::string> consumers,
    const bool persistent = false,
    const bool temporary = false,
    const bool host_visible = false,
    const std::string& alias_group = std::string()) {
    graph.tensors.push_back(WorkloadTensor{
        id,
        alias_group,
        producer,
        std::move(consumers),
        bytes,
        persistent,
        temporary,
        host_visible});
}

void finalize_workload_graph(WorkloadGraph& graph) {
    const auto indices = operation_indices(graph);
    graph.dependencies.clear();
    graph.lifetimes.clear();
    for (auto& operation : graph.operations) {
        operation.dependency_operation_names.clear();
    }

    for (const auto& tensor : graph.tensors) {
        std::uint32_t first_index = 0u;
        if (!tensor.producer_operation.empty()) {
            first_index = lookup_index(indices, tensor.producer_operation, 0u);
        } else if (!tensor.consumer_operations.empty()) {
            first_index = lookup_index(indices, tensor.consumer_operations.front(), 0u);
        }

        std::uint32_t last_index = first_index;
        for (const auto& consumer : tensor.consumer_operations) {
            last_index = std::max(last_index, lookup_index(indices, consumer, first_index));
            if (!tensor.producer_operation.empty()) {
                graph.dependencies.push_back(WorkloadDependency{
                    tensor.producer_operation,
                    consumer,
                    tensor.id,
                    true});
            }
        }

        graph.lifetimes.push_back(TensorLifetime{
            tensor.id,
            first_index,
            last_index,
            tensor.bytes,
            tensor.persistent});
    }

    for (auto& operation : graph.operations) {
        for (const auto& dependency : graph.dependencies) {
            if (dependency.target_operation_name == operation.name &&
                std::find(
                    operation.dependency_operation_names.begin(),
                    operation.dependency_operation_names.end(),
                    dependency.source_operation_name) == operation.dependency_operation_names.end()) {
                operation.dependency_operation_names.push_back(dependency.source_operation_name);
            }
        }
    }
}

}  // namespace

std::vector<CanonicalWorkloadPreset> canonical_workload_presets() {
    return {
        CanonicalWorkloadPreset{
            WorkloadSpec{
                "gaming-upscale-1080p",
                WorkloadKind::gaming,
                "gaming-fsr-like-720p-to-1080p",
                768ull * 1024ull * 1024ull,
                96ull * 1024ull * 1024ull,
                1.2e12,
                1,
                true,
                true,
                false},
            "Realtime render reconstruction and post-processing chain from 1280x720 to 1920x1080.",
            "single-path frame pipeline"},
        CanonicalWorkloadPreset{
            WorkloadSpec{
                "ai-vision-inference-lite",
                WorkloadKind::inference,
                "ai-vision-inference-224",
                1024ull * 1024ull * 1024ull,
                128ull * 1024ull * 1024ull,
                4.5e12,
                8,
                false,
                false,
                true},
            "Vision encoder style inference chain with convolution stem, projection, attention-like GEMMs, and MLP blocks.",
            "single-device reference kernels"},
        CanonicalWorkloadPreset{
            WorkloadSpec{
                "ai-train-step-lite",
                WorkloadKind::training,
                "ai-transformer-train-step-lite",
                1536ull * 1024ull * 1024ull,
                256ull * 1024ull * 1024ull,
                7.5e12,
                16,
                false,
                false,
                true},
            "Compact training-step surrogate with forward, reduction, gradient GEMMs, and optimizer-style updates.",
            "single-device reference kernels"}};
}

std::vector<CpuDeepLearningExplorationPreset> cpu_deep_learning_exploration_presets() {
    return {
        CpuDeepLearningExplorationPreset{
            WorkloadSpec{
                "llm-prefill-context-lite",
                WorkloadKind::inference,
                "llm-prefill-context-lite",
                896ull * kMiB,
                48ull * kMiB,
                2.8e11,
                1,
                false,
                false,
                true},
            "Longer-context transformer prefill surrogate with several medium GEMMs plus norm and reduction stages.",
            "Large projection work should stay on GPU, but norm, score reduction, and logits post-process can be candidates for host execution or overlap.",
            "Most matmuls avoid the host while lighter reductions and elementwise stages can land on the CPU without hurting total latency."},
        CpuDeepLearningExplorationPreset{
            WorkloadSpec{
                "llm-decode-token-lite",
                WorkloadKind::inference,
                "llm-decode-token-lite",
                640ull * kMiB,
                12ull * kMiB,
                3.8e10,
                1,
                true,
                true,
                true},
            "Single-token decode surrogate with tiny GEMMs, persistent KV cache, and strict latency sensitivity.",
            "Batch-1 decode is where the CPU may become useful again because dispatch overhead can dominate very small GEMMs and cache-touching work.",
            "The planner or executor chooses host or mixed execution for several decode stages instead of forcing every op through a GPU path."},
        CpuDeepLearningExplorationPreset{
            WorkloadSpec{
                "llm-kv-cache-update-lite",
                WorkloadKind::inference,
                "llm-kv-cache-update-lite",
                1024ull * kMiB,
                192ull * kMiB,
                4.5e10,
                1,
                true,
                true,
                false},
            "KV-cache maintenance surrogate with persistent cache pages, cache-window scans, and lightweight attention bookkeeping.",
            "CPU and unified memory may be better for cache paging, eviction, and scan-heavy work than pushing every update through a discrete device.",
            "Persistent cache tensors remain host-visible and the runtime keeps a meaningful fraction of cache maintenance work on the CPU side."},
        CpuDeepLearningExplorationPreset{
            WorkloadSpec{
                "llm-int4-dequant-lite",
                WorkloadKind::inference,
                "llm-int4-dequant-lite",
                768ull * kMiB,
                96ull * kMiB,
                7.2e10,
                4,
                true,
                true,
                true},
            "Quantized weight pipeline surrogate with unpack, dequantize, fused matmul, and residual update stages.",
            "CPU-side unpacking or dequant staging may reduce GPU pressure if it overlaps well with the main matrix multiply.",
            "Elementwise unpack and dequant stages show as host-friendly while the main fused matmul still prefers GPU or mixed placement."}};
}

WorkloadGraph default_workload_graph(const WorkloadSpec& workload) {
    WorkloadGraph graph;
    graph.signature = workload.name + "|" + to_string(workload.kind) + "|" + workload.dataset_tag +
                      "|" + to_string(canonical_workload_phase(workload)) +
                      "|" + canonical_workload_shape_bucket(workload);

    const std::uint64_t working_set =
        workload.working_set_bytes == 0 ? (32ull * kMiB) : workload.working_set_bytes;
    const std::uint64_t sample_bytes = clamp_u64(working_set / 12ull, 2ull * kMiB, 16ull * kMiB);
    const std::uint64_t vector_count = std::max<std::uint64_t>(sample_bytes / sizeof(float), 64ull * 1024ull);
    const auto matmul_side = round_down_to_multiple(
        static_cast<std::uint32_t>(std::clamp(std::sqrt(static_cast<double>(sample_bytes) / 12.0), 32.0, 96.0)),
        16u);
    const auto conv_side = round_down_to_multiple(
        static_cast<std::uint32_t>(std::clamp(std::sqrt(static_cast<double>(sample_bytes) / 8.0), 32.0, 80.0)),
        8u);
    const auto resample_src = round_down_to_multiple(
        static_cast<std::uint32_t>(std::clamp(std::sqrt(static_cast<double>(sample_bytes) / 4.0), 64.0, 160.0)),
        16u);
    const auto resample_dst = std::max<std::uint32_t>(96u, (resample_src * 3u) / 2u);

    if (workload.dataset_tag == "gaming-fsr-like-720p-to-1080p" || workload.kind == WorkloadKind::gaming) {
        constexpr std::uint32_t src_h = 720u;
        constexpr std::uint32_t src_w = 1280u;
        constexpr std::uint32_t dst_h = 1080u;
        constexpr std::uint32_t dst_w = 1920u;
        const auto src_bytes = 1ull * src_h * src_w * sizeof(float);
        const auto dst_bytes = 1ull * dst_h * dst_w * sizeof(float);
        const auto conv_bytes = 1ull * (src_h - 2u) * (src_w - 2u) * sizeof(float);
        add_tensor(graph, "frame-src", src_bytes, "", {"frame-pre-tonemap"}, false, false, true);
        add_tensor(graph, "history-buffer", src_bytes, "", {"frame-pre-tonemap", "history-reconstruction"}, true, false, true, "history");
        add_tensor(graph, "frame-pre", src_bytes, "frame-pre-tonemap", {"reactive-mask", "exposure-luma", "history-reconstruction"});
        add_tensor(graph, "reactive-mask", src_bytes, "reactive-mask", {"history-reconstruction"});
        add_tensor(graph, "exposure-luma", sizeof(float), "exposure-luma", {"post-tonemap"}, true);
        add_tensor(graph, "history-reconstruct", conv_bytes, "history-reconstruction", {"detail-sharpen"});
        add_tensor(graph, "detail-sharp", conv_bytes, "detail-sharpen", {"upscale-resolve"});
        add_tensor(graph, "frame-upscaled", dst_bytes, "upscale-resolve", {"post-tonemap"});
        add_tensor(graph, "present-frame", dst_bytes, "post-tonemap", {});
        add_tensor(graph, "history-reconstruction.ws", 9ull * sizeof(float), "history-reconstruction", {}, false, true);
        add_tensor(graph, "detail-sharpen.ws", 9ull * sizeof(float), "detail-sharpen", {}, false, true);
        graph.operations = {
            make_elementwise("frame-pre-tonemap", src_h * src_w, {"frame-src", "history-buffer"}, {"frame-pre"}),
            make_elementwise("reactive-mask", src_h * src_w, {"frame-pre", "history-buffer"}, {"reactive-mask"}),
            make_reduction("exposure-luma", src_h * src_w, {"frame-pre"}, {"exposure-luma"}, {}, 1.2e-3),
            make_convolution("history-reconstruction", src_h, src_w, {"frame-pre", "reactive-mask", "history-buffer"}, {"history-reconstruct"}, {"history-reconstruction.ws"}),
            make_convolution("detail-sharpen", src_h, src_w, {"history-reconstruct"}, {"detail-sharp"}, {"detail-sharpen.ws"}),
            make_resample("upscale-resolve", src_h, src_w, dst_h, dst_w, {"detail-sharp"}, {"frame-upscaled"}),
            make_elementwise("post-tonemap", dst_h * dst_w, {"frame-upscaled", "exposure-luma"}, {"present-frame"})};
    } else if (workload.dataset_tag == "llm-prefill-context-lite") {
        constexpr std::uint32_t token_count = 192u;
        constexpr std::uint32_t hidden = 320u;
        constexpr std::uint32_t qkv = hidden * 3u;
        constexpr std::uint32_t mlp = 896u;
        constexpr std::uint32_t vocab_slice = 512u;
        add_tensor(graph, "token-embeddings", 1ull * token_count * hidden * sizeof(float), "", {"token-rmsnorm"}, false, false, true);
        add_tensor(graph, "rms-weights", 1ull * hidden * sizeof(float), "", {"token-rmsnorm"}, true, false, true, "weights");
        add_tensor(graph, "normed-tokens", 1ull * token_count * hidden * sizeof(float), "token-rmsnorm", {"attention-qkv"});
        add_tensor(graph, "qkv-weights", 1ull * hidden * qkv * sizeof(float), "", {"attention-qkv"}, true, false, true, "weights");
        add_tensor(graph, "qkv-out", 1ull * token_count * qkv * sizeof(float), "attention-qkv", {"attention-score-reduce", "context-proj"});
        add_tensor(graph, "attn-score", sizeof(float), "attention-score-reduce", {"mlp-gate", "logits-proj"}, true);
        add_tensor(graph, "context-weights", 1ull * hidden * hidden * sizeof(float), "", {"context-proj"}, true, false, true, "weights");
        add_tensor(graph, "context-out", 1ull * token_count * hidden * sizeof(float), "context-proj", {"mlp-up"});
        add_tensor(graph, "mlp-up-weights", 1ull * hidden * mlp * sizeof(float), "", {"mlp-up"}, true, false, true, "weights");
        add_tensor(graph, "mlp-up-out", 1ull * token_count * mlp * sizeof(float), "mlp-up", {"mlp-gate"});
        add_tensor(graph, "mlp-gated", 1ull * token_count * mlp * sizeof(float), "mlp-gate", {"mlp-down"});
        add_tensor(graph, "mlp-down-weights", 1ull * hidden * mlp * sizeof(float), "", {"mlp-down"}, true, false, true, "weights");
        add_tensor(graph, "decode-state", 1ull * token_count * hidden * sizeof(float), "mlp-down", {"logits-proj"});
        add_tensor(graph, "logits-weights", 1ull * hidden * vocab_slice * sizeof(float), "", {"logits-proj"}, true, false, true, "weights");
        add_tensor(graph, "logits-out", 1ull * token_count * vocab_slice * sizeof(float), "logits-proj", {"logits-max"});
        add_tensor(graph, "token-logit-max", sizeof(float), "logits-max", {});
        graph.operations = {
            make_elementwise("token-rmsnorm", token_count * hidden, {"token-embeddings", "rms-weights"}, {"normed-tokens"}),
            make_matmul("attention-qkv", token_count, qkv, hidden, {"normed-tokens", "qkv-weights"}, {"qkv-out"}),
            make_reduction("attention-score-reduce", token_count * qkv, {"qkv-out"}, {"attn-score"}, {}, 1.5e-3),
            make_matmul("context-proj", token_count, hidden, hidden, {"qkv-out", "context-weights"}, {"context-out"}),
            make_matmul("mlp-up", token_count, mlp, hidden, {"context-out", "mlp-up-weights"}, {"mlp-up-out"}),
            make_elementwise("mlp-gate", token_count * mlp, {"mlp-up-out", "attn-score"}, {"mlp-gated"}, {}, 7.5e-4),
            make_matmul("mlp-down", token_count, hidden, mlp, {"mlp-gated", "mlp-down-weights"}, {"decode-state"}),
            make_matmul("logits-proj", token_count, vocab_slice, hidden, {"decode-state", "logits-weights"}, {"logits-out"}),
            make_reduction("logits-max", token_count * vocab_slice, {"logits-out"}, {"token-logit-max"}, {}, 1.5e-3)};
    } else if (workload.dataset_tag == "llm-decode-token-lite") {
        constexpr std::uint32_t token_count = 1u;
        constexpr std::uint32_t hidden = 320u;
        constexpr std::uint32_t qkv = hidden * 3u;
        constexpr std::uint32_t mlp = 896u;
        constexpr std::uint32_t cache_tokens = 2048u;
        add_tensor(graph, "decode-token", 1ull * hidden * sizeof(float), "", {"decode-rmsnorm"}, false, false, true);
        add_tensor(graph, "decode-rms-weights", 1ull * hidden * sizeof(float), "", {"decode-rmsnorm"}, true, false, true, "weights");
        add_tensor(graph, "decode-normed", 1ull * hidden * sizeof(float), "decode-rmsnorm", {"decode-qkv"});
        add_tensor(graph, "decode-qkv-weights", 1ull * hidden * qkv * sizeof(float), "", {"decode-qkv"}, true, false, true, "weights");
        add_tensor(graph, "decode-qkv-out", 1ull * qkv * sizeof(float), "decode-qkv", {"kv-append", "decode-score-reduce", "decode-context"});
        add_tensor(graph, "kv-cache", 1ull * cache_tokens * hidden * 2ull * sizeof(float), "", {"kv-append", "decode-score-reduce", "decode-context"}, true, false, true, "cache");
        add_tensor(graph, "kv-cache-next", 1ull * cache_tokens * hidden * 2ull * sizeof(float), "kv-append", {"decode-context"}, true, false, true, "cache");
        add_tensor(graph, "decode-score", sizeof(float), "decode-score-reduce", {"decode-mlp-gate", "decode-sample"}, true);
        add_tensor(graph, "decode-context-weights", 1ull * hidden * hidden * sizeof(float), "", {"decode-context"}, true, false, true, "weights");
        add_tensor(graph, "decode-context-out", 1ull * hidden * sizeof(float), "decode-context", {"decode-mlp-up"});
        add_tensor(graph, "decode-mlp-up-weights", 1ull * hidden * mlp * sizeof(float), "", {"decode-mlp-up"}, true, false, true, "weights");
        add_tensor(graph, "decode-mlp-up-out", 1ull * mlp * sizeof(float), "decode-mlp-up", {"decode-mlp-gate"});
        add_tensor(graph, "decode-mlp-gated", 1ull * mlp * sizeof(float), "decode-mlp-gate", {"decode-mlp-down"});
        add_tensor(graph, "decode-mlp-down-weights", 1ull * hidden * mlp * sizeof(float), "", {"decode-mlp-down"}, true, false, true, "weights");
        add_tensor(graph, "decode-state", 1ull * hidden * sizeof(float), "decode-mlp-down", {"decode-sample"});
        add_tensor(graph, "sampled-logit", sizeof(float), "decode-sample", {});
        graph.operations = {
            make_elementwise("decode-rmsnorm", hidden, {"decode-token", "decode-rms-weights"}, {"decode-normed"}),
            make_matmul("decode-qkv", token_count, qkv, hidden, {"decode-normed", "decode-qkv-weights"}, {"decode-qkv-out"}),
            make_elementwise("kv-append", hidden * 2u, {"decode-qkv-out", "kv-cache"}, {"kv-cache-next"}, {}, 7.5e-4),
            make_reduction("decode-score-reduce", static_cast<std::uint64_t>(cache_tokens) * hidden, {"decode-qkv-out", "kv-cache"}, {"decode-score"}, {}, 1.5e-3),
            make_matmul("decode-context", token_count, hidden, hidden, {"decode-qkv-out", "decode-context-weights", "kv-cache-next"}, {"decode-context-out"}),
            make_matmul("decode-mlp-up", token_count, mlp, hidden, {"decode-context-out", "decode-mlp-up-weights"}, {"decode-mlp-up-out"}),
            make_elementwise("decode-mlp-gate", mlp, {"decode-mlp-up-out", "decode-score"}, {"decode-mlp-gated"}, {}, 7.5e-4),
            make_matmul("decode-mlp-down", token_count, hidden, mlp, {"decode-mlp-gated", "decode-mlp-down-weights"}, {"decode-state"}),
            make_reduction("decode-sample", hidden, {"decode-state", "decode-score"}, {"sampled-logit"}, {}, 1.5e-3)};
    } else if (workload.dataset_tag == "llm-kv-cache-update-lite") {
        constexpr std::uint32_t cache_tokens = 4096u;
        constexpr std::uint32_t hidden = 256u;
        constexpr std::uint32_t cache_scan = 8192u;
        add_tensor(graph, "cache-pages", 1ull * cache_tokens * hidden * sizeof(float), "", {"cache-window-scan", "cache-page-read"}, true, false, true, "cache");
        add_tensor(graph, "cache-metadata", 1ull * cache_tokens * sizeof(float), "", {"cache-window-scan", "cache-evict-score"}, true, false, true, "cache-meta");
        add_tensor(graph, "query-state", 1ull * hidden * sizeof(float), "", {"cache-page-read", "cache-value-merge"}, false, false, true);
        add_tensor(graph, "scan-score", sizeof(float), "cache-window-scan", {"cache-page-read", "cache-value-merge"}, true);
        add_tensor(graph, "cache-page", 1ull * hidden * sizeof(float), "cache-page-read", {"cache-rope-rotate", "cache-value-merge"});
        add_tensor(graph, "cache-rotated", 1ull * hidden * sizeof(float), "cache-rope-rotate", {"cache-value-merge"});
        add_tensor(graph, "cache-merged", 1ull * hidden * sizeof(float), "cache-value-merge", {"cache-evict-score", "cache-writeback"});
        add_tensor(graph, "evict-score", sizeof(float), "cache-evict-score", {"cache-writeback"}, true);
        add_tensor(graph, "cache-pages-next", 1ull * cache_tokens * hidden * sizeof(float), "cache-writeback", {}, true, false, true, "cache");
        graph.operations = {
            make_reduction("cache-window-scan", static_cast<std::uint64_t>(cache_scan) * hidden, {"cache-pages", "cache-metadata"}, {"scan-score"}, {}, 1.5e-3),
            make_elementwise("cache-page-read", hidden, {"cache-pages", "query-state", "scan-score"}, {"cache-page"}),
            make_elementwise("cache-rope-rotate", hidden, {"cache-page", "query-state"}, {"cache-rotated"}, {}, 7.5e-4),
            make_elementwise("cache-value-merge", hidden, {"cache-rotated", "query-state", "scan-score"}, {"cache-merged"}, {}, 7.5e-4),
            make_reduction("cache-evict-score", static_cast<std::uint64_t>(cache_tokens), {"cache-merged", "cache-metadata"}, {"evict-score"}, {}, 1.5e-3),
            make_elementwise("cache-writeback", hidden, {"cache-merged", "evict-score", "cache-pages"}, {"cache-pages-next"}, {}, 7.5e-4)};
    } else if (workload.dataset_tag == "llm-int4-dequant-lite") {
        constexpr std::uint32_t token_count = 16u;
        constexpr std::uint32_t hidden = 256u;
        constexpr std::uint32_t blocks = hidden * hidden;
        add_tensor(graph, "quant-input", 1ull * token_count * hidden * sizeof(float), "", {"unpack-nibbles"}, false, false, true);
        add_tensor(graph, "packed-weights", 1ull * blocks, "", {"unpack-nibbles"}, true, false, true, "weights-packed");
        add_tensor(graph, "quant-scales", 1ull * hidden * sizeof(float), "", {"dequant-blocks"}, true, false, true, "weights-meta");
        add_tensor(graph, "unpacked-weights", 1ull * blocks * sizeof(float), "unpack-nibbles", {"dequant-blocks"});
        add_tensor(graph, "dequant-weights", 1ull * blocks * sizeof(float), "dequant-blocks", {"fused-int4-matmul"});
        add_tensor(graph, "fused-out", 1ull * token_count * hidden * sizeof(float), "fused-int4-matmul", {"residual-add", "logit-reduce"});
        add_tensor(graph, "residual-state", 1ull * token_count * hidden * sizeof(float), "", {"residual-add"}, true, false, true);
        add_tensor(graph, "residual-out", 1ull * token_count * hidden * sizeof(float), "residual-add", {"logit-reduce"});
        add_tensor(graph, "block-max", sizeof(float), "logit-reduce", {});
        graph.operations = {
            make_elementwise("unpack-nibbles", blocks, {"quant-input", "packed-weights"}, {"unpacked-weights"}, {}, 7.5e-4),
            make_elementwise("dequant-blocks", blocks, {"unpacked-weights", "quant-scales"}, {"dequant-weights"}, {}, 7.5e-4),
            make_matmul("fused-int4-matmul", token_count, hidden, hidden, {"quant-input", "dequant-weights"}, {"fused-out"}),
            make_elementwise("residual-add", token_count * hidden, {"fused-out", "residual-state"}, {"residual-out"}, {}, 7.5e-4),
            make_reduction("logit-reduce", token_count * hidden, {"residual-out"}, {"block-max"}, {}, 1.5e-3)};
    } else if (workload.dataset_tag == "ai-vision-inference-224" || workload.kind == WorkloadKind::inference) {
        constexpr std::uint32_t image_side = 224u;
        constexpr std::uint32_t token_count = 196u;
        constexpr std::uint32_t hidden = 384u;
        constexpr std::uint32_t mlp = 768u;
        add_tensor(graph, "image-input", 1ull * image_side * image_side * sizeof(float), "", {"stem-conv3x3"}, false, false, true);
        add_tensor(graph, "stem-out", 1ull * (image_side - 2u) * (image_side - 2u) * sizeof(float), "stem-conv3x3", {"patch-proj"});
        add_tensor(graph, "patch-weights", 1ull * hidden * hidden * sizeof(float), "", {"patch-proj"}, true, false, true, "weights");
        add_tensor(graph, "patch-out", 1ull * token_count * hidden * sizeof(float), "patch-proj", {"attention-qkv"});
        add_tensor(graph, "attn-weights", 1ull * hidden * hidden * sizeof(float), "", {"attention-qkv"}, true, false, true, "weights");
        add_tensor(graph, "attention-qkv", 1ull * token_count * hidden * sizeof(float), "attention-qkv", {"attention-score-reduce", "mlp-up"});
        add_tensor(graph, "attn-score", sizeof(float), "attention-score-reduce", {"mlp-activation", "token-pool"});
        add_tensor(graph, "mlp-up-weights", 1ull * hidden * mlp * sizeof(float), "", {"mlp-up"}, true, false, true, "weights");
        add_tensor(graph, "mlp-up-out", 1ull * token_count * mlp * sizeof(float), "mlp-up", {"mlp-activation"});
        add_tensor(graph, "mlp-activation-out", 1ull * token_count * mlp * sizeof(float), "mlp-activation", {"mlp-down"});
        add_tensor(graph, "mlp-down-weights", 1ull * hidden * mlp * sizeof(float), "", {"mlp-down"}, true, false, true, "weights");
        add_tensor(graph, "mlp-down-out", 1ull * token_count * hidden * sizeof(float), "mlp-down", {"token-pool"});
        add_tensor(graph, "pooled-token", sizeof(float), "token-pool", {});
        add_tensor(graph, "stem-conv3x3.ws", 9ull * sizeof(float), "stem-conv3x3", {}, false, true);
        graph.operations = {
            make_convolution("stem-conv3x3", image_side, image_side, {"image-input"}, {"stem-out"}, {"stem-conv3x3.ws"}),
            make_matmul("patch-proj", token_count, hidden, hidden, {"stem-out", "patch-weights"}, {"patch-out"}),
            make_matmul("attention-qkv", token_count, hidden, hidden, {"patch-out", "attn-weights"}, {"attention-qkv"}),
            make_reduction("attention-score-reduce", token_count * hidden, {"attention-qkv"}, {"attn-score"}, {}, 1.5e-3),
            make_matmul("mlp-up", token_count, mlp, hidden, {"attention-qkv", "mlp-up-weights"}, {"mlp-up-out"}),
            make_elementwise("mlp-activation", token_count * mlp, {"mlp-up-out", "attn-score"}, {"mlp-activation-out"}, {}, 7.5e-4),
            make_matmul("mlp-down", token_count, hidden, mlp, {"mlp-activation-out", "mlp-down-weights"}, {"mlp-down-out"}),
            make_reduction("token-pool", token_count * hidden, {"mlp-down-out", "attn-score"}, {"pooled-token"}, {}, 1.5e-3)};
    } else if (workload.dataset_tag == "ai-transformer-train-step-lite" || workload.kind == WorkloadKind::training) {
        constexpr std::uint32_t batch = 128u;
        constexpr std::uint32_t hidden = 384u;
        const auto hidden_bytes = 1ull * batch * hidden * sizeof(float);
        const auto weight_bytes = 1ull * hidden * hidden * sizeof(float);
        add_tensor(graph, "train-batch", hidden_bytes, "", {"fwd-proj"}, false, false, true);
        add_tensor(graph, "train-weights", weight_bytes, "", {"fwd-proj", "fwd-head", "grad-head", "grad-input", "adam-update"}, true, false, true, "weights");
        add_tensor(graph, "adam-m1", weight_bytes, "", {"adam-moment"}, true, false, true, "moments");
        add_tensor(graph, "adam-m2", weight_bytes, "", {"adam-update"}, true, false, true, "moments");
        add_tensor(graph, "fwd-proj-out", hidden_bytes, "fwd-proj", {"fwd-activation"});
        add_tensor(graph, "fwd-activation-out", hidden_bytes, "fwd-activation", {"fwd-head", "grad-input"});
        add_tensor(graph, "fwd-head-out", hidden_bytes, "fwd-head", {"loss-reduce", "loss-scale"});
        add_tensor(graph, "loss-scalar", sizeof(float), "loss-reduce", {"loss-scale"});
        add_tensor(graph, "loss-scale-out", hidden_bytes, "loss-scale", {"grad-head"});
        add_tensor(graph, "grad-head-out", weight_bytes, "grad-head", {"grad-input", "grad-norm", "adam-moment", "adam-update"});
        add_tensor(graph, "grad-input-out", hidden_bytes, "grad-input", {"grad-norm"});
        add_tensor(graph, "grad-norm-scalar", sizeof(float), "grad-norm", {"adam-update"});
        add_tensor(graph, "adam-m1-next", weight_bytes, "adam-moment", {"adam-update"}, true, false, false, "moments");
        add_tensor(graph, "train-weights-next", weight_bytes, "adam-update", {}, true, false, false, "weights");
        graph.operations = {
            make_matmul("fwd-proj", batch, hidden, hidden, {"train-batch", "train-weights"}, {"fwd-proj-out"}),
            make_elementwise("fwd-activation", batch * hidden, {"fwd-proj-out", "train-batch"}, {"fwd-activation-out"}, {}, 7.5e-4),
            make_matmul("fwd-head", batch, hidden, hidden, {"fwd-activation-out", "train-weights"}, {"fwd-head-out"}),
            make_reduction("loss-reduce", batch * hidden, {"fwd-head-out"}, {"loss-scalar"}, {}, 1.5e-3),
            make_elementwise("loss-scale", batch * hidden, {"fwd-head-out", "loss-scalar"}, {"loss-scale-out"}, {}, 7.5e-4),
            make_matmul("grad-head", hidden, hidden, batch, {"loss-scale-out", "train-weights"}, {"grad-head-out"}, {}, 2.5e-3),
            make_matmul("grad-input", batch, hidden, hidden, {"fwd-activation-out", "grad-head-out", "train-weights"}, {"grad-input-out"}, {}, 2.5e-3),
            make_reduction("grad-norm", hidden * hidden, {"grad-head-out", "grad-input-out"}, {"grad-norm-scalar"}, {}, 1.5e-3),
            make_elementwise("adam-moment", hidden * hidden, {"grad-head-out", "adam-m1"}, {"adam-m1-next"}, {}, 7.5e-4),
            make_elementwise("adam-update", hidden * hidden, {"grad-head-out", "grad-norm-scalar", "train-weights", "adam-m1-next", "adam-m2"}, {"train-weights-next"}, {}, 7.5e-4)};
    } else {
        const auto vector_bytes = vector_count * sizeof(float);
        const auto matmul_bytes = 1ull * matmul_side * matmul_side * sizeof(float);
        const auto conv_bytes = 1ull * conv_side * conv_side * sizeof(float);
        const auto conv_out_bytes = 1ull * (conv_side - 2u) * (conv_side - 2u) * sizeof(float);
        const auto resample_out_bytes = 1ull * resample_dst * resample_dst * sizeof(float);
        add_tensor(graph, "tensor-input", vector_bytes, "", {"elementwise-map"}, false, false, true);
        add_tensor(graph, "tensor-bias", vector_bytes, "", {"elementwise-map"}, true, false, true);
        add_tensor(graph, "tensor-eltwise", vector_bytes, "elementwise-map", {"reduction-sum", "blocked-matmul"});
        add_tensor(graph, "tensor-sum", sizeof(float), "reduction-sum", {"bilinear-resample"}, true);
        add_tensor(graph, "tensor-weights", matmul_bytes, "", {"blocked-matmul"}, true, false, true, "weights");
        add_tensor(graph, "tensor-matmul", matmul_bytes, "blocked-matmul", {"conv3x3"});
        add_tensor(graph, "tensor-image", conv_bytes, "", {"conv3x3"}, false, false, true);
        add_tensor(graph, "tensor-conv", conv_out_bytes, "conv3x3", {"bilinear-resample"});
        add_tensor(graph, "tensor-resample", resample_out_bytes, "bilinear-resample", {});
        add_tensor(graph, "conv3x3.ws", 9ull * sizeof(float), "conv3x3", {}, false, true);
        graph.operations = {
            make_elementwise("elementwise-map", vector_count, {"tensor-input", "tensor-bias"}, {"tensor-eltwise"}),
            make_reduction("reduction-sum", vector_count, {"tensor-eltwise"}, {"tensor-sum"}),
            make_matmul("blocked-matmul", matmul_side, matmul_side, matmul_side, {"tensor-eltwise", "tensor-weights"}, {"tensor-matmul"}),
            make_convolution("conv3x3", conv_side, conv_side, {"tensor-image", "tensor-matmul"}, {"tensor-conv"}, {"conv3x3.ws"}),
            make_resample("bilinear-resample", resample_src, resample_src, resample_dst, resample_dst, {"tensor-conv", "tensor-sum"}, {"tensor-resample"})};
    }

    finalize_workload_graph(graph);
    return graph;
}

std::vector<OperationSpec> default_operation_suite(const WorkloadSpec& workload) {
    return default_workload_graph(workload).operations;
}

}  // namespace jakal
