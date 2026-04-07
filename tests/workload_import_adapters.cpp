#include "jakal/runtime.hpp"
#include "jakal/workloads.hpp"

#include <chrono>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::filesystem::path unique_temp_file(const std::string& stem, const std::string& extension) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / (stem + "-" + std::to_string(nonce) + extension);
}

void write_text_file(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream output(path, std::ios::trunc);
    output << contents;
}

template <typename T>
void write_binary_value(std::ostream& output, const T value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_gguf_string(std::ostream& output, const std::string& value) {
    write_binary_value<std::uint64_t>(output, static_cast<std::uint64_t>(value.size()));
    if (!value.empty()) {
        output.write(value.data(), static_cast<std::streamsize>(value.size()));
    }
}

void write_gguf_u32_metadata(std::ostream& output, const std::string& key, const std::uint32_t value) {
    write_gguf_string(output, key);
    write_binary_value<std::uint32_t>(output, 4u);
    write_binary_value<std::uint32_t>(output, value);
}

void write_gguf_string_metadata(std::ostream& output, const std::string& key, const std::string& value) {
    write_gguf_string(output, key);
    write_binary_value<std::uint32_t>(output, 8u);
    write_gguf_string(output, value);
}

void pad_stream_to_alignment(std::ostream& output, const std::uint64_t alignment) {
    const auto position = static_cast<std::uint64_t>(output.tellp());
    const auto remainder = position % alignment;
    if (remainder == 0u) {
        return;
    }
    const auto padding = alignment - remainder;
    for (std::uint64_t index = 0u; index < padding; ++index) {
        output.put('\0');
    }
}

void write_tiny_gguf_file(const std::filesystem::path& path) {
    struct TensorBlob {
        std::string name;
        std::vector<std::uint64_t> dims;
        std::uint32_t type = 0u;
        std::vector<std::uint8_t> data;
    };

    std::vector<TensorBlob> tensors;
    tensors.push_back(TensorBlob{
        "blk.0.attn_q.weight",
        {320u, 320u},
        2u,
        std::vector<std::uint8_t>(40960u, 0x11u)});
    tensors.push_back(TensorBlob{
        "blk.0.ffn_up.weight",
        {320u, 896u},
        1u,
        std::vector<std::uint8_t>(320u * 896u * 2u, 0x22u)});
    tensors.push_back(TensorBlob{
        "output.weight",
        {320u, 320u},
        1u,
        std::vector<std::uint8_t>(320u * 320u * 2u, 0x33u)});

    constexpr std::uint32_t version = 3u;
    constexpr std::uint64_t alignment = 32u;

    std::vector<std::uint64_t> offsets;
    offsets.reserve(tensors.size());
    std::uint64_t running_offset = 0u;
    for (const auto& tensor : tensors) {
        offsets.push_back(running_offset);
        running_offset += static_cast<std::uint64_t>(tensor.data.size());
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write("GGUF", 4);
    write_binary_value<std::uint32_t>(output, version);
    write_binary_value<std::uint64_t>(output, static_cast<std::uint64_t>(tensors.size()));
    write_binary_value<std::uint64_t>(output, 7u);

    write_gguf_string_metadata(output, "general.architecture", "llama");
    write_gguf_string_metadata(output, "general.name", "tiny-gguf");
    write_gguf_u32_metadata(output, "general.alignment", static_cast<std::uint32_t>(alignment));
    write_gguf_u32_metadata(output, "llama.context_length", 1024u);
    write_gguf_u32_metadata(output, "llama.embedding_length", 320u);
    write_gguf_u32_metadata(output, "llama.feed_forward_length", 896u);
    write_gguf_u32_metadata(output, "llama.block_count", 4u);

    for (std::size_t index = 0u; index < tensors.size(); ++index) {
        write_gguf_string(output, tensors[index].name);
        write_binary_value<std::uint32_t>(output, static_cast<std::uint32_t>(tensors[index].dims.size()));
        for (const auto dim : tensors[index].dims) {
            write_binary_value<std::uint64_t>(output, dim);
        }
        write_binary_value<std::uint32_t>(output, tensors[index].type);
        write_binary_value<std::uint64_t>(output, offsets[index]);
    }

    pad_stream_to_alignment(output, alignment);
    for (const auto& tensor : tensors) {
        output.write(reinterpret_cast<const char*>(tensor.data.data()), static_cast<std::streamsize>(tensor.data.size()));
    }
}

void append_varint(std::vector<std::uint8_t>& out, std::uint64_t value) {
    while (value >= 0x80u) {
        out.push_back(static_cast<std::uint8_t>((value & 0x7fu) | 0x80u));
        value >>= 7u;
    }
    out.push_back(static_cast<std::uint8_t>(value));
}

void append_key(std::vector<std::uint8_t>& out, const std::uint32_t field_number, const std::uint32_t wire_type) {
    append_varint(out, (static_cast<std::uint64_t>(field_number) << 3u) | wire_type);
}

void append_string_field(std::vector<std::uint8_t>& out, const std::uint32_t field_number, const std::string& value) {
    append_key(out, field_number, 2u);
    append_varint(out, static_cast<std::uint64_t>(value.size()));
    out.insert(out.end(), value.begin(), value.end());
}

void append_bytes_field(
    std::vector<std::uint8_t>& out,
    const std::uint32_t field_number,
    const std::vector<std::uint8_t>& value) {
    append_key(out, field_number, 2u);
    append_varint(out, static_cast<std::uint64_t>(value.size()));
    out.insert(out.end(), value.begin(), value.end());
}

void append_varint_field(std::vector<std::uint8_t>& out, const std::uint32_t field_number, const std::uint64_t value) {
    append_key(out, field_number, 0u);
    append_varint(out, value);
}

std::vector<std::uint8_t> make_onnx_dim(const std::uint64_t dim_value) {
    std::vector<std::uint8_t> message;
    append_varint_field(message, 1u, dim_value);
    return message;
}

std::vector<std::uint8_t> make_onnx_shape(const std::vector<std::uint64_t>& dims) {
    std::vector<std::uint8_t> message;
    for (const auto dim : dims) {
        append_bytes_field(message, 1u, make_onnx_dim(dim));
    }
    return message;
}

std::vector<std::uint8_t> make_onnx_type(const std::uint32_t elem_type, const std::vector<std::uint64_t>& dims) {
    std::vector<std::uint8_t> tensor_type;
    append_varint_field(tensor_type, 1u, elem_type);
    append_bytes_field(tensor_type, 2u, make_onnx_shape(dims));

    std::vector<std::uint8_t> type;
    append_bytes_field(type, 1u, tensor_type);
    return type;
}

std::vector<std::uint8_t> make_onnx_value_info(
    const std::string& name,
    const std::uint32_t elem_type,
    const std::vector<std::uint64_t>& dims) {
    std::vector<std::uint8_t> message;
    append_string_field(message, 1u, name);
    append_bytes_field(message, 2u, make_onnx_type(elem_type, dims));
    return message;
}

std::vector<std::uint8_t> make_onnx_tensor(
    const std::string& name,
    const std::uint32_t elem_type,
    const std::vector<std::uint64_t>& dims,
    const std::size_t raw_bytes,
    const std::uint8_t fill) {
    std::vector<std::uint8_t> message;
    for (const auto dim : dims) {
        append_varint_field(message, 1u, dim);
    }
    append_varint_field(message, 2u, elem_type);
    append_string_field(message, 8u, name);
    append_bytes_field(message, 9u, std::vector<std::uint8_t>(raw_bytes, fill));
    return message;
}

std::vector<std::uint8_t> make_onnx_tensor_i64(
    const std::string& name,
    const std::vector<std::uint64_t>& dims,
    const std::vector<std::int64_t>& values) {
    std::vector<std::uint8_t> message;
    for (const auto dim : dims) {
        append_varint_field(message, 1u, dim);
    }
    append_varint_field(message, 2u, 7u);
    append_string_field(message, 8u, name);
    for (const auto value : values) {
        append_varint_field(message, 7u, static_cast<std::uint64_t>(value));
    }
    return message;
}

std::vector<std::uint8_t> make_onnx_attribute_int(const std::string& name, const std::int64_t value) {
    std::vector<std::uint8_t> message;
    append_string_field(message, 1u, name);
    append_varint_field(message, 3u, static_cast<std::uint64_t>(value));
    append_varint_field(message, 20u, 2u);
    return message;
}

std::vector<std::uint8_t> make_onnx_attribute_ints(
    const std::string& name,
    const std::vector<std::int64_t>& values) {
    std::vector<std::uint8_t> message;
    append_string_field(message, 1u, name);
    for (const auto value : values) {
        append_varint_field(message, 8u, static_cast<std::uint64_t>(value));
    }
    append_varint_field(message, 20u, 7u);
    return message;
}

void append_fixed32(std::vector<std::uint8_t>& out, const std::uint32_t value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(value));
}

std::vector<std::uint8_t> make_onnx_attribute_floats(
    const std::string& name,
    const std::vector<float>& values) {
    std::vector<std::uint8_t> message;
    append_string_field(message, 1u, name);
    for (const auto value : values) {
        std::uint32_t bits = 0u;
        std::memcpy(&bits, &value, sizeof(bits));
        append_key(message, 7u, 5u);
        append_fixed32(message, bits);
    }
    append_varint_field(message, 20u, 6u);
    return message;
}

std::vector<std::uint8_t> make_onnx_node(
    const std::string& name,
    const std::string& op_type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<std::vector<std::uint8_t>>& attributes = {}) {
    std::vector<std::uint8_t> message;
    for (const auto& input : inputs) {
        append_string_field(message, 1u, input);
    }
    for (const auto& output : outputs) {
        append_string_field(message, 2u, output);
    }
    append_string_field(message, 3u, name);
    append_string_field(message, 4u, op_type);
    for (const auto& attribute : attributes) {
        append_bytes_field(message, 5u, attribute);
    }
    return message;
}

void write_tiny_onnx_file(const std::filesystem::path& path) {
    std::vector<std::uint8_t> graph;
    append_bytes_field(
        graph,
        1u,
        make_onnx_node(
            "stem_conv",
            "Conv",
            {"image"},
            {"stem_out"},
            {make_onnx_attribute_ints("kernel_shape", {3, 3}), make_onnx_attribute_ints("strides", {1, 1})}));
    append_bytes_field(
        graph,
        1u,
        make_onnx_node(
            "patch_proj",
            "Gemm",
            {"stem_out", "proj_w"},
            {"patch_out"},
            {make_onnx_attribute_int("transB", 1)}));
    append_bytes_field(graph, 1u, make_onnx_node("token_pool", "ReduceMean", {"patch_out"}, {"pooled"}));
    append_string_field(graph, 2u, "forward");
    append_bytes_field(graph, 5u, make_onnx_tensor("proj_w", 1u, {192u, 384u}, 192u * 384u * 4u, 0x3a));
    append_bytes_field(graph, 11u, make_onnx_value_info("image", 1u, {1u, 3u, 224u, 224u}));
    append_bytes_field(graph, 12u, make_onnx_value_info("pooled", 1u, {1u}));
    append_bytes_field(graph, 13u, make_onnx_value_info("stem_out", 1u, {1u, 1u, 222u, 222u}));
    append_bytes_field(graph, 13u, make_onnx_value_info("patch_out", 1u, {196u, 192u}));

    std::vector<std::uint8_t> model;
    append_varint_field(model, 1u, 9u);
    append_string_field(model, 2u, "jakal-tests");
    append_bytes_field(model, 7u, graph);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
}

void write_tiny_resize_onnx_file(const std::filesystem::path& path) {
    std::vector<std::uint8_t> graph;
    append_bytes_field(
        graph,
        1u,
        make_onnx_node(
            "upscale",
            "Resize",
            {"frame"},
            {"upscaled"},
            {make_onnx_attribute_floats("scales", {1.0f, 1.0f, 2.0f, 2.0f})}));
    append_string_field(graph, 2u, "resize_forward");
    append_bytes_field(graph, 11u, make_onnx_value_info("frame", 1u, {1u, 1u, 64u, 64u}));
    append_bytes_field(graph, 12u, make_onnx_value_info("upscaled", 1u, {1u, 1u, 128u, 128u}));

    std::vector<std::uint8_t> model;
    append_varint_field(model, 1u, 9u);
    append_string_field(model, 2u, "jakal-tests");
    append_bytes_field(model, 7u, graph);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
}

void write_tiny_shape_inference_onnx_file(const std::filesystem::path& path) {
    std::vector<std::uint8_t> graph;
    append_bytes_field(
        graph,
        1u,
        make_onnx_node(
            "channels_last",
            "Transpose",
            {"tokens"},
            {"tokens_t"},
            {make_onnx_attribute_ints("perm", {0, 2, 1})}));
    append_bytes_field(
        graph,
        1u,
        make_onnx_node(
            "projection",
            "Gemm",
            {"tokens_t", "proj_t"},
            {"projected"},
            {make_onnx_attribute_int("transB", 1)}));
    append_bytes_field(
        graph,
        1u,
        make_onnx_node(
            "collapse_heads",
            "ReduceMean",
            {"attention"},
            {"reduced"},
            {make_onnx_attribute_ints("axes", {-1}), make_onnx_attribute_int("keepdims", 0)}));
    append_bytes_field(
        graph,
        1u,
        make_onnx_node(
            "decode_head",
            "Gemm",
            {"reduced", "decode_w"},
            {"logits"}));
    append_string_field(graph, 2u, "shape_forward");
    append_bytes_field(graph, 5u, make_onnx_tensor("proj_t", 1u, {5u, 2u}, 5u * 2u * 4u, 0x37));
    append_bytes_field(graph, 5u, make_onnx_tensor("decode_w", 1u, {4u, 6u}, 4u * 6u * 4u, 0x41));
    append_bytes_field(graph, 11u, make_onnx_value_info("tokens", 1u, {1u, 2u, 3u}));
    append_bytes_field(graph, 11u, make_onnx_value_info("attention", 1u, {3u, 4u, 8u}));
    append_bytes_field(graph, 12u, make_onnx_value_info("projected", 1u, {3u, 5u}));
    append_bytes_field(graph, 12u, make_onnx_value_info("logits", 1u, {3u, 6u}));

    std::vector<std::uint8_t> model;
    append_varint_field(model, 1u, 9u);
    append_string_field(model, 2u, "jakal-tests");
    append_bytes_field(model, 7u, graph);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
}

void write_tiny_graph_ops_onnx_file(const std::filesystem::path& path) {
    std::vector<std::uint8_t> graph;
    append_bytes_field(graph, 1u, make_onnx_node("reshape_tokens", "Reshape", {"tokens", "shape_spec"}, {"reshaped"}));
    append_bytes_field(graph, 1u, make_onnx_node("expand_batch", "Unsqueeze", {"reshaped", "unsqueeze_axes"}, {"expanded"}));
    append_bytes_field(
        graph,
        1u,
        make_onnx_node("squeeze_batch", "Squeeze", {"expanded"}, {"squeezed"}, {make_onnx_attribute_ints("axes", {0})}));
    append_bytes_field(
        graph,
        1u,
        make_onnx_node("concat_tail", "Concat", {"squeezed", "tail"}, {"concat_out"}, {make_onnx_attribute_int("axis", -1)}));
    append_bytes_field(graph, 1u, make_onnx_node("score", "MatMul", {"concat_out", "proj"}, {"scores"}));
    append_string_field(graph, 2u, "graph_ops_forward");
    append_bytes_field(graph, 5u, make_onnx_tensor_i64("shape_spec", {2u}, {6, 4}));
    append_bytes_field(graph, 5u, make_onnx_tensor_i64("unsqueeze_axes", {1u}, {0}));
    append_bytes_field(graph, 5u, make_onnx_tensor("proj", 1u, {6u, 7u}, 6u * 7u * 4u, 0x2b));
    append_bytes_field(graph, 11u, make_onnx_value_info("tokens", 1u, {2u, 3u, 4u}));
    append_bytes_field(graph, 11u, make_onnx_value_info("tail", 1u, {6u, 2u}));
    append_bytes_field(graph, 12u, make_onnx_value_info("scores", 1u, {6u, 7u}));

    std::vector<std::uint8_t> model;
    append_varint_field(model, 1u, 9u);
    append_string_field(model, 2u, "jakal-tests");
    append_bytes_field(model, 7u, graph);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
}

void write_tiny_broadcast_matmul_onnx_file(const std::filesystem::path& path) {
    std::vector<std::uint8_t> graph;
    append_bytes_field(graph, 1u, make_onnx_node("batched_scores", "MatMul", {"q", "k"}, {"scores"}));
    append_string_field(graph, 2u, "broadcast_matmul_forward");
    append_bytes_field(graph, 11u, make_onnx_value_info("q", 1u, {2u, 3u, 4u}));
    append_bytes_field(graph, 11u, make_onnx_value_info("k", 1u, {1u, 4u, 5u}));
    append_bytes_field(graph, 12u, make_onnx_value_info("scores", 1u, {2u, 3u, 5u}));

    std::vector<std::uint8_t> model;
    append_varint_field(model, 1u, 9u);
    append_string_field(model, 2u, "jakal-tests");
    append_bytes_field(model, 7u, graph);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
}

void write_tiny_indexing_onnx_file(const std::filesystem::path& path) {
    std::vector<std::uint8_t> graph;
    append_bytes_field(
        graph,
        1u,
        make_onnx_node("head_select", "Gather", {"tokens", "head_idx"}, {"selected"}, {make_onnx_attribute_int("axis", 1)}));
    append_bytes_field(
        graph,
        1u,
        make_onnx_node("trim_tail", "Slice", {"selected", "slice_starts", "slice_ends", "slice_axes", "slice_steps"}, {"trimmed"}));
    append_bytes_field(
        graph,
        1u,
        make_onnx_node("flatten_tokens", "Flatten", {"trimmed"}, {"flattened"}, {make_onnx_attribute_int("axis", 1)}));
    append_bytes_field(graph, 1u, make_onnx_node("expand_bias", "Expand", {"bias_seed", "expand_shape"}, {"bias"}));
    append_bytes_field(
        graph,
        1u,
        make_onnx_node("merge_bias", "Concat", {"flattened", "bias"}, {"features"}, {make_onnx_attribute_int("axis", -1)}));
    append_bytes_field(graph, 1u, make_onnx_node("classify", "MatMul", {"features", "proj"}, {"scores"}));
    append_string_field(graph, 2u, "indexing_forward");
    append_bytes_field(graph, 5u, make_onnx_tensor_i64("head_idx", {2u}, {0, 2}));
    append_bytes_field(graph, 5u, make_onnx_tensor_i64("slice_starts", {1u}, {1}));
    append_bytes_field(graph, 5u, make_onnx_tensor_i64("slice_ends", {1u}, {4}));
    append_bytes_field(graph, 5u, make_onnx_tensor_i64("slice_axes", {1u}, {-1}));
    append_bytes_field(graph, 5u, make_onnx_tensor_i64("slice_steps", {1u}, {1}));
    append_bytes_field(graph, 5u, make_onnx_tensor_i64("expand_shape", {2u}, {2, 6}));
    append_bytes_field(graph, 5u, make_onnx_tensor("proj", 1u, {12u, 5u}, 12u * 5u * 4u, 0x55));
    append_bytes_field(graph, 11u, make_onnx_value_info("tokens", 1u, {2u, 3u, 4u}));
    append_bytes_field(graph, 11u, make_onnx_value_info("bias_seed", 1u, {1u, 6u}));
    append_bytes_field(graph, 12u, make_onnx_value_info("scores", 1u, {2u, 5u}));

    std::vector<std::uint8_t> model;
    append_varint_field(model, 1u, 9u);
    append_string_field(model, 2u, "jakal-tests");
    append_bytes_field(model, 7u, graph);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
}

void write_tiny_branching_onnx_file(const std::filesystem::path& path) {
    std::vector<std::uint8_t> graph;
    append_bytes_field(graph, 1u, make_onnx_node("pick_heads", "GatherElements", {"tokens", "gather_idx"}, {"picked"}));
    append_bytes_field(graph, 1u, make_onnx_node("choose", "Where", {"cond", "picked", "fallback"}, {"selected"}));
    append_bytes_field(graph, 1u, make_onnx_node("pad_tail", "Pad", {"selected", "pads"}, {"padded"}));
    append_bytes_field(graph, 1u, make_onnx_node("tile_context", "Tile", {"padded", "repeats"}, {"tiled"}));
    append_bytes_field(
        graph,
        1u,
        make_onnx_node(
            "split_heads",
            "Split",
            {"tiled", "split_sizes"},
            {"left", "right"},
            {make_onnx_attribute_int("axis", 1)}));
    append_bytes_field(graph, 1u, make_onnx_node("score_right", "MatMul", {"right", "proj"}, {"scores"}));
    append_string_field(graph, 2u, "branching_forward");
    append_bytes_field(graph, 5u, make_onnx_tensor_i64("gather_idx", {2u, 3u}, {0, 1, 2, 2, 1, 0}));
    append_bytes_field(graph, 5u, make_onnx_tensor_i64("pads", {4u}, {1, 0, 0, 2}));
    append_bytes_field(graph, 5u, make_onnx_tensor_i64("repeats", {2u}, {1, 2}));
    append_bytes_field(graph, 5u, make_onnx_tensor_i64("split_sizes", {2u}, {4, 6}));
    append_bytes_field(graph, 5u, make_onnx_tensor("proj", 1u, {6u, 5u}, 6u * 5u * 4u, 0x61));
    append_bytes_field(graph, 11u, make_onnx_value_info("tokens", 1u, {2u, 3u}));
    append_bytes_field(graph, 11u, make_onnx_value_info("cond", 9u, {2u, 1u}));
    append_bytes_field(graph, 11u, make_onnx_value_info("fallback", 1u, {1u, 3u}));
    append_bytes_field(graph, 12u, make_onnx_value_info("scores", 1u, {3u, 5u}));

    std::vector<std::uint8_t> model;
    append_varint_field(model, 1u, 9u);
    append_string_field(model, 2u, "jakal-tests");
    append_bytes_field(model, 7u, graph);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
}

std::string pytorch_export_text() {
    return
        "[source]\n"
        "format=pytorch_export\n"
        "name=torch-decode-lite\n"
        "kind=inference\n"
        "dataset_tag=torch-decode-lite\n"
        "phase=decode\n"
        "entry=decode_step\n"
        "batch_size=1\n"
        "latency_sensitive=true\n"
        "prefer_unified_memory=true\n"
        "matrix_friendly=true\n\n"
        "[value]\n"
        "id=token\n"
        "shape=1,320\n"
        "dtype=f32\n"
        "host_visible=true\n"
        "consumers=rmsnorm\n\n"
        "[value]\n"
        "id=rms_w\n"
        "shape=320\n"
        "dtype=f32\n"
        "initializer=true\n"
        "consumers=rmsnorm\n\n"
        "[value]\n"
        "id=normed\n"
        "shape=1,320\n"
        "dtype=f32\n"
        "producer=rmsnorm\n"
        "consumers=qkv_linear\n\n"
        "[value]\n"
        "id=qkv_w\n"
        "shape=320,960\n"
        "dtype=f16\n"
        "initializer=true\n"
        "consumers=qkv_linear\n\n"
        "[value]\n"
        "id=qkv\n"
        "shape=1,960\n"
        "dtype=f16\n"
        "producer=qkv_linear\n"
        "consumers=token_score\n\n"
        "[value]\n"
        "id=score\n"
        "shape=1\n"
        "dtype=f32\n"
        "producer=token_score\n\n"
        "[node]\n"
        "name=rmsnorm\n"
        "op_type=LayerNormalization\n"
        "inputs=token,rms_w\n"
        "outputs=normed\n"
        "shape=1,320\n\n"
        "[node]\n"
        "name=qkv_linear\n"
        "op_type=Linear\n"
        "inputs=normed,qkv_w\n"
        "outputs=qkv\n"
        "shape=1,960\n\n"
        "[node]\n"
        "name=token_score\n"
        "op_type=Softmax\n"
        "inputs=qkv\n"
        "outputs=score\n"
        "shape=1\n";
}

std::string ggml_export_text() {
    return
        "[source]\n"
        "format=ggml\n"
        "name=ggml-decode-lite\n"
        "kind=inference\n"
        "dataset_tag=ggml-decode-lite\n"
        "phase=decode\n"
        "entry=decode_step\n"
        "batch_size=1\n"
        "latency_sensitive=true\n"
        "prefer_unified_memory=true\n\n"
        "[value]\n"
        "id=token\n"
        "shape=1,320\n"
        "dtype=f32\n"
        "host_visible=true\n"
        "consumers=mul_mat\n\n"
        "[value]\n"
        "id=q4_w\n"
        "shape=320,960\n"
        "dtype=q4_0\n"
        "initializer=true\n"
        "consumers=dequant_w\n\n"
        "[value]\n"
        "id=dequant_w\n"
        "shape=320,960\n"
        "dtype=f16\n"
        "producer=dequant_weights\n"
        "consumers=mul_mat\n\n"
        "[value]\n"
        "id=qkv\n"
        "shape=1,960\n"
        "dtype=f16\n"
        "producer=mul_mat\n"
        "consumers=token_sample\n\n"
        "[value]\n"
        "id=sampled\n"
        "shape=1\n"
        "dtype=f32\n"
        "producer=token_sample\n\n"
        "[node]\n"
        "name=dequant_weights\n"
        "op_type=DequantizeLinear\n"
        "inputs=q4_w\n"
        "outputs=dequant_w\n"
        "shape=320,960\n\n"
        "[node]\n"
        "name=mul_mat\n"
        "op_type=mul_mat\n"
        "inputs=token,dequant_w\n"
        "outputs=qkv\n"
        "shape=1,960\n\n"
        "[node]\n"
        "name=token_sample\n"
        "op_type=Softmax\n"
        "inputs=qkv\n"
        "outputs=sampled\n"
        "shape=1\n";
}

}  // namespace

int main() {
    const auto onnx_path = unique_temp_file("workload-import-onnx", ".onnx");
    const auto resize_onnx_path = unique_temp_file("workload-import-onnx-resize", ".onnx");
    const auto shape_onnx_path = unique_temp_file("workload-import-onnx-shape", ".onnx");
    const auto graph_ops_onnx_path = unique_temp_file("workload-import-onnx-graph-ops", ".onnx");
    const auto matmul_broadcast_onnx_path = unique_temp_file("workload-import-onnx-matmul", ".onnx");
    const auto indexing_onnx_path = unique_temp_file("workload-import-onnx-indexing", ".onnx");
    const auto branching_onnx_path = unique_temp_file("workload-import-onnx-branching", ".onnx");
    const auto pytorch_path = unique_temp_file("workload-import-torch", ".workload");
    const auto ggml_path = unique_temp_file("workload-import-ggml", ".workload");
    const auto gguf_path = unique_temp_file("workload-import-gguf", ".gguf");
    const auto telemetry_path = unique_temp_file("workload-import-runtime", ".telemetry.tsv");

    write_tiny_onnx_file(onnx_path);
    write_tiny_resize_onnx_file(resize_onnx_path);
    write_tiny_shape_inference_onnx_file(shape_onnx_path);
    write_tiny_graph_ops_onnx_file(graph_ops_onnx_path);
    write_tiny_broadcast_matmul_onnx_file(matmul_broadcast_onnx_path);
    write_tiny_indexing_onnx_file(indexing_onnx_path);
    write_tiny_branching_onnx_file(branching_onnx_path);
    write_text_file(pytorch_path, pytorch_export_text());
    write_text_file(ggml_path, ggml_export_text());
    write_tiny_gguf_file(gguf_path);

    try {
        const auto onnx = jakal::load_workload_source(onnx_path);
        if (!onnx.imported || onnx.source_format != "onnx" || !onnx.has_graph) {
            std::cerr << "onnx source import metadata missing\n";
            return 1;
        }
        if (onnx.graph.operations.size() != 3u || onnx.graph.tensors.size() != 5u) {
            std::cerr << "onnx source import graph shape mismatch\n";
            return 1;
        }
        if (onnx.graph.operations[0].op_class != jakal::OperationClass::convolution_2d ||
            onnx.graph.operations[1].op_class != jakal::OperationClass::matmul ||
            onnx.graph.operations[2].op_class != jakal::OperationClass::reduction) {
            std::cerr << "onnx op type mapping mismatch\n";
            return 1;
        }
        if (onnx.graph.operations[1].extents.size() != 3u || onnx.graph.operations[1].extents[0] != 196u ||
            onnx.graph.operations[1].extents[1] != 192u || onnx.graph.operations[1].extents[2] != 384u) {
            std::cerr << "onnx attribute-aware extent inference mismatch\n";
            return 1;
        }
        if (onnx.source_entry != "forward" || onnx.workload.name != "forward") {
            std::cerr << "onnx source entry metadata mismatch\n";
            return 1;
        }
        if (onnx.workload.working_set_bytes == 0u || onnx.workload.estimated_flops <= 0.0) {
            std::cerr << "onnx source import workload metadata missing\n";
            return 1;
        }

        const auto resize_onnx = jakal::load_workload_source(resize_onnx_path);
        if (resize_onnx.graph.operations.size() != 1u ||
            resize_onnx.graph.operations[0].op_class != jakal::OperationClass::resample_2d) {
            std::cerr << "onnx resize op mapping mismatch\n";
            return 1;
        }
        if (resize_onnx.graph.operations[0].extents.size() != 4u ||
            resize_onnx.graph.operations[0].extents[0] != 64u ||
            resize_onnx.graph.operations[0].extents[1] != 64u ||
            resize_onnx.graph.operations[0].extents[2] != 128u ||
            resize_onnx.graph.operations[0].extents[3] != 128u) {
            std::cerr << "onnx resize attribute-aware extent inference mismatch\n";
            return 1;
        }

        const auto shape_onnx = jakal::load_workload_source(shape_onnx_path);
        if (shape_onnx.graph.operations.size() != 4u) {
            std::cerr << "onnx inferred-shape graph size mismatch\n";
            return 1;
        }
        if (shape_onnx.graph.operations[0].op_class != jakal::OperationClass::elementwise_map ||
            shape_onnx.graph.operations[1].op_class != jakal::OperationClass::matmul ||
            shape_onnx.graph.operations[2].op_class != jakal::OperationClass::reduction ||
            shape_onnx.graph.operations[3].op_class != jakal::OperationClass::matmul) {
            std::cerr << "onnx inferred-shape op mapping mismatch\n";
            return 1;
        }
        if (shape_onnx.graph.operations[1].extents.size() != 3u ||
            shape_onnx.graph.operations[1].extents[0] != 3u ||
            shape_onnx.graph.operations[1].extents[1] != 5u ||
            shape_onnx.graph.operations[1].extents[2] != 2u) {
            std::cerr << "onnx transpose perm shape inference mismatch\n";
            return 1;
        }
        if (shape_onnx.graph.operations[3].extents.size() != 3u ||
            shape_onnx.graph.operations[3].extents[0] != 3u ||
            shape_onnx.graph.operations[3].extents[1] != 6u ||
            shape_onnx.graph.operations[3].extents[2] != 4u) {
            std::cerr << "onnx reduce axes keepdims shape inference mismatch\n";
            return 1;
        }

        const auto graph_ops_onnx = jakal::load_workload_source(graph_ops_onnx_path);
        if (graph_ops_onnx.graph.operations.size() != 5u) {
            std::cerr << "onnx graph ops shape inference size mismatch\n";
            return 1;
        }
        if (graph_ops_onnx.graph.operations[4].extents.size() != 3u ||
            graph_ops_onnx.graph.operations[4].extents[0] != 6u ||
            graph_ops_onnx.graph.operations[4].extents[1] != 7u ||
            graph_ops_onnx.graph.operations[4].extents[2] != 6u) {
            std::cerr << "onnx reshape/concat/matmul shape inference mismatch\n";
            return 1;
        }
        const auto concat_tensor = std::find_if(
            graph_ops_onnx.graph.tensors.begin(),
            graph_ops_onnx.graph.tensors.end(),
            [](const jakal::WorkloadTensor& tensor) {
                return tensor.id == "concat_out";
            });
        if (concat_tensor == graph_ops_onnx.graph.tensors.end() || concat_tensor->bytes != 144u) {
            std::cerr << "onnx concat output tensor bytes mismatch\n";
            return 1;
        }

        const auto matmul_broadcast_onnx = jakal::load_workload_source(matmul_broadcast_onnx_path);
        if (matmul_broadcast_onnx.graph.operations.size() != 1u ||
            matmul_broadcast_onnx.graph.operations[0].op_class != jakal::OperationClass::matmul) {
            std::cerr << "onnx broadcast matmul op mapping mismatch\n";
            return 1;
        }
        if (matmul_broadcast_onnx.graph.operations[0].extents.size() != 3u ||
            matmul_broadcast_onnx.graph.operations[0].extents[0] != 6u ||
            matmul_broadcast_onnx.graph.operations[0].extents[1] != 5u ||
            matmul_broadcast_onnx.graph.operations[0].extents[2] != 4u) {
            std::cerr << "onnx broadcast matmul extent inference mismatch\n";
            return 1;
        }
        const auto scores_tensor = std::find_if(
            matmul_broadcast_onnx.graph.tensors.begin(),
            matmul_broadcast_onnx.graph.tensors.end(),
            [](const jakal::WorkloadTensor& tensor) {
                return tensor.id == "scores";
            });
        if (scores_tensor == matmul_broadcast_onnx.graph.tensors.end() || scores_tensor->bytes != 120u) {
            std::cerr << "onnx broadcast matmul output tensor bytes mismatch\n";
            return 1;
        }

        const auto indexing_onnx = jakal::load_workload_source(indexing_onnx_path);
        if (indexing_onnx.graph.operations.size() != 6u) {
            std::cerr << "onnx indexing graph size mismatch\n";
            return 1;
        }
        if (indexing_onnx.graph.operations[5].extents.size() != 3u ||
            indexing_onnx.graph.operations[5].extents[0] != 2u ||
            indexing_onnx.graph.operations[5].extents[1] != 5u ||
            indexing_onnx.graph.operations[5].extents[2] != 12u) {
            std::cerr << "onnx gather/slice/expand/flatten shape inference mismatch\n";
            return 1;
        }
        const auto features_tensor = std::find_if(
            indexing_onnx.graph.tensors.begin(),
            indexing_onnx.graph.tensors.end(),
            [](const jakal::WorkloadTensor& tensor) {
                return tensor.id == "features";
            });
        if (features_tensor == indexing_onnx.graph.tensors.end() || features_tensor->bytes != 96u) {
            std::cerr << "onnx indexing feature tensor bytes mismatch\n";
            return 1;
        }
        const auto bias_tensor = std::find_if(
            indexing_onnx.graph.tensors.begin(),
            indexing_onnx.graph.tensors.end(),
            [](const jakal::WorkloadTensor& tensor) {
                return tensor.id == "bias";
            });
        if (bias_tensor == indexing_onnx.graph.tensors.end() || bias_tensor->bytes != 48u) {
            std::cerr << "onnx expand output tensor bytes mismatch\n";
            return 1;
        }

        const auto branching_onnx = jakal::load_workload_source(branching_onnx_path);
        if (branching_onnx.graph.operations.size() != 6u) {
            std::cerr << "onnx branching graph size mismatch\n";
            return 1;
        }
        if (branching_onnx.graph.operations[5].extents.size() != 3u ||
            branching_onnx.graph.operations[5].extents[0] != 3u ||
            branching_onnx.graph.operations[5].extents[1] != 5u ||
            branching_onnx.graph.operations[5].extents[2] != 6u) {
            std::cerr << "onnx gatherelements/where/pad/tile/split shape inference mismatch\n";
            return 1;
        }
        const auto right_tensor = std::find_if(
            branching_onnx.graph.tensors.begin(),
            branching_onnx.graph.tensors.end(),
            [](const jakal::WorkloadTensor& tensor) {
                return tensor.id == "right";
            });
        if (right_tensor == branching_onnx.graph.tensors.end() || right_tensor->bytes != 72u) {
            std::cerr << "onnx split output tensor bytes mismatch\n";
            return 1;
        }
        const auto tiled_tensor = std::find_if(
            branching_onnx.graph.tensors.begin(),
            branching_onnx.graph.tensors.end(),
            [](const jakal::WorkloadTensor& tensor) {
                return tensor.id == "tiled";
            });
        if (tiled_tensor == branching_onnx.graph.tensors.end() || tiled_tensor->bytes != 120u) {
            std::cerr << "onnx tile output tensor bytes mismatch\n";
            return 1;
        }

        const auto pytorch = jakal::load_workload_source(pytorch_path);
        if (pytorch.source_format != "pytorch_export" || pytorch.source_entry != "decode_step") {
            std::cerr << "pytorch source import metadata mismatch\n";
            return 1;
        }
        if (pytorch.graph.operations.size() != 3u ||
            pytorch.graph.operations[1].op_class != jakal::OperationClass::matmul) {
            std::cerr << "pytorch source import op mapping mismatch\n";
            return 1;
        }
        if (!pytorch.workload.latency_sensitive || !pytorch.workload.prefer_unified_memory) {
            std::cerr << "pytorch source import workload flags missing\n";
            return 1;
        }

        const auto ggml = jakal::load_workload_source(ggml_path);
        if (ggml.source_format != "ggml" || ggml.graph.operations.size() != 3u) {
            std::cerr << "ggml source import metadata mismatch\n";
            return 1;
        }
        if (ggml.graph.operations[0].op_class != jakal::OperationClass::elementwise_map ||
            ggml.graph.operations[1].op_class != jakal::OperationClass::matmul ||
            ggml.graph.operations[2].op_class != jakal::OperationClass::reduction) {
            std::cerr << "ggml source import op mapping mismatch\n";
            return 1;
        }
        const auto ggml_weight = std::find_if(
            ggml.graph.tensors.begin(),
            ggml.graph.tensors.end(),
            [](const jakal::WorkloadTensor& tensor) {
                return tensor.id == "q4_w";
            });
        if (ggml_weight == ggml.graph.tensors.end() || !ggml_weight->persistent || ggml_weight->bytes != 153600u) {
            std::cerr << "ggml quantized tensor import mismatch\n";
            return 1;
        }

        const auto gguf = jakal::load_workload_source(gguf_path);
        if (!gguf.imported || gguf.source_format != "gguf" || gguf.graph.operations.empty()) {
            std::cerr << "gguf binary import metadata missing\n";
            return 1;
        }
        if (gguf.workload.dataset_tag != "gguf-llama" || gguf.workload.phase != jakal::WorkloadPhase::decode) {
            std::cerr << "gguf binary import workload metadata mismatch\n";
            return 1;
        }
        const auto gguf_tensor = std::find_if(
            gguf.graph.tensors.begin(),
            gguf.graph.tensors.end(),
            [](const jakal::WorkloadTensor& tensor) {
                return tensor.id == "blk.0.ffn_up.weight";
            });
        if (gguf_tensor == gguf.graph.tensors.end() || !gguf_tensor->persistent || gguf_tensor->bytes != 573440u) {
            std::cerr << "gguf tensor table parsing mismatch\n";
            return 1;
        }

        jakal::RuntimeOptions runtime_options;
        runtime_options.enable_host_probe = true;
        runtime_options.enable_opencl_probe = false;
        runtime_options.enable_level_zero_probe = false;
        runtime_options.enable_cuda_probe = false;
        runtime_options.enable_rocm_probe = false;
        runtime_options.product.observability.telemetry_path = telemetry_path;

        jakal::Runtime runtime(runtime_options);
        const auto executed = runtime.execute_manifest(onnx_path);
        if (!executed.executed || executed.execution.optimization.operations.size() != onnx.graph.operations.size()) {
            std::cerr << "runtime execute did not accept imported onnx source\n";
            return 1;
        }
        if (!std::filesystem::exists(executed.telemetry_path)) {
            std::cerr << "runtime execute did not emit imported source telemetry\n";
            return 1;
        }
        const auto executed_gguf = runtime.execute_manifest(gguf_path);
        if (!executed_gguf.executed || executed_gguf.execution.optimization.operations.empty()) {
            std::cerr << "runtime execute did not accept imported gguf source\n";
            return 1;
        }

        std::error_code ec;
        std::filesystem::remove(onnx_path, ec);
        std::filesystem::remove(resize_onnx_path, ec);
        std::filesystem::remove(shape_onnx_path, ec);
        std::filesystem::remove(graph_ops_onnx_path, ec);
        std::filesystem::remove(matmul_broadcast_onnx_path, ec);
        std::filesystem::remove(indexing_onnx_path, ec);
        std::filesystem::remove(branching_onnx_path, ec);
        std::filesystem::remove(pytorch_path, ec);
        std::filesystem::remove(ggml_path, ec);
        std::filesystem::remove(gguf_path, ec);
        std::filesystem::remove(executed.telemetry_path, ec);
        std::filesystem::remove(executed_gguf.telemetry_path, ec);

        std::cout << "workload import adapters ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "exception: " << error.what() << '\n';
        return 1;
    }
}
