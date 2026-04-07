#include "jakal/execution.hpp"
#include "jakal/workloads.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace jakal {
namespace {

constexpr std::uint64_t kKiB = 1024ull;
constexpr std::uint64_t kMiB = 1024ull * 1024ull;

struct ImportedSourceSpec {
    std::string format = "imported";
    std::string entry;
};

struct ImportedValueSpec {
    std::string id;
    std::vector<std::uint64_t> shape;
    std::vector<std::int64_t> int64_data;
    std::string dtype = "f32";
    std::string alias_group;
    std::string producer_operation;
    std::vector<std::string> consumer_operations;
    std::uint64_t bytes = 0;
    bool initializer = false;
    bool persistent = false;
    bool temporary = false;
    bool host_visible = false;
};

struct ImportedNodeSpec {
    std::string name;
    std::string op_type;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint64_t> extents;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> temporaries;
    std::vector<std::string> dependencies;
    std::uint64_t input_bytes = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t temporary_bytes = 0;
    double estimated_flops = 0.0;
    double max_relative_error = 0.0;
    bool parallelizable = true;
    bool reduction_like = false;
    bool streaming_friendly = false;
    bool matrix_friendly = false;
    std::unordered_map<std::string, std::int64_t> int_attributes;
    std::unordered_map<std::string, std::vector<std::int64_t>> int_list_attributes;
    std::unordered_map<std::string, float> float_attributes;
    std::unordered_map<std::string, std::vector<float>> float_list_attributes;
    std::unordered_map<std::string, std::string> string_attributes;
};

struct GgufTensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type = 0;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

struct GgufFileContents {
    std::unordered_map<std::string, std::string> metadata;
    std::vector<GgufTensorInfo> tensors;
    std::uint64_t alignment = 32u;
    std::uint64_t data_offset = 0u;
    std::uint64_t total_tensor_bytes = 0u;
};

struct OnnxValueInfo {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::vector<std::int64_t> int64_data;
    std::string dtype = "f32";
    bool initializer = false;
    bool persistent = false;
    std::uint64_t bytes = 0;
    bool host_visible = false;
};

struct OnnxNodeInfo {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, std::int64_t> int_attributes;
    std::unordered_map<std::string, std::vector<std::int64_t>> int_list_attributes;
    std::unordered_map<std::string, float> float_attributes;
    std::unordered_map<std::string, std::vector<float>> float_list_attributes;
    std::unordered_map<std::string, std::string> string_attributes;
};

struct OnnxGraphInfo {
    std::string name;
    std::vector<OnnxValueInfo> values;
    std::vector<OnnxNodeInfo> nodes;
};

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

std::string trim_copy(const std::string& input) {
    std::size_t first = 0u;
    while (first < input.size() && std::isspace(static_cast<unsigned char>(input[first])) != 0) {
        ++first;
    }
    std::size_t last = input.size();
    while (last > first && std::isspace(static_cast<unsigned char>(input[last - 1u])) != 0) {
        --last;
    }
    return input.substr(first, last - first);
}

std::string lowercase_copy(const std::string& input) {
    auto lowered = input;
    std::transform(
        lowered.begin(),
        lowered.end(),
        lowered.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
    return lowered;
}

std::vector<std::string> split_csv_strings(const std::string& input) {
    std::vector<std::string> values;
    std::stringstream stream(input);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto trimmed = trim_copy(item);
        if (!trimmed.empty()) {
            values.push_back(trimmed);
        }
    }
    return values;
}

std::vector<std::uint64_t> split_csv_u64(const std::string& input) {
    std::vector<std::uint64_t> values;
    for (const auto& item : split_csv_strings(input)) {
        values.push_back(static_cast<std::uint64_t>(std::stoull(item)));
    }
    return values;
}

std::uint64_t safe_product(const std::vector<std::uint64_t>& values) {
    if (values.empty()) {
        return 0;
    }
    std::uint64_t product = 1u;
    for (const auto value : values) {
        if (value == 0u) {
            return 0;
        }
        if (product > (std::numeric_limits<std::uint64_t>::max() / value)) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        product *= value;
    }
    return product;
}

std::uint64_t shape_elements(const std::vector<std::uint64_t>& shape) {
    return safe_product(shape);
}

std::uint64_t dtype_bit_width(const std::string& dtype) {
    auto lowered = dtype;
    std::transform(
        lowered.begin(),
        lowered.end(),
        lowered.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
    if (lowered == "f16" || lowered == "fp16" || lowered == "half" || lowered == "float16" ||
        lowered == "bf16" || lowered == "bfloat16" || lowered == "i16" || lowered == "u16") {
        return 16u;
    }
    if (lowered == "i8" || lowered == "u8" || lowered == "int8" || lowered == "uint8" ||
        lowered == "q8" || lowered == "q8_0") {
        return 8u;
    }
    if (lowered == "q6" || lowered == "q6_k") {
        return 6u;
    }
    if (lowered == "q5" || lowered == "q5_0" || lowered == "q5_k") {
        return 5u;
    }
    if (lowered == "q4" || lowered == "q4_0" || lowered == "q4_k") {
        return 4u;
    }
    if (lowered == "i32" || lowered == "u32" || lowered == "int32" || lowered == "uint32" ||
        lowered == "f32" || lowered == "fp32" || lowered == "float" || lowered == "float32") {
        return 32u;
    }
    return 32u;
}

std::uint64_t bytes_for_shape(const std::vector<std::uint64_t>& shape, const std::string& dtype) {
    const auto elements = shape_elements(shape);
    if (elements == 0u) {
        return 0u;
    }
    const auto bits = dtype_bit_width(dtype);
    if (elements > (std::numeric_limits<std::uint64_t>::max() / bits)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return (elements * bits + 7u) / 8u;
}

std::uint64_t bytes_per_element_for_dtype(const std::string& dtype) {
    return std::max<std::uint64_t>(1u, (dtype_bit_width(dtype) + 7u) / 8u);
}

std::uint64_t leading_shape_extent(const std::vector<std::uint64_t>& shape) {
    if (shape.empty()) {
        return 0u;
    }
    if (shape.size() == 1u) {
        return shape.front();
    }
    return safe_product(std::vector<std::uint64_t>(shape.begin(), shape.end() - 1));
}

std::uint64_t trailing_shape_extent(const std::vector<std::uint64_t>& shape) {
    return shape.empty() ? 0u : shape.back();
}

std::pair<std::uint64_t, std::uint64_t> spatial_extents(const std::vector<std::uint64_t>& shape) {
    if (shape.size() >= 2u) {
        return {shape[shape.size() - 2u], shape[shape.size() - 1u]};
    }
    if (shape.size() == 1u) {
        return {shape.front(), shape.front()};
    }
    return {0u, 0u};
}

WorkloadManifest finalize_imported_manifest(
    WorkloadManifest manifest,
    const ImportedSourceSpec& source,
    const std::vector<ImportedValueSpec>& values,
    const std::vector<ImportedNodeSpec>& nodes);

std::uint64_t align_up_u64(const std::uint64_t value, const std::uint64_t alignment) {
    if (alignment == 0u) {
        return value;
    }
    const auto remainder = value % alignment;
    return remainder == 0u ? value : (value + alignment - remainder);
}

template <typename T>
T read_binary_value(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("unexpected end of GGUF stream");
    }
    return value;
}

std::string read_gguf_string(std::istream& input) {
    const auto length = read_binary_value<std::uint64_t>(input);
    std::string value(length, '\0');
    if (length > 0u) {
        input.read(value.data(), static_cast<std::streamsize>(length));
        if (!input) {
            throw std::runtime_error("unexpected end of GGUF string");
        }
    }
    return value;
}

std::string gguf_scalar_value_to_string(std::istream& input, const std::uint32_t type);

std::string gguf_array_value_to_string(std::istream& input) {
    const auto element_type = read_binary_value<std::uint32_t>(input);
    const auto count = read_binary_value<std::uint64_t>(input);
    std::ostringstream stream;
    stream << '[' << count << ']';
    const auto preview = std::min<std::uint64_t>(count, 4u);
    if (preview > 0u) {
        stream << '=';
    }
    for (std::uint64_t index = 0u; index < count; ++index) {
        const auto value = gguf_scalar_value_to_string(input, element_type);
        if (index < preview) {
            if (index > 0u) {
                stream << ',';
            }
            stream << value;
        }
    }
    return stream.str();
}

std::string gguf_scalar_value_to_string(std::istream& input, const std::uint32_t type) {
    switch (type) {
    case 0u:
        return std::to_string(read_binary_value<std::uint8_t>(input));
    case 1u:
        return std::to_string(read_binary_value<std::int8_t>(input));
    case 2u:
        return std::to_string(read_binary_value<std::uint16_t>(input));
    case 3u:
        return std::to_string(read_binary_value<std::int16_t>(input));
    case 4u:
        return std::to_string(read_binary_value<std::uint32_t>(input));
    case 5u:
        return std::to_string(read_binary_value<std::int32_t>(input));
    case 6u: {
        std::ostringstream stream;
        stream << read_binary_value<float>(input);
        return stream.str();
    }
    case 7u:
        return read_binary_value<std::uint8_t>(input) != 0u ? "true" : "false";
    case 8u:
        return read_gguf_string(input);
    case 9u:
        return gguf_array_value_to_string(input);
    case 10u:
        return std::to_string(read_binary_value<std::uint64_t>(input));
    case 11u:
        return std::to_string(read_binary_value<std::int64_t>(input));
    case 12u: {
        std::ostringstream stream;
        stream << read_binary_value<double>(input);
        return stream.str();
    }
    default:
        throw std::runtime_error("unsupported GGUF metadata type: " + std::to_string(type));
    }
}

std::uint64_t parse_u64_string(const std::string& value, const std::uint64_t fallback = 0u) {
    try {
        return static_cast<std::uint64_t>(std::stoull(value));
    } catch (const std::exception&) {
        return fallback;
    }
}

std::string metadata_value_or(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::string& key,
    const std::string& fallback = std::string()) {
    const auto it = metadata.find(key);
    return it == metadata.end() ? fallback : it->second;
}

std::uint64_t metadata_u64_by_suffix(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::vector<std::string>& suffixes,
    const std::uint64_t fallback) {
    for (const auto& [key, value] : metadata) {
        for (const auto& suffix : suffixes) {
            if (key == suffix ||
                (key.size() >= suffix.size() &&
                 key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0)) {
                return parse_u64_string(value, fallback);
            }
        }
    }
    return fallback;
}

std::string gguf_tensor_type_name(const std::uint32_t type) {
    switch (type) {
    case 0u:
        return "f32";
    case 1u:
        return "f16";
    case 2u:
        return "q4_0";
    case 3u:
        return "q4_1";
    case 6u:
        return "q5_0";
    case 7u:
        return "q5_1";
    case 8u:
        return "q8_0";
    case 14u:
        return "q6_k";
    default:
        return "ggml_type_" + std::to_string(type);
    }
}

std::string onnx_dtype_name(const std::uint32_t type) {
    switch (type) {
    case 1u:
        return "f32";
    case 2u:
        return "u8";
    case 3u:
        return "i8";
    case 4u:
        return "u16";
    case 5u:
        return "i16";
    case 6u:
        return "i32";
    case 7u:
        return "i64";
    case 9u:
        return "bool";
    case 10u:
        return "f16";
    case 11u:
        return "f64";
    case 16u:
        return "bf16";
    default:
        return "f32";
    }
}

std::int64_t node_int_attribute_or(
    const ImportedNodeSpec& node,
    const std::string& key,
    const std::int64_t fallback = 0) {
    const auto it = node.int_attributes.find(key);
    return it == node.int_attributes.end() ? fallback : it->second;
}

std::vector<std::int64_t> node_i64_list_attribute(
    const ImportedNodeSpec& node,
    const std::string& key) {
    const auto it = node.int_list_attributes.find(key);
    return it == node.int_list_attributes.end() ? std::vector<std::int64_t>{} : it->second;
}

std::vector<std::uint64_t> node_u64_list_attribute(
    const ImportedNodeSpec& node,
    const std::string& key) {
    std::vector<std::uint64_t> values;
    const auto source_values = node_i64_list_attribute(node, key);
    values.reserve(source_values.size());
    for (const auto value : source_values) {
        values.push_back(value < 0 ? 0u : static_cast<std::uint64_t>(value));
    }
    return values;
}

std::vector<float> node_float_list_attribute(
    const ImportedNodeSpec& node,
    const std::string& key) {
    const auto it = node.float_list_attributes.find(key);
    return it == node.float_list_attributes.end() ? std::vector<float>{} : it->second;
}

std::int64_t normalize_axis_index(const std::int64_t axis, const std::size_t rank) {
    const auto rank_i64 = static_cast<std::int64_t>(rank);
    const auto normalized = axis < 0 ? axis + rank_i64 : axis;
    if (normalized < 0 || normalized >= rank_i64) {
        return -1;
    }
    return normalized;
}

const ImportedValueSpec* node_input_value(
    const ImportedNodeSpec& node,
    const std::unordered_map<std::string, ImportedValueSpec>& values,
    const std::size_t input_index) {
    if (input_index >= node.inputs.size()) {
        return nullptr;
    }
    const auto it = values.find(node.inputs[input_index]);
    return it == values.end() ? nullptr : &it->second;
}

std::vector<std::int64_t> node_i64_input_or_attribute(
    const ImportedNodeSpec& node,
    const std::unordered_map<std::string, ImportedValueSpec>& values,
    const std::string& key,
    const std::size_t input_index) {
    auto data = node_i64_list_attribute(node, key);
    if (!data.empty()) {
        return data;
    }
    if (const auto* input = node_input_value(node, values, input_index);
        input != nullptr && !input->int64_data.empty()) {
        return input->int64_data;
    }
    return {};
}

std::vector<std::uint64_t> resolve_reshape_target_shape(
    const std::vector<std::uint64_t>& input_shape,
    const std::vector<std::int64_t>& raw_target_shape) {
    if (input_shape.empty() || raw_target_shape.empty()) {
        return {};
    }

    std::vector<std::uint64_t> output_shape;
    output_shape.reserve(raw_target_shape.size());
    std::optional<std::size_t> infer_index;
    std::uint64_t known_product = 1u;
    for (std::size_t index = 0u; index < raw_target_shape.size(); ++index) {
        const auto dim = raw_target_shape[index];
        if (dim == 0) {
            const auto copied = index < input_shape.size() ? input_shape[index] : 1u;
            output_shape.push_back(copied);
            known_product *= std::max<std::uint64_t>(1u, copied);
            continue;
        }
        if (dim == -1) {
            if (infer_index.has_value()) {
                return {};
            }
            infer_index = index;
            output_shape.push_back(1u);
            continue;
        }
        if (dim < -1) {
            return {};
        }
        output_shape.push_back(static_cast<std::uint64_t>(dim));
        known_product *= std::max<std::uint64_t>(1u, output_shape.back());
    }

    if (infer_index.has_value()) {
        const auto input_elements = std::max<std::uint64_t>(1u, shape_elements(input_shape));
        output_shape[*infer_index] = known_product == 0u ? 1u : std::max<std::uint64_t>(1u, input_elements / known_product);
    }
    return output_shape;
}

std::vector<std::uint64_t> normalize_axes_for_rank(
    const std::vector<std::int64_t>& axes,
    const std::size_t rank) {
    std::vector<std::uint64_t> normalized;
    normalized.reserve(axes.size());
    for (const auto axis : axes) {
        const auto normalized_axis = normalize_axis_index(axis, rank);
        if (normalized_axis >= 0) {
            normalized.push_back(static_cast<std::uint64_t>(normalized_axis));
        }
    }
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
    return normalized;
}

std::vector<std::uint64_t> broadcast_batch_shape(
    const std::vector<std::uint64_t>& lhs,
    const std::vector<std::uint64_t>& rhs) {
    const auto rank = std::max(lhs.size(), rhs.size());
    std::vector<std::uint64_t> output(rank, 1u);
    for (std::size_t index = 0u; index < rank; ++index) {
        const auto lhs_dim = index < rank - lhs.size() ? 1u : lhs[index - (rank - lhs.size())];
        const auto rhs_dim = index < rank - rhs.size() ? 1u : rhs[index - (rank - rhs.size())];
        output[index] = std::max(lhs_dim, rhs_dim);
    }
    return output;
}

std::vector<std::uint64_t> broadcast_full_shape(
    const std::vector<std::uint64_t>& lhs,
    const std::vector<std::uint64_t>& rhs) {
    return broadcast_batch_shape(lhs, rhs);
}

std::int64_t normalize_slice_bound(
    std::int64_t value,
    const std::int64_t dim,
    const bool positive_step,
    const bool end_bound) {
    if (dim <= 0) {
        return 0;
    }
    if (value < 0) {
        value += dim;
    }
    if (positive_step) {
        const auto upper = dim;
        const auto lower = std::int64_t{0};
        if (end_bound) {
            return std::clamp(value, lower, upper);
        }
        return std::clamp(value, lower, upper);
    }
    const auto upper = dim - 1;
    const auto lower = std::int64_t{-1};
    return std::clamp(value, lower, upper);
}

std::uint64_t slice_extent(
    const std::uint64_t input_extent,
    const std::int64_t raw_start,
    const std::int64_t raw_end,
    const std::int64_t raw_step) {
    if (input_extent == 0u || raw_step == 0) {
        return 0u;
    }

    const auto dim = static_cast<std::int64_t>(input_extent);
    const auto positive_step = raw_step > 0;
    const auto step = positive_step ? raw_step : -raw_step;
    const auto start = normalize_slice_bound(raw_start, dim, positive_step, false);
    const auto end = normalize_slice_bound(raw_end, dim, positive_step, true);
    if (positive_step) {
        if (end <= start) {
            return 0u;
        }
        return static_cast<std::uint64_t>((end - start + step - 1) / step);
    }
    if (start <= end) {
        return 0u;
    }
    return static_cast<std::uint64_t>((start - end + step - 1) / step);
}

std::vector<std::vector<std::uint64_t>> infer_imported_split_output_shapes(
    const ImportedNodeSpec& node,
    const std::unordered_map<std::string, ImportedValueSpec>& values,
    const ImportedValueSpec& primary_input) {
    if (node.outputs.empty() || primary_input.shape.empty()) {
        return {};
    }

    const auto axis = normalize_axis_index(node_int_attribute_or(node, "axis", 0), primary_input.shape.size());
    if (axis < 0) {
        return {};
    }

    auto split_sizes = node_i64_input_or_attribute(node, values, "split", 1u);
    if (split_sizes.empty()) {
        const auto total = primary_input.shape[static_cast<std::size_t>(axis)];
        const auto count = std::max<std::uint64_t>(1u, static_cast<std::uint64_t>(node.outputs.size()));
        const auto base = total / count;
        auto remainder = total % count;
        split_sizes.assign(node.outputs.size(), static_cast<std::int64_t>(base));
        for (auto& size : split_sizes) {
            if (remainder > 0u) {
                ++size;
                --remainder;
            }
        }
    }

    std::vector<std::vector<std::uint64_t>> output_shapes;
    output_shapes.reserve(node.outputs.size());
    for (std::size_t index = 0u; index < node.outputs.size(); ++index) {
        auto shape = primary_input.shape;
        if (index < split_sizes.size() && split_sizes[index] > 0) {
            shape[static_cast<std::size_t>(axis)] = static_cast<std::uint64_t>(split_sizes[index]);
        }
        output_shapes.push_back(std::move(shape));
    }
    return output_shapes;
}

std::vector<std::uint64_t> infer_imported_output_shape(
    const ImportedNodeSpec& node,
    const std::unordered_map<std::string, ImportedValueSpec>& values) {
    if (!node.shape.empty()) {
        return node.shape;
    }

    const auto primary_input_it =
        !node.inputs.empty() ? values.find(node.inputs.front()) : values.end();
    const auto* primary_input = primary_input_it == values.end() ? nullptr : &primary_input_it->second;
    if (primary_input == nullptr || primary_input->shape.empty()) {
        return {};
    }

    const auto lowered_op = lowercase_copy(node.op_type);
    if (lowered_op == "gatherelements") {
        if (const auto* indices = node_input_value(node, values, 1u); indices != nullptr && !indices->shape.empty()) {
            return indices->shape;
        }
        return primary_input->shape;
    }

    if (lowered_op == "gather") {
        const auto* indices = node_input_value(node, values, 1u);
        if (indices == nullptr || indices->shape.empty()) {
            return primary_input->shape;
        }
        const auto axis = normalize_axis_index(node_int_attribute_or(node, "axis", 0), primary_input->shape.size());
        if (axis < 0) {
            return {};
        }
        std::vector<std::uint64_t> output_shape;
        output_shape.reserve(primary_input->shape.size() + indices->shape.size());
        output_shape.insert(
            output_shape.end(),
            primary_input->shape.begin(),
            primary_input->shape.begin() + axis);
        output_shape.insert(output_shape.end(), indices->shape.begin(), indices->shape.end());
        output_shape.insert(
            output_shape.end(),
            primary_input->shape.begin() + axis + 1,
            primary_input->shape.end());
        return output_shape;
    }

    if (lowered_op == "where") {
        auto output_shape = primary_input->shape;
        if (const auto* x = node_input_value(node, values, 1u); x != nullptr && !x->shape.empty()) {
            output_shape = broadcast_full_shape(output_shape, x->shape);
        }
        if (const auto* y = node_input_value(node, values, 2u); y != nullptr && !y->shape.empty()) {
            output_shape = broadcast_full_shape(output_shape, y->shape);
        }
        return output_shape;
    }

    if (lowered_op == "slice") {
        auto output_shape = primary_input->shape;
        const auto starts = node_i64_input_or_attribute(node, values, "starts", 1u);
        const auto ends = node_i64_input_or_attribute(node, values, "ends", 2u);
        auto axes = node_i64_input_or_attribute(node, values, "axes", 3u);
        auto steps = node_i64_input_or_attribute(node, values, "steps", 4u);
        if (starts.empty() || ends.empty()) {
            return output_shape;
        }
        if (axes.empty()) {
            axes.resize(starts.size());
            std::iota(axes.begin(), axes.end(), 0);
        }
        if (steps.empty()) {
            steps.assign(starts.size(), 1);
        }
        for (std::size_t index = 0u; index < starts.size() && index < ends.size() && index < axes.size(); ++index) {
            const auto axis = normalize_axis_index(axes[index], output_shape.size());
            if (axis < 0) {
                continue;
            }
            const auto step = index < steps.size() ? steps[index] : 1;
            output_shape[static_cast<std::size_t>(axis)] = slice_extent(
                output_shape[static_cast<std::size_t>(axis)],
                starts[index],
                ends[index],
                step == 0 ? 1 : step);
        }
        return output_shape;
    }

    if (lowered_op == "pad") {
        auto output_shape = primary_input->shape;
        auto pads = node_i64_input_or_attribute(node, values, "pads", 1u);
        if (pads.empty()) {
            pads = node_i64_list_attribute(node, "pads");
        }
        if (pads.size() < output_shape.size() * 2u) {
            return output_shape;
        }
        for (std::size_t index = 0u; index < output_shape.size(); ++index) {
            const auto begin_pad = std::max<std::int64_t>(0, pads[index]);
            const auto end_pad = std::max<std::int64_t>(0, pads[index + output_shape.size()]);
            output_shape[index] += static_cast<std::uint64_t>(begin_pad + end_pad);
        }
        return output_shape;
    }

    if (lowered_op == "reshape") {
        return resolve_reshape_target_shape(
            primary_input->shape,
            node_i64_input_or_attribute(node, values, "shape", 1u));
    }

    if (lowered_op == "expand") {
        const auto target_shape = resolve_reshape_target_shape(
            std::vector<std::uint64_t>(node_i64_input_or_attribute(node, values, "shape", 1u).size(), 1u),
            node_i64_input_or_attribute(node, values, "shape", 1u));
        if (target_shape.empty()) {
            return primary_input->shape;
        }
        return broadcast_full_shape(primary_input->shape, target_shape);
    }

    if (lowered_op == "tile") {
        auto output_shape = primary_input->shape;
        const auto repeats = node_i64_input_or_attribute(node, values, "repeats", 1u);
        if (repeats.empty()) {
            return output_shape;
        }
        if (output_shape.size() < repeats.size()) {
            output_shape.insert(output_shape.begin(), repeats.size() - output_shape.size(), 1u);
        }
        const auto offset = output_shape.size() - std::min(output_shape.size(), repeats.size());
        for (std::size_t index = 0u; index < repeats.size(); ++index) {
            output_shape[offset + index] *= static_cast<std::uint64_t>(std::max<std::int64_t>(1, repeats[index]));
        }
        return output_shape;
    }

    if (lowered_op == "unsqueeze") {
        const auto axes = normalize_axes_for_rank(
            node_i64_input_or_attribute(node, values, "axes", 1u),
            primary_input->shape.size() + node_i64_input_or_attribute(node, values, "axes", 1u).size());
        if (axes.empty()) {
            return primary_input->shape;
        }
        std::vector<std::uint64_t> output_shape;
        output_shape.reserve(primary_input->shape.size() + axes.size());
        std::size_t input_index = 0u;
        for (std::size_t output_index = 0u; output_index < primary_input->shape.size() + axes.size(); ++output_index) {
            if (std::binary_search(axes.begin(), axes.end(), static_cast<std::uint64_t>(output_index))) {
                output_shape.push_back(1u);
            } else if (input_index < primary_input->shape.size()) {
                output_shape.push_back(primary_input->shape[input_index++]);
            }
        }
        return output_shape;
    }

    if (lowered_op == "squeeze") {
        const auto axes = normalize_axes_for_rank(
            node_i64_input_or_attribute(node, values, "axes", 1u),
            primary_input->shape.size());
        std::vector<std::uint64_t> output_shape;
        output_shape.reserve(primary_input->shape.size());
        for (std::size_t index = 0u; index < primary_input->shape.size(); ++index) {
            const auto selected = std::binary_search(axes.begin(), axes.end(), static_cast<std::uint64_t>(index));
            if ((axes.empty() && primary_input->shape[index] == 1u) || (selected && primary_input->shape[index] == 1u)) {
                continue;
            }
            output_shape.push_back(primary_input->shape[index]);
        }
        return output_shape.empty() ? std::vector<std::uint64_t>{1u} : output_shape;
    }

    if (lowered_op == "split") {
        const auto shapes = infer_imported_split_output_shapes(node, values, *primary_input);
        return shapes.empty() ? primary_input->shape : shapes.front();
    }

    if (lowered_op == "flatten") {
        auto axis = node_int_attribute_or(node, "axis", 1);
        const auto rank = static_cast<std::int64_t>(primary_input->shape.size());
        if (axis < 0) {
            axis += rank;
        }
        axis = std::clamp<std::int64_t>(axis, 0, rank);
        std::vector<std::uint64_t> output_shape;
        output_shape.reserve(2u);
        output_shape.push_back(axis == 0 ? 1u : safe_product(std::vector<std::uint64_t>(
                                                  primary_input->shape.begin(),
                                                  primary_input->shape.begin() + axis)));
        output_shape.push_back(
            axis >= rank
                ? 1u
                : safe_product(std::vector<std::uint64_t>(
                      primary_input->shape.begin() + axis,
                      primary_input->shape.end())));
        return output_shape;
    }

    if (lowered_op == "concat") {
        auto output_shape = primary_input->shape;
        const auto axis = normalize_axis_index(node_int_attribute_or(node, "axis", 0), output_shape.size());
        if (axis < 0) {
            return output_shape;
        }
        for (std::size_t input_index = 1u; input_index < node.inputs.size(); ++input_index) {
            if (const auto* value = node_input_value(node, values, input_index); value != nullptr &&
                value->shape.size() == output_shape.size()) {
                output_shape[static_cast<std::size_t>(axis)] += value->shape[static_cast<std::size_t>(axis)];
            }
        }
        return output_shape;
    }

    if (lowered_op == "matmul") {
        const auto* rhs = node_input_value(node, values, 1u);
        if (rhs == nullptr || rhs->shape.empty()) {
            return {};
        }

        auto lhs_shape = primary_input->shape;
        auto rhs_shape = rhs->shape;
        const bool lhs_was_vector = lhs_shape.size() == 1u;
        const bool rhs_was_vector = rhs_shape.size() == 1u;
        if (lhs_was_vector) {
            lhs_shape.insert(lhs_shape.begin(), 1u);
        }
        if (rhs_was_vector) {
            rhs_shape.push_back(1u);
        }
        if (lhs_shape.size() < 2u || rhs_shape.size() < 2u) {
            return {};
        }

        std::vector<std::uint64_t> output_shape = broadcast_batch_shape(
            std::vector<std::uint64_t>(lhs_shape.begin(), lhs_shape.end() - 2),
            std::vector<std::uint64_t>(rhs_shape.begin(), rhs_shape.end() - 2));
        output_shape.push_back(lhs_shape[lhs_shape.size() - 2u]);
        output_shape.push_back(rhs_shape.back());

        if (lhs_was_vector && !output_shape.empty()) {
            output_shape.erase(output_shape.end() - 2);
        }
        if (rhs_was_vector && !output_shape.empty()) {
            output_shape.pop_back();
        }
        return output_shape.empty() ? std::vector<std::uint64_t>{1u} : output_shape;
    }

    if (lowered_op == "transpose") {
        auto perm = node_u64_list_attribute(node, "perm");
        if (perm.empty()) {
            perm.resize(primary_input->shape.size());
            for (std::size_t index = 0u; index < perm.size(); ++index) {
                perm[index] = static_cast<std::uint64_t>(perm.size() - 1u - index);
            }
        }
        std::vector<std::uint64_t> output_shape;
        output_shape.reserve(perm.size());
        for (const auto axis : perm) {
            if (axis < primary_input->shape.size()) {
                output_shape.push_back(primary_input->shape[static_cast<std::size_t>(axis)]);
            }
        }
        return output_shape;
    }

    if (lowered_op.find("reduce") != std::string::npos) {
        const auto axes = node_i64_list_attribute(node, "axes");
        const auto keepdims = node_int_attribute_or(node, "keepdims", 1) != 0;
        std::vector<bool> reduce_mask(primary_input->shape.size(), false);
        if (axes.empty()) {
            std::fill(reduce_mask.begin(), reduce_mask.end(), true);
        } else {
            for (const auto axis : axes) {
                const auto normalized = normalize_axis_index(axis, reduce_mask.size());
                if (normalized >= 0) {
                    reduce_mask[static_cast<std::size_t>(normalized)] = true;
                }
            }
        }

        std::vector<std::uint64_t> output_shape;
        output_shape.reserve(primary_input->shape.size());
        for (std::size_t index = 0u; index < primary_input->shape.size(); ++index) {
            if (reduce_mask[index]) {
                if (keepdims) {
                    output_shape.push_back(1u);
                }
            } else {
                output_shape.push_back(primary_input->shape[index]);
            }
        }
        if (output_shape.empty()) {
            output_shape.push_back(1u);
        }
        return output_shape;
    }

    if (lowered_op.find("resize") != std::string::npos ||
        lowered_op.find("upsample") != std::string::npos ||
        lowered_op.find("interpolate") != std::string::npos) {
        auto output_shape = primary_input->shape;
        const auto sizes = node_u64_list_attribute(node, "sizes");
        if (!sizes.empty()) {
            return sizes;
        }
        const auto scales = node_float_list_attribute(node, "scales");
        if (scales.size() == primary_input->shape.size()) {
            for (std::size_t index = 0u; index < output_shape.size(); ++index) {
                const auto safe_scale = std::max(scales[index], 0.0f);
                output_shape[index] = std::max<std::uint64_t>(
                    1u,
                    static_cast<std::uint64_t>(std::llround(
                        static_cast<double>(primary_input->shape[index]) * static_cast<double>(safe_scale))));
            }
        }
        return output_shape;
    }

    return {};
}

std::vector<std::vector<std::uint64_t>> infer_imported_output_shapes(
    const ImportedNodeSpec& node,
    const std::unordered_map<std::string, ImportedValueSpec>& values) {
    const auto lowered_op = lowercase_copy(node.op_type);
    if (lowered_op == "split") {
        const auto primary_input_it =
            !node.inputs.empty() ? values.find(node.inputs.front()) : values.end();
        if (primary_input_it == values.end()) {
            return {};
        }
        return infer_imported_split_output_shapes(node, values, primary_input_it->second);
    }
    const auto output_shape = infer_imported_output_shape(node, values);
    if (output_shape.empty()) {
        return {};
    }
    return std::vector<std::vector<std::uint64_t>>(node.outputs.size(), output_shape);
}

std::uint64_t read_protobuf_varint(const std::vector<std::uint8_t>& bytes, std::size_t& offset) {
    std::uint64_t value = 0u;
    int shift = 0;
    while (offset < bytes.size() && shift <= 63) {
        const auto byte = bytes[offset++];
        value |= static_cast<std::uint64_t>(byte & 0x7fu) << shift;
        if ((byte & 0x80u) == 0u) {
            return value;
        }
        shift += 7;
    }
    throw std::runtime_error("invalid protobuf varint");
}

std::uint32_t read_protobuf_fixed32(const std::vector<std::uint8_t>& bytes, std::size_t& offset) {
    if (offset + 4u > bytes.size()) {
        throw std::runtime_error("invalid protobuf fixed32 field");
    }
    std::uint32_t value = 0u;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    offset += 4u;
    return value;
}

std::int64_t decode_protobuf_int64(const std::uint64_t value) {
    return std::bit_cast<std::int64_t>(value);
}

std::vector<std::uint8_t> read_protobuf_length_delimited(
    const std::vector<std::uint8_t>& bytes,
    std::size_t& offset) {
    const auto length = read_protobuf_varint(bytes, offset);
    if (offset + length > bytes.size()) {
        throw std::runtime_error("invalid protobuf length-delimited field");
    }
    const auto begin = bytes.begin() + static_cast<std::ptrdiff_t>(offset);
    const auto end = begin + static_cast<std::ptrdiff_t>(length);
    offset += static_cast<std::size_t>(length);
    return std::vector<std::uint8_t>(begin, end);
}

std::string read_protobuf_string(const std::vector<std::uint8_t>& bytes, std::size_t& offset) {
    const auto value = read_protobuf_length_delimited(bytes, offset);
    return std::string(value.begin(), value.end());
}

void skip_protobuf_field(
    const std::vector<std::uint8_t>& bytes,
    std::size_t& offset,
    const std::uint32_t wire_type) {
    switch (wire_type) {
    case 0u:
        (void)read_protobuf_varint(bytes, offset);
        return;
    case 1u:
        if (offset + 8u > bytes.size()) {
            throw std::runtime_error("invalid protobuf fixed64 field");
        }
        offset += 8u;
        return;
    case 2u: {
        const auto length = read_protobuf_varint(bytes, offset);
        if (offset + length > bytes.size()) {
            throw std::runtime_error("invalid protobuf length field");
        }
        offset += static_cast<std::size_t>(length);
        return;
    }
    case 5u:
        if (offset + 4u > bytes.size()) {
            throw std::runtime_error("invalid protobuf fixed32 field");
        }
        offset += 4u;
        return;
    default:
        throw std::runtime_error("unsupported protobuf wire type: " + std::to_string(wire_type));
    }
}

std::vector<std::uint64_t> parse_onnx_shape_message(const std::vector<std::uint8_t>& bytes);

std::pair<std::string, std::vector<std::uint64_t>> parse_onnx_type_proto(const std::vector<std::uint8_t>& bytes) {
    std::string dtype = "f32";
    std::vector<std::uint64_t> shape;
    std::size_t offset = 0u;
    while (offset < bytes.size()) {
        const auto key = read_protobuf_varint(bytes, offset);
        const auto field_number = static_cast<std::uint32_t>(key >> 3u);
        const auto wire_type = static_cast<std::uint32_t>(key & 0x7u);
        if (field_number == 1u && wire_type == 2u) {
            const auto tensor_type = read_protobuf_length_delimited(bytes, offset);
            std::size_t tensor_offset = 0u;
            while (tensor_offset < tensor_type.size()) {
                const auto tensor_key = read_protobuf_varint(tensor_type, tensor_offset);
                const auto tensor_field = static_cast<std::uint32_t>(tensor_key >> 3u);
                const auto tensor_wire = static_cast<std::uint32_t>(tensor_key & 0x7u);
                if (tensor_field == 1u && tensor_wire == 0u) {
                    dtype = onnx_dtype_name(static_cast<std::uint32_t>(read_protobuf_varint(tensor_type, tensor_offset)));
                } else if (tensor_field == 2u && tensor_wire == 2u) {
                    shape = parse_onnx_shape_message(read_protobuf_length_delimited(tensor_type, tensor_offset));
                } else {
                    skip_protobuf_field(tensor_type, tensor_offset, tensor_wire);
                }
            }
        } else {
            skip_protobuf_field(bytes, offset, wire_type);
        }
    }
    return {dtype, shape};
}

std::vector<std::uint64_t> parse_onnx_shape_message(const std::vector<std::uint8_t>& bytes) {
    std::vector<std::uint64_t> shape;
    std::size_t offset = 0u;
    while (offset < bytes.size()) {
        const auto key = read_protobuf_varint(bytes, offset);
        const auto field_number = static_cast<std::uint32_t>(key >> 3u);
        const auto wire_type = static_cast<std::uint32_t>(key & 0x7u);
        if (field_number == 1u && wire_type == 2u) {
            const auto dim = read_protobuf_length_delimited(bytes, offset);
            std::size_t dim_offset = 0u;
            std::uint64_t dim_value = 1u;
            while (dim_offset < dim.size()) {
                const auto dim_key = read_protobuf_varint(dim, dim_offset);
                const auto dim_field = static_cast<std::uint32_t>(dim_key >> 3u);
                const auto dim_wire = static_cast<std::uint32_t>(dim_key & 0x7u);
                if (dim_field == 1u && dim_wire == 0u) {
                    dim_value = read_protobuf_varint(dim, dim_offset);
                } else {
                    skip_protobuf_field(dim, dim_offset, dim_wire);
                }
            }
            shape.push_back(dim_value == 0u ? 1u : dim_value);
        } else {
            skip_protobuf_field(bytes, offset, wire_type);
        }
    }
    return shape;
}

OnnxValueInfo parse_onnx_value_info(const std::vector<std::uint8_t>& bytes) {
    OnnxValueInfo value;
    std::size_t offset = 0u;
    while (offset < bytes.size()) {
        const auto key = read_protobuf_varint(bytes, offset);
        const auto field_number = static_cast<std::uint32_t>(key >> 3u);
        const auto wire_type = static_cast<std::uint32_t>(key & 0x7u);
        if (field_number == 1u && wire_type == 2u) {
            value.name = read_protobuf_string(bytes, offset);
        } else if (field_number == 2u && wire_type == 2u) {
            const auto [dtype, shape] = parse_onnx_type_proto(read_protobuf_length_delimited(bytes, offset));
            value.dtype = dtype;
            value.shape = shape;
        } else {
            skip_protobuf_field(bytes, offset, wire_type);
        }
    }
    value.bytes = bytes_for_shape(value.shape, value.dtype);
    return value;
}

OnnxValueInfo parse_onnx_tensor_proto(const std::vector<std::uint8_t>& bytes) {
    OnnxValueInfo value;
    value.initializer = true;
    value.persistent = true;
    std::size_t offset = 0u;
    std::vector<std::uint64_t> dims;
    std::vector<std::uint8_t> raw_data;
    std::uint64_t raw_bytes = 0u;
    while (offset < bytes.size()) {
        const auto key = read_protobuf_varint(bytes, offset);
        const auto field_number = static_cast<std::uint32_t>(key >> 3u);
        const auto wire_type = static_cast<std::uint32_t>(key & 0x7u);
        if (field_number == 1u && wire_type == 0u) {
            dims.push_back(read_protobuf_varint(bytes, offset));
        } else if (field_number == 1u && wire_type == 2u) {
            auto packed_offset = std::size_t{0u};
            const auto packed = read_protobuf_length_delimited(bytes, offset);
            while (packed_offset < packed.size()) {
                dims.push_back(read_protobuf_varint(packed, packed_offset));
            }
        } else if (field_number == 2u && wire_type == 0u) {
            value.dtype = onnx_dtype_name(static_cast<std::uint32_t>(read_protobuf_varint(bytes, offset)));
        } else if (field_number == 7u && wire_type == 0u) {
            value.int64_data.push_back(decode_protobuf_int64(read_protobuf_varint(bytes, offset)));
        } else if (field_number == 7u && wire_type == 2u) {
            auto packed_offset = std::size_t{0u};
            const auto packed = read_protobuf_length_delimited(bytes, offset);
            while (packed_offset < packed.size()) {
                value.int64_data.push_back(decode_protobuf_int64(read_protobuf_varint(packed, packed_offset)));
            }
        } else if (field_number == 8u && wire_type == 2u) {
            value.name = read_protobuf_string(bytes, offset);
        } else if (field_number == 9u && wire_type == 2u) {
            raw_data = read_protobuf_length_delimited(bytes, offset);
            raw_bytes = static_cast<std::uint64_t>(raw_data.size());
        } else {
            skip_protobuf_field(bytes, offset, wire_type);
        }
    }
    value.shape = std::move(dims);
    if (value.dtype == "i64" && value.int64_data.empty() && raw_data.size() >= sizeof(std::int64_t) &&
        raw_data.size() % sizeof(std::int64_t) == 0u) {
        for (std::size_t data_offset = 0u; data_offset < raw_data.size(); data_offset += sizeof(std::int64_t)) {
            std::int64_t element = 0;
            std::memcpy(&element, raw_data.data() + data_offset, sizeof(element));
            value.int64_data.push_back(element);
        }
    }
    value.bytes = raw_bytes == 0u ? bytes_for_shape(value.shape, value.dtype) : raw_bytes;
    return value;
}

void parse_onnx_attribute_proto(
    const std::vector<std::uint8_t>& bytes,
    OnnxNodeInfo& node) {
    std::string name;
    std::uint32_t type = 0u;
    std::optional<std::int64_t> int_value;
    std::optional<float> float_value;
    std::vector<std::int64_t> int_values;
    std::vector<float> float_values;
    std::optional<std::string> string_value;

    std::size_t offset = 0u;
    while (offset < bytes.size()) {
        const auto key = read_protobuf_varint(bytes, offset);
        const auto field_number = static_cast<std::uint32_t>(key >> 3u);
        const auto wire_type = static_cast<std::uint32_t>(key & 0x7u);
        if (field_number == 1u && wire_type == 2u) {
            name = read_protobuf_string(bytes, offset);
        } else if (field_number == 2u && wire_type == 5u) {
            float_value = std::bit_cast<float>(read_protobuf_fixed32(bytes, offset));
        } else if (field_number == 3u && wire_type == 0u) {
            int_value = decode_protobuf_int64(read_protobuf_varint(bytes, offset));
        } else if (field_number == 4u && wire_type == 2u) {
            string_value = read_protobuf_string(bytes, offset);
        } else if (field_number == 20u && wire_type == 0u) {
            type = static_cast<std::uint32_t>(read_protobuf_varint(bytes, offset));
        } else if (field_number == 7u && wire_type == 0u) {
            if (const auto value = read_protobuf_varint(bytes, offset);
                value <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
                float_values.push_back(static_cast<float>(value));
            }
        } else if (field_number == 7u && wire_type == 5u) {
            float_values.push_back(std::bit_cast<float>(read_protobuf_fixed32(bytes, offset)));
        } else if (field_number == 7u && wire_type == 2u) {
            std::size_t packed_offset = 0u;
            const auto packed = read_protobuf_length_delimited(bytes, offset);
            while (packed_offset < packed.size()) {
                float_values.push_back(std::bit_cast<float>(read_protobuf_fixed32(packed, packed_offset)));
            }
        } else if (field_number == 8u && wire_type == 0u) {
            int_values.push_back(decode_protobuf_int64(read_protobuf_varint(bytes, offset)));
        } else if (field_number == 8u && wire_type == 2u) {
            std::size_t packed_offset = 0u;
            const auto packed = read_protobuf_length_delimited(bytes, offset);
            while (packed_offset < packed.size()) {
                int_values.push_back(decode_protobuf_int64(read_protobuf_varint(packed, packed_offset)));
            }
        } else {
            skip_protobuf_field(bytes, offset, wire_type);
        }
    }

    if (name.empty()) {
        return;
    }
    if (!float_values.empty()) {
        node.float_list_attributes.emplace(name, std::move(float_values));
        return;
    }
    if (!int_values.empty()) {
        node.int_list_attributes.emplace(name, std::move(int_values));
        return;
    }
    if (float_value.has_value()) {
        node.float_attributes.emplace(name, *float_value);
        return;
    }
    if (int_value.has_value()) {
        node.int_attributes.emplace(name, *int_value);
        return;
    }
    if (string_value.has_value()) {
        node.string_attributes.emplace(name, *string_value);
        return;
    }
    if (type == 7u) {
        node.int_list_attributes.emplace(name, std::vector<std::int64_t>{});
    }
}

OnnxNodeInfo parse_onnx_node_proto(const std::vector<std::uint8_t>& bytes) {
    OnnxNodeInfo node;
    std::size_t offset = 0u;
    while (offset < bytes.size()) {
        const auto key = read_protobuf_varint(bytes, offset);
        const auto field_number = static_cast<std::uint32_t>(key >> 3u);
        const auto wire_type = static_cast<std::uint32_t>(key & 0x7u);
        if (field_number == 1u && wire_type == 2u) {
            node.inputs.push_back(read_protobuf_string(bytes, offset));
        } else if (field_number == 2u && wire_type == 2u) {
            node.outputs.push_back(read_protobuf_string(bytes, offset));
        } else if (field_number == 3u && wire_type == 2u) {
            node.name = read_protobuf_string(bytes, offset);
        } else if (field_number == 4u && wire_type == 2u) {
            node.op_type = read_protobuf_string(bytes, offset);
        } else if (field_number == 5u && wire_type == 2u) {
            parse_onnx_attribute_proto(read_protobuf_length_delimited(bytes, offset), node);
        } else {
            skip_protobuf_field(bytes, offset, wire_type);
        }
    }
    return node;
}

OnnxGraphInfo parse_onnx_graph_proto(const std::vector<std::uint8_t>& bytes) {
    OnnxGraphInfo graph;
    std::size_t offset = 0u;
    while (offset < bytes.size()) {
        const auto key = read_protobuf_varint(bytes, offset);
        const auto field_number = static_cast<std::uint32_t>(key >> 3u);
        const auto wire_type = static_cast<std::uint32_t>(key & 0x7u);
        if (field_number == 1u && wire_type == 2u) {
            graph.nodes.push_back(parse_onnx_node_proto(read_protobuf_length_delimited(bytes, offset)));
        } else if (field_number == 2u && wire_type == 2u) {
            graph.name = read_protobuf_string(bytes, offset);
        } else if (field_number == 5u && wire_type == 2u) {
            graph.values.push_back(parse_onnx_tensor_proto(read_protobuf_length_delimited(bytes, offset)));
        } else if ((field_number == 11u || field_number == 12u || field_number == 13u) && wire_type == 2u) {
            auto value = parse_onnx_value_info(read_protobuf_length_delimited(bytes, offset));
            if (field_number == 11u) {
                value.host_visible = true;
            }
            graph.values.push_back(std::move(value));
        } else {
            skip_protobuf_field(bytes, offset, wire_type);
        }
    }
    return graph;
}

OnnxGraphInfo parse_onnx_model(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("unable to open ONNX file: " + path.string());
    }
    std::vector<std::uint8_t> bytes{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    OnnxGraphInfo graph;
    std::size_t offset = 0u;
    while (offset < bytes.size()) {
        const auto key = read_protobuf_varint(bytes, offset);
        const auto field_number = static_cast<std::uint32_t>(key >> 3u);
        const auto wire_type = static_cast<std::uint32_t>(key & 0x7u);
        if (field_number == 7u && wire_type == 2u) {
            graph = parse_onnx_graph_proto(read_protobuf_length_delimited(bytes, offset));
        } else {
            skip_protobuf_field(bytes, offset, wire_type);
        }
    }
    if (graph.nodes.empty()) {
        throw std::runtime_error("ONNX graph contained no nodes: " + path.string());
    }
    return graph;
}

WorkloadManifest load_onnx_workload_source(const std::filesystem::path& path) {
    const auto onnx = parse_onnx_model(path);

    WorkloadManifest manifest;
    manifest.source_path = path;
    manifest.source_format = "onnx";
    manifest.source_entry = onnx.name;
    manifest.imported = true;

    ImportedSourceSpec source;
    source.format = "onnx";
    source.entry = onnx.name;

    manifest.workload.name = onnx.name.empty() ? path.stem().string() : onnx.name;
    manifest.workload.kind = WorkloadKind::inference;
    manifest.workload.dataset_tag = "onnx-" + manifest.workload.name;
    manifest.workload.matrix_friendly = true;
    manifest.workload.phase = WorkloadPhase::prefill;

    std::unordered_map<std::string, ImportedValueSpec> value_map;
    value_map.reserve(onnx.values.size());
    for (const auto& value : onnx.values) {
        auto& slot = value_map[value.name];
        if (slot.id.empty()) {
            slot.id = value.name;
        }
        if (!value.shape.empty()) {
            slot.shape = value.shape;
        }
        if (!value.dtype.empty()) {
            slot.dtype = value.dtype;
        }
        if (!value.int64_data.empty()) {
            slot.int64_data = value.int64_data;
        }
        if (value.bytes > 0u) {
            slot.bytes = value.bytes;
        }
        slot.initializer = slot.initializer || value.initializer;
        slot.persistent = slot.persistent || value.initializer;
        slot.host_visible = slot.host_visible || value.host_visible;
    }

    std::vector<ImportedNodeSpec> nodes;
    nodes.reserve(onnx.nodes.size());
    std::size_t synthetic_index = 0u;
    for (const auto& raw_node : onnx.nodes) {
        ImportedNodeSpec node;
        node.name = raw_node.name.empty() ? (lowercase_copy(raw_node.op_type) + "-" + std::to_string(++synthetic_index))
                                          : raw_node.name;
        node.op_type = raw_node.op_type;
        node.inputs = raw_node.inputs;
        node.outputs = raw_node.outputs;
        node.int_attributes = raw_node.int_attributes;
        node.int_list_attributes = raw_node.int_list_attributes;
        node.float_attributes = raw_node.float_attributes;
        node.float_list_attributes = raw_node.float_list_attributes;
        node.string_attributes = raw_node.string_attributes;

        for (const auto& input_id : node.inputs) {
            if (input_id.empty()) {
                continue;
            }
            auto& value = value_map[input_id];
            if (value.id.empty()) {
                value.id = input_id;
                value.host_visible = true;
            }
            if (std::find(value.consumer_operations.begin(), value.consumer_operations.end(), node.name) ==
                value.consumer_operations.end()) {
                value.consumer_operations.push_back(node.name);
            }
        }
        for (const auto& output_id : node.outputs) {
            if (output_id.empty()) {
                continue;
            }
            auto& value = value_map[output_id];
            if (value.id.empty()) {
                value.id = output_id;
            }
            if (value.producer_operation.empty()) {
                value.producer_operation = node.name;
            }
            if (node.shape.empty() && !value.shape.empty()) {
                node.shape = value.shape;
            }
        }
        const auto inferred_output_shapes = infer_imported_output_shapes(node, value_map);
        if (node.shape.empty() && !inferred_output_shapes.empty()) {
            node.shape = inferred_output_shapes.front();
        }
        const auto primary_input_it =
            !node.inputs.empty() ? value_map.find(node.inputs.front()) : value_map.end();
        const auto* primary_input = primary_input_it == value_map.end() ? nullptr : &primary_input_it->second;
        for (std::size_t output_index = 0u; output_index < node.outputs.size(); ++output_index) {
            const auto& output_id = node.outputs[output_index];
            if (output_id.empty()) {
                continue;
            }
            auto& value = value_map[output_id];
            if (value.shape.empty()) {
                if (output_index < inferred_output_shapes.size() && !inferred_output_shapes[output_index].empty()) {
                    value.shape = inferred_output_shapes[output_index];
                } else if (!node.shape.empty()) {
                    value.shape = node.shape;
                }
            }
            if (value.dtype.empty() && primary_input != nullptr && !primary_input->dtype.empty()) {
                value.dtype = primary_input->dtype;
            }
            if (value.bytes == 0u) {
                value.bytes = bytes_for_shape(value.shape, value.dtype);
            }
        }
        nodes.push_back(std::move(node));
    }

    bool saw_initializer = false;
    std::vector<ImportedValueSpec> values;
    values.reserve(value_map.size());
    for (auto& [id, value] : value_map) {
        if (value.id.empty()) {
            value.id = id;
        }
        if (value.bytes == 0u) {
            value.bytes = bytes_for_shape(value.shape, value.dtype);
        }
        if (value.initializer) {
            saw_initializer = true;
        }
        values.push_back(std::move(value));
    }

    manifest.workload.prefer_unified_memory = saw_initializer;
    return finalize_imported_manifest(std::move(manifest), source, values, nodes);
}

bool parse_bool_string(const std::string& input, const bool fallback = false) {
    const auto lowered = lowercase_copy(trim_copy(input));
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
        return false;
    }
    return fallback;
}

WorkloadKind parse_workload_kind_string(const std::string& input) {
    const auto lowered = lowercase_copy(input);
    if (lowered == "inference") {
        return WorkloadKind::inference;
    }
    if (lowered == "image") {
        return WorkloadKind::image;
    }
    if (lowered == "tensor") {
        return WorkloadKind::tensor;
    }
    if (lowered == "gaming") {
        return WorkloadKind::gaming;
    }
    if (lowered == "training") {
        return WorkloadKind::training;
    }
    return WorkloadKind::custom;
}

WorkloadPhase parse_workload_phase_string(const std::string& input) {
    const auto lowered = lowercase_copy(input);
    if (lowered == "prefill") {
        return WorkloadPhase::prefill;
    }
    if (lowered == "decode") {
        return WorkloadPhase::decode;
    }
    if (lowered == "cache_update" || lowered == "cache-update" || lowered == "cacheupdate") {
        return WorkloadPhase::cache_update;
    }
    if (lowered == "dequantize" || lowered == "dequant" || lowered == "quantize") {
        return WorkloadPhase::dequantize;
    }
    if (lowered == "training_step" || lowered == "training-step" || lowered == "train") {
        return WorkloadPhase::training_step;
    }
    return WorkloadPhase::unknown;
}

PartitionStrategy parse_partition_strategy_string(const std::string& input) {
    const auto lowered = lowercase_copy(input);
    if (lowered == "blind_sharded" || lowered == "blind-sharded") {
        return PartitionStrategy::blind_sharded;
    }
    if (lowered == "role_split" || lowered == "role-split") {
        return PartitionStrategy::role_split;
    }
    if (lowered == "reduce_on_gpu" || lowered == "reduce-on-gpu") {
        return PartitionStrategy::reduce_on_gpu;
    }
    if (lowered == "projection_sharded" || lowered == "projection-sharded") {
        return PartitionStrategy::projection_sharded;
    }
    if (lowered == "tpu_like" || lowered == "tpu-like") {
        return PartitionStrategy::tpu_like;
    }
    return PartitionStrategy::auto_balanced;
}

OperationClass parse_operation_class_string(const std::string& input) {
    const auto lowered = lowercase_copy(input);
    if (lowered == "reduction") {
        return OperationClass::reduction;
    }
    if (lowered == "matmul") {
        return OperationClass::matmul;
    }
    if (lowered == "convolution_2d" || lowered == "convolution-2d" || lowered == "conv2d") {
        return OperationClass::convolution_2d;
    }
    if (lowered == "resample_2d" || lowered == "resample-2d" || lowered == "resample") {
        return OperationClass::resample_2d;
    }
    return OperationClass::elementwise_map;
}

void apply_workload_fields(
    const std::unordered_map<std::string, std::string>& fields,
    WorkloadSpec& workload) {
    if (const auto it = fields.find("name"); it != fields.end()) {
        workload.name = it->second;
    }
    if (const auto it = fields.find("kind"); it != fields.end()) {
        workload.kind = parse_workload_kind_string(it->second);
    }
    if (const auto it = fields.find("dataset_tag"); it != fields.end()) {
        workload.dataset_tag = it->second;
    }
    if (const auto it = fields.find("working_set_bytes"); it != fields.end()) {
        workload.working_set_bytes = static_cast<std::uint64_t>(std::stoull(it->second));
    }
    if (const auto it = fields.find("host_exchange_bytes"); it != fields.end()) {
        workload.host_exchange_bytes = static_cast<std::uint64_t>(std::stoull(it->second));
    }
    if (const auto it = fields.find("estimated_flops"); it != fields.end()) {
        workload.estimated_flops = std::stod(it->second);
    }
    if (const auto it = fields.find("batch_size"); it != fields.end()) {
        workload.batch_size = static_cast<std::uint32_t>(std::stoul(it->second));
    }
    if (const auto it = fields.find("latency_sensitive"); it != fields.end()) {
        workload.latency_sensitive = parse_bool_string(it->second);
    }
    if (const auto it = fields.find("prefer_unified_memory"); it != fields.end()) {
        workload.prefer_unified_memory = parse_bool_string(it->second);
    }
    if (const auto it = fields.find("matrix_friendly"); it != fields.end()) {
        workload.matrix_friendly = parse_bool_string(it->second);
    }
    if (const auto it = fields.find("partition_strategy"); it != fields.end()) {
        workload.partition_strategy = parse_partition_strategy_string(it->second);
    }
    if (const auto it = fields.find("phase"); it != fields.end()) {
        workload.phase = parse_workload_phase_string(it->second);
    }
    if (const auto it = fields.find("shape_bucket"); it != fields.end()) {
        workload.shape_bucket = it->second;
    }
}

OperationClass infer_imported_operation_class(
    const std::string& op_type,
    const std::string& source_format) {
    const auto lowered = lowercase_copy(op_type);
    if (lowered.find("dequant") != std::string::npos || lowered.find("quantize") != std::string::npos ||
        lowered.find("norm") != std::string::npos || lowered.find("rope") != std::string::npos) {
        return OperationClass::elementwise_map;
    }
    if (lowered.find("matmul") != std::string::npos || lowered.find("gemm") != std::string::npos ||
        lowered.find("linear") != std::string::npos || lowered == "mm" || lowered == "mul_mat") {
        return OperationClass::matmul;
    }
    if (lowered.find("conv") != std::string::npos) {
        return OperationClass::convolution_2d;
    }
    if (lowered.find("resize") != std::string::npos || lowered.find("upsample") != std::string::npos ||
        lowered.find("interpolate") != std::string::npos || lowered.find("resample") != std::string::npos) {
        return OperationClass::resample_2d;
    }
    if (lowered.find("reduce") != std::string::npos || lowered.find("softmax") != std::string::npos ||
        lowered.find("argmax") != std::string::npos || lowered.find("argmin") != std::string::npos ||
        lowered.find("pool") != std::string::npos || lowered.find("sample") != std::string::npos) {
        return OperationClass::reduction;
    }
    if (lowered == "ggml_mul_mat" || source_format == "ggml" && lowered == "mul_mat") {
        return OperationClass::matmul;
    }
    return OperationClass::elementwise_map;
}

const ImportedValueSpec* find_imported_value(
    const std::unordered_map<std::string, ImportedValueSpec>& values,
    const std::string& id) {
    const auto it = values.find(id);
    return it == values.end() ? nullptr : &it->second;
}

std::uint64_t sum_imported_value_bytes(
    const std::vector<std::string>& ids,
    const std::unordered_map<std::string, ImportedValueSpec>& values) {
    std::uint64_t total = 0u;
    for (const auto& id : ids) {
        if (const auto* value = find_imported_value(values, id); value != nullptr) {
            total += value->bytes;
        }
    }
    return total;
}

std::vector<std::uint64_t> infer_imported_extents(
    const ImportedNodeSpec& node,
    const OperationClass op_class,
    const std::unordered_map<std::string, ImportedValueSpec>& values) {
    if (!node.extents.empty()) {
        return node.extents;
    }

    const auto* primary_input = !node.inputs.empty() ? find_imported_value(values, node.inputs.front()) : nullptr;
    const auto* secondary_input = node.inputs.size() > 1u ? find_imported_value(values, node.inputs[1u]) : nullptr;

    switch (op_class) {
    case OperationClass::matmul: {
        const auto lowered_op = lowercase_copy(node.op_type);
        const bool transpose_a = node_int_attribute_or(node, "transA", 0) != 0;
        const bool transpose_b = node_int_attribute_or(node, "transB", 0) != 0;
        const auto lhs_shape =
            primary_input != nullptr ? primary_input->shape : std::vector<std::uint64_t>{1u, 1u};
        const auto rhs_shape =
            secondary_input != nullptr ? secondary_input->shape : std::vector<std::uint64_t>{1u, 1u};
        const auto rows = std::max<std::uint64_t>(1u, leading_shape_extent(!node.shape.empty() ? node.shape
                                                                                             : lhs_shape));
        auto columns = std::max<std::uint64_t>(1u, trailing_shape_extent(!node.shape.empty() ? node.shape : rhs_shape));
        if (lowered_op == "gemm" && transpose_b && rhs_shape.size() >= 2u) {
            columns = std::max<std::uint64_t>(1u, leading_shape_extent(rhs_shape));
        }
        std::uint64_t depth = 0u;
        if (lowered_op == "gemm" && rhs_shape.size() >= 2u) {
            depth = transpose_b ? trailing_shape_extent(rhs_shape) : rhs_shape[rhs_shape.size() - 2u];
        }
        if (depth == 0u && lowered_op == "gemm" && secondary_input != nullptr && secondary_input->bytes > 0u && columns > 0u) {
            const auto element_bytes = bytes_per_element_for_dtype(secondary_input->dtype);
            const auto total_elements = secondary_input->bytes / element_bytes;
            depth = std::max<std::uint64_t>(1u, total_elements / columns);
        }
        if (!lhs_shape.empty()) {
            const auto lhs_depth =
                transpose_a && lhs_shape.size() >= 2u ? leading_shape_extent(lhs_shape) : trailing_shape_extent(lhs_shape);
            if (depth == 0u || lhs_shape.size() <= 2u) {
                depth = lhs_depth;
            }
        }
        if (depth == 0u && rhs_shape.size() >= 2u) {
            depth = transpose_b ? trailing_shape_extent(rhs_shape) : rhs_shape[rhs_shape.size() - 2u];
        }
        depth = std::max<std::uint64_t>(1u, depth == 0u ? columns : depth);
        return {rows, columns, depth};
    }
    case OperationClass::convolution_2d: {
        const auto input_shape = primary_input != nullptr ? primary_input->shape : node.shape;
        auto [height, width] = spatial_extents(input_shape);
        if ((height == 0u || width == 0u) && !node.shape.empty()) {
            const auto kernel = node_u64_list_attribute(node, "kernel_shape");
            const auto strides = node_u64_list_attribute(node, "strides");
            const auto pads = node_u64_list_attribute(node, "pads");
            const auto dilations = node_u64_list_attribute(node, "dilations");
            const auto [out_h, out_w] = spatial_extents(node.shape);
            const auto kernel_h = kernel.size() >= 1u ? std::max<std::uint64_t>(1u, kernel[0u]) : 3u;
            const auto kernel_w = kernel.size() >= 2u ? std::max<std::uint64_t>(1u, kernel[1u]) : kernel_h;
            const auto stride_h = strides.size() >= 1u ? std::max<std::uint64_t>(1u, strides[0u]) : 1u;
            const auto stride_w = strides.size() >= 2u ? std::max<std::uint64_t>(1u, strides[1u]) : stride_h;
            const auto dilation_h = dilations.size() >= 1u ? std::max<std::uint64_t>(1u, dilations[0u]) : 1u;
            const auto dilation_w = dilations.size() >= 2u ? std::max<std::uint64_t>(1u, dilations[1u]) : dilation_h;
            const auto pad_top = pads.size() >= 1u ? pads[0u] : 0u;
            const auto pad_left = pads.size() >= 2u ? pads[1u] : 0u;
            const auto pad_bottom = pads.size() >= 3u ? pads[2u] : pad_top;
            const auto pad_right = pads.size() >= 4u ? pads[3u] : pad_left;
            height = std::max<std::uint64_t>(
                1u,
                ((out_h == 0u ? 1u : out_h) - 1u) * stride_h + ((kernel_h - 1u) * dilation_h + 1u) - pad_top - pad_bottom);
            width = std::max<std::uint64_t>(
                1u,
                ((out_w == 0u ? 1u : out_w) - 1u) * stride_w + ((kernel_w - 1u) * dilation_w + 1u) - pad_left - pad_right);
        }
        return {std::max<std::uint64_t>(3u, height), std::max<std::uint64_t>(3u, width)};
    }
    case OperationClass::resample_2d: {
        const auto [src_h, src_w] = spatial_extents(primary_input != nullptr ? primary_input->shape : node.shape);
        auto [dst_h, dst_w] = spatial_extents(node.shape);
        const auto sizes = node_u64_list_attribute(node, "sizes");
        if (sizes.size() >= 2u) {
            dst_h = std::max<std::uint64_t>(1u, sizes[sizes.size() - 2u]);
            dst_w = std::max<std::uint64_t>(1u, sizes[sizes.size() - 1u]);
        }
        const auto scales = node_float_list_attribute(node, "scales");
        if ((dst_h == 0u || dst_w == 0u) && scales.size() >= 2u) {
            const auto scale_h = std::max(1.0f, scales[scales.size() - 2u]);
            const auto scale_w = std::max(1.0f, scales[scales.size() - 1u]);
            dst_h = std::max<std::uint64_t>(1u, static_cast<std::uint64_t>(std::llround(static_cast<double>(src_h) * scale_h)));
            dst_w = std::max<std::uint64_t>(1u, static_cast<std::uint64_t>(std::llround(static_cast<double>(src_w) * scale_w)));
        }
        return {
            std::max<std::uint64_t>(1u, src_h),
            std::max<std::uint64_t>(1u, src_w),
            std::max<std::uint64_t>(1u, dst_h == 0u ? src_h : dst_h),
            std::max<std::uint64_t>(1u, dst_w == 0u ? src_w : dst_w)};
    }
    case OperationClass::reduction: {
        const auto elements = std::max<std::uint64_t>(
            1u,
            primary_input != nullptr ? shape_elements(primary_input->shape) : shape_elements(node.shape));
        return {elements};
    }
    case OperationClass::elementwise_map:
    default: {
        const auto elements = std::max<std::uint64_t>(
            1u,
            !node.shape.empty() ? shape_elements(node.shape)
                                : (primary_input != nullptr ? shape_elements(primary_input->shape) : 1u));
        return {elements};
    }
    }
}

double infer_operation_flops(
    const OperationClass op_class,
    const std::vector<std::uint64_t>& extents) {
    switch (op_class) {
    case OperationClass::reduction:
        return extents.empty() ? 0.0 : static_cast<double>(extents.front());
    case OperationClass::matmul:
        if (extents.size() < 3u) {
            return 0.0;
        }
        return 2.0 * static_cast<double>(extents[0u]) * static_cast<double>(extents[1u]) * static_cast<double>(extents[2u]);
    case OperationClass::convolution_2d:
        if (extents.size() < 2u || extents[0u] < 3u || extents[1u] < 3u) {
            return 0.0;
        }
        return 18.0 * static_cast<double>(extents[0u] - 2u) * static_cast<double>(extents[1u] - 2u);
    case OperationClass::resample_2d:
        if (extents.size() < 4u) {
            return 0.0;
        }
        return 8.0 * static_cast<double>(extents[2u]) * static_cast<double>(extents[3u]);
    case OperationClass::elementwise_map:
    default:
        return extents.empty() ? 0.0 : 3.0 * static_cast<double>(extents.front());
    }
}

WorkloadManifest finalize_imported_manifest(
    WorkloadManifest manifest,
    const ImportedSourceSpec& source,
    const std::vector<ImportedValueSpec>& values,
    const std::vector<ImportedNodeSpec>& nodes) {
    manifest.source_format = source.format.empty() ? "imported" : source.format;
    manifest.source_entry = source.entry;
    manifest.imported = true;

    std::unordered_map<std::string, ImportedValueSpec> value_lookup;
    value_lookup.reserve(values.size());
    for (auto value : values) {
        if (value.bytes == 0u) {
            value.bytes = bytes_for_shape(value.shape, value.dtype);
        }
        value_lookup.emplace(value.id, std::move(value));
    }

    for (const auto& [id, value] : value_lookup) {
        manifest.graph.tensors.push_back(WorkloadTensor{
            value.id,
            value.alias_group,
            value.producer_operation,
            value.consumer_operations,
            value.bytes,
            value.persistent || value.initializer,
            value.temporary,
            value.host_visible});
    }

    for (const auto& node : nodes) {
        OperationSpec operation;
        operation.name = node.name;
        operation.op_class = infer_imported_operation_class(node.op_type, manifest.source_format);
        operation.extents = infer_imported_extents(node, operation.op_class, value_lookup);
        operation.input_tensor_ids = node.inputs;
        operation.output_tensor_ids = node.outputs;
        operation.temporary_tensor_ids = node.temporaries;
        operation.dependency_operation_names = node.dependencies;
        operation.input_bytes = node.input_bytes == 0u ? sum_imported_value_bytes(node.inputs, value_lookup) : node.input_bytes;
        operation.output_bytes =
            node.output_bytes == 0u ? sum_imported_value_bytes(node.outputs, value_lookup) : node.output_bytes;
        operation.temporary_bytes =
            node.temporary_bytes == 0u ? sum_imported_value_bytes(node.temporaries, value_lookup) : node.temporary_bytes;
        operation.estimated_flops =
            node.estimated_flops == 0.0 ? infer_operation_flops(operation.op_class, operation.extents) : node.estimated_flops;
        operation.max_relative_error = node.max_relative_error == 0.0 ? 1.5e-3 : node.max_relative_error;
        operation.parallelizable = node.parallelizable;
        operation.reduction_like = node.reduction_like || operation.op_class == OperationClass::reduction;
        operation.streaming_friendly =
            node.streaming_friendly || operation.op_class == OperationClass::convolution_2d ||
            operation.op_class == OperationClass::resample_2d;
        operation.matrix_friendly = node.matrix_friendly || operation.op_class == OperationClass::matmul;
        manifest.graph.operations.push_back(std::move(operation));
    }

    manifest.has_graph = !manifest.graph.operations.empty() || !manifest.graph.tensors.empty();
    if (manifest.workload.name.empty()) {
        manifest.workload.name = manifest.source_path.stem().string();
    }
    if (manifest.workload.dataset_tag.empty()) {
        manifest.workload.dataset_tag = manifest.source_format + "-" + manifest.workload.name;
    }
    if (manifest.workload.shape_bucket.empty() && !manifest.source_entry.empty()) {
        manifest.workload.shape_bucket = manifest.source_entry;
    }
    if (manifest.workload.estimated_flops == 0.0) {
        manifest.workload.estimated_flops = std::accumulate(
            manifest.graph.operations.begin(),
            manifest.graph.operations.end(),
            0.0,
            [](const double total, const OperationSpec& operation) {
                return total + operation.estimated_flops;
            });
    }
    if (manifest.workload.working_set_bytes == 0u) {
        manifest.workload.working_set_bytes = std::accumulate(
            manifest.graph.tensors.begin(),
            manifest.graph.tensors.end(),
            std::uint64_t{0},
            [](const std::uint64_t total, const WorkloadTensor& tensor) {
                return total + tensor.bytes;
            });
    }
    if (manifest.workload.host_exchange_bytes == 0u) {
        manifest.workload.host_exchange_bytes = std::accumulate(
            manifest.graph.tensors.begin(),
            manifest.graph.tensors.end(),
            std::uint64_t{0},
            [](const std::uint64_t total, const WorkloadTensor& tensor) {
                return total + (tensor.host_visible ? tensor.bytes : 0u);
            });
    }
    if (manifest.has_graph) {
        manifest.graph.signature = manifest.workload.name + "|" + to_string(manifest.workload.kind) + "|" +
                                   manifest.workload.dataset_tag + "|" + manifest.source_format;
        normalize_workload_graph(manifest.graph);
    }
    return manifest;
}

GgufFileContents parse_gguf_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("unable to open GGUF file: " + path.string());
    }

    std::array<char, 4> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || std::string_view(magic.data(), magic.size()) != "GGUF") {
        throw std::runtime_error("not a GGUF file: " + path.string());
    }

    const auto version = read_binary_value<std::uint32_t>(input);
    if (version < 2u || version > 3u) {
        throw std::runtime_error("unsupported GGUF version: " + std::to_string(version));
    }

    const auto tensor_count = read_binary_value<std::uint64_t>(input);
    const auto metadata_count = read_binary_value<std::uint64_t>(input);

    GgufFileContents contents;
    contents.metadata.reserve(static_cast<std::size_t>(metadata_count));
    contents.tensors.reserve(static_cast<std::size_t>(tensor_count));

    for (std::uint64_t index = 0u; index < metadata_count; ++index) {
        const auto key = read_gguf_string(input);
        const auto value_type = read_binary_value<std::uint32_t>(input);
        contents.metadata.emplace(key, gguf_scalar_value_to_string(input, value_type));
    }

    contents.alignment = metadata_u64_by_suffix(contents.metadata, {"general.alignment", ".alignment"}, 32u);
    contents.alignment = std::max<std::uint64_t>(1u, contents.alignment);

    for (std::uint64_t index = 0u; index < tensor_count; ++index) {
        GgufTensorInfo tensor;
        tensor.name = read_gguf_string(input);
        const auto dimension_count = read_binary_value<std::uint32_t>(input);
        tensor.dimensions.reserve(dimension_count);
        for (std::uint32_t dim = 0u; dim < dimension_count; ++dim) {
            tensor.dimensions.push_back(read_binary_value<std::uint64_t>(input));
        }
        tensor.type = read_binary_value<std::uint32_t>(input);
        tensor.offset = read_binary_value<std::uint64_t>(input);
        contents.tensors.push_back(std::move(tensor));
    }

    const auto tensor_info_end = static_cast<std::uint64_t>(input.tellg());
    contents.data_offset = align_up_u64(tensor_info_end, contents.alignment);

    input.seekg(0, std::ios::end);
    const auto file_size = static_cast<std::uint64_t>(input.tellg());
    for (std::size_t index = 0u; index < contents.tensors.size(); ++index) {
        const auto next_offset =
            index + 1u < contents.tensors.size()
                ? contents.tensors[index + 1u].offset
                : (file_size > contents.data_offset ? file_size - contents.data_offset : 0u);
        if (next_offset < contents.tensors[index].offset) {
            throw std::runtime_error("GGUF tensor offsets are not monotonic");
        }
        contents.tensors[index].bytes = next_offset - contents.tensors[index].offset;
        contents.total_tensor_bytes += contents.tensors[index].bytes;
    }

    return contents;
}

std::string choose_weight_tensor_id(
    std::size_t preferred_index,
    const std::vector<std::string>& ranked_ids,
    WorkloadGraph& graph,
    const std::string& fallback_id,
    const std::uint64_t fallback_bytes) {
    if (preferred_index < ranked_ids.size()) {
        return ranked_ids[preferred_index];
    }
    const auto existing = std::find_if(
        graph.tensors.begin(),
        graph.tensors.end(),
        [&](const WorkloadTensor& tensor) {
            return tensor.id == fallback_id;
        });
    if (existing == graph.tensors.end()) {
        add_tensor(graph, fallback_id, fallback_bytes, "", {}, true, false, false, "imported-fallback");
    }
    return fallback_id;
}

WorkloadManifest load_gguf_workload_source(const std::filesystem::path& path) {
    const auto gguf = parse_gguf_file(path);

    WorkloadManifest manifest;
    manifest.source_path = path;
    manifest.source_format = "gguf";
    manifest.source_entry = "decode_step";
    manifest.imported = true;
    manifest.has_graph = true;

    const auto architecture = metadata_value_or(gguf.metadata, "general.architecture", "llama");
    const auto model_name = metadata_value_or(gguf.metadata, "general.name", path.stem().string());
    const auto hidden = std::max<std::uint64_t>(
        128u,
        metadata_u64_by_suffix(gguf.metadata, {".embedding_length", "embedding_length"}, 320u));
    const auto feed_forward = std::max<std::uint64_t>(
        hidden * 2u,
        metadata_u64_by_suffix(gguf.metadata, {".feed_forward_length", "feed_forward_length"}, hidden * 3u));
    const auto context_length = std::max<std::uint64_t>(
        256u,
        metadata_u64_by_suffix(gguf.metadata, {".context_length", "context_length"}, 2048u));
    const auto block_count = std::max<std::uint64_t>(
        1u,
        metadata_u64_by_suffix(gguf.metadata, {".block_count", "block_count"}, 1u));

    manifest.workload.name = model_name;
    manifest.workload.kind = WorkloadKind::inference;
    manifest.workload.dataset_tag = "gguf-" + architecture;
    manifest.workload.working_set_bytes = std::max<std::uint64_t>(gguf.total_tensor_bytes, 32u * kMiB);
    manifest.workload.host_exchange_bytes = std::max<std::uint64_t>(hidden * sizeof(float) * 2u, 4096u);
    manifest.workload.batch_size = 1u;
    manifest.workload.latency_sensitive = true;
    manifest.workload.prefer_unified_memory = true;
    manifest.workload.matrix_friendly = true;
    manifest.workload.phase = WorkloadPhase::decode;
    manifest.workload.shape_bucket = "ctx-" + std::to_string(context_length);

    auto& graph = manifest.graph;
    graph.signature = model_name + "|inference|" + manifest.workload.dataset_tag + "|gguf";

    std::vector<std::pair<std::uint64_t, std::string>> ranked_weights;
    ranked_weights.reserve(gguf.tensors.size());
    for (const auto& tensor : gguf.tensors) {
        const auto persistent = true;
        const auto host_visible = tensor.offset == 0u;
        add_tensor(
            graph,
            tensor.name,
            std::max<std::uint64_t>(tensor.bytes, 1u),
            "",
            {},
            persistent,
            false,
            host_visible,
            gguf_tensor_type_name(tensor.type));
        ranked_weights.emplace_back(tensor.bytes, tensor.name);
    }
    std::sort(
        ranked_weights.begin(),
        ranked_weights.end(),
        [](const auto& left, const auto& right) {
            if (left.first != right.first) {
                return left.first > right.first;
            }
            return left.second < right.second;
        });
    std::vector<std::string> ranked_ids;
    ranked_ids.reserve(ranked_weights.size());
    for (const auto& [bytes, id] : ranked_weights) {
        if (bytes > 0u) {
            ranked_ids.push_back(id);
        }
    }

    add_tensor(graph, "decode-token", hidden * sizeof(float), "", {"decode-rmsnorm"}, false, false, true);
    add_tensor(graph, "decode-rms-weight", hidden * sizeof(float), "", {"decode-rmsnorm"}, true, false, true, "norm");
    add_tensor(graph, "decode-normed", hidden * sizeof(float), "decode-rmsnorm", {"decode-qkv"});
    add_tensor(graph, "decode-qkv-out", hidden * 3u * sizeof(float), "decode-qkv", {"decode-score-reduce", "decode-context"});
    add_tensor(graph, "kv-cache", context_length * hidden * 2u * sizeof(float), "", {"decode-score-reduce", "decode-context"}, true, false, true, "cache");
    add_tensor(graph, "decode-score", sizeof(float), "decode-score-reduce", {"decode-mlp-gate", "decode-sample"}, true);
    add_tensor(graph, "decode-context-out", hidden * sizeof(float), "decode-context", {"decode-mlp-up"});
    add_tensor(graph, "decode-mlp-up-out", feed_forward * sizeof(float), "decode-mlp-up", {"decode-mlp-gate"});
    add_tensor(graph, "decode-mlp-gated", feed_forward * sizeof(float), "decode-mlp-gate", {"decode-mlp-down"});
    add_tensor(graph, "decode-state", hidden * sizeof(float), "decode-mlp-down", {"decode-sample"});
    add_tensor(graph, "sampled-logit", sizeof(float), "decode-sample", {});

    const auto qkv_weight_id = choose_weight_tensor_id(0u, ranked_ids, graph, "gguf-qkv-weight", hidden * hidden * 3u);
    const auto context_weight_id = choose_weight_tensor_id(1u, ranked_ids, graph, "gguf-context-weight", hidden * hidden);
    const auto mlp_up_weight_id = choose_weight_tensor_id(2u, ranked_ids, graph, "gguf-mlp-up-weight", hidden * feed_forward);
    const auto mlp_down_weight_id = choose_weight_tensor_id(3u, ranked_ids, graph, "gguf-mlp-down-weight", hidden * feed_forward);

    graph.operations = {
        make_elementwise("decode-rmsnorm", hidden, {"decode-token", "decode-rms-weight"}, {"decode-normed"}),
        make_matmul("decode-qkv", 1u, static_cast<std::uint32_t>(hidden * 3u), static_cast<std::uint32_t>(hidden), {"decode-normed", qkv_weight_id}, {"decode-qkv-out"}),
        make_reduction("decode-score-reduce", context_length * hidden, {"decode-qkv-out", "kv-cache"}, {"decode-score"}, {}, 1.5e-3),
        make_matmul("decode-context", 1u, static_cast<std::uint32_t>(hidden), static_cast<std::uint32_t>(hidden), {"decode-qkv-out", context_weight_id, "kv-cache"}, {"decode-context-out"}),
        make_matmul("decode-mlp-up", 1u, static_cast<std::uint32_t>(feed_forward), static_cast<std::uint32_t>(hidden), {"decode-context-out", mlp_up_weight_id}, {"decode-mlp-up-out"}),
        make_elementwise("decode-mlp-gate", feed_forward, {"decode-mlp-up-out", "decode-score"}, {"decode-mlp-gated"}, {}, 7.5e-4),
        make_matmul("decode-mlp-down", 1u, static_cast<std::uint32_t>(hidden), static_cast<std::uint32_t>(feed_forward), {"decode-mlp-gated", mlp_down_weight_id}, {"decode-state"}),
        make_reduction("decode-sample", hidden, {"decode-state", "decode-score"}, {"sampled-logit"}, {}, 1.5e-3)};

    normalize_workload_graph(graph);
    manifest.workload.estimated_flops = std::accumulate(
        graph.operations.begin(),
        graph.operations.end(),
        static_cast<double>(block_count),
        [](const double total, const OperationSpec& operation) {
            return total + operation.estimated_flops;
        }) * static_cast<double>(block_count);
    return manifest;
}

void finalize_workload_graph(WorkloadGraph& graph) {
    const auto indices = operation_indices(graph);
    const auto explicit_dependencies = graph.dependencies;
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

    for (const auto& dependency : explicit_dependencies) {
        const auto duplicate = std::find_if(
            graph.dependencies.begin(),
            graph.dependencies.end(),
            [&](const WorkloadDependency& existing) {
                return existing.source_operation_name == dependency.source_operation_name &&
                       existing.target_operation_name == dependency.target_operation_name &&
                       existing.tensor_id == dependency.tensor_id;
            });
        if (duplicate == graph.dependencies.end()) {
            graph.dependencies.push_back(dependency);
        }
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

void normalize_workload_graph(WorkloadGraph& graph) {
    finalize_workload_graph(graph);
}

WorkloadManifest load_workload_source(const std::filesystem::path& path) {
    const auto extension = lowercase_copy(path.extension().string());
    if (extension == ".gguf") {
        return load_gguf_workload_source(path);
    }
    if (extension == ".onnx") {
        return load_onnx_workload_source(path);
    }

    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("unable to open workload manifest: " + path.string());
    }

    WorkloadManifest manifest;
    manifest.source_path = path;
    ImportedSourceSpec imported_source;
    std::vector<ImportedValueSpec> imported_values;
    std::vector<ImportedNodeSpec> imported_nodes;
    bool saw_import_sections = false;

    enum class Section {
        none,
        source,
        workload,
        tensor,
        operation,
        dependency,
        value,
        node
    };

    Section current_section = Section::none;
    std::unordered_map<std::string, std::string> fields;

    const auto flush_section = [&]() {
        if (fields.empty()) {
            return;
        }

        switch (current_section) {
        case Section::source: {
            apply_workload_fields(fields, manifest.workload);
            if (const auto it = fields.find("format"); it != fields.end()) {
                imported_source.format = lowercase_copy(it->second);
            }
            if (const auto it = fields.find("entry"); it != fields.end()) {
                imported_source.entry = it->second;
            }
            if (const auto it = fields.find("entry_graph"); it != fields.end()) {
                imported_source.entry = it->second;
            }
            manifest.source_format = imported_source.format;
            manifest.source_entry = imported_source.entry;
            manifest.imported = true;
            saw_import_sections = true;
            break;
        }
        case Section::workload: {
            apply_workload_fields(fields, manifest.workload);
            break;
        }
        case Section::tensor: {
            WorkloadTensor tensor;
            tensor.id = fields.at("id");
            if (const auto it = fields.find("alias_group"); it != fields.end()) {
                tensor.alias_group = it->second;
            }
            if (const auto it = fields.find("producer"); it != fields.end()) {
                tensor.producer_operation = it->second;
            }
            if (const auto it = fields.find("consumers"); it != fields.end()) {
                tensor.consumer_operations = split_csv_strings(it->second);
            }
            if (const auto it = fields.find("bytes"); it != fields.end()) {
                tensor.bytes = static_cast<std::uint64_t>(std::stoull(it->second));
            }
            tensor.persistent = parse_bool_string(fields["persistent"], false);
            tensor.temporary = parse_bool_string(fields["temporary"], false);
            tensor.host_visible = parse_bool_string(fields["host_visible"], false);
            manifest.graph.tensors.push_back(std::move(tensor));
            manifest.has_graph = true;
            break;
        }
        case Section::operation: {
            OperationSpec operation;
            operation.name = fields.at("name");
            if (const auto it = fields.find("class"); it != fields.end()) {
                operation.op_class = parse_operation_class_string(it->second);
            }
            if (const auto it = fields.find("extents"); it != fields.end()) {
                operation.extents = split_csv_u64(it->second);
            }
            if (const auto it = fields.find("input_bytes"); it != fields.end()) {
                operation.input_bytes = static_cast<std::uint64_t>(std::stoull(it->second));
            }
            if (const auto it = fields.find("output_bytes"); it != fields.end()) {
                operation.output_bytes = static_cast<std::uint64_t>(std::stoull(it->second));
            }
            if (const auto it = fields.find("temporary_bytes"); it != fields.end()) {
                operation.temporary_bytes = static_cast<std::uint64_t>(std::stoull(it->second));
            }
            if (const auto it = fields.find("estimated_flops"); it != fields.end()) {
                operation.estimated_flops = std::stod(it->second);
            }
            if (const auto it = fields.find("max_relative_error"); it != fields.end()) {
                operation.max_relative_error = std::stod(it->second);
            }
            operation.parallelizable = parse_bool_string(fields["parallelizable"], true);
            operation.reduction_like = parse_bool_string(fields["reduction_like"], false);
            operation.streaming_friendly = parse_bool_string(fields["streaming_friendly"], false);
            operation.matrix_friendly = parse_bool_string(fields["matrix_friendly"], false);
            if (const auto it = fields.find("inputs"); it != fields.end()) {
                operation.input_tensor_ids = split_csv_strings(it->second);
            }
            if (const auto it = fields.find("outputs"); it != fields.end()) {
                operation.output_tensor_ids = split_csv_strings(it->second);
            }
            if (const auto it = fields.find("temporaries"); it != fields.end()) {
                operation.temporary_tensor_ids = split_csv_strings(it->second);
            }
            if (const auto it = fields.find("dependencies"); it != fields.end()) {
                operation.dependency_operation_names = split_csv_strings(it->second);
            }
            manifest.graph.operations.push_back(std::move(operation));
            manifest.has_graph = true;
            break;
        }
        case Section::dependency: {
            WorkloadDependency dependency;
            dependency.source_operation_name = fields.at("source");
            dependency.target_operation_name = fields.at("target");
            if (const auto it = fields.find("tensor_id"); it != fields.end()) {
                dependency.tensor_id = it->second;
            }
            dependency.requires_residency = parse_bool_string(fields["requires_residency"], true);
            manifest.graph.dependencies.push_back(std::move(dependency));
            manifest.has_graph = true;
            break;
        }
        case Section::value: {
            ImportedValueSpec value;
            value.id = fields.at("id");
            if (const auto it = fields.find("shape"); it != fields.end()) {
                value.shape = split_csv_u64(it->second);
            }
            if (const auto it = fields.find("dtype"); it != fields.end()) {
                value.dtype = it->second;
            }
            if (const auto it = fields.find("alias_group"); it != fields.end()) {
                value.alias_group = it->second;
            }
            if (const auto it = fields.find("producer"); it != fields.end()) {
                value.producer_operation = it->second;
            }
            if (const auto it = fields.find("consumers"); it != fields.end()) {
                value.consumer_operations = split_csv_strings(it->second);
            }
            if (const auto it = fields.find("bytes"); it != fields.end()) {
                value.bytes = static_cast<std::uint64_t>(std::stoull(it->second));
            }
            value.initializer = parse_bool_string(fields["initializer"], false);
            value.persistent = parse_bool_string(fields["persistent"], value.initializer);
            value.temporary = parse_bool_string(fields["temporary"], false);
            value.host_visible = parse_bool_string(fields["host_visible"], false);
            imported_values.push_back(std::move(value));
            saw_import_sections = true;
            break;
        }
        case Section::node: {
            ImportedNodeSpec node;
            node.name = fields.at("name");
            if (const auto it = fields.find("op_type"); it != fields.end()) {
                node.op_type = it->second;
            }
            if (const auto it = fields.find("shape"); it != fields.end()) {
                node.shape = split_csv_u64(it->second);
            }
            if (const auto it = fields.find("extents"); it != fields.end()) {
                node.extents = split_csv_u64(it->second);
            }
            if (const auto it = fields.find("inputs"); it != fields.end()) {
                node.inputs = split_csv_strings(it->second);
            }
            if (const auto it = fields.find("outputs"); it != fields.end()) {
                node.outputs = split_csv_strings(it->second);
            }
            if (const auto it = fields.find("temporaries"); it != fields.end()) {
                node.temporaries = split_csv_strings(it->second);
            }
            if (const auto it = fields.find("dependencies"); it != fields.end()) {
                node.dependencies = split_csv_strings(it->second);
            }
            if (const auto it = fields.find("input_bytes"); it != fields.end()) {
                node.input_bytes = static_cast<std::uint64_t>(std::stoull(it->second));
            }
            if (const auto it = fields.find("output_bytes"); it != fields.end()) {
                node.output_bytes = static_cast<std::uint64_t>(std::stoull(it->second));
            }
            if (const auto it = fields.find("temporary_bytes"); it != fields.end()) {
                node.temporary_bytes = static_cast<std::uint64_t>(std::stoull(it->second));
            }
            if (const auto it = fields.find("estimated_flops"); it != fields.end()) {
                node.estimated_flops = std::stod(it->second);
            }
            if (const auto it = fields.find("max_relative_error"); it != fields.end()) {
                node.max_relative_error = std::stod(it->second);
            }
            node.parallelizable = parse_bool_string(fields["parallelizable"], true);
            node.reduction_like = parse_bool_string(fields["reduction_like"], false);
            node.streaming_friendly = parse_bool_string(fields["streaming_friendly"], false);
            node.matrix_friendly = parse_bool_string(fields["matrix_friendly"], false);
            imported_nodes.push_back(std::move(node));
            saw_import_sections = true;
            break;
        }
        case Section::none:
        default:
            break;
        }

        fields.clear();
    };

    std::string line;
    while (std::getline(input, line)) {
        const auto trimmed = trim_copy(line);
        if (trimmed.empty() || trimmed[0] == '#' || trimmed[0] == ';') {
            continue;
        }
        if (trimmed.front() == '[' && trimmed.back() == ']') {
            flush_section();
            const auto section_name = lowercase_copy(trimmed.substr(1u, trimmed.size() - 2u));
            if (section_name == "source" || section_name == "model" || section_name == "import") {
                current_section = Section::source;
            } else if (section_name == "workload") {
                current_section = Section::workload;
            } else if (section_name == "tensor") {
                current_section = Section::tensor;
            } else if (section_name == "operation") {
                current_section = Section::operation;
            } else if (section_name == "dependency") {
                current_section = Section::dependency;
            } else if (section_name == "value" || section_name == "initializer" || section_name == "value_info") {
                current_section = Section::value;
            } else if (section_name == "node" || section_name == "op" || section_name == "operator") {
                current_section = Section::node;
            } else {
                current_section = Section::none;
            }
            continue;
        }

        const auto delimiter = trimmed.find('=');
        if (delimiter == std::string::npos) {
            continue;
        }
        const auto key = lowercase_copy(trim_copy(trimmed.substr(0u, delimiter)));
        const auto value = trim_copy(trimmed.substr(delimiter + 1u));
        fields[key] = value;
    }
    flush_section();

    if (manifest.workload.name.empty()) {
        manifest.workload.name = path.stem().string();
    }
    if (saw_import_sections) {
        return finalize_imported_manifest(
            std::move(manifest),
            imported_source,
            imported_values,
            imported_nodes);
    }
    if (manifest.has_graph) {
        manifest.graph.signature = manifest.workload.name + "|" + to_string(manifest.workload.kind) + "|" +
                                   manifest.workload.dataset_tag + "|manifest";
        normalize_workload_graph(manifest.graph);
    }
    return manifest;
}

WorkloadManifest load_workload_manifest(const std::filesystem::path& path) {
    return load_workload_source(path);
}

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
