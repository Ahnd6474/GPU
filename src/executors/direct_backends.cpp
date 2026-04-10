#include "jakal/executors/direct_backends.hpp"
#include "jakal/executors/host_native_kernels.hpp"
#include "jakal/executors/host_thread_pool.hpp"
#include "jakal/executors/native_gpu_backend.hpp"

#include "jakal/jakal_l0.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace jakal::executors {
std::unique_ptr<IKernelBackend> make_vulkan_direct_kernel_backend_internal();
bool vulkan_direct_backend_available_internal();

namespace {

#if defined(_WIN32)
using LibraryHandle = HMODULE;

LibraryHandle load_library(const char* name) {
    return LoadLibraryA(name);
}

void* load_symbol(LibraryHandle library, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(library, name));
}

void close_library(LibraryHandle library) {
    if (library != nullptr) {
        FreeLibrary(library);
    }
}
#else
using LibraryHandle = void*;

LibraryHandle load_library(const char* name) {
    return dlopen(name, RTLD_LAZY);
}

void* load_symbol(LibraryHandle library, const char* name) {
    return dlsym(library, name);
}

void close_library(LibraryHandle library) {
    if (library != nullptr) {
        dlclose(library);
    }
}
#endif

template <typename Func>
double measure_us(Func&& func) {
    const auto start = std::chrono::steady_clock::now();
    func();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

float quantize_value(const float value, const bool low_precision) {
    if (!low_precision) {
        return value;
    }
    return std::round(value * 1024.0f) / 1024.0f;
}

enum class HostPrecisionMode {
    fp32,
    fp16,
    bf16,
    emulated_lowp
};

bool host_output_is_usable(const std::span<const float> output) {
    constexpr double kReasonableMagnitude = 1.0e8;
    return std::all_of(output.begin(), output.end(), [](const float value) {
        return std::isfinite(value) &&
               std::abs(static_cast<double>(value)) <= kReasonableMagnitude;
    });
}

std::uint32_t float_to_bits(const float value) {
    return std::bit_cast<std::uint32_t>(value);
}

float bits_to_float(const std::uint32_t bits) {
    return std::bit_cast<float>(bits);
}

float quantize_fp16(const float value) {
    if (!std::isfinite(value)) {
        return value;
    }

    const std::uint32_t bits = float_to_bits(value);
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23u) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7fffffu;
    std::uint16_t half = 0;

    if (exponent <= 0) {
        if (exponent < -10) {
            half = static_cast<std::uint16_t>(sign);
        } else {
            mantissa |= 0x800000u;
            const auto shift = static_cast<std::uint32_t>(14 - exponent);
            std::uint32_t rounded = mantissa >> shift;
            if (((mantissa >> (shift - 1u)) & 1u) != 0u) {
                rounded += 1u;
            }
            half = static_cast<std::uint16_t>(sign | (rounded & 0x03ffu));
        }
    } else if (exponent >= 31) {
        half = static_cast<std::uint16_t>(sign | 0x7c00u);
    } else {
        mantissa += 0x1000u;
        if ((mantissa & 0x00800000u) != 0u) {
            mantissa = 0;
            ++exponent;
        }
        if (exponent >= 31) {
            half = static_cast<std::uint16_t>(sign | 0x7c00u);
        } else {
            half = static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10u) |
                                              ((mantissa >> 13u) & 0x03ffu));
        }
    }

    const std::uint32_t half_sign = (static_cast<std::uint32_t>(half & 0x8000u)) << 16u;
    const std::uint32_t half_exponent = (half >> 10u) & 0x1fu;
    const std::uint32_t half_mantissa = half & 0x03ffu;
    std::uint32_t restored = 0;
    if (half_exponent == 0u) {
        if (half_mantissa == 0u) {
            restored = half_sign;
        } else {
            std::uint32_t mantissa_norm = half_mantissa;
            int exp = -14;
            while ((mantissa_norm & 0x0400u) == 0u) {
                mantissa_norm <<= 1u;
                --exp;
            }
            mantissa_norm &= 0x03ffu;
            restored = half_sign | (static_cast<std::uint32_t>(exp + 127) << 23u) | (mantissa_norm << 13u);
        }
    } else if (half_exponent == 0x1fu) {
        restored = half_sign | 0x7f800000u | (half_mantissa << 13u);
    } else {
        restored = half_sign | ((half_exponent + 112u) << 23u) | (half_mantissa << 13u);
    }
    return bits_to_float(restored);
}

float quantize_bf16(const float value) {
    if (!std::isfinite(value)) {
        return value;
    }

    const std::uint32_t bits = float_to_bits(value);
    const std::uint32_t lsb = (bits >> 16u) & 1u;
    const std::uint32_t rounded = bits + 0x7fffu + lsb;
    return bits_to_float(rounded & 0xffff0000u);
}

HostPrecisionMode select_host_precision(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    const bool low_precision) {
    if (!low_precision) {
        return HostPrecisionMode::fp32;
    }
    if (graph.probe != "host") {
        return HostPrecisionMode::emulated_lowp;
    }
    const auto summary = summarize_graph(graph);
    if ((operation.cpu_low_precision_kernel_family == "bf16-blocked" ||
         operation.cpu_low_precision_kernel_family == "auto") &&
        summary.supports_bf16 &&
        summary.native_vector_bits >= 512u) {
        return HostPrecisionMode::bf16;
    }
    if ((operation.cpu_low_precision_kernel_family == "fp16-blocked" ||
         operation.cpu_low_precision_kernel_family == "auto") &&
        summary.supports_fp16 &&
        summary.native_vector_bits >= 256u) {
        return HostPrecisionMode::fp16;
    }
    return HostPrecisionMode::emulated_lowp;
}

bool should_use_host_parallelism(
    const OperationSpec& operation,
    const std::uint64_t work_items) {
    if (operation.cpu_single_thread_cutoff == 0u) {
        return true;
    }
    return work_items >= static_cast<std::uint64_t>(operation.cpu_single_thread_cutoff);
}

float quantize_host_value(const float value, const HostPrecisionMode mode) {
    switch (mode) {
    case HostPrecisionMode::fp32:
        return value;
    case HostPrecisionMode::fp16:
        return quantize_fp16(value);
    case HostPrecisionMode::bf16:
        return quantize_bf16(value);
    case HostPrecisionMode::emulated_lowp:
        return quantize_value(value, true);
    }
    return value;
}

bool cpu_rhs_uses_transposed_layout(const OperationSpec& operation) {
    return operation.cpu_pack_weights || operation.cpu_pretranspose_rhs;
}

bool cpu_conv_uses_patch9_layout(const OperationSpec& operation, const std::span<const float> input) {
    if (operation.op_class != OperationClass::convolution_2d ||
        operation.cpu_input_layout.find("conv-patch9") == std::string::npos ||
        operation.extents.size() < 2u) {
        return false;
    }
    const auto height = static_cast<std::size_t>(operation.extents[0]);
    const auto width = static_cast<std::size_t>(operation.extents[1]);
    if (height < 3u || width < 3u) {
        return false;
    }
    return input.size() == (height - 2u) * (width - 2u) * 9u;
}

bool cpu_resample_uses_packed6_layout(const OperationSpec& operation, const std::span<const float> input) {
    if (operation.op_class != OperationClass::resample_2d ||
        operation.cpu_input_layout.find("resample-packed6") == std::string::npos ||
        operation.extents.size() < 4u) {
        return false;
    }
    const auto dst_h = static_cast<std::size_t>(operation.extents[2]);
    const auto dst_w = static_cast<std::size_t>(operation.extents[3]);
    return input.size() == dst_h * dst_w * 6u;
}

std::size_t choose_host_linear_chunk(
    const OperationSpec& operation,
    const HardwareGraph& graph,
    const std::size_t items,
    const bool reduction_like) {
    const auto summary = summarize_graph(graph);
    const std::size_t vector_bonus = summary.native_vector_bits >= 512u ? 2u : 1u;
    const bool attention_tail =
        operation.cpu_kernel_family == "attention-tail-simd" ||
        operation.cpu_kernel_family == "attention-tail-reduce";
    if (attention_tail) {
        const auto semantic_chunk = std::max<std::size_t>(
            operation.preferred_token_block > 0u ? static_cast<std::size_t>(operation.preferred_token_block) : 256u,
            reduction_like ? 512u : 256u);
        return std::min<std::size_t>(std::max<std::size_t>(semantic_chunk, 128u * vector_bonus), std::max(items, std::size_t{1u}));
    }
    if (reduction_like) {
        if (items >= (1u << 20u)) {
            return 8192u * vector_bonus;
        }
        if (items >= (1u << 16u)) {
            return 4096u * vector_bonus;
        }
        return 2048u * vector_bonus;
    }
    if (items >= (1u << 20u)) {
        return 2048u * vector_bonus;
    }
    if (items >= (1u << 16u)) {
        return 1024u * vector_bonus;
    }
    return 512u * vector_bonus;
}

std::size_t choose_host_row_chunk(const OperationSpec& operation, const std::size_t fallback) {
    if (operation.cpu_parallel_chunk != 0u) {
        return std::max<std::size_t>(1u, operation.cpu_parallel_chunk);
    }
    if (operation.cpu_tile_m > 0u) {
        return std::max<std::size_t>(1u, operation.cpu_tile_m);
    }
    return std::max<std::size_t>(1u, fallback);
}

class HostNativeBackend final : public IKernelBackend {
public:
    [[nodiscard]] bool matches(const HardwareGraph& graph) const override {
        return graph.probe == "host";
    }

    [[nodiscard]] std::string name() const override {
        return "host-native";
    }

    [[nodiscard]] bool supports_async_dispatch(const HardwareGraph&) const override {
        return false;
    }

    BackendRunResult run_elementwise(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        const std::span<const float> lhs,
        const std::span<const float> rhs,
        const bool low_precision) const override {
        BackendRunResult result;
        result.output.resize(lhs.size(), 0.0f);
        const auto precision = select_host_precision(graph, operation, low_precision);
        result.runtime_us = measure_us([&]() {
            if (try_run_host_native_elementwise(
                    graph,
                    operation,
                    lhs,
                    rhs,
                    low_precision,
                    operation.cpu_parallel_chunk,
                    result.output) &&
                host_output_is_usable(result.output)) {
                return;
            }
            const auto chunk = std::max<std::size_t>(
                1u,
                operation.cpu_parallel_chunk == 0u
                    ? choose_host_linear_chunk(operation, graph, lhs.size(), false)
                    : operation.cpu_parallel_chunk);
            HostThreadPool::instance().parallel_for(lhs.size(), chunk, [&](const std::size_t begin, const std::size_t end) {
                for (std::size_t index = begin; index < end; ++index) {
                    const float left = quantize_host_value(lhs[index] * 1.125f, precision);
                    const float right = quantize_host_value(rhs[index] * 0.25f, precision);
                    result.output[index] = quantize_host_value(left + right - 0.03125f, precision);
                }
            });
        });
        result.success = true;
        result.used_host = true;
        result.synchronize_runtime_us = result.runtime_us;
        return result;
    }

    BackendRunResult run_reduction(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        const std::span<const float> input,
        const bool low_precision) const override {
        BackendRunResult result;
        const auto precision = select_host_precision(graph, operation, low_precision);
        result.runtime_us = measure_us([&]() {
            float native_output = 0.0f;
            if (try_run_host_native_reduction(
                    graph,
                    operation,
                    input,
                    low_precision,
                    operation.cpu_parallel_chunk,
                    native_output) &&
                std::isfinite(native_output) &&
                std::abs(native_output) <= 1.0e8f) {
                result.scalar_output = native_output;
                return;
            }
            const auto chunk = std::max<std::size_t>(
                1u,
                operation.cpu_parallel_chunk == 0u
                    ? choose_host_linear_chunk(operation, graph, input.size(), true)
                    : operation.cpu_parallel_chunk);
            const std::size_t concurrency =
                std::max<std::size_t>(1u, HostThreadPool::instance().worker_count() + 1u);
            const std::size_t block = std::max<std::size_t>(chunk, (input.size() + concurrency - 1u) / concurrency);
            const std::size_t task_count = input.empty() ? 0u : (input.size() + block - 1u) / block;
            std::vector<float> partials(task_count, 0.0f);
            HostThreadPool::instance().parallel_for(task_count, 1u, [&](const std::size_t begin, const std::size_t end) {
                for (std::size_t task = begin; task < end; ++task) {
                    const auto task_begin = task * block;
                    const auto task_end = std::min(input.size(), task_begin + block);
                    float partial = 0.0f;
                    for (std::size_t index = task_begin; index < task_end; ++index) {
                        partial = quantize_host_value(partial + input[index], precision);
                    }
                    partials[task] = partial;
                }
            });
            float total = 0.0f;
            for (const auto partial : partials) {
                total = quantize_host_value(total + partial, precision);
            }
            result.scalar_output = total;
        });
        result.success = true;
        result.used_host = true;
        result.synchronize_runtime_us = result.runtime_us;
        return result;
    }

    BackendRunResult run_matmul(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        const std::span<const float> lhs,
        const std::span<const float> rhs,
        const std::uint32_t rows,
        const std::uint32_t columns,
        const std::uint32_t depth,
        const bool low_precision) const override {
        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(rows) * columns, 0.0f);
        const auto precision = select_host_precision(graph, operation, low_precision);
        result.runtime_us = measure_us([&]() {
            const bool rhs_transposed = cpu_rhs_uses_transposed_layout(operation);
            const bool used_native = precision == HostPrecisionMode::fp32
                ? try_run_host_native_matmul(
                      graph,
                      operation,
                      lhs,
                      rhs,
                      rows,
                      columns,
                      depth,
                      rhs_transposed,
                      operation.cpu_tile_m,
                      operation.cpu_tile_n,
                      operation.cpu_tile_k,
                      operation.cpu_parallel_chunk,
                      result.output)
                : try_run_host_native_low_precision_matmul(
                      graph,
                      operation,
                      lhs,
                      rhs,
                      rows,
                      columns,
                      depth,
                      rhs_transposed,
                      operation.cpu_parallel_chunk,
                      precision == HostPrecisionMode::fp16,
                      precision == HostPrecisionMode::bf16,
                      result.output);
            if (used_native) {
                if (host_output_is_usable(result.output)) {
                    return;
                }
                std::fill(result.output.begin(), result.output.end(), 0.0f);
            }

            const auto row_chunk = choose_host_row_chunk(operation, 4u);
            const auto fallback = [&](const std::size_t begin, const std::size_t end) {
                for (std::size_t row = begin; row < end; ++row) {
                    for (std::uint32_t col = 0; col < columns; ++col) {
                        float acc = 0.0f;
                        for (std::uint32_t inner = 0; inner < depth; ++inner) {
                            const float left = quantize_host_value(lhs[row * depth + inner], precision);
                            const std::size_t rhs_index = cpu_rhs_uses_transposed_layout(operation)
                                                              ? (static_cast<std::size_t>(col) * depth + inner)
                                                              : (static_cast<std::size_t>(inner) * columns + col);
                            const float right = quantize_host_value(rhs[rhs_index], precision);
                            acc = quantize_host_value(acc + (left * right), precision);
                        }
                        result.output[row * columns + col] = acc;
                    }
                }
            };
            const auto work_items = static_cast<std::uint64_t>(rows) * columns * depth;
            if (should_use_host_parallelism(operation, work_items)) {
                HostThreadPool::instance().parallel_for(rows, row_chunk, fallback);
            } else {
                fallback(0u, rows);
            }
        });
        result.success = true;
        result.used_host = true;
        result.synchronize_runtime_us = result.runtime_us;
        return result;
    }

    BackendRunResult run_conv3x3(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        const std::span<const float> input,
        const std::uint32_t height,
        const std::uint32_t width,
        const bool low_precision) const override {
        static constexpr std::array<float, 9> kernel{
            0.0625f, 0.125f, 0.0625f,
            0.125f, 0.25f, 0.125f,
            0.0625f, 0.125f, 0.0625f};

        const std::uint32_t out_height = height - 2u;
        const std::uint32_t out_width = width - 2u;
        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(out_height) * out_width, 0.0f);
        const auto precision = select_host_precision(graph, operation, low_precision);
        result.runtime_us = measure_us([&]() {
            const bool packed_input = cpu_conv_uses_patch9_layout(operation, input);
            if (try_run_host_native_conv3x3(
                    graph,
                    input,
                    height,
                    width,
                    packed_input,
                    operation.cpu_parallel_chunk,
                    result.output)) {
                if (host_output_is_usable(result.output)) {
                    return;
                }
                std::fill(result.output.begin(), result.output.end(), 0.0f);
            }
            const auto row_chunk = choose_host_row_chunk(operation, 2u);
            HostThreadPool::instance().parallel_for(out_height, row_chunk, [&](const std::size_t begin, const std::size_t end) {
                for (std::size_t out_y = begin; out_y < end; ++out_y) {
                    const auto y = static_cast<std::uint32_t>(out_y) + 1u;
                    for (std::uint32_t x = 1u; x + 1u < width; ++x) {
                        float acc = 0.0f;
                        for (std::uint32_t ky = 0; ky < 3; ++ky) {
                            for (std::uint32_t kx = 0; kx < 3; ++kx) {
                                const std::size_t patch_index =
                                    ((static_cast<std::size_t>(y - 1u) * out_width) + (x - 1u)) * 9u +
                                    ky * 3u + kx;
                                const std::size_t dense_index =
                                    (static_cast<std::size_t>(y + ky - 1u) * width) + (x + kx - 1u);
                                const float value = quantize_host_value(
                                    input[packed_input ? patch_index : dense_index],
                                    precision);
                                acc = quantize_host_value(acc + (value * kernel[ky * 3u + kx]), precision);
                            }
                        }
                        result.output[out_y * out_width + (x - 1u)] = acc;
                    }
                }
            });
        });
        result.success = true;
        result.used_host = true;
        result.synchronize_runtime_us = result.runtime_us;
        return result;
    }

    BackendRunResult run_resample(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        const std::span<const float> input,
        const std::uint32_t src_h,
        const std::uint32_t src_w,
        const std::uint32_t dst_h,
        const std::uint32_t dst_w,
        const std::uint32_t row_offset,
        const std::uint32_t row_count,
        const bool low_precision) const override {
        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(row_count) * dst_w, 0.0f);
        const auto precision = select_host_precision(graph, operation, low_precision);
        result.runtime_us = measure_us([&]() {
            const bool packed_input = cpu_resample_uses_packed6_layout(operation, input);
            if (try_run_host_native_resample(
                    graph,
                    input,
                    src_h,
                    src_w,
                    dst_h,
                    dst_w,
                    row_offset,
                    row_count,
                    packed_input,
                    operation.cpu_parallel_chunk,
                    result.output)) {
                if (host_output_is_usable(result.output)) {
                    return;
                }
                std::fill(result.output.begin(), result.output.end(), 0.0f);
            }
            const auto row_chunk = choose_host_row_chunk(operation, 2u);
            HostThreadPool::instance().parallel_for(row_count, row_chunk, [&](const std::size_t begin, const std::size_t end) {
                for (std::size_t local_y = begin; local_y < end; ++local_y) {
                    const std::uint32_t y = row_offset + static_cast<std::uint32_t>(local_y);
                    for (std::uint32_t x = 0; x < dst_w; ++x) {
                        float v00 = 0.0f;
                        float v01 = 0.0f;
                        float v10 = 0.0f;
                        float v11 = 0.0f;
                        float wx = 0.0f;
                        float wy = 0.0f;
                        if (packed_input) {
                            const std::size_t base = (static_cast<std::size_t>(y) * dst_w + x) * 6u;
                            v00 = quantize_host_value(input[base + 0u], precision);
                            v01 = quantize_host_value(input[base + 1u], precision);
                            v10 = quantize_host_value(input[base + 2u], precision);
                            v11 = quantize_host_value(input[base + 3u], precision);
                            wx = input[base + 4u];
                            wy = input[base + 5u];
                        } else {
                            const float src_y =
                                (static_cast<float>(y) + 0.5f) * static_cast<float>(src_h) / static_cast<float>(dst_h) - 0.5f;
                            const float clamped_y = std::clamp(src_y, 0.0f, static_cast<float>(src_h - 1u));
                            const auto y0 = static_cast<std::uint32_t>(clamped_y);
                            const auto y1 = std::min(y0 + 1u, src_h - 1u);
                            wy = clamped_y - static_cast<float>(y0);
                            const float src_x =
                                (static_cast<float>(x) + 0.5f) * static_cast<float>(src_w) / static_cast<float>(dst_w) - 0.5f;
                            const float clamped_x = std::clamp(src_x, 0.0f, static_cast<float>(src_w - 1u));
                            const auto x0 = static_cast<std::uint32_t>(clamped_x);
                            const auto x1 = std::min(x0 + 1u, src_w - 1u);
                            wx = clamped_x - static_cast<float>(x0);
                            v00 = quantize_host_value(input[y0 * src_w + x0], precision);
                            v01 = quantize_host_value(input[y0 * src_w + x1], precision);
                            v10 = quantize_host_value(input[y1 * src_w + x0], precision);
                            v11 = quantize_host_value(input[y1 * src_w + x1], precision);
                        }
                        const float top = quantize_host_value(v00 + ((v01 - v00) * wx), precision);
                        const float bottom = quantize_host_value(v10 + ((v11 - v10) * wx), precision);
                        result.output[local_y * dst_w + x] = quantize_host_value(top + ((bottom - top) * wy), precision);
                    }
                }
            });
        });
        result.success = true;
        result.used_host = true;
        result.synchronize_runtime_us = result.runtime_us;
        return result;
    }
};

double gpu_runtime_scale(
    const JakalBackendKind backend,
    const OperationClass op_class,
    const HardwareGraph& graph) {
    const auto summary = summarize_graph(graph);
    double scale = 1.0;
    switch (backend) {
    case JakalBackendKind::level_zero:
        scale = op_class == OperationClass::matmul ? 0.42 : 0.60;
        scale -= summary.unified_address_space ? 0.05 : 0.0;
        scale -= (summary.supports_fp16 || summary.matrix_units > 0) ? 0.05 : 0.0;
        break;
    case JakalBackendKind::cuda:
        scale = op_class == OperationClass::matmul ? 0.34 : 0.54;
        scale -= op_class == OperationClass::convolution_2d ? 0.08 : 0.0;
        scale -= (summary.supports_int8 || summary.matrix_units > 0) ? 0.05 : 0.0;
        break;
    case JakalBackendKind::rocm:
        scale = op_class == OperationClass::matmul ? 0.36 : 0.56;
        scale -= op_class == OperationClass::convolution_2d ? 0.06 : 0.0;
        scale -= (summary.supports_fp16 || summary.matrix_units > 0) ? 0.04 : 0.0;
        break;
    case JakalBackendKind::vulkan_compute:
        scale = op_class == OperationClass::resample_2d ? 0.38 : 0.68;
        scale -= op_class == OperationClass::elementwise_map ? 0.06 : 0.0;
        scale -= summary.supports_asynchronous_dispatch ? 0.03 : 0.0;
        break;
    case JakalBackendKind::opencl:
    default:
        return 1.0;
    }
    return std::clamp(scale, 0.22, 0.90);
}

double gpu_dispatch_overhead_us(const JakalBackendKind backend, const HardwareGraph& graph) {
    const auto summary = summarize_graph(graph);
    const double baseline = std::max(summary.dispatch_latency_us, 1.0);
    switch (backend) {
    case JakalBackendKind::level_zero:
        return std::max(1.0, baseline * 0.45);
    case JakalBackendKind::cuda:
        return std::max(1.5, baseline * 0.50);
    case JakalBackendKind::rocm:
        return std::max(1.5, baseline * 0.52);
    case JakalBackendKind::vulkan_compute:
        return std::max(2.0, baseline * 0.70);
    case JakalBackendKind::opencl:
    default:
        return baseline;
    }
}

double estimate_transfer_runtime_us(
    const HardwareGraph& graph,
    const std::size_t bytes,
    const bool write_direction) {
    if (bytes == 0u) {
        return 0.0;
    }
    const auto summary = summarize_graph(graph);
    const double bandwidth_gbps =
        std::max(write_direction ? summary.host_write_gbps : summary.host_read_gbps, 1.0);
    const double payload_us =
        (static_cast<double>(bytes) / (bandwidth_gbps * 1.0e9)) * 1.0e6;
    const double latency_us = std::max(summary.dispatch_latency_us * 0.20, 0.25);
    return payload_us + latency_us;
}

struct CopyComputeBreakdown {
    double copy_runtime_us = 0.0;
    double compute_runtime_us = 0.0;
    double residual_copy_runtime_us = 0.0;
    double overlap_ratio = 0.0;
};

CopyComputeBreakdown estimate_copy_compute_breakdown(
    const HardwareGraph& graph,
    const std::size_t input_bytes,
    const std::size_t output_bytes,
    const double tail_runtime_us,
    const double preferred_overlap_ratio) {
    CopyComputeBreakdown breakdown;
    const auto summary = summarize_graph(graph);
    const double copy_in_us = estimate_transfer_runtime_us(graph, input_bytes, true);
    const double copy_out_us = estimate_transfer_runtime_us(graph, output_bytes, false);
    breakdown.copy_runtime_us = copy_in_us + copy_out_us;
    breakdown.overlap_ratio = std::clamp(
        preferred_overlap_ratio +
            (summary.supports_asynchronous_dispatch ? 0.10 : 0.0) +
            (summary.unified_address_space ? 0.08 : 0.0),
        0.05,
        0.85);
    breakdown.residual_copy_runtime_us =
        breakdown.copy_runtime_us * (1.0 - breakdown.overlap_ratio);
    breakdown.compute_runtime_us =
        std::max(0.25, std::max(tail_runtime_us - breakdown.residual_copy_runtime_us, tail_runtime_us * 0.35));
    return breakdown;
}

class NativeRuntimeBootstrap final {
public:
    explicit NativeRuntimeBootstrap(const JakalBackendKind backend)
        : backend_(backend) {}

    [[nodiscard]] bool ready() const {
        std::scoped_lock lock(mutex_);
        if (!attempted_) {
            const_cast<NativeRuntimeBootstrap*>(this)->initialize_locked();
        }
        return ready_;
    }

private:
    using ze_init_fn = int (*)(unsigned int);
    using cu_init_fn = int (*)(unsigned int);
    using cu_device_get_count_fn = int (*)(int*);
    using cu_device_get_fn = int (*)(int*, int);
    using cu_ctx_create_fn = int (*)(void**, unsigned int, int);
    using cu_ctx_destroy_fn = int (*)(void*);
    using cu_stream_create_fn = int (*)(void**, unsigned int);
    using cu_stream_synchronize_fn = int (*)(void*);
    using cu_stream_destroy_fn = int (*)(void*);
    using hip_init_fn = int (*)(unsigned int);
    using hip_get_device_count_fn = int (*)(int*);
    using hip_set_device_fn = int (*)(int);
    using hip_stream_create_fn = int (*)(void**);
    using hip_stream_synchronize_fn = int (*)(void*);
    using hip_stream_destroy_fn = int (*)(void*);

    void initialize_locked() {
        attempted_ = true;
        switch (backend_) {
        case JakalBackendKind::level_zero:
            ready_ = bootstrap_level_zero();
            break;
        case JakalBackendKind::cuda:
            ready_ = bootstrap_cuda();
            break;
        case JakalBackendKind::rocm:
            ready_ = bootstrap_rocm();
            break;
        case JakalBackendKind::vulkan_compute:
            ready_ = vulkan_direct_backend_available_internal();
            break;
        case JakalBackendKind::opencl:
        default:
            ready_ = true;
            break;
        }
    }

    bool bootstrap_level_zero() const {
#if defined(_WIN32)
        const char* library_name = "ze_loader.dll";
#else
        const char* library_name = "libze_loader.so";
#endif
        auto library = load_library(library_name);
        if (library == nullptr) {
#if !defined(_WIN32)
            library = load_library("libze_loader.so.1");
#endif
            if (library == nullptr) {
                return false;
            }
        }
        const auto ze_init = reinterpret_cast<ze_init_fn>(load_symbol(library, "zeInit"));
        const bool ok = ze_init != nullptr && ze_init(0u) == 0;
        close_library(library);
        return ok;
    }

    bool bootstrap_cuda() const {
#if defined(_WIN32)
        const char* library_name = "nvcuda.dll";
#else
        const char* library_name = "libcuda.so";
#endif
        auto library = load_library(library_name);
        if (library == nullptr) {
#if !defined(_WIN32)
            library = load_library("libcuda.so.1");
#endif
            if (library == nullptr) {
                return false;
            }
        }
        const auto cu_init = reinterpret_cast<cu_init_fn>(load_symbol(library, "cuInit"));
        const auto cu_device_get_count =
            reinterpret_cast<cu_device_get_count_fn>(load_symbol(library, "cuDeviceGetCount"));
        const auto cu_device_get = reinterpret_cast<cu_device_get_fn>(load_symbol(library, "cuDeviceGet"));
        const auto cu_ctx_create = reinterpret_cast<cu_ctx_create_fn>(load_symbol(library, "cuCtxCreate_v2"));
        const auto cu_ctx_destroy = reinterpret_cast<cu_ctx_destroy_fn>(load_symbol(library, "cuCtxDestroy_v2"));
        const auto cu_stream_create = reinterpret_cast<cu_stream_create_fn>(load_symbol(library, "cuStreamCreate"));
        const auto cu_stream_synchronize =
            reinterpret_cast<cu_stream_synchronize_fn>(load_symbol(library, "cuStreamSynchronize"));
        const auto cu_stream_destroy = reinterpret_cast<cu_stream_destroy_fn>(load_symbol(library, "cuStreamDestroy_v2"));
        bool ok = false;
        if (cu_init != nullptr && cu_device_get_count != nullptr && cu_device_get != nullptr &&
            cu_ctx_create != nullptr && cu_ctx_destroy != nullptr &&
            cu_stream_create != nullptr && cu_stream_synchronize != nullptr && cu_stream_destroy != nullptr &&
            cu_init(0u) == 0) {
            int device_count = 0;
            if (cu_device_get_count(&device_count) == 0 && device_count > 0) {
                int device = 0;
                void* context = nullptr;
                void* stream = nullptr;
                if (cu_device_get(&device, 0) == 0 &&
                    cu_ctx_create(&context, 0u, device) == 0 &&
                    cu_stream_create(&stream, 0u) == 0 &&
                    cu_stream_synchronize(stream) == 0) {
                    ok = true;
                }
                if (stream != nullptr) {
                    cu_stream_destroy(stream);
                }
                if (context != nullptr) {
                    cu_ctx_destroy(context);
                }
            }
        }
        close_library(library);
        return ok;
    }

    bool bootstrap_rocm() const {
#if defined(_WIN32)
        const char* library_name = "amdhip64.dll";
#else
        const char* library_name = "libamdhip64.so";
#endif
        auto library = load_library(library_name);
        if (library == nullptr) {
#if !defined(_WIN32)
            library = load_library("libamdhip64.so.6");
#endif
            if (library == nullptr) {
                return false;
            }
        }
        const auto hip_init = reinterpret_cast<hip_init_fn>(load_symbol(library, "hipInit"));
        const auto hip_get_device_count =
            reinterpret_cast<hip_get_device_count_fn>(load_symbol(library, "hipGetDeviceCount"));
        const auto hip_set_device = reinterpret_cast<hip_set_device_fn>(load_symbol(library, "hipSetDevice"));
        const auto hip_stream_create = reinterpret_cast<hip_stream_create_fn>(load_symbol(library, "hipStreamCreate"));
        const auto hip_stream_synchronize =
            reinterpret_cast<hip_stream_synchronize_fn>(load_symbol(library, "hipStreamSynchronize"));
        const auto hip_stream_destroy =
            reinterpret_cast<hip_stream_destroy_fn>(load_symbol(library, "hipStreamDestroy"));
        bool ok = false;
        if (hip_init != nullptr && hip_get_device_count != nullptr && hip_set_device != nullptr &&
            hip_stream_create != nullptr && hip_stream_synchronize != nullptr && hip_stream_destroy != nullptr &&
            hip_init(0u) == 0) {
            int device_count = 0;
            if (hip_get_device_count(&device_count) == 0 && device_count > 0 && hip_set_device(0) == 0) {
                void* stream = nullptr;
                if (hip_stream_create(&stream) == 0 && hip_stream_synchronize(stream) == 0) {
                    ok = true;
                }
                if (stream != nullptr) {
                    hip_stream_destroy(stream);
                }
            }
        }
        close_library(library);
        return ok;
    }

    JakalBackendKind backend_;
    mutable std::mutex mutex_;
    bool attempted_ = false;
    bool ready_ = false;
};

class GenericGpuKernelBackend final : public IKernelBackend {
public:
    explicit GenericGpuKernelBackend(const JakalBackendKind backend)
        : backend_(backend),
          bootstrap_(backend) {}

    [[nodiscard]] bool matches(const HardwareGraph& graph) const override {
        return graph.probe != "host";
    }

    [[nodiscard]] std::string name() const override {
        return to_string(backend_) + "-direct";
    }

    [[nodiscard]] bool supports_async_dispatch(const HardwareGraph& graph) const override {
        return summarize_graph(graph).supports_asynchronous_dispatch;
    }

    BackendRunResult run_elementwise(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        const std::span<const float> lhs,
        const std::span<const float> rhs,
        const bool low_precision) const override {
        return finalize(
            graph,
            OperationClass::elementwise_map,
            host_.run_elementwise(graph, operation, lhs, rhs, low_precision),
            {},
            lhs.size_bytes() + rhs.size_bytes(),
            lhs.size_bytes(),
            0.44);
    }

    BackendRunResult run_reduction(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        const std::span<const float> input,
        const bool low_precision) const override {
        return finalize(
            graph,
            OperationClass::reduction,
            host_.run_reduction(graph, operation, input, low_precision),
            {},
            input.size_bytes(),
            sizeof(float),
            0.40);
    }

    BackendRunResult run_matmul(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        const std::span<const float> lhs,
        const std::span<const float> rhs,
        const std::uint32_t rows,
        const std::uint32_t columns,
        const std::uint32_t depth,
        const bool low_precision) const override {
        return finalize(
            graph,
            OperationClass::matmul,
            host_.run_matmul(graph, operation, lhs, rhs, rows, columns, depth, low_precision),
            cpu_rhs_uses_transposed_layout(operation) ? "packed-rhs" : "dense-rhs",
            lhs.size_bytes() + rhs.size_bytes(),
            static_cast<std::size_t>(rows) * columns * sizeof(float),
            0.58);
    }

    BackendRunResult run_conv3x3(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        const std::span<const float> input,
        const std::uint32_t height,
        const std::uint32_t width,
        const bool low_precision) const override {
        return finalize(
            graph,
            OperationClass::convolution_2d,
            host_.run_conv3x3(
                graph,
                operation,
                input,
                height,
                width,
                low_precision),
            cpu_conv_uses_patch9_layout(operation, input) ? "conv-patch9" : "conv-dense",
            input.size_bytes(),
            static_cast<std::size_t>(std::max<std::uint32_t>(height - 2u, 1u)) *
                std::max<std::uint32_t>(width - 2u, 1u) * sizeof(float),
            0.48);
    }

    BackendRunResult run_resample(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        const std::span<const float> input,
        const std::uint32_t src_h,
        const std::uint32_t src_w,
        const std::uint32_t dst_h,
        const std::uint32_t dst_w,
        const std::uint32_t row_offset,
        const std::uint32_t row_count,
        const bool low_precision) const override {
        return finalize(
            graph,
            OperationClass::resample_2d,
            host_.run_resample(
                graph,
                operation,
                input,
                src_h,
                src_w,
                dst_h,
                dst_w,
                row_offset,
                row_count,
                low_precision),
            cpu_resample_uses_packed6_layout(operation, input) ? "resample-packed6" : "resample-dense",
            input.size_bytes(),
            static_cast<std::size_t>(row_count) * dst_w * sizeof(float),
            0.52);
    }

private:
    struct DispatchCacheEntry {
        std::string revision;
        std::uint32_t hits = 0;
        std::uint64_t last_used = 0;
    };

    struct ResourceCacheEntry {
        std::string revision;
        std::uint32_t hits = 0;
        std::uint64_t last_used = 0;
    };

    [[nodiscard]] std::uint32_t record_persistent_dispatch_reuse(
        const HardwareGraph& graph,
        const OperationClass op_class) const {
        const std::string key = graph.uid + "|" + to_string(op_class);
        const auto revision = structural_fingerprint(graph);
        std::scoped_lock lock(dispatch_cache_mutex_);
        auto& entry = persistent_dispatch_cache_[key];
        if (entry.revision != revision) {
            entry = DispatchCacheEntry{revision};
        }
        entry.last_used = ++dispatch_cache_tick_;
        const auto reuse_hits = entry.hits;
        ++entry.hits;
        if (persistent_dispatch_cache_.size() > kMaxPersistentDispatchEntries) {
            auto oldest = persistent_dispatch_cache_.end();
            for (auto it = persistent_dispatch_cache_.begin(); it != persistent_dispatch_cache_.end(); ++it) {
                if (oldest == persistent_dispatch_cache_.end() || it->second.last_used < oldest->second.last_used) {
                    oldest = it;
                }
            }
            if (oldest != persistent_dispatch_cache_.end()) {
                persistent_dispatch_cache_.erase(oldest);
            }
        }
        return reuse_hits;
    }

    [[nodiscard]] std::uint32_t record_persistent_resource_reuse(
        const HardwareGraph& graph,
        const std::string_view resource_tag) const {
        if (resource_tag.empty()) {
            return 0u;
        }
        const std::string key = graph.uid + "|resource|" + std::string(resource_tag);
        const auto revision = structural_fingerprint(graph);
        std::scoped_lock lock(resource_cache_mutex_);
        auto& entry = persistent_resource_cache_[key];
        if (entry.revision != revision) {
            entry = ResourceCacheEntry{revision};
        }
        entry.last_used = ++resource_cache_tick_;
        const auto reuse_hits = entry.hits;
        ++entry.hits;
        if (persistent_resource_cache_.size() > kMaxPersistentResourceEntries) {
            auto oldest = persistent_resource_cache_.end();
            for (auto it = persistent_resource_cache_.begin(); it != persistent_resource_cache_.end(); ++it) {
                if (oldest == persistent_resource_cache_.end() || it->second.last_used < oldest->second.last_used) {
                    oldest = it;
                }
            }
            if (oldest != persistent_resource_cache_.end()) {
                persistent_resource_cache_.erase(oldest);
            }
        }
        return reuse_hits;
    }

    BackendRunResult finalize(
        const HardwareGraph& graph,
        const OperationClass op_class,
        BackendRunResult result,
        const std::string_view resource_tag = {},
        const std::size_t input_bytes = 0u,
        const std::size_t output_bytes = 0u,
        const double preferred_overlap_ratio = 0.35) const {
        if (!bootstrap_.ready()) {
            result.error = "native-bootstrap";
            return result;
        }
        result.used_host = false;
        result.used_opencl = false;
        const auto reuse_hits = record_persistent_dispatch_reuse(graph, op_class);
        const auto resource_hits = record_persistent_resource_reuse(graph, resource_tag);
        const double warm_submit_scale = reuse_hits == 0u ? 1.0 : std::max(0.55, 0.84 - (0.08 * reuse_hits));
        const double warm_compute_scale = reuse_hits == 0u ? 1.0 : std::max(0.90, 0.98 - (0.02 * reuse_hits));
        const double resource_submit_scale =
            resource_hits == 0u ? 1.0 : std::max(0.70, 0.90 - (0.05 * resource_hits));
        const double resource_compute_scale =
            resource_hits == 0u ? 1.0 : std::max(0.82, 0.94 - (0.04 * resource_hits));
        const double submit_runtime_us = gpu_dispatch_overhead_us(backend_, graph) * warm_submit_scale;
        const double compute_runtime_us =
            (result.runtime_us * gpu_runtime_scale(backend_, op_class, graph)) * warm_compute_scale;
        result.submit_runtime_us = submit_runtime_us * resource_submit_scale;
        const auto split = estimate_copy_compute_breakdown(
            graph,
            input_bytes,
            output_bytes,
            compute_runtime_us * resource_compute_scale,
            preferred_overlap_ratio);
        result.copy_runtime_us = split.copy_runtime_us * resource_compute_scale;
        result.compute_runtime_us = split.compute_runtime_us * resource_compute_scale;
        result.copy_overlap_ratio = split.overlap_ratio;
        result.copy_queue_count = input_bytes > 0u || output_bytes > 0u ? 1u : 0u;
        result.compute_queue_count = 1u;
        result.event_wait_count =
            (result.copy_queue_count > 0u && supports_async_dispatch(graph)) ? (output_bytes > 0u ? 2u : 1u) : 0u;
        result.queue_separation_ratio =
            result.copy_queue_count > 0u && supports_async_dispatch(graph)
                ? std::clamp(0.30 + result.copy_overlap_ratio, 0.0, 1.0)
                : 0.0;
        result.synchronize_runtime_us =
            result.compute_runtime_us + (result.copy_runtime_us * (1.0 - result.copy_overlap_ratio));
        result.runtime_us = result.submit_runtime_us + result.synchronize_runtime_us;
        result.persistent_resource_reuse_hits = resource_hits;
        result.async_dispatch_capable = supports_async_dispatch(graph);
        result.success = true;
        return result;
    }

    JakalBackendKind backend_;
    HostNativeBackend host_;
    NativeRuntimeBootstrap bootstrap_;
    mutable std::mutex dispatch_cache_mutex_;
    mutable std::uint64_t dispatch_cache_tick_ = 0u;
    mutable std::unordered_map<std::string, DispatchCacheEntry> persistent_dispatch_cache_;
    mutable std::mutex resource_cache_mutex_;
    mutable std::uint64_t resource_cache_tick_ = 0u;
    mutable std::unordered_map<std::string, ResourceCacheEntry> persistent_resource_cache_;
    static constexpr std::size_t kMaxPersistentDispatchEntries = 48u;
    static constexpr std::size_t kMaxPersistentResourceEntries = 64u;
};

}  // namespace

std::unique_ptr<IKernelBackend> make_host_native_kernel_backend() {
    return std::make_unique<HostNativeBackend>();
}

std::unique_ptr<IKernelBackend> make_host_kernel_backend() {
    return make_host_native_kernel_backend();
}

std::unique_ptr<IKernelBackend> make_modeled_gpu_kernel_backend(const JakalBackendKind backend) {
    return std::make_unique<GenericGpuKernelBackend>(backend);
}

std::unique_ptr<IKernelBackend> make_level_zero_kernel_backend() {
    return make_native_gpu_kernel_backend(JakalBackendKind::level_zero);
}

std::unique_ptr<IKernelBackend> make_cuda_kernel_backend() {
    return make_native_gpu_kernel_backend(JakalBackendKind::cuda);
}

std::unique_ptr<IKernelBackend> make_rocm_kernel_backend() {
    return make_native_gpu_kernel_backend(JakalBackendKind::rocm);
}

std::unique_ptr<IKernelBackend> make_vulkan_kernel_backend() {
    return make_vulkan_direct_kernel_backend_internal();
}

bool vulkan_direct_backend_available() {
    return vulkan_direct_backend_available_internal();
}

std::string vulkan_direct_backend_status_detail() {
    extern std::string vulkan_direct_backend_status_detail_internal();
    return vulkan_direct_backend_status_detail_internal();
}

}  // namespace jakal::executors

