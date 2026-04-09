#include "jakal/executors/host_native_kernels.hpp"
#include "jakal/executors/host_thread_pool.hpp"
#include "jakal/execution.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include <immintrin.h>

namespace jakal::executors {
namespace {

std::uint32_t float_to_bits(const float value) {
    return std::bit_cast<std::uint32_t>(value);
}

float bits_to_float(const std::uint32_t bits) {
    return std::bit_cast<float>(bits);
}

float quantize_fp16_scalar(const float value) {
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
            half = static_cast<std::uint16_t>(
                sign |
                (static_cast<std::uint32_t>(exponent) << 10u) |
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
            restored = half_sign |
                       (static_cast<std::uint32_t>(exp + 127) << 23u) |
                       (mantissa_norm << 13u);
        }
    } else if (half_exponent == 0x1fu) {
        restored = half_sign | 0x7f800000u | (half_mantissa << 13u);
    } else {
        restored = half_sign | ((half_exponent + 112u) << 23u) | (half_mantissa << 13u);
    }
    return bits_to_float(restored);
}

#if defined(__AVX2__) || defined(_MSC_VER)

template <typename Func>
void run_parallel_rows(const std::uint32_t rows, const std::size_t min_chunk, Func&& func) {
    HostThreadPool::instance().parallel_for(rows, min_chunk, std::forward<Func>(func));
}

float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

__m256 gather_strided_8(const float* data, const std::int32_t base, const std::int32_t stride) {
    const __m256i indices = _mm256_setr_epi32(
        base + (0 * stride),
        base + (1 * stride),
        base + (2 * stride),
        base + (3 * stride),
        base + (4 * stride),
        base + (5 * stride),
        base + (6 * stride),
        base + (7 * stride));
    return _mm256_i32gather_ps(data, indices, 4);
}

constexpr std::uint32_t kHostMatmulMBlock = 48u;
constexpr std::uint32_t kHostMatmulNBlock = 64u;
constexpr std::uint32_t kHostMatmulKBlock = 128u;
constexpr std::uint32_t kHostMatmulMr = 4u;
constexpr std::uint32_t kHostMatmulNr = 8u;
#if defined(__AVX512F__)
constexpr std::uint32_t kHostMatmulAvx512Mr = 8u;
constexpr std::uint32_t kHostMatmulAvx512Nr = 16u;
#endif

std::uint32_t effective_tile_hint(
    const std::uint32_t requested,
    const std::uint32_t fallback,
    const std::uint32_t multiple,
    const std::uint32_t minimum) {
    const auto value = requested == 0u ? fallback : requested;
    return std::max(minimum, ((std::max(value, minimum) + multiple - 1u) / multiple) * multiple);
}

std::size_t choose_linear_chunk(
    const HardwareGraphSummary& summary,
    const std::size_t items,
    const bool reduction_like) {
    const std::size_t vector_bonus = summary.native_vector_bits >= 512u ? 2u : 1u;
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

bool has_fused_name(const OperationSpec& operation, const std::string_view name) {
    return std::find(
               operation.fused_operation_names.begin(),
               operation.fused_operation_names.end(),
               name) != operation.fused_operation_names.end();
}

enum class HostElementwisePattern {
    generic,
    fused_bias_relu,
    fused_mlp_activation
};

HostElementwisePattern select_elementwise_pattern(const OperationSpec& operation) {
    if (has_fused_name(operation, "mlp-activation")) {
        return HostElementwisePattern::fused_mlp_activation;
    }
    if (has_fused_name(operation, "bias") ||
        has_fused_name(operation, "relu") ||
        has_fused_name(operation, "ew-post") ||
        has_fused_name(operation, "post-tonemap")) {
        return HostElementwisePattern::fused_bias_relu;
    }
    return HostElementwisePattern::generic;
}

enum class HostReductionPattern {
    generic,
    fused_epilogue,
    streaming_chain
};

HostReductionPattern select_reduction_pattern(const OperationSpec& operation) {
    if (has_fused_name(operation, "bias") ||
        has_fused_name(operation, "relu") ||
        has_fused_name(operation, "post-tonemap")) {
        return HostReductionPattern::fused_epilogue;
    }
    if (operation.streaming_friendly || has_fused_name(operation, "ew-post")) {
        return HostReductionPattern::streaming_chain;
    }
    return HostReductionPattern::generic;
}

void pack_rhs_panel(
    std::span<const float> rhs,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const std::uint32_t k_begin,
    const std::uint32_t k_count,
    const std::uint32_t col_begin,
    const std::uint32_t col_count,
    const bool rhs_transposed,
    std::span<float> packed_rhs) {
    for (std::uint32_t k = 0u; k < k_count; ++k) {
        for (std::uint32_t n = 0u; n < col_count; ++n) {
            const auto rhs_index = rhs_transposed
                ? static_cast<std::size_t>(col_begin + n) * depth + (k_begin + k)
                : static_cast<std::size_t>(k_begin + k) * columns + (col_begin + n);
            packed_rhs[static_cast<std::size_t>(k) * col_count + n] = rhs[rhs_index];
        }
    }
}

void microkernel_fp32_4x8(
    std::span<const float> lhs,
    const std::uint32_t depth,
    std::span<const float> packed_rhs,
    const std::uint32_t k_count,
    const std::uint32_t k_begin,
    const std::uint32_t row,
    const std::uint32_t col_count,
    const std::uint32_t col_offset,
    const std::uint32_t rows_in_tile,
    const std::uint32_t cols_in_tile,
    std::span<float> output,
    const std::uint32_t columns) {
    __m256 acc[kHostMatmulMr];
    for (std::uint32_t m = 0u; m < kHostMatmulMr; ++m) {
        if (m < rows_in_tile) {
            if (cols_in_tile == kHostMatmulNr) {
                acc[m] = _mm256_loadu_ps(
                    output.data() +
                    (static_cast<std::size_t>(row + m) * columns) +
                    col_offset);
            } else {
                alignas(32) float partial[kHostMatmulNr]{};
                for (std::uint32_t n = 0u; n < cols_in_tile; ++n) {
                    partial[n] =
                        output[(static_cast<std::size_t>(row + m) * columns) + col_offset + n];
                }
                acc[m] = _mm256_load_ps(partial);
            }
        } else {
            acc[m] = _mm256_setzero_ps();
        }
    }

    for (std::uint32_t k = 0u; k < k_count; ++k) {
        alignas(32) float rhs_lane[kHostMatmulNr]{};
        const float* rhs_ptr = packed_rhs.data() + static_cast<std::size_t>(k) * col_count;
        for (std::uint32_t n = 0u; n < cols_in_tile; ++n) {
            rhs_lane[n] = rhs_ptr[n];
        }
        const __m256 rhs_vec = _mm256_load_ps(rhs_lane);
        for (std::uint32_t m = 0u; m < rows_in_tile; ++m) {
            const float lhs_scalar =
                lhs[(static_cast<std::size_t>(row + m) * depth) + k_begin + k];
            acc[m] = _mm256_add_ps(acc[m], _mm256_mul_ps(_mm256_set1_ps(lhs_scalar), rhs_vec));
        }
    }

    for (std::uint32_t m = 0u; m < rows_in_tile; ++m) {
        if (cols_in_tile == kHostMatmulNr) {
            _mm256_storeu_ps(
                output.data() +
                    (static_cast<std::size_t>(row + m) * columns) +
                    col_offset,
                acc[m]);
        } else {
            alignas(32) float partial[kHostMatmulNr]{};
            _mm256_store_ps(partial, acc[m]);
            for (std::uint32_t n = 0u; n < cols_in_tile; ++n) {
                output[(static_cast<std::size_t>(row + m) * columns) + col_offset + n] =
                    partial[n];
            }
        }
    }
}

#if defined(__AVX512F__)
void microkernel_fp32_8x16_avx512(
    std::span<const float> lhs,
    const std::uint32_t depth,
    std::span<const float> packed_rhs,
    const std::uint32_t k_count,
    const std::uint32_t k_begin,
    const std::uint32_t row,
    const std::uint32_t col_count,
    const std::uint32_t col_offset,
    const std::uint32_t rows_in_tile,
    const std::uint32_t cols_in_tile,
    std::span<float> output,
    const std::uint32_t columns) {
    __m512 acc[kHostMatmulAvx512Mr];
    for (std::uint32_t m = 0u; m < kHostMatmulAvx512Mr; ++m) {
        if (m < rows_in_tile) {
            if (cols_in_tile == kHostMatmulAvx512Nr) {
                acc[m] = _mm512_loadu_ps(
                    output.data() +
                    (static_cast<std::size_t>(row + m) * columns) +
                    col_offset);
            } else {
                alignas(64) float partial[kHostMatmulAvx512Nr]{};
                for (std::uint32_t n = 0u; n < cols_in_tile; ++n) {
                    partial[n] =
                        output[(static_cast<std::size_t>(row + m) * columns) + col_offset + n];
                }
                acc[m] = _mm512_load_ps(partial);
            }
        } else {
            acc[m] = _mm512_setzero_ps();
        }
    }

    for (std::uint32_t k = 0u; k < k_count; ++k) {
        alignas(64) float rhs_lane[kHostMatmulAvx512Nr]{};
        const float* rhs_ptr = packed_rhs.data() + static_cast<std::size_t>(k) * col_count;
        for (std::uint32_t n = 0u; n < cols_in_tile; ++n) {
            rhs_lane[n] = rhs_ptr[n];
        }
        const __m512 rhs_vec = _mm512_load_ps(rhs_lane);
        for (std::uint32_t m = 0u; m < rows_in_tile; ++m) {
            const float lhs_scalar =
                lhs[(static_cast<std::size_t>(row + m) * depth) + k_begin + k];
            acc[m] = _mm512_add_ps(acc[m], _mm512_mul_ps(_mm512_set1_ps(lhs_scalar), rhs_vec));
        }
    }

    for (std::uint32_t m = 0u; m < rows_in_tile; ++m) {
        if (cols_in_tile == kHostMatmulAvx512Nr) {
            _mm512_storeu_ps(
                output.data() +
                    (static_cast<std::size_t>(row + m) * columns) +
                    col_offset,
                acc[m]);
        } else {
            alignas(64) float partial[kHostMatmulAvx512Nr]{};
            _mm512_store_ps(partial, acc[m]);
            for (std::uint32_t n = 0u; n < cols_in_tile; ++n) {
                output[(static_cast<std::size_t>(row + m) * columns) + col_offset + n] =
                    partial[n];
            }
        }
    }
}
#endif

void run_fp32_blocked_matmul_rows(
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const bool rhs_transposed,
    const std::uint32_t tile_m,
    const std::uint32_t tile_n,
    const std::uint32_t tile_k,
    const std::size_t row_begin,
    const std::size_t row_end,
    std::span<float> output) {
    (void)rows;
    const auto row_start = static_cast<std::uint32_t>(row_begin);
    const auto row_limit = static_cast<std::uint32_t>(row_end);
    for (std::uint32_t col_block = 0u; col_block < columns; col_block += tile_n) {
        const auto col_count = std::min(tile_n, columns - col_block);
        std::vector<float> packed_rhs(static_cast<std::size_t>(tile_k) * col_count, 0.0f);
        for (std::uint32_t k_block = 0u; k_block < depth; k_block += tile_k) {
            const auto k_count = std::min(tile_k, depth - k_block);
            pack_rhs_panel(
                rhs,
                columns,
                depth,
                k_block,
                k_count,
                col_block,
                col_count,
                rhs_transposed,
                std::span<float>(packed_rhs.data(), static_cast<std::size_t>(k_count) * col_count));

            for (std::uint32_t row_block = row_start; row_block < row_limit; row_block += tile_m) {
                const auto row_block_end = std::min(row_limit, row_block + tile_m);
                for (std::uint32_t row = row_block; row < row_block_end; row += kHostMatmulMr) {
                    const auto rows_in_tile = std::min(kHostMatmulMr, row_block_end - row);
                    for (std::uint32_t n = 0u; n < col_count; n += kHostMatmulNr) {
                        const auto cols_in_tile = std::min(kHostMatmulNr, col_count - n);
                        microkernel_fp32_4x8(
                            lhs,
                            depth,
                            std::span<const float>(
                                packed_rhs.data() + n,
                                static_cast<std::size_t>(k_count) * col_count - n),
                            k_count,
                            k_block,
                            row,
                            col_count,
                            col_block + n,
                            rows_in_tile,
                            cols_in_tile,
                            output,
                            columns);
                    }
                }
            }
        }
    }
}

#if defined(__AVX512F__)
void run_fp32_blocked_matmul_rows_avx512(
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const bool rhs_transposed,
    const std::uint32_t tile_m,
    const std::uint32_t tile_n,
    const std::uint32_t tile_k,
    const std::size_t row_begin,
    const std::size_t row_end,
    std::span<float> output) {
    (void)rows;
    const auto row_start = static_cast<std::uint32_t>(row_begin);
    const auto row_limit = static_cast<std::uint32_t>(row_end);
    for (std::uint32_t col_block = 0u; col_block < columns; col_block += tile_n) {
        const auto col_count = std::min(tile_n, columns - col_block);
        std::vector<float> packed_rhs(static_cast<std::size_t>(tile_k) * col_count, 0.0f);
        for (std::uint32_t k_block = 0u; k_block < depth; k_block += tile_k) {
            const auto k_count = std::min(tile_k, depth - k_block);
            pack_rhs_panel(
                rhs,
                columns,
                depth,
                k_block,
                k_count,
                col_block,
                col_count,
                rhs_transposed,
                std::span<float>(packed_rhs.data(), static_cast<std::size_t>(k_count) * col_count));

            for (std::uint32_t row_block = row_start; row_block < row_limit; row_block += tile_m) {
                const auto row_block_end = std::min(row_limit, row_block + tile_m);
                for (std::uint32_t row = row_block; row < row_block_end; row += kHostMatmulAvx512Mr) {
                    const auto rows_in_tile = std::min(kHostMatmulAvx512Mr, row_block_end - row);
                    for (std::uint32_t n = 0u; n < col_count; n += kHostMatmulAvx512Nr) {
                        const auto cols_in_tile = std::min(kHostMatmulAvx512Nr, col_count - n);
                        microkernel_fp32_8x16_avx512(
                            lhs,
                            depth,
                            std::span<const float>(
                                packed_rhs.data() + n,
                                static_cast<std::size_t>(k_count) * col_count - n),
                            k_count,
                            k_block,
                            row,
                            col_count,
                            col_block + n,
                            rows_in_tile,
                            cols_in_tile,
                            output,
                            columns);
                    }
                }
            }
        }
    }
}
#endif

void run_fp16_matmul_rows(
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const bool rhs_transposed,
    const std::size_t row_begin,
    const std::size_t row_end,
    std::span<float> output) {
    constexpr int rounding = _MM_FROUND_TO_NEAREST_INT;
    for (std::size_t row = row_begin; row < row_end; ++row) {
        const auto lhs_base = static_cast<std::size_t>(row) * depth;
        for (std::uint32_t col = 0; col < columns; ++col) {
            __m256 acc = _mm256_setzero_ps();
            std::uint32_t inner = 0;
            const auto rhs_base = static_cast<std::size_t>(col) * depth;
            for (; inner + 8u <= depth; inner += 8u) {
                const __m256 lhs_vec = _mm256_loadu_ps(lhs.data() + lhs_base + inner);
                const __m256 rhs_vec = rhs_transposed
                    ? _mm256_loadu_ps(rhs.data() + rhs_base + inner)
                    : _mm256_i32gather_ps(
                          rhs.data(),
                          _mm256_setr_epi32(
                              static_cast<int>((inner + 0u) * columns + col),
                              static_cast<int>((inner + 1u) * columns + col),
                              static_cast<int>((inner + 2u) * columns + col),
                              static_cast<int>((inner + 3u) * columns + col),
                              static_cast<int>((inner + 4u) * columns + col),
                              static_cast<int>((inner + 5u) * columns + col),
                              static_cast<int>((inner + 6u) * columns + col),
                              static_cast<int>((inner + 7u) * columns + col)),
                          4);
                const __m128i lhs_half = _mm256_cvtps_ph(lhs_vec, rounding);
                const __m128i rhs_half = _mm256_cvtps_ph(rhs_vec, rounding);
                const __m256 lhs_quant = _mm256_cvtph_ps(lhs_half);
                const __m256 rhs_quant = _mm256_cvtph_ps(rhs_half);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(lhs_quant, rhs_quant));
            }

            float scalar_acc = horizontal_sum(acc);
            for (; inner < depth; ++inner) {
                const float right = rhs_transposed
                    ? rhs[rhs_base + inner]
                    : rhs[static_cast<std::size_t>(inner) * columns + col];
                scalar_acc = quantize_fp16_scalar(
                    scalar_acc +
                    (quantize_fp16_scalar(lhs[lhs_base + inner]) *
                     quantize_fp16_scalar(right)));
            }
            output[static_cast<std::size_t>(row) * columns + col] = scalar_acc;
        }
    }
}

void run_dense_conv_rows(
    std::span<const float> input,
    const std::uint32_t width,
    const std::uint32_t out_width,
    const std::size_t row_begin,
    const std::size_t row_end,
    std::span<float> output) {
    static constexpr std::array<float, 9> kernel{
        0.0625f, 0.125f, 0.0625f,
        0.125f, 0.25f, 0.125f,
        0.0625f, 0.125f, 0.0625f};

    for (std::size_t y = row_begin; y < row_end; ++y) {
        const float* row0 = input.data() + y * width;
        const float* row1 = row0 + width;
        const float* row2 = row1 + width;
        std::size_t x = 0;
        for (; x + 8u <= out_width; x += 8u) {
            __m256 acc = _mm256_setzero_ps();
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(row0 + x + 0u), _mm256_set1_ps(kernel[0])));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(row0 + x + 1u), _mm256_set1_ps(kernel[1])));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(row0 + x + 2u), _mm256_set1_ps(kernel[2])));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(row1 + x + 0u), _mm256_set1_ps(kernel[3])));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(row1 + x + 1u), _mm256_set1_ps(kernel[4])));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(row1 + x + 2u), _mm256_set1_ps(kernel[5])));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(row2 + x + 0u), _mm256_set1_ps(kernel[6])));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(row2 + x + 1u), _mm256_set1_ps(kernel[7])));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(row2 + x + 2u), _mm256_set1_ps(kernel[8])));
            _mm256_storeu_ps(output.data() + (y * out_width) + x, acc);
        }

        for (; x < out_width; ++x) {
            float acc = 0.0f;
            for (std::uint32_t ky = 0; ky < 3u; ++ky) {
                for (std::uint32_t kx = 0; kx < 3u; ++kx) {
                    const auto index = (y + ky) * width + (x + kx);
                    acc += input[index] * kernel[ky * 3u + kx];
                }
            }
            output[(y * out_width) + x] = acc;
        }
    }
}

void run_patch9_conv_rows(
    std::span<const float> input,
    const std::uint32_t out_width,
    const std::size_t row_begin,
    const std::size_t row_end,
    std::span<float> output) {
    static constexpr std::array<float, 9> kernel{
        0.0625f, 0.125f, 0.0625f,
        0.125f, 0.25f, 0.125f,
        0.0625f, 0.125f, 0.0625f};

    for (std::size_t y = row_begin; y < row_end; ++y) {
        std::size_t x = 0;
        for (; x + 8u <= out_width; x += 8u) {
            __m256 acc = _mm256_setzero_ps();
            const auto base = static_cast<std::int32_t>(((y * out_width) + x) * 9u);
            for (std::int32_t k = 0; k < 9; ++k) {
                const __m256 values = gather_strided_8(input.data(), base + k, 9);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(values, _mm256_set1_ps(kernel[static_cast<std::size_t>(k)])));
            }
            _mm256_storeu_ps(output.data() + (y * out_width) + x, acc);
        }

        for (; x < out_width; ++x) {
            const auto base = ((y * out_width) + x) * 9u;
            float acc = 0.0f;
            for (std::size_t k = 0; k < 9u; ++k) {
                acc += input[base + k] * kernel[k];
            }
            output[(y * out_width) + x] = acc;
        }
    }
}

void run_dense_resample_rows(
    std::span<const float> input,
    const std::uint32_t src_h,
    const std::uint32_t src_w,
    const std::uint32_t dst_h,
    const std::uint32_t dst_w,
    const std::uint32_t row_offset,
    const std::size_t row_begin,
    const std::size_t row_end,
    std::span<float> output) {
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 max_x = _mm256_set1_ps(static_cast<float>(src_w - 1u));
    const __m256 src_w_scale = _mm256_set1_ps(static_cast<float>(src_w) / static_cast<float>(dst_w));
    const __m256i max_x_i = _mm256_set1_epi32(static_cast<int>(src_w - 1u));
    const __m256 lane_offsets = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);

    for (std::size_t local_y = row_begin; local_y < row_end; ++local_y) {
        const auto y = row_offset + static_cast<std::uint32_t>(local_y);
        const float src_y =
            (static_cast<float>(y) + 0.5f) * static_cast<float>(src_h) / static_cast<float>(dst_h) - 0.5f;
        const float clamped_y = std::clamp(src_y, 0.0f, static_cast<float>(src_h - 1u));
        const auto y0 = static_cast<std::uint32_t>(clamped_y);
        const auto y1 = std::min(y0 + 1u, src_h - 1u);
        const float wy = clamped_y - static_cast<float>(y0);
        const __m256 wy_vec = _mm256_set1_ps(wy);
        const auto y0_base = static_cast<int>(y0 * src_w);
        const auto y1_base = static_cast<int>(y1 * src_w);

        std::size_t x = 0;
        for (; x + 8u <= dst_w; x += 8u) {
            const __m256 x_vec = _mm256_add_ps(_mm256_set1_ps(static_cast<float>(x)), lane_offsets);
            __m256 src_x = _mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(x_vec, half), src_w_scale), half);
            src_x = _mm256_min_ps(_mm256_max_ps(src_x, zero), max_x);
            const __m256 floored = _mm256_floor_ps(src_x);
            const __m256i x0 = _mm256_cvttps_epi32(floored);
            const __m256i x1 = _mm256_min_epi32(_mm256_add_epi32(x0, _mm256_set1_epi32(1)), max_x_i);
            const __m256 wx = _mm256_sub_ps(src_x, floored);

            const __m256 v00 = _mm256_i32gather_ps(input.data(), _mm256_add_epi32(x0, _mm256_set1_epi32(y0_base)), 4);
            const __m256 v01 = _mm256_i32gather_ps(input.data(), _mm256_add_epi32(x1, _mm256_set1_epi32(y0_base)), 4);
            const __m256 v10 = _mm256_i32gather_ps(input.data(), _mm256_add_epi32(x0, _mm256_set1_epi32(y1_base)), 4);
            const __m256 v11 = _mm256_i32gather_ps(input.data(), _mm256_add_epi32(x1, _mm256_set1_epi32(y1_base)), 4);
            const __m256 top = _mm256_add_ps(v00, _mm256_mul_ps(_mm256_sub_ps(v01, v00), wx));
            const __m256 bottom = _mm256_add_ps(v10, _mm256_mul_ps(_mm256_sub_ps(v11, v10), wx));
            const __m256 value = _mm256_add_ps(top, _mm256_mul_ps(_mm256_sub_ps(bottom, top), wy_vec));
            _mm256_storeu_ps(output.data() + (local_y * dst_w) + x, value);
        }

        for (; x < dst_w; ++x) {
            const float src_x =
                (static_cast<float>(x) + 0.5f) * static_cast<float>(src_w) / static_cast<float>(dst_w) - 0.5f;
            const float clamped_x = std::clamp(src_x, 0.0f, static_cast<float>(src_w - 1u));
            const auto x0 = static_cast<std::uint32_t>(clamped_x);
            const auto x1 = std::min(x0 + 1u, src_w - 1u);
            const float wx = clamped_x - static_cast<float>(x0);
            const float v00 = input[y0 * src_w + x0];
            const float v01 = input[y0 * src_w + x1];
            const float v10 = input[y1 * src_w + x0];
            const float v11 = input[y1 * src_w + x1];
            const float top = v00 + ((v01 - v00) * wx);
            const float bottom = v10 + ((v11 - v10) * wx);
            output[(local_y * dst_w) + x] = top + ((bottom - top) * wy);
        }
    }
}

void run_packed_resample_rows(
    std::span<const float> input,
    const std::uint32_t dst_w,
    const std::size_t row_begin,
    const std::size_t row_end,
    std::span<float> output) {
    for (std::size_t local_y = row_begin; local_y < row_end; ++local_y) {
        std::size_t x = 0;
        for (; x + 8u <= dst_w; x += 8u) {
            const auto base = static_cast<std::int32_t>(((local_y * dst_w) + x) * 6u);
            const __m256 v00 = gather_strided_8(input.data(), base + 0, 6);
            const __m256 v01 = gather_strided_8(input.data(), base + 1, 6);
            const __m256 v10 = gather_strided_8(input.data(), base + 2, 6);
            const __m256 v11 = gather_strided_8(input.data(), base + 3, 6);
            const __m256 wx = gather_strided_8(input.data(), base + 4, 6);
            const __m256 wy = gather_strided_8(input.data(), base + 5, 6);
            const __m256 top = _mm256_add_ps(v00, _mm256_mul_ps(_mm256_sub_ps(v01, v00), wx));
            const __m256 bottom = _mm256_add_ps(v10, _mm256_mul_ps(_mm256_sub_ps(v11, v10), wx));
            const __m256 value = _mm256_add_ps(top, _mm256_mul_ps(_mm256_sub_ps(bottom, top), wy));
            _mm256_storeu_ps(output.data() + (local_y * dst_w) + x, value);
        }

        for (; x < dst_w; ++x) {
            const auto base = ((local_y * dst_w) + x) * 6u;
            const float v00 = input[base + 0u];
            const float v01 = input[base + 1u];
            const float v10 = input[base + 2u];
            const float v11 = input[base + 3u];
            const float wx = input[base + 4u];
            const float wy = input[base + 5u];
            const float top = v00 + ((v01 - v00) * wx);
            const float bottom = v10 + ((v11 - v10) * wx);
            output[(local_y * dst_w) + x] = top + ((bottom - top) * wy);
        }
    }
}

void run_elementwise_fp32_generic(
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::size_t begin,
    const std::size_t end,
    std::span<float> output) {
    const __m256 scale_l = _mm256_set1_ps(1.125f);
    const __m256 scale_r = _mm256_set1_ps(0.25f);
    const __m256 bias = _mm256_set1_ps(0.03125f);
    std::size_t index = begin;
    for (; index + 8u <= end; index += 8u) {
        const __m256 left = _mm256_mul_ps(_mm256_loadu_ps(lhs.data() + index), scale_l);
        const __m256 right = _mm256_mul_ps(_mm256_loadu_ps(rhs.data() + index), scale_r);
        const __m256 value = _mm256_sub_ps(_mm256_add_ps(left, right), bias);
        _mm256_storeu_ps(output.data() + index, value);
    }
    for (; index < end; ++index) {
        output[index] = (lhs[index] * 1.125f) + (rhs[index] * 0.25f) - 0.03125f;
    }
}

void run_elementwise_fp32_bias_relu(
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::size_t begin,
    const std::size_t end,
    std::span<float> output) {
    const __m256 scale_l = _mm256_set1_ps(1.125f);
    const __m256 scale_r = _mm256_set1_ps(0.25f);
    const __m256 neg_bias = _mm256_set1_ps(-0.03125f);
    std::size_t index = begin;
    for (; index + 16u <= end; index += 16u) {
        const __m256 lhs0 = _mm256_mul_ps(_mm256_loadu_ps(lhs.data() + index), scale_l);
        const __m256 rhs0 = _mm256_mul_ps(_mm256_loadu_ps(rhs.data() + index), scale_r);
        const __m256 lhs1 = _mm256_mul_ps(_mm256_loadu_ps(lhs.data() + index + 8u), scale_l);
        const __m256 rhs1 = _mm256_mul_ps(_mm256_loadu_ps(rhs.data() + index + 8u), scale_r);
        _mm256_storeu_ps(output.data() + index, _mm256_add_ps(_mm256_add_ps(lhs0, rhs0), neg_bias));
        _mm256_storeu_ps(output.data() + index + 8u, _mm256_add_ps(_mm256_add_ps(lhs1, rhs1), neg_bias));
    }
    run_elementwise_fp32_generic(lhs, rhs, index, end, output);
}

void run_elementwise_fp32_mlp_activation(
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::size_t begin,
    const std::size_t end,
    std::span<float> output) {
    const __m256 scale_l = _mm256_set1_ps(1.125f);
    const __m256 scale_r = _mm256_set1_ps(0.25f);
    const __m256 bias = _mm256_set1_ps(0.03125f);
    std::size_t index = begin;
    for (; index + 24u <= end; index += 24u) {
        for (std::size_t lane = 0u; lane < 24u; lane += 8u) {
            const __m256 left = _mm256_mul_ps(_mm256_loadu_ps(lhs.data() + index + lane), scale_l);
            const __m256 right = _mm256_mul_ps(_mm256_loadu_ps(rhs.data() + index + lane), scale_r);
            _mm256_storeu_ps(
                output.data() + index + lane,
                _mm256_sub_ps(_mm256_add_ps(left, right), bias));
        }
    }
    run_elementwise_fp32_generic(lhs, rhs, index, end, output);
}

void run_elementwise_fp32(
    const HostElementwisePattern pattern,
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::size_t begin,
    const std::size_t end,
    std::span<float> output) {
    switch (pattern) {
    case HostElementwisePattern::fused_bias_relu:
        run_elementwise_fp32_bias_relu(lhs, rhs, begin, end, output);
        return;
    case HostElementwisePattern::fused_mlp_activation:
        run_elementwise_fp32_mlp_activation(lhs, rhs, begin, end, output);
        return;
    case HostElementwisePattern::generic:
    default:
        run_elementwise_fp32_generic(lhs, rhs, begin, end, output);
        return;
    }
}

float run_reduction_fp32_generic(std::span<const float> input, const std::size_t begin, const std::size_t end) {
    __m256 acc = _mm256_setzero_ps();
    std::size_t index = begin;
    for (; index + 8u <= end; index += 8u) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(input.data() + index));
    }
    float total = horizontal_sum(acc);
    for (; index < end; ++index) {
        total += input[index];
    }
    return total;
}

float run_reduction_fp32_fused_epilogue(
    std::span<const float> input,
    const std::size_t begin,
    const std::size_t end) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::size_t index = begin;
    for (; index + 16u <= end; index += 16u) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(input.data() + index));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(input.data() + index + 8u));
    }
    float total = horizontal_sum(_mm256_add_ps(acc0, acc1));
    for (; index < end; ++index) {
        total += input[index];
    }
    return total;
}

float run_reduction_fp32_streaming_chain(
    std::span<const float> input,
    const std::size_t begin,
    const std::size_t end) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    std::size_t index = begin;
    for (; index + 32u <= end; index += 32u) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(input.data() + index));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(input.data() + index + 8u));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(input.data() + index + 16u));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(input.data() + index + 24u));
    }
    float total = horizontal_sum(
        _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; index < end; ++index) {
        total += input[index];
    }
    return total;
}

float run_reduction_fp32(
    const HostReductionPattern pattern,
    std::span<const float> input,
    const std::size_t begin,
    const std::size_t end) {
    switch (pattern) {
    case HostReductionPattern::fused_epilogue:
        return run_reduction_fp32_fused_epilogue(input, begin, end);
    case HostReductionPattern::streaming_chain:
        return run_reduction_fp32_streaming_chain(input, begin, end);
    case HostReductionPattern::generic:
    default:
        return run_reduction_fp32_generic(input, begin, end);
    }
}

#endif

}  // namespace

bool try_run_host_native_elementwise(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    std::span<const float> lhs,
    std::span<const float> rhs,
    const bool low_precision,
    const std::uint32_t parallel_chunk,
    std::span<float> output) {
#if defined(__AVX2__) || defined(_MSC_VER)
    const auto summary = summarize_graph(graph);
    if (graph.probe != "host" || low_precision || summary.native_vector_bits < 256u || lhs.size() != rhs.size()) {
        return false;
    }
    const auto chunk = std::max<std::size_t>(
        1u,
        parallel_chunk == 0u ? choose_linear_chunk(summary, lhs.size(), false) : parallel_chunk);
    const auto pattern = select_elementwise_pattern(operation);
    HostThreadPool::instance().parallel_for(lhs.size(), chunk, [&](const std::size_t begin, const std::size_t end) {
        run_elementwise_fp32(pattern, lhs, rhs, begin, end, output);
    });
    return true;
#else
    (void)graph;
    (void)operation;
    (void)lhs;
    (void)rhs;
    (void)low_precision;
    (void)parallel_chunk;
    (void)output;
    return false;
#endif
}

bool try_run_host_native_reduction(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    std::span<const float> input,
    const bool low_precision,
    const std::uint32_t parallel_chunk,
    float& output) {
#if defined(__AVX2__) || defined(_MSC_VER)
    const auto summary = summarize_graph(graph);
    if (graph.probe != "host" || low_precision || summary.native_vector_bits < 256u || input.empty()) {
        return false;
    }

    const std::size_t concurrency =
        std::max<std::size_t>(1u, HostThreadPool::instance().worker_count() + 1u);
    const std::size_t chunk = std::max<std::size_t>(
        1u,
        parallel_chunk == 0u ? choose_linear_chunk(summary, input.size(), true) : parallel_chunk);
    const std::size_t block = std::max<std::size_t>(chunk, (input.size() + concurrency - 1u) / concurrency);
    const std::size_t task_count = (input.size() + block - 1u) / block;
    std::vector<float> partials(task_count, 0.0f);
    const auto pattern = select_reduction_pattern(operation);
    HostThreadPool::instance().parallel_for(task_count, 1u, [&](const std::size_t begin, const std::size_t end) {
        for (std::size_t task = begin; task < end; ++task) {
            const auto task_begin = task * block;
            const auto task_end = std::min(input.size(), task_begin + block);
            partials[task] = run_reduction_fp32(pattern, input, task_begin, task_end);
        }
    });
    output = 0.0f;
    for (const auto partial : partials) {
        output += partial;
    }
    return true;
#else
    (void)graph;
    (void)operation;
    (void)input;
    (void)low_precision;
    (void)parallel_chunk;
    (void)output;
    return false;
#endif
}

bool try_run_host_native_matmul(
    const HardwareGraph& graph,
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const bool rhs_transposed,
    const std::uint32_t tile_m,
    const std::uint32_t tile_n,
    const std::uint32_t tile_k,
    const std::uint32_t parallel_chunk_rows,
    std::span<float> output) {
#if defined(__AVX2__) || defined(_MSC_VER)
    const auto summary = summarize_graph(graph);
    if (graph.probe != "host" || summary.native_vector_bits < 256u || depth < 8u) {
        return false;
    }

    const auto effective_tile_m_value = effective_tile_hint(tile_m, kHostMatmulMBlock, kHostMatmulMr, kHostMatmulMr);
    const auto effective_tile_n_value = effective_tile_hint(tile_n, kHostMatmulNBlock, kHostMatmulNr, kHostMatmulNr);
    const auto effective_tile_k_value = effective_tile_hint(tile_k, kHostMatmulKBlock, kHostMatmulNr, kHostMatmulNr);
    const auto min_chunk = std::max<std::size_t>(1u, parallel_chunk_rows == 0u ? 4u : parallel_chunk_rows);
    run_parallel_rows(rows, min_chunk, [&](const std::size_t begin, const std::size_t end) {
#if defined(__AVX512F__)
        if (summary.native_vector_bits >= 512u && columns >= kHostMatmulAvx512Nr && depth >= 16u) {
            const auto avx512_tile_m = effective_tile_hint(tile_m, 64u, kHostMatmulAvx512Mr, kHostMatmulAvx512Mr);
            const auto avx512_tile_n = effective_tile_hint(tile_n, 96u, kHostMatmulAvx512Nr, kHostMatmulAvx512Nr);
            const auto avx512_tile_k = effective_tile_hint(tile_k, 128u, 16u, 16u);
            run_fp32_blocked_matmul_rows_avx512(
                lhs,
                rhs,
                rows,
                columns,
                depth,
                rhs_transposed,
                avx512_tile_m,
                avx512_tile_n,
                avx512_tile_k,
                begin,
                end,
                output);
            return;
        }
#endif
        run_fp32_blocked_matmul_rows(
            lhs,
            rhs,
            rows,
            columns,
            depth,
            rhs_transposed,
            effective_tile_m_value,
            effective_tile_n_value,
            effective_tile_k_value,
            begin,
            end,
            output);
    });
    return true;
#else
    (void)graph;
    (void)lhs;
    (void)rhs;
    (void)rows;
    (void)columns;
    (void)depth;
    (void)rhs_transposed;
    (void)tile_m;
    (void)tile_n;
    (void)tile_k;
    (void)parallel_chunk_rows;
    (void)output;
    return false;
#endif
}

bool try_run_host_native_low_precision_matmul(
    const HardwareGraph& graph,
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const bool rhs_transposed,
    const std::uint32_t parallel_chunk_rows,
    const bool prefer_fp16,
    const bool prefer_bf16,
    std::span<float> output) {
    if (prefer_bf16 &&
        try_run_host_native_bf16_matmul(
            graph,
            lhs,
            rhs,
            rows,
            columns,
            depth,
            rhs_transposed,
            parallel_chunk_rows,
            output)) {
        return true;
    }
#if defined(__AVX2__) || defined(_MSC_VER)
    const auto summary = summarize_graph(graph);
    if (!prefer_fp16 ||
        graph.probe != "host" ||
        !summary.supports_fp16 ||
        summary.native_vector_bits < 256u ||
        depth < 8u) {
        return false;
    }

    const auto min_chunk = std::max<std::size_t>(1u, parallel_chunk_rows == 0u ? 4u : parallel_chunk_rows);
    run_parallel_rows(rows, min_chunk, [&](const std::size_t begin, const std::size_t end) {
        run_fp16_matmul_rows(lhs, rhs, columns, depth, rhs_transposed, begin, end, output);
    });
    return true;
#else
    (void)graph;
    (void)lhs;
    (void)rhs;
    (void)rows;
    (void)columns;
    (void)depth;
    (void)rhs_transposed;
    (void)parallel_chunk_rows;
    (void)prefer_fp16;
    (void)prefer_bf16;
    (void)output;
    return false;
#endif
}

bool try_run_host_native_conv3x3(
    const HardwareGraph& graph,
    std::span<const float> input,
    const std::uint32_t height,
    const std::uint32_t width,
    const bool packed_input,
    const std::uint32_t parallel_chunk_rows,
    std::span<float> output) {
#if defined(__AVX2__) || defined(_MSC_VER)
    const auto summary = summarize_graph(graph);
    if (graph.probe != "host" || summary.native_vector_bits < 256u || height < 3u || width < 3u) {
        return false;
    }

    const auto out_height = height - 2u;
    const auto out_width = width - 2u;
    if ((!packed_input && input.size() < static_cast<std::size_t>(height) * width) ||
        (packed_input && input.size() < static_cast<std::size_t>(out_height) * out_width * 9u)) {
        return false;
    }

    const auto min_chunk = std::max<std::size_t>(1u, parallel_chunk_rows == 0u ? 2u : parallel_chunk_rows);
    run_parallel_rows(out_height, min_chunk, [&](const std::size_t begin, const std::size_t end) {
        if (packed_input) {
            run_patch9_conv_rows(input, out_width, begin, end, output);
        } else {
            run_dense_conv_rows(input, width, out_width, begin, end, output);
        }
    });
    return true;
#else
    (void)graph;
    (void)input;
    (void)height;
    (void)width;
    (void)packed_input;
    (void)parallel_chunk_rows;
    (void)output;
    return false;
#endif
}

bool try_run_host_native_resample(
    const HardwareGraph& graph,
    std::span<const float> input,
    const std::uint32_t src_h,
    const std::uint32_t src_w,
    const std::uint32_t dst_h,
    const std::uint32_t dst_w,
    const std::uint32_t row_offset,
    const std::uint32_t row_count,
    const bool packed_input,
    const std::uint32_t parallel_chunk_rows,
    std::span<float> output) {
#if defined(__AVX2__) || defined(_MSC_VER)
    const auto summary = summarize_graph(graph);
    if (graph.probe != "host" ||
        summary.native_vector_bits < 256u ||
        src_h == 0u ||
        src_w == 0u ||
        dst_h == 0u ||
        dst_w == 0u ||
        row_count == 0u) {
        return false;
    }

    if ((!packed_input && input.size() < static_cast<std::size_t>(src_h) * src_w) ||
        (packed_input && input.size() < static_cast<std::size_t>(row_count) * dst_w * 6u)) {
        return false;
    }

    const auto min_chunk = std::max<std::size_t>(1u, parallel_chunk_rows == 0u ? 2u : parallel_chunk_rows);
    run_parallel_rows(row_count, min_chunk, [&](const std::size_t begin, const std::size_t end) {
        if (packed_input) {
            run_packed_resample_rows(input, dst_w, begin, end, output);
        } else {
            run_dense_resample_rows(input, src_h, src_w, dst_h, dst_w, row_offset, begin, end, output);
        }
    });
    return true;
#else
    (void)graph;
    (void)input;
    (void)src_h;
    (void)src_w;
    (void)dst_h;
    (void)dst_w;
    (void)row_offset;
    (void)row_count;
    (void)packed_input;
    (void)parallel_chunk_rows;
    (void)output;
    return false;
#endif
}

}  // namespace jakal::executors
