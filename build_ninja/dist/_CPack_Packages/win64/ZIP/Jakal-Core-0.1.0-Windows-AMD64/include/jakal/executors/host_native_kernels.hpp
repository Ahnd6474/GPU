#pragma once

#include "jakal/device.hpp"

#include <cstdint>
#include <span>

namespace jakal {
struct OperationSpec;
}

namespace jakal::executors {

bool host_native_kernels_compiled_with_avx512();
bool host_native_avx512_available();

bool try_run_host_native_elementwise(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    std::span<const float> lhs,
    std::span<const float> rhs,
    bool low_precision,
    std::uint32_t parallel_chunk,
    std::span<float> output);

bool try_run_host_native_reduction(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    std::span<const float> input,
    bool low_precision,
    std::uint32_t parallel_chunk,
    float& output);

bool try_run_host_native_matmul(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    std::span<const float> lhs,
    std::span<const float> rhs,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t depth,
    bool rhs_transposed,
    std::uint32_t tile_m,
    std::uint32_t tile_n,
    std::uint32_t tile_k,
    std::uint32_t parallel_chunk_rows,
    std::span<float> output);

bool try_run_host_native_matmul_avx512(
    const HardwareGraph& graph,
    std::span<const float> lhs,
    std::span<const float> rhs,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t depth,
    bool rhs_transposed,
    std::uint32_t tile_m,
    std::uint32_t tile_n,
    std::uint32_t tile_k,
    std::uint32_t parallel_chunk_rows,
    std::span<float> output);

bool try_run_host_native_low_precision_matmul(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    std::span<const float> lhs,
    std::span<const float> rhs,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t depth,
    bool rhs_transposed,
    std::uint32_t parallel_chunk_rows,
    bool prefer_fp16,
    bool prefer_bf16,
    std::span<float> output);

bool try_run_host_native_bf16_matmul(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    std::span<const float> lhs,
    std::span<const float> rhs,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t depth,
    bool rhs_transposed,
    std::uint32_t parallel_chunk_rows,
    std::span<float> output);

bool try_run_host_native_conv3x3(
    const HardwareGraph& graph,
    std::span<const float> input,
    std::uint32_t height,
    std::uint32_t width,
    bool packed_input,
    std::uint32_t parallel_chunk_rows,
    std::span<float> output);

bool try_run_host_native_resample(
    const HardwareGraph& graph,
    std::span<const float> input,
    std::uint32_t src_h,
    std::uint32_t src_w,
    std::uint32_t dst_h,
    std::uint32_t dst_w,
    std::uint32_t row_offset,
    std::uint32_t row_count,
    bool packed_input,
    std::uint32_t parallel_chunk_rows,
    std::span<float> output);

}  // namespace jakal::executors
