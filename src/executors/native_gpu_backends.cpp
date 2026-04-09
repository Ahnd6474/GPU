#include "jakal/executors/native_gpu_backend.hpp"

#include "jakal/executors/direct_backends.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace jakal::executors {
namespace {

#if defined(_WIN32)
using LibraryHandle = HMODULE;
LibraryHandle load_library(const char* name) { return LoadLibraryA(name); }
std::optional<std::string> env_value(const char* name) {
    char* buffer = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&buffer, &length, name) != 0 || buffer == nullptr || length <= 1) {
        std::free(buffer);
        return std::nullopt;
    }
    std::string value(buffer);
    std::free(buffer);
    return value;
}
std::vector<std::filesystem::path> candidate_cuda_binary_dirs() {
    std::vector<std::filesystem::path> paths;
    if (const auto cuda_path = env_value("CUDA_PATH")) {
        paths.emplace_back(*cuda_path);
    }
    if (const auto cuda_path = env_value("CUDA_PATH_V13_2")) {
        paths.emplace_back(*cuda_path);
    }
    paths.emplace_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2");
    paths.emplace_back("D:\\");

    std::vector<std::filesystem::path> binary_dirs;
    binary_dirs.reserve(paths.size() * 2u);
    for (const auto& root : paths) {
        if (root.empty()) {
            continue;
        }
        binary_dirs.push_back(root / "bin");
        binary_dirs.push_back(root / "bin" / "x64");
    }
    binary_dirs.push_back("D:\\bin");
    binary_dirs.push_back("D:\\bin\\x64");
    return binary_dirs;
}
LibraryHandle load_library_with_fallbacks(std::initializer_list<const char*> names) {
    for (const char* name : names) {
        if (LibraryHandle handle = load_library(name); handle != nullptr) {
            return handle;
        }
        for (const auto& directory : candidate_cuda_binary_dirs()) {
            const auto candidate = directory / name;
            if (!std::filesystem::exists(candidate)) {
                continue;
            }
            if (LibraryHandle handle = LoadLibraryA(candidate.string().c_str()); handle != nullptr) {
                return handle;
            }
        }
    }
    return nullptr;
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
LibraryHandle load_library(const char* name) { return dlopen(name, RTLD_LAZY); }
void* load_symbol(LibraryHandle library, const char* name) { return dlsym(library, name); }
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

template <typename T>
T round_up(const T value, const T alignment) {
    return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}

constexpr std::uint32_t kNativeTileSize = 16u;
constexpr std::uint32_t kNativeReductionGroupSize = 256u;

std::vector<std::filesystem::path> candidate_ocloc_paths() {
    std::vector<std::filesystem::path> paths;
#if defined(_WIN32)
    paths.emplace_back("C:\\Program Files (x86)\\Intel\\oneAPI\\2025.1\\bin\\ocloc.exe");
    paths.emplace_back("C:\\Program Files (x86)\\Intel\\oneAPI\\ocloc\\2025.1\\bin\\ocloc.exe");
    paths.emplace_back("C:\\Program Files (x86)\\Intel\\oneAPI\\latest\\bin\\ocloc.exe");
#else
    paths.emplace_back("/opt/intel/oneapi/compiler/latest/bin/ocloc");
#endif
    return paths;
}

std::optional<std::filesystem::path> locate_ocloc() {
    for (const auto& path : candidate_ocloc_paths()) {
        std::error_code error;
        if (std::filesystem::exists(path, error)) {
            return path;
        }
    }
    return std::nullopt;
}

bool write_text_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return false;
    }
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    return stream.good();
}

std::vector<std::uint8_t> read_binary_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return {};
    }
    stream.seekg(0, std::ios::end);
    const auto size = stream.tellg();
    if (size <= 0) {
        return {};
    }
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!stream.good() && !stream.eof()) {
        return {};
    }
    return bytes;
}

int run_command(const std::string& command) {
    return std::system(command.c_str());
}

bool gpu_rhs_uses_transposed_layout(const OperationSpec& operation) {
    return operation.gpu_pack_weights || operation.gpu_pretranspose_rhs;
}

bool gpu_conv_uses_patch9_layout(const OperationSpec& operation) {
    return operation.op_class == OperationClass::convolution_2d &&
           operation.gpu_input_layout.find("conv-patch9") != std::string::npos;
}

bool gpu_resample_uses_packed6_layout(const OperationSpec& operation) {
    return operation.op_class == OperationClass::resample_2d &&
           operation.gpu_input_layout.find("resample-packed6") != std::string::npos;
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
            (summary.supports_asynchronous_dispatch ? 0.12 : 0.0) +
            (summary.unified_address_space ? 0.10 : 0.0),
        0.05,
        0.88);
    breakdown.residual_copy_runtime_us =
        breakdown.copy_runtime_us * (1.0 - breakdown.overlap_ratio);
    breakdown.compute_runtime_us =
        std::max(0.25, std::max(tail_runtime_us - breakdown.residual_copy_runtime_us, tail_runtime_us * 0.35));
    return breakdown;
}

constexpr const char* kCudaLikeProgramSource = R"GPU(
extern "C" __device__ __forceinline__ float q(float value, int low_precision) {
  if (!low_precision) return value;
  const float scaled = value * 1024.0f;
  const float rounded = scaled >= 0.0f ? floorf(scaled + 0.5f) : -floorf((-scaled) + 0.5f);
  return rounded / 1024.0f;
}
extern "C" __global__ void elementwise_map(const float* lhs,const float* rhs,float* out,unsigned int count,int low_precision) {
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  const float left = q(lhs[gid] * 1.125f, low_precision);
  const float right = q(rhs[gid] * 0.25f, low_precision);
  out[gid] = q(left + right - 0.03125f, low_precision);
}
extern "C" __global__ void reduce_sum(const float* input,float* partials,unsigned int count,int low_precision) {
  extern __shared__ float scratch[];
  const unsigned int lid = threadIdx.x;
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;
  float value = 0.0f;
  for (unsigned int index = gid; index < count; index += stride) value = q(value + input[index], low_precision);
  scratch[lid] = value;
  __syncthreads();
  for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (lid < offset) scratch[lid] = q(scratch[lid] + scratch[lid + offset], low_precision);
    __syncthreads();
  }
  if (lid == 0) partials[blockIdx.x] = scratch[0];
}
extern "C" __global__ void matmul_tiled(const float* lhs,const float* rhs,float* out,unsigned int rows,unsigned int columns,unsigned int depth,int low_precision) {
  __shared__ float lhs_tile[16][16];
  __shared__ float rhs_tile[16][16];
  const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  float acc = 0.0f;
  for (unsigned int base = 0; base < depth; base += 16) {
    lhs_tile[threadIdx.y][threadIdx.x] = (row < rows && (base + threadIdx.x) < depth) ? lhs[row * depth + base + threadIdx.x] : 0.0f;
    rhs_tile[threadIdx.y][threadIdx.x] = (col < columns && (base + threadIdx.y) < depth) ? rhs[(base + threadIdx.y) * columns + col] : 0.0f;
    __syncthreads();
    for (unsigned int inner = 0; inner < 16; ++inner) acc = q(acc + lhs_tile[threadIdx.y][inner] * rhs_tile[inner][threadIdx.x], low_precision);
    __syncthreads();
  }
  if (row < rows && col < columns) out[row * columns + col] = q(acc, low_precision);
}
extern "C" __global__ void matmul_tiled_rhs_t(const float* lhs,const float* rhs,float* out,unsigned int rows,unsigned int columns,unsigned int depth,int low_precision) {
  __shared__ float lhs_tile[16][16];
  __shared__ float rhs_tile[16][16];
  const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  float acc = 0.0f;
  for (unsigned int base = 0; base < depth; base += 16) {
    lhs_tile[threadIdx.y][threadIdx.x] = (row < rows && (base + threadIdx.x) < depth) ? lhs[row * depth + base + threadIdx.x] : 0.0f;
    rhs_tile[threadIdx.y][threadIdx.x] = (col < columns && (base + threadIdx.y) < depth) ? rhs[col * depth + base + threadIdx.y] : 0.0f;
    __syncthreads();
    for (unsigned int inner = 0; inner < 16; ++inner) acc = q(acc + lhs_tile[threadIdx.y][inner] * rhs_tile[inner][threadIdx.x], low_precision);
    __syncthreads();
  }
  if (row < rows && col < columns) out[row * columns + col] = q(acc, low_precision);
}
extern "C" __global__ void conv3x3_valid(const float* input,float* output,unsigned int height,unsigned int width,int low_precision) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int out_width = width - 2u;
  const unsigned int out_height = height - 2u;
  if (x >= out_width || y >= out_height) return;
  const float kernel[9] = {0.0625f,0.125f,0.0625f,0.125f,0.25f,0.125f,0.0625f,0.125f,0.0625f};
  float acc = 0.0f;
  for (unsigned int ky = 0; ky < 3; ++ky) for (unsigned int kx = 0; kx < 3; ++kx) acc = q(acc + q(input[(y + ky) * width + (x + kx)], low_precision) * kernel[ky * 3u + kx], low_precision);
  output[y * out_width + x] = acc;
}
extern "C" __global__ void conv3x3_valid_patch9(const float* input,float* output,unsigned int height,unsigned int width,int low_precision) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int out_width = width - 2u;
  const unsigned int out_height = height - 2u;
  if (x >= out_width || y >= out_height) return;
  const float kernel[9] = {0.0625f,0.125f,0.0625f,0.125f,0.25f,0.125f,0.0625f,0.125f,0.0625f};
  const unsigned int base = (y * out_width + x) * 9u;
  float acc = 0.0f;
  for (unsigned int index = 0; index < 9u; ++index) acc = q(acc + q(input[base + index], low_precision) * kernel[index], low_precision);
  output[y * out_width + x] = acc;
}
extern "C" __global__ void bilinear_resample(const float* input,float* output,unsigned int src_h,unsigned int src_w,unsigned int dst_h,unsigned int dst_w,unsigned int dst_y_offset,unsigned int shard_rows,int low_precision) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= shard_rows) return;
  float global_y = (float)(y + dst_y_offset);
  float src_y = ((global_y + 0.5f) * (float)src_h / (float)dst_h) - 0.5f;
  src_y = src_y < 0.0f ? 0.0f : (src_y > (float)(src_h - 1u) ? (float)(src_h - 1u) : src_y);
  const unsigned int y0 = (unsigned int)src_y;
  const unsigned int y1 = y0 + 1u < src_h ? y0 + 1u : src_h - 1u;
  const float wy = src_y - (float)y0;
  float src_x = (((float)x + 0.5f) * (float)src_w / (float)dst_w) - 0.5f;
  src_x = src_x < 0.0f ? 0.0f : (src_x > (float)(src_w - 1u) ? (float)(src_w - 1u) : src_x);
  const unsigned int x0 = (unsigned int)src_x;
  const unsigned int x1 = x0 + 1u < src_w ? x0 + 1u : src_w - 1u;
  const float wx = src_x - (float)x0;
  const float v00 = q(input[y0 * src_w + x0], low_precision);
  const float v01 = q(input[y0 * src_w + x1], low_precision);
  const float v10 = q(input[y1 * src_w + x0], low_precision);
  const float v11 = q(input[y1 * src_w + x1], low_precision);
  const float top = q(v00 + ((v01 - v00) * wx), low_precision);
  const float bottom = q(v10 + ((v11 - v10) * wx), low_precision);
  output[y * dst_w + x] = q(top + ((bottom - top) * wy), low_precision);
}
extern "C" __global__ void bilinear_resample_packed6(const float* input,float* output,unsigned int src_h,unsigned int src_w,unsigned int dst_h,unsigned int dst_w,unsigned int dst_y_offset,unsigned int shard_rows,int low_precision) {
  (void)src_h; (void)src_w; (void)dst_h; (void)dst_y_offset;
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= shard_rows) return;
  const unsigned int base = (y * dst_w + x) * 6u;
  const float v00 = q(input[base + 0u], low_precision);
  const float v01 = q(input[base + 1u], low_precision);
  const float v10 = q(input[base + 2u], low_precision);
  const float v11 = q(input[base + 3u], low_precision);
  const float wx = input[base + 4u];
  const float wy = input[base + 5u];
  const float top = q(v00 + ((v01 - v00) * wx), low_precision);
  const float bottom = q(v10 + ((v11 - v10) * wx), low_precision);
  output[y * dst_w + x] = q(top + ((bottom - top) * wy), low_precision);
}
)GPU";

constexpr const char* kOpenClProgramSource = R"CLC(
#ifndef JAKAL_ALWAYS_LOW_PRECISION
#define JAKAL_ALWAYS_LOW_PRECISION 0
#endif
inline float q(float value, int low_precision) {
  (void)low_precision;
  if (!JAKAL_ALWAYS_LOW_PRECISION) return value;
  float scaled = value * 1024.0f;
  int rounded = (int)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
  return ((float)rounded) / 1024.0f;
}
__kernel void elementwise_map(__global const float* lhs,__global const float* rhs,__global float* out,uint count,int low_precision) {
  uint gid = get_global_id(0); if (gid >= count) return;
  float left = q(lhs[gid] * 1.125f, low_precision); float right = q(rhs[gid] * 0.25f, low_precision);
  out[gid] = q(left + right - 0.03125f, low_precision);
}
__kernel void reduce_sum(__global const float* input,__global float* partials,uint count,int low_precision,__local float* scratch) {
  uint lid = get_local_id(0); uint gid = get_global_id(0); uint global_size = get_global_size(0); float value = 0.0f;
  for (uint index = gid; index < count; index += global_size) value = q(value + input[index], low_precision);
  scratch[lid] = value; barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = get_local_size(0) / 2; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] = q(scratch[lid] + scratch[lid + stride], low_precision); barrier(CLK_LOCAL_MEM_FENCE); }
  if (lid == 0) partials[get_group_id(0)] = scratch[0];
}
__kernel void matmul_tiled(__global const float* lhs,__global const float* rhs,__global float* out,uint rows,uint columns,uint depth,int low_precision,__local float* lhs_tile,__local float* rhs_tile) {
  uint col = get_global_id(0); uint row = get_global_id(1); uint lcol = get_local_id(0); uint lrow = get_local_id(1); uint tile = get_local_size(0); float acc = 0.0f;
  for (uint base = 0; base < depth; base += tile) {
    uint lhs_index = row * depth + base + lcol; uint rhs_index = (base + lrow) * columns + col;
    lhs_tile[lrow * tile + lcol] = (row < rows && (base + lcol) < depth) ? lhs[lhs_index] : 0.0f;
    rhs_tile[lrow * tile + lcol] = (col < columns && (base + lrow) < depth) ? rhs[rhs_index] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint inner = 0; inner < tile; ++inner) acc = q(acc + lhs_tile[lrow * tile + inner] * rhs_tile[inner * tile + lcol], low_precision);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (row < rows && col < columns) out[row * columns + col] = q(acc, low_precision);
}
__kernel void matmul_tiled_rhs_t(__global const float* lhs,__global const float* rhs,__global float* out,uint rows,uint columns,uint depth,int low_precision,__local float* lhs_tile,__local float* rhs_tile) {
  uint col = get_global_id(0); uint row = get_global_id(1); uint lcol = get_local_id(0); uint lrow = get_local_id(1); uint tile = get_local_size(0); float acc = 0.0f;
  for (uint base = 0; base < depth; base += tile) {
    uint lhs_index = row * depth + base + lcol; uint rhs_index = col * depth + base + lrow;
    lhs_tile[lrow * tile + lcol] = (row < rows && (base + lcol) < depth) ? lhs[lhs_index] : 0.0f;
    rhs_tile[lrow * tile + lcol] = (col < columns && (base + lrow) < depth) ? rhs[rhs_index] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint inner = 0; inner < tile; ++inner) acc = q(acc + lhs_tile[lrow * tile + inner] * rhs_tile[inner * tile + lcol], low_precision);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (row < rows && col < columns) out[row * columns + col] = q(acc, low_precision);
}
__kernel void conv3x3_valid(__global const float* input,__global float* output,uint height,uint width,int low_precision) {
  uint x = get_global_id(0); uint y = get_global_id(1); uint out_width = width - 2u; uint out_height = height - 2u; if (x >= out_width || y >= out_height) return;
  float acc = 0.0f;
  for (uint ky = 0; ky < 3; ++ky) {
    for (uint kx = 0; kx < 3; ++kx) {
      float weight = 0.125f;
      if ((ky == 1u) && (kx == 1u)) weight = 0.25f;
      else if ((ky != 1u) && (kx != 1u)) weight = 0.0625f;
      acc = q(acc + q(input[(y + ky) * width + (x + kx)], low_precision) * weight, low_precision);
    }
  }
  output[y * out_width + x] = acc;
}
__kernel void conv3x3_valid_patch9(__global const float* input,__global float* output,uint height,uint width,int low_precision) {
  uint x = get_global_id(0); uint y = get_global_id(1); uint out_width = width - 2u; uint out_height = height - 2u; if (x >= out_width || y >= out_height) return;
  float acc = 0.0f; uint base = (y * out_width + x) * 9u;
  for (uint index = 0; index < 9u; ++index) {
    float weight = 0.125f;
    uint ky = index / 3u; uint kx = index % 3u;
    if ((ky == 1u) && (kx == 1u)) weight = 0.25f;
    else if ((ky != 1u) && (kx != 1u)) weight = 0.0625f;
    acc = q(acc + q(input[base + index], low_precision) * weight, low_precision);
  }
  output[y * out_width + x] = acc;
}
__kernel void bilinear_resample(__global const float* input,__global float* output,uint src_h,uint src_w,uint dst_h,uint dst_w,uint dst_y_offset,uint shard_rows,int low_precision) {
  uint x = get_global_id(0); uint y = get_global_id(1); if (x >= dst_w || y >= shard_rows) return;
  float global_y = (float)(y + dst_y_offset); float src_y = ((global_y + 0.5f) * (float)src_h / (float)dst_h) - 0.5f; src_y = clamp(src_y, 0.0f, (float)(src_h - 1u));
  uint y0 = (uint)src_y; uint y1 = min(y0 + 1u, src_h - 1u); float wy = src_y - (float)y0;
  float src_x = (((float)x + 0.5f) * (float)src_w / (float)dst_w) - 0.5f; src_x = clamp(src_x, 0.0f, (float)(src_w - 1u));
  uint x0 = (uint)src_x; uint x1 = min(x0 + 1u, src_w - 1u); float wx = src_x - (float)x0;
  float v00 = q(input[y0 * src_w + x0], low_precision); float v01 = q(input[y0 * src_w + x1], low_precision); float v10 = q(input[y1 * src_w + x0], low_precision); float v11 = q(input[y1 * src_w + x1], low_precision);
  float top = q(v00 + ((v01 - v00) * wx), low_precision); float bottom = q(v10 + ((v11 - v10) * wx), low_precision);
  output[y * dst_w + x] = q(top + ((bottom - top) * wy), low_precision);
}
__kernel void bilinear_resample_packed6(__global const float* input,__global float* output,uint src_h,uint src_w,uint dst_h,uint dst_w,uint dst_y_offset,uint shard_rows,int low_precision) {
  (void)src_h; (void)src_w; (void)dst_h; (void)dst_y_offset;
  uint x = get_global_id(0); uint y = get_global_id(1); if (x >= dst_w || y >= shard_rows) return;
  uint base = (y * dst_w + x) * 6u;
  float v00 = q(input[base + 0u], low_precision); float v01 = q(input[base + 1u], low_precision); float v10 = q(input[base + 2u], low_precision); float v11 = q(input[base + 3u], low_precision);
  float wx = input[base + 4u]; float wy = input[base + 5u];
  float top = q(v00 + ((v01 - v00) * wx), low_precision); float bottom = q(v10 + ((v11 - v10) * wx), low_precision);
  output[y * dst_w + x] = q(top + ((bottom - top) * wy), low_precision);
}
)CLC";

constexpr const char* kOpenClFp16ProgramSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
inline float q(float value) {
  return convert_float(convert_half_rte(value));
}
__kernel void elementwise_map(__global const float* lhs,__global const float* rhs,__global float* out,uint count,int low_precision) {
  (void)low_precision;
  uint gid = get_global_id(0); if (gid >= count) return;
  float left = q(lhs[gid] * 1.125f); float right = q(rhs[gid] * 0.25f);
  out[gid] = q(left + right - 0.03125f);
}
__kernel void reduce_sum(__global const float* input,__global float* partials,uint count,int low_precision,__local float* scratch) {
  (void)low_precision;
  uint lid = get_local_id(0); uint gid = get_global_id(0); uint global_size = get_global_size(0); float value = 0.0f;
  for (uint index = gid; index < count; index += global_size) value = q(value + q(input[index]));
  scratch[lid] = value; barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = get_local_size(0) / 2; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] = q(scratch[lid] + scratch[lid + stride]); barrier(CLK_LOCAL_MEM_FENCE); }
  if (lid == 0) partials[get_group_id(0)] = scratch[0];
}
__kernel void matmul_tiled(__global const float* lhs,__global const float* rhs,__global float* out,uint rows,uint columns,uint depth,int low_precision,__local float* lhs_tile,__local float* rhs_tile) {
  (void)low_precision;
  uint col = get_global_id(0); uint row = get_global_id(1); uint lcol = get_local_id(0); uint lrow = get_local_id(1); uint tile = get_local_size(0); float acc = 0.0f;
  for (uint base = 0; base < depth; base += tile) {
    uint lhs_index = row * depth + base + lcol; uint rhs_index = (base + lrow) * columns + col;
    lhs_tile[lrow * tile + lcol] = (row < rows && (base + lcol) < depth) ? q(lhs[lhs_index]) : 0.0f;
    rhs_tile[lrow * tile + lcol] = (col < columns && (base + lrow) < depth) ? q(rhs[rhs_index]) : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint inner = 0; inner < tile; ++inner) acc = q(acc + q(lhs_tile[lrow * tile + inner] * rhs_tile[inner * tile + lcol]));
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (row < rows && col < columns) out[row * columns + col] = q(acc);
}
__kernel void matmul_tiled_rhs_t(__global const float* lhs,__global const float* rhs,__global float* out,uint rows,uint columns,uint depth,int low_precision,__local float* lhs_tile,__local float* rhs_tile) {
  (void)low_precision;
  uint col = get_global_id(0); uint row = get_global_id(1); uint lcol = get_local_id(0); uint lrow = get_local_id(1); uint tile = get_local_size(0); float acc = 0.0f;
  for (uint base = 0; base < depth; base += tile) {
    uint lhs_index = row * depth + base + lcol; uint rhs_index = col * depth + base + lrow;
    lhs_tile[lrow * tile + lcol] = (row < rows && (base + lcol) < depth) ? q(lhs[lhs_index]) : 0.0f;
    rhs_tile[lrow * tile + lcol] = (col < columns && (base + lrow) < depth) ? q(rhs[rhs_index]) : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint inner = 0; inner < tile; ++inner) acc = q(acc + q(lhs_tile[lrow * tile + inner] * rhs_tile[inner * tile + lcol]));
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (row < rows && col < columns) out[row * columns + col] = q(acc);
}
__kernel void conv3x3_valid(__global const float* input,__global float* output,uint height,uint width,int low_precision) {
  (void)low_precision;
  uint x = get_global_id(0); uint y = get_global_id(1); uint out_width = width - 2u; uint out_height = height - 2u; if (x >= out_width || y >= out_height) return;
  float acc = 0.0f;
  for (uint ky = 0; ky < 3; ++ky) {
    for (uint kx = 0; kx < 3; ++kx) {
      float weight = 0.125f;
      if ((ky == 1u) && (kx == 1u)) weight = 0.25f;
      else if ((ky != 1u) && (kx != 1u)) weight = 0.0625f;
      acc = q(acc + q(q(input[(y + ky) * width + (x + kx)]) * q(weight)));
    }
  }
  output[y * out_width + x] = q(acc);
}
__kernel void conv3x3_valid_patch9(__global const float* input,__global float* output,uint height,uint width,int low_precision) {
  (void)low_precision;
  uint x = get_global_id(0); uint y = get_global_id(1); uint out_width = width - 2u; uint out_height = height - 2u; if (x >= out_width || y >= out_height) return;
  float acc = 0.0f; uint base = (y * out_width + x) * 9u;
  for (uint index = 0; index < 9u; ++index) {
    float weight = 0.125f;
    uint ky = index / 3u; uint kx = index % 3u;
    if ((ky == 1u) && (kx == 1u)) weight = 0.25f;
    else if ((ky != 1u) && (kx != 1u)) weight = 0.0625f;
    acc = q(acc + q(q(input[base + index]) * q(weight)));
  }
  output[y * out_width + x] = q(acc);
}
__kernel void bilinear_resample(__global const float* input,__global float* output,uint src_h,uint src_w,uint dst_h,uint dst_w,uint dst_y_offset,uint shard_rows,int low_precision) {
  (void)low_precision;
  uint x = get_global_id(0); uint y = get_global_id(1); if (x >= dst_w || y >= shard_rows) return;
  float global_y = (float)(y + dst_y_offset); float src_y = ((global_y + 0.5f) * (float)src_h / (float)dst_h) - 0.5f; src_y = clamp(src_y, 0.0f, (float)(src_h - 1u));
  uint y0 = (uint)src_y; uint y1 = min(y0 + 1u, src_h - 1u); float wy = q(src_y - (float)y0);
  float src_x = (((float)x + 0.5f) * (float)src_w / (float)dst_w) - 0.5f; src_x = clamp(src_x, 0.0f, (float)(src_w - 1u));
  uint x0 = (uint)src_x; uint x1 = min(x0 + 1u, src_w - 1u); float wx = q(src_x - (float)x0);
  float v00 = q(input[y0 * src_w + x0]); float v01 = q(input[y0 * src_w + x1]); float v10 = q(input[y1 * src_w + x0]); float v11 = q(input[y1 * src_w + x1]);
  float top = q(v00 + q((v01 - v00) * wx)); float bottom = q(v10 + q((v11 - v10) * wx));
  output[y * dst_w + x] = q(top + q((bottom - top) * wy));
}
__kernel void bilinear_resample_packed6(__global const float* input,__global float* output,uint src_h,uint src_w,uint dst_h,uint dst_w,uint dst_y_offset,uint shard_rows,int low_precision) {
  (void)low_precision; (void)src_h; (void)src_w; (void)dst_h; (void)dst_y_offset;
  uint x = get_global_id(0); uint y = get_global_id(1); if (x >= dst_w || y >= shard_rows) return;
  uint base = (y * dst_w + x) * 6u;
  float v00 = q(input[base + 0u]); float v01 = q(input[base + 1u]); float v10 = q(input[base + 2u]); float v11 = q(input[base + 3u]);
  float wx = input[base + 4u]; float wy = input[base + 5u];
  float top = q(v00 + q((v01 - v00) * wx)); float bottom = q(v10 + q((v11 - v10) * wx));
  output[y * dst_w + x] = q(top + q((bottom - top) * wy));
}
)CLC";

class NativeKernelBackendBase : public IKernelBackend {
public:
    NativeKernelBackendBase(const JakalBackendKind backend, std::string probe_name)
        : backend_(backend),
          probe_name_(std::move(probe_name)),
          host_(make_host_kernel_backend()) {}

    [[nodiscard]] bool matches(const HardwareGraph& graph) const override { return graph.probe == probe_name_; }
    [[nodiscard]] std::string name() const override { return to_string(backend_) + "-native"; }
    [[nodiscard]] bool supports_async_dispatch(const HardwareGraph& graph) const override {
        return summarize_graph(graph).supports_asynchronous_dispatch;
    }

    BackendRunResult run_elementwise(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> lhs, const std::span<const float> rhs, const bool low_precision) const override {
        auto result = run_elementwise_native(graph, lhs, rhs, low_precision);
        if (result.success) {
            return mark_async_result(
                graph,
                "elementwise",
                std::move(result),
                {},
                lhs.size_bytes() + rhs.size_bytes(),
                lhs.size_bytes(),
                0.44);
        }
        auto fallback = host_->run_elementwise(graph, operation, lhs, rhs, low_precision);
        if (fallback.error.empty()) fallback.error = result.error;
        return fallback;
    }
    BackendRunResult run_reduction(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> input, const bool low_precision) const override {
        auto result = run_reduction_native(graph, input, low_precision);
        if (result.success) {
            return mark_async_result(
                graph,
                "reduction",
                std::move(result),
                {},
                input.size_bytes(),
                sizeof(float),
                0.40);
        }
        auto fallback = host_->run_reduction(graph, operation, input, low_precision);
        if (fallback.error.empty()) fallback.error = result.error;
        return fallback;
    }
    BackendRunResult run_matmul(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> lhs, const std::span<const float> rhs, const std::uint32_t rows, const std::uint32_t columns, const std::uint32_t depth, const bool low_precision) const override {
        auto result = run_matmul_native(graph, operation, lhs, rhs, rows, columns, depth, low_precision);
        if (result.success) {
            return mark_async_result(
                graph,
                "matmul",
                std::move(result),
                gpu_rhs_uses_transposed_layout(operation) ? "packed-rhs" : "dense-rhs",
                lhs.size_bytes() + rhs.size_bytes(),
                static_cast<std::size_t>(rows) * columns * sizeof(float),
                0.60);
        }
        auto fallback = host_->run_matmul(graph, operation, lhs, rhs, rows, columns, depth, low_precision);
        if (fallback.error.empty()) fallback.error = result.error;
        return fallback;
    }
    BackendRunResult run_conv3x3(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> input, const std::uint32_t height, const std::uint32_t width, const bool low_precision) const override {
        auto result = run_conv3x3_native(graph, operation, input, height, width, low_precision);
        if (result.success) {
            return mark_async_result(
                graph,
                "conv3x3",
                std::move(result),
                gpu_conv_uses_patch9_layout(operation) ? "conv-patch9" : "conv-dense",
                input.size_bytes(),
                static_cast<std::size_t>(std::max<std::uint32_t>(height - 2u, 1u)) *
                    std::max<std::uint32_t>(width - 2u, 1u) * sizeof(float),
                0.50);
        }
        auto fallback = host_->run_conv3x3(graph, operation, input, height, width, low_precision);
        if (fallback.error.empty()) fallback.error = result.error;
        return fallback;
    }
    BackendRunResult run_resample(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> input, const std::uint32_t src_h, const std::uint32_t src_w, const std::uint32_t dst_h, const std::uint32_t dst_w, const std::uint32_t row_offset, const std::uint32_t row_count, const bool low_precision) const override {
        auto result = run_resample_native(graph, operation, input, src_h, src_w, dst_h, dst_w, row_offset, row_count, low_precision);
        if (result.success) {
            return mark_async_result(
                graph,
                "resample",
                std::move(result),
                gpu_resample_uses_packed6_layout(operation) ? "resample-packed6" : "resample-dense",
                input.size_bytes(),
                static_cast<std::size_t>(row_count) * dst_w * sizeof(float),
                0.54);
        }
        auto fallback = host_->run_resample(graph, operation, input, src_h, src_w, dst_h, dst_w, row_offset, row_count, low_precision);
        if (fallback.error.empty()) fallback.error = result.error;
        return fallback;
    }

protected:
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

    [[nodiscard]] BackendRunResult failure(std::string error) const { BackendRunResult result; result.error = std::move(error); return result; }
    [[nodiscard]] std::uint32_t record_persistent_dispatch_reuse(
        const HardwareGraph& graph,
        const std::string_view kernel_tag) const {
        const std::string key = graph.uid + "|" + std::string(kernel_tag);
        const auto revision = structural_fingerprint(graph);
        std::scoped_lock lock(persistent_dispatch_mutex_);
        auto& entry = persistent_dispatch_cache_[key];
        if (entry.revision != revision) {
            entry = DispatchCacheEntry{revision};
        }
        entry.last_used = ++persistent_dispatch_tick_;
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
        std::scoped_lock lock(persistent_resource_mutex_);
        auto& entry = persistent_resource_cache_[key];
        if (entry.revision != revision) {
            entry = ResourceCacheEntry{revision};
        }
        entry.last_used = ++persistent_resource_tick_;
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

    [[nodiscard]] std::uint32_t preferred_stream_groups(
        const HardwareGraph& graph,
        const std::string_view kernel_tag,
        const std::size_t input_bytes,
        const std::size_t output_bytes) const {
        if (!supports_async_dispatch(graph) || (input_bytes == 0u && output_bytes == 0u)) {
            return 1u;
        }
        const auto summary = summarize_graph(graph);
        std::uint32_t groups = summary.supports_asynchronous_dispatch ? 2u : 1u;
        const auto transfer_bytes = input_bytes + output_bytes;
        if (transfer_bytes >= (8ull * 1024ull * 1024ull) ||
            kernel_tag == "matmul" ||
            kernel_tag == "conv3x3" ||
            kernel_tag == "resample") {
            groups = std::max<std::uint32_t>(groups, 2u);
        }
        if (summary.unified_address_space && transfer_bytes >= (2ull * 1024ull * 1024ull)) {
            groups = std::max<std::uint32_t>(groups, 2u);
        }
        return std::clamp<std::uint32_t>(groups, 1u, 2u);
    }

    [[nodiscard]] double preferred_overlap_with_policy(
        const HardwareGraph& graph,
        const std::string_view kernel_tag,
        const std::size_t input_bytes,
        const std::size_t output_bytes,
        const double preferred_overlap_ratio,
        const std::uint32_t resource_hits) const {
        double overlap = preferred_overlap_ratio;
        const auto stream_groups = preferred_stream_groups(graph, kernel_tag, input_bytes, output_bytes);
        if (stream_groups > 1u) {
            overlap += 0.08;
        }
        if (resource_hits > 0u) {
            overlap += std::min(0.08, 0.03 * static_cast<double>(resource_hits));
        }
        if (kernel_tag == "matmul") {
            overlap += 0.06;
        } else if (kernel_tag == "resample" || kernel_tag == "conv3x3") {
            overlap += 0.04;
        } else if (kernel_tag == "reduction" && output_bytes <= sizeof(float)) {
            overlap += 0.02;
        }
        return std::clamp(
            overlap + (supports_async_dispatch(graph) ? 0.05 : 0.0),
            0.05,
            0.92);
    }

    [[nodiscard]] BackendRunResult mark_async_result(
        const HardwareGraph& graph,
        const std::string_view kernel_tag,
        BackendRunResult result,
        const std::string_view resource_tag = {},
        const std::size_t input_bytes = 0u,
        const std::size_t output_bytes = 0u,
        const double preferred_overlap_ratio = 0.35) const {
        result.async_dispatch_capable = supports_async_dispatch(graph);
        if (result.submit_runtime_us <= 0.0 && result.synchronize_runtime_us <= 0.0) {
            const auto summary = summarize_graph(graph);
            result.submit_runtime_us = std::max(1.0, summary.dispatch_latency_us * 0.5);
            result.synchronize_runtime_us = std::max(0.0, result.runtime_us - result.submit_runtime_us);
        }
        const auto reuse_hits = record_persistent_dispatch_reuse(graph, kernel_tag);
        const auto resource_hits = record_persistent_resource_reuse(graph, resource_tag);
        const auto stream_groups =
            preferred_stream_groups(graph, kernel_tag, input_bytes, output_bytes);
        const double overlap_hint = preferred_overlap_with_policy(
            graph,
            kernel_tag,
            input_bytes,
            output_bytes,
            preferred_overlap_ratio,
            resource_hits);
        if (reuse_hits > 0u) {
            const double submit_scale = std::max(0.50, 0.82 - (0.08 * reuse_hits));
            const double sync_scale = std::max(0.88, 0.97 - (0.03 * reuse_hits));
            result.submit_runtime_us *= submit_scale;
            result.synchronize_runtime_us *= sync_scale;
        }
        if (resource_hits > 0u) {
            const double submit_scale = std::max(0.72, 0.92 - (0.05 * resource_hits));
            const double sync_scale = std::max(0.80, 0.93 - (0.04 * resource_hits));
            result.submit_runtime_us *= submit_scale;
            result.synchronize_runtime_us *= sync_scale;
        }
        if (stream_groups > 1u && result.async_dispatch_capable) {
            result.submit_runtime_us *= 0.92;
            result.synchronize_runtime_us *= 0.94;
        }
        if (result.copy_runtime_us <= 0.0 || result.compute_runtime_us <= 0.0) {
            const auto split = estimate_copy_compute_breakdown(
                graph,
                input_bytes,
                output_bytes,
                result.synchronize_runtime_us,
                overlap_hint);
            result.copy_runtime_us = split.copy_runtime_us;
            result.compute_runtime_us = split.compute_runtime_us;
            result.copy_overlap_ratio = split.overlap_ratio;
            result.synchronize_runtime_us =
                result.compute_runtime_us + (result.copy_runtime_us * (1.0 - result.copy_overlap_ratio));
        } else {
            result.copy_overlap_ratio = overlap_hint;
            result.synchronize_runtime_us =
                result.compute_runtime_us + (result.copy_runtime_us * (1.0 - result.copy_overlap_ratio));
        }
        result.copy_queue_count = input_bytes > 0u || output_bytes > 0u ? 1u : 0u;
        result.compute_queue_count = 1u;
        result.event_wait_count =
            (result.copy_queue_count > 0u && result.async_dispatch_capable)
                ? ((output_bytes > 0u ? 2u : 1u) * (stream_groups > 1u ? 1u : 0u))
                : 0u;
        result.queue_separation_ratio =
            result.copy_queue_count > 0u && result.async_dispatch_capable
                ? std::clamp(0.24 + (0.08 * static_cast<double>(stream_groups - 1u)) + result.copy_overlap_ratio, 0.0, 1.0)
                : 0.0;
        result.persistent_resource_reuse_hits = resource_hits;
        result.runtime_us = result.submit_runtime_us + result.synchronize_runtime_us;
        return result;
    }

    virtual BackendRunResult run_elementwise_native(const HardwareGraph&, std::span<const float>, std::span<const float>, bool) const = 0;
    virtual BackendRunResult run_reduction_native(const HardwareGraph&, std::span<const float>, bool) const = 0;
    virtual BackendRunResult run_matmul_native(const HardwareGraph&, const OperationSpec&, std::span<const float>, std::span<const float>, std::uint32_t, std::uint32_t, std::uint32_t, bool) const = 0;
    virtual BackendRunResult run_conv3x3_native(const HardwareGraph&, const OperationSpec&, std::span<const float>, std::uint32_t, std::uint32_t, bool) const = 0;
    virtual BackendRunResult run_resample_native(const HardwareGraph&, const OperationSpec&, std::span<const float>, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool) const = 0;

    JakalBackendKind backend_;
    std::string probe_name_;
    std::unique_ptr<IKernelBackend> host_;
    mutable std::mutex persistent_dispatch_mutex_;
    mutable std::uint64_t persistent_dispatch_tick_ = 0u;
    mutable std::unordered_map<std::string, DispatchCacheEntry> persistent_dispatch_cache_;
    mutable std::mutex persistent_resource_mutex_;
    mutable std::uint64_t persistent_resource_tick_ = 0u;
    mutable std::unordered_map<std::string, ResourceCacheEntry> persistent_resource_cache_;
    static constexpr std::size_t kMaxPersistentDispatchEntries = 64u;
    static constexpr std::size_t kMaxPersistentResourceEntries = 96u;
};

class FallbackNativeBackend final : public NativeKernelBackendBase {
public:
    FallbackNativeBackend(const JakalBackendKind backend, std::string probe_name)
        : NativeKernelBackendBase(backend, std::move(probe_name)) {}

private:
    BackendRunResult run_elementwise_native(const HardwareGraph&, std::span<const float>, std::span<const float>, bool) const override { return failure("native-unimplemented"); }
    BackendRunResult run_reduction_native(const HardwareGraph&, std::span<const float>, bool) const override { return failure("native-unimplemented"); }
    BackendRunResult run_matmul_native(const HardwareGraph&, const OperationSpec&, std::span<const float>, std::span<const float>, std::uint32_t, std::uint32_t, std::uint32_t, bool) const override { return failure("native-unimplemented"); }
    BackendRunResult run_conv3x3_native(const HardwareGraph&, const OperationSpec&, std::span<const float>, std::uint32_t, std::uint32_t, bool) const override { return failure("native-unimplemented"); }
    BackendRunResult run_resample_native(const HardwareGraph&, const OperationSpec&, std::span<const float>, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool) const override { return failure("native-unimplemented"); }
};

class CudaNativeBackend final : public NativeKernelBackendBase {
public:
    CudaNativeBackend()
        : NativeKernelBackendBase(JakalBackendKind::cuda, "cuda") {}

private:
    using cu_result_t = int;
    using cu_device_t = int;
    using cu_device_ptr_t = std::uint64_t;
    using cu_context_t = void*;
    using cu_stream_t = void*;
    using cu_event_t = void*;
    using cu_module_t = void*;
    using cu_function_t = void*;
    using nvrtc_result_t = int;
    using nvrtc_program_t = void*;

    struct Api {
        using cu_init_fn = cu_result_t (*)(unsigned int);
        using cu_device_get_fn = cu_result_t (*)(cu_device_t*, int);
        using cu_ctx_create_fn = cu_result_t (*)(cu_context_t*, unsigned int, cu_device_t);
        using cu_ctx_destroy_fn = cu_result_t (*)(cu_context_t);
        using cu_ctx_set_current_fn = cu_result_t (*)(cu_context_t);
        using cu_stream_create_fn = cu_result_t (*)(cu_stream_t*, unsigned int);
        using cu_stream_synchronize_fn = cu_result_t (*)(cu_stream_t);
        using cu_stream_wait_event_fn = cu_result_t (*)(cu_stream_t, cu_event_t, unsigned int);
        using cu_stream_destroy_fn = cu_result_t (*)(cu_stream_t);
        using cu_event_create_fn = cu_result_t (*)(cu_event_t*, unsigned int);
        using cu_event_record_fn = cu_result_t (*)(cu_event_t, cu_stream_t);
        using cu_event_synchronize_fn = cu_result_t (*)(cu_event_t);
        using cu_event_destroy_fn = cu_result_t (*)(cu_event_t);
        using cu_module_load_data_ex_fn = cu_result_t (*)(cu_module_t*, const void*, unsigned int, void*, void*);
        using cu_module_unload_fn = cu_result_t (*)(cu_module_t);
        using cu_module_get_function_fn = cu_result_t (*)(cu_function_t*, cu_module_t, const char*);
        using cu_mem_alloc_fn = cu_result_t (*)(cu_device_ptr_t*, std::size_t);
        using cu_mem_free_fn = cu_result_t (*)(cu_device_ptr_t);
        using cu_memcpy_htod_fn = cu_result_t (*)(cu_device_ptr_t, const void*, std::size_t);
        using cu_memcpy_dtoh_fn = cu_result_t (*)(void*, cu_device_ptr_t, std::size_t);
        using cu_memcpy_htod_async_fn = cu_result_t (*)(cu_device_ptr_t, const void*, std::size_t, cu_stream_t);
        using cu_memcpy_dtoh_async_fn = cu_result_t (*)(void*, cu_device_ptr_t, std::size_t, cu_stream_t);
        using cu_launch_kernel_fn = cu_result_t (*)(cu_function_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, cu_stream_t, void**, void**);
        using nvrtc_create_program_fn = nvrtc_result_t (*)(nvrtc_program_t*, const char*, const char*, int, const char* const*, const char* const*);
        using nvrtc_compile_program_fn = nvrtc_result_t (*)(nvrtc_program_t, int, const char* const*);
        using nvrtc_get_ptx_size_fn = nvrtc_result_t (*)(nvrtc_program_t, std::size_t*);
        using nvrtc_get_ptx_fn = nvrtc_result_t (*)(nvrtc_program_t, char*);
        using nvrtc_destroy_program_fn = nvrtc_result_t (*)(nvrtc_program_t*);

        LibraryHandle driver_library = nullptr;
        LibraryHandle nvrtc_library = nullptr;
        bool ready = false;
        cu_init_fn cu_init = nullptr;
        cu_device_get_fn cu_device_get = nullptr;
        cu_ctx_create_fn cu_ctx_create = nullptr;
        cu_ctx_destroy_fn cu_ctx_destroy = nullptr;
        cu_ctx_set_current_fn cu_ctx_set_current = nullptr;
        cu_stream_create_fn cu_stream_create = nullptr;
        cu_stream_synchronize_fn cu_stream_synchronize = nullptr;
        cu_stream_wait_event_fn cu_stream_wait_event = nullptr;
        cu_stream_destroy_fn cu_stream_destroy = nullptr;
        cu_event_create_fn cu_event_create = nullptr;
        cu_event_record_fn cu_event_record = nullptr;
        cu_event_synchronize_fn cu_event_synchronize = nullptr;
        cu_event_destroy_fn cu_event_destroy = nullptr;
        cu_module_load_data_ex_fn cu_module_load_data_ex = nullptr;
        cu_module_unload_fn cu_module_unload = nullptr;
        cu_module_get_function_fn cu_module_get_function = nullptr;
        cu_mem_alloc_fn cu_mem_alloc = nullptr;
        cu_mem_free_fn cu_mem_free = nullptr;
        cu_memcpy_htod_fn cu_memcpy_htod = nullptr;
        cu_memcpy_dtoh_fn cu_memcpy_dtoh = nullptr;
        cu_memcpy_htod_async_fn cu_memcpy_htod_async = nullptr;
        cu_memcpy_dtoh_async_fn cu_memcpy_dtoh_async = nullptr;
        cu_launch_kernel_fn cu_launch_kernel = nullptr;
        nvrtc_create_program_fn nvrtc_create_program = nullptr;
        nvrtc_compile_program_fn nvrtc_compile_program = nullptr;
        nvrtc_get_ptx_size_fn nvrtc_get_ptx_size = nullptr;
        nvrtc_get_ptx_fn nvrtc_get_ptx = nullptr;
        nvrtc_destroy_program_fn nvrtc_destroy_program = nullptr;

        Api() {
#if defined(_WIN32)
            driver_library = load_library("nvcuda.dll");
            nvrtc_library = load_library_with_fallbacks({
                "nvrtc64_130_0.dll",
                "nvrtc64_125_0.dll",
                "nvrtc64_124_0.dll",
                "nvrtc64_122_0.dll",
                "nvrtc64_121_0.dll",
                "nvrtc64_120_0.dll",
            });
#else
            driver_library = load_library("libcuda.so");
            if (driver_library == nullptr) driver_library = load_library("libcuda.so.1");
            nvrtc_library = load_library("libnvrtc.so");
            if (nvrtc_library == nullptr) nvrtc_library = load_library("libnvrtc.so.12");
#endif
            if (driver_library == nullptr || nvrtc_library == nullptr) {
                return;
            }
            cu_init = reinterpret_cast<cu_init_fn>(load_symbol(driver_library, "cuInit"));
            cu_device_get = reinterpret_cast<cu_device_get_fn>(load_symbol(driver_library, "cuDeviceGet"));
            cu_ctx_create = reinterpret_cast<cu_ctx_create_fn>(load_symbol(driver_library, "cuCtxCreate_v2"));
            cu_ctx_destroy = reinterpret_cast<cu_ctx_destroy_fn>(load_symbol(driver_library, "cuCtxDestroy_v2"));
            cu_ctx_set_current = reinterpret_cast<cu_ctx_set_current_fn>(load_symbol(driver_library, "cuCtxSetCurrent"));
            cu_stream_create = reinterpret_cast<cu_stream_create_fn>(load_symbol(driver_library, "cuStreamCreate"));
            cu_stream_synchronize = reinterpret_cast<cu_stream_synchronize_fn>(load_symbol(driver_library, "cuStreamSynchronize"));
            cu_stream_wait_event = reinterpret_cast<cu_stream_wait_event_fn>(load_symbol(driver_library, "cuStreamWaitEvent"));
            cu_stream_destroy = reinterpret_cast<cu_stream_destroy_fn>(load_symbol(driver_library, "cuStreamDestroy_v2"));
            cu_event_create = reinterpret_cast<cu_event_create_fn>(load_symbol(driver_library, "cuEventCreate"));
            cu_event_record = reinterpret_cast<cu_event_record_fn>(load_symbol(driver_library, "cuEventRecord"));
            cu_event_synchronize = reinterpret_cast<cu_event_synchronize_fn>(load_symbol(driver_library, "cuEventSynchronize"));
            cu_event_destroy = reinterpret_cast<cu_event_destroy_fn>(load_symbol(driver_library, "cuEventDestroy_v2"));
            cu_module_load_data_ex = reinterpret_cast<cu_module_load_data_ex_fn>(load_symbol(driver_library, "cuModuleLoadDataEx"));
            cu_module_unload = reinterpret_cast<cu_module_unload_fn>(load_symbol(driver_library, "cuModuleUnload"));
            cu_module_get_function = reinterpret_cast<cu_module_get_function_fn>(load_symbol(driver_library, "cuModuleGetFunction"));
            cu_mem_alloc = reinterpret_cast<cu_mem_alloc_fn>(load_symbol(driver_library, "cuMemAlloc_v2"));
            cu_mem_free = reinterpret_cast<cu_mem_free_fn>(load_symbol(driver_library, "cuMemFree_v2"));
            cu_memcpy_htod = reinterpret_cast<cu_memcpy_htod_fn>(load_symbol(driver_library, "cuMemcpyHtoD_v2"));
            cu_memcpy_dtoh = reinterpret_cast<cu_memcpy_dtoh_fn>(load_symbol(driver_library, "cuMemcpyDtoH_v2"));
            cu_memcpy_htod_async =
                reinterpret_cast<cu_memcpy_htod_async_fn>(load_symbol(driver_library, "cuMemcpyHtoDAsync_v2"));
            cu_memcpy_dtoh_async =
                reinterpret_cast<cu_memcpy_dtoh_async_fn>(load_symbol(driver_library, "cuMemcpyDtoHAsync_v2"));
            cu_launch_kernel = reinterpret_cast<cu_launch_kernel_fn>(load_symbol(driver_library, "cuLaunchKernel"));
            nvrtc_create_program = reinterpret_cast<nvrtc_create_program_fn>(load_symbol(nvrtc_library, "nvrtcCreateProgram"));
            nvrtc_compile_program = reinterpret_cast<nvrtc_compile_program_fn>(load_symbol(nvrtc_library, "nvrtcCompileProgram"));
            nvrtc_get_ptx_size = reinterpret_cast<nvrtc_get_ptx_size_fn>(load_symbol(nvrtc_library, "nvrtcGetPTXSize"));
            nvrtc_get_ptx = reinterpret_cast<nvrtc_get_ptx_fn>(load_symbol(nvrtc_library, "nvrtcGetPTX"));
            nvrtc_destroy_program = reinterpret_cast<nvrtc_destroy_program_fn>(load_symbol(nvrtc_library, "nvrtcDestroyProgram"));
            ready = cu_init != nullptr && cu_device_get != nullptr && cu_ctx_create != nullptr && cu_ctx_destroy != nullptr && cu_ctx_set_current != nullptr && cu_stream_create != nullptr && cu_stream_synchronize != nullptr && cu_stream_destroy != nullptr && cu_module_load_data_ex != nullptr && cu_module_unload != nullptr && cu_module_get_function != nullptr && cu_mem_alloc != nullptr && cu_mem_free != nullptr && cu_memcpy_htod != nullptr && cu_memcpy_dtoh != nullptr && cu_launch_kernel != nullptr && nvrtc_create_program != nullptr && nvrtc_compile_program != nullptr && nvrtc_get_ptx_size != nullptr && nvrtc_get_ptx != nullptr && nvrtc_destroy_program != nullptr;
        }

        ~Api() {
            close_library(nvrtc_library);
            close_library(driver_library);
        }
    };

    struct Context {
        cu_context_t context = nullptr;
        cu_stream_t stream = nullptr;
        cu_stream_t copy_stream = nullptr;
        cu_event_t copy_ready_event = nullptr;
        cu_event_t compute_ready_event = nullptr;
        cu_module_t module = nullptr;
        std::array<std::vector<float>, 2> staging_buffers;
        std::unordered_map<std::string, cu_function_t> kernels;
        std::unordered_map<std::string, cu_device_ptr_t> reusable_buffers;
        std::unordered_map<std::string, std::size_t> reusable_buffer_sizes;
        std::unordered_map<std::string, std::uint64_t> reusable_buffer_last_used;
        std::uint64_t reusable_buffer_bytes = 0u;
        std::uint64_t reusable_buffer_budget_bytes = 0u;
        std::uint64_t reusable_buffer_tick = 0u;
        std::string revision;
        std::mutex execution_mutex;
    };

    const Api& api() const {
        static const Api instance;
        return instance;
    }

    bool activate(const Context& context) const {
        return api().ready && api().cu_ctx_set_current(context.context) == 0;
    }

    bool supports_split_streams() const {
        return api().cu_stream_wait_event != nullptr &&
               api().cu_event_create != nullptr &&
               api().cu_event_record != nullptr &&
               api().cu_event_destroy != nullptr &&
               api().cu_memcpy_htod_async != nullptr &&
               api().cu_memcpy_dtoh_async != nullptr;
    }

    bool can_split_dispatch(const Context& context) const {
        return supports_split_streams() &&
               context.copy_stream != nullptr &&
               context.copy_ready_event != nullptr &&
               context.compute_ready_event != nullptr;
    }

    void free_device(const cu_device_ptr_t ptr) const {
        if (ptr != 0) {
            (void)api().cu_mem_free(ptr);
        }
    }

    std::uint64_t reusable_buffer_budget_bytes(const HardwareGraph& graph) const {
        const auto summary = summarize_graph(graph);
        const auto capacity = summary.directly_attached_bytes > 0u
                                  ? summary.directly_attached_bytes
                                  : std::max(summary.addressable_bytes, summary.shared_host_bytes);
        return std::clamp<std::uint64_t>(capacity / 16u, 16ull * 1024ull * 1024ull, 256ull * 1024ull * 1024ull);
    }

    void touch_buffer(Context& context, const std::string& slot) const {
        context.reusable_buffer_last_used[slot] = ++context.reusable_buffer_tick;
    }

    void trim_reusable_buffers(Context& context, const std::string& preserve_slot, const std::size_t incoming_bytes) const {
        while (!context.reusable_buffers.empty() &&
               context.reusable_buffer_budget_bytes > 0u &&
               context.reusable_buffer_bytes + incoming_bytes > context.reusable_buffer_budget_bytes) {
            auto oldest = context.reusable_buffers.end();
            for (auto it = context.reusable_buffers.begin(); it != context.reusable_buffers.end(); ++it) {
                if (it->first == preserve_slot) {
                    continue;
                }
                if (oldest == context.reusable_buffers.end() ||
                    context.reusable_buffer_last_used[it->first] < context.reusable_buffer_last_used[oldest->first]) {
                    oldest = it;
                }
            }
            if (oldest == context.reusable_buffers.end()) {
                break;
            }
            context.reusable_buffer_bytes -= context.reusable_buffer_sizes[oldest->first];
            free_device(oldest->second);
            context.reusable_buffer_sizes.erase(oldest->first);
            context.reusable_buffer_last_used.erase(oldest->first);
            context.reusable_buffers.erase(oldest);
        }
    }

    void destroy_context(Context& context) const {
        for (const auto& [slot, ptr] : context.reusable_buffers) {
            (void)slot;
            free_device(ptr);
        }
        context.reusable_buffers.clear();
        context.reusable_buffer_sizes.clear();
        context.reusable_buffer_last_used.clear();
        context.reusable_buffer_bytes = 0u;
        context.kernels.clear();
        if (context.module != nullptr) {
            (void)api().cu_module_unload(context.module);
            context.module = nullptr;
        }
        if (context.stream != nullptr) {
            (void)api().cu_stream_destroy(context.stream);
            context.stream = nullptr;
        }
        if (context.copy_ready_event != nullptr) {
            (void)api().cu_event_destroy(context.copy_ready_event);
            context.copy_ready_event = nullptr;
        }
        if (context.compute_ready_event != nullptr) {
            (void)api().cu_event_destroy(context.compute_ready_event);
            context.compute_ready_event = nullptr;
        }
        if (context.copy_stream != nullptr) {
            (void)api().cu_stream_destroy(context.copy_stream);
            context.copy_stream = nullptr;
        }
        if (context.context != nullptr) {
            (void)api().cu_ctx_destroy(context.context);
            context.context = nullptr;
        }
    }

    bool ensure_buffer(Context& context, const std::string& slot, const std::size_t bytes, cu_device_ptr_t& ptr) const {
        if (bytes == 0u) {
            ptr = 0;
            return true;
        }
        auto& slot_ptr = context.reusable_buffers[slot];
        auto& slot_size = context.reusable_buffer_sizes[slot];
        if (slot_ptr == 0 || slot_size < bytes) {
            if (slot_ptr != 0) {
                context.reusable_buffer_bytes -= slot_size;
            }
            free_device(slot_ptr);
            slot_ptr = 0;
            slot_size = 0u;
            trim_reusable_buffers(context, slot, bytes);
            if (api().cu_mem_alloc(&slot_ptr, bytes) != 0) {
                return false;
            }
            slot_size = bytes;
            context.reusable_buffer_bytes += bytes;
        }
        touch_buffer(context, slot);
        ptr = slot_ptr;
        return true;
    }

    bool alloc_and_copy_input(Context& context, const std::string& slot, cu_device_ptr_t& ptr, std::span<const float> input) const {
        if (!ensure_buffer(context, slot, input.size_bytes(), ptr)) {
            return false;
        }
        if (api().cu_memcpy_htod(ptr, input.data(), input.size_bytes()) != 0) {
            return false;
        }
        return true;
    }

    bool alloc_and_copy_input_async(
        Context& context,
        const std::string& slot,
        cu_device_ptr_t& ptr,
        std::span<const float> input,
        const cu_stream_t stream,
        const std::size_t staging_slot = 0u) const {
        if (!ensure_buffer(context, slot, input.size_bytes(), ptr)) {
            return false;
        }
        const void* source = input.data();
        if (!input.empty()) {
            auto& staging = context.staging_buffers[staging_slot % context.staging_buffers.size()];
            staging.resize(input.size());
            std::memcpy(staging.data(), input.data(), input.size_bytes());
            source = staging.data();
        }
        return api().cu_memcpy_htod_async(ptr, source, input.size_bytes(), stream) == 0;
    }

    std::shared_ptr<Context> acquire_context(const HardwareGraph& graph) const {
        std::scoped_lock lock(mutex_);
        if (!api().ready || api().cu_init(0u) != 0) {
            return {};
        }
        const auto revision = structural_fingerprint(graph);
        if (const auto existing = contexts_.find(graph.uid); existing != contexts_.end()) {
            if (existing->second->revision == revision) {
                return existing->second;
            }
            destroy_context(*existing->second);
            contexts_.erase(existing);
        }
        cu_device_t device = 0;
        if (api().cu_device_get(&device, static_cast<int>(graph.ordinal)) != 0) {
            return {};
        }
        auto context = std::make_shared<Context>();
        if (api().cu_ctx_create(&context->context, 0u, device) != 0 ||
            api().cu_ctx_set_current(context->context) != 0 ||
            api().cu_stream_create(&context->stream, 0u) != 0) {
            if (context->stream != nullptr) api().cu_stream_destroy(context->stream);
            if (context->context != nullptr) api().cu_ctx_destroy(context->context);
            return {};
        }
        if (supports_split_streams()) {
            if (api().cu_stream_create(&context->copy_stream, 0u) != 0 ||
                api().cu_event_create(&context->copy_ready_event, 0u) != 0 ||
                api().cu_event_create(&context->compute_ready_event, 0u) != 0) {
                destroy_context(*context);
                return {};
            }
        }
        if (!compile_module(*context)) {
            destroy_context(*context);
            return {};
        }
        context->revision = revision;
        context->reusable_buffer_budget_bytes = reusable_buffer_budget_bytes(graph);
        contexts_.emplace(graph.uid, context);
        return context;
    }

    bool compile_module(Context& context) const {
        nvrtc_program_t program = nullptr;
        if (api().nvrtc_create_program(&program, kCudaLikeProgramSource, "jakal_native.cu", 0, nullptr, nullptr) != 0) {
            return false;
        }
        const char* options[] = {"--gpu-architecture=compute_52"};
        if (api().nvrtc_compile_program(program, 1, options) != 0) {
            api().nvrtc_destroy_program(&program);
            return false;
        }
        std::size_t ptx_size = 0;
        if (api().nvrtc_get_ptx_size(program, &ptx_size) != 0 || ptx_size == 0) {
            api().nvrtc_destroy_program(&program);
            return false;
        }
        std::string ptx(ptx_size, '\0');
        const bool ok = api().nvrtc_get_ptx(program, ptx.data()) == 0 && api().cu_module_load_data_ex(&context.module, ptx.data(), 0u, nullptr, nullptr) == 0;
        api().nvrtc_destroy_program(&program);
        return ok;
    }

    cu_function_t get_kernel(Context& context, const char* name) const {
        if (const auto existing = context.kernels.find(name); existing != context.kernels.end()) {
            return existing->second;
        }
        cu_function_t kernel = nullptr;
        if (api().cu_module_get_function(&kernel, context.module, name) != 0 || kernel == nullptr) {
            return nullptr;
        }
        context.kernels.emplace(name, kernel);
        return kernel;
    }

    BackendRunResult launch_unary_2d(const HardwareGraph& graph, const char* kernel_name, const std::span<const float> input, const std::size_t output_count, const std::uint32_t grid_items_x, const std::uint32_t grid_items_y, std::vector<void*> args) const {
        auto context = acquire_context(graph);
        if (context == nullptr) return failure("cuda-context");
        BackendRunResult result;
        result.output.resize(output_count, 0.0f);
        double copy_in_us = 0.0;
        double copy_out_us = 0.0;
        double compute_us = 0.0;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock execution_lock(context->execution_mutex);
            if (!activate(*context)) { result.error = "cuda-activate"; return; }
            auto kernel = get_kernel(*context, kernel_name);
            if (kernel == nullptr) { result.error = "cuda-kernel"; return; }
            cu_device_ptr_t d_in = 0, d_out = 0;
            if (supports_split_streams() && context->copy_stream != nullptr &&
                context->copy_ready_event != nullptr && context->compute_ready_event != nullptr) {
                if (!alloc_and_copy_input_async(*context, "unary-in", d_in, input, context->copy_stream) ||
                    !ensure_buffer(*context, "unary-out", result.output.size() * sizeof(float), d_out)) {
                    result.error = "cuda-memory";
                    return;
                }
                args[0] = &d_in;
                args[1] = &d_out;
                const unsigned int block_x = kNativeTileSize;
                const unsigned int block_y = kNativeTileSize;
                const unsigned int grid_x = grid_items_x == 0 ? 1u : (grid_items_x + block_x - 1u) / block_x;
                const unsigned int grid_y = grid_items_y == 0 ? 1u : (grid_items_y + block_y - 1u) / block_y;
                if (api().cu_event_record(context->copy_ready_event, context->copy_stream) != 0 ||
                    api().cu_stream_wait_event(context->stream, context->copy_ready_event, 0u) != 0 ||
                    api().cu_launch_kernel(kernel, grid_x, grid_y, 1, block_x, block_y, 1, 0u, context->stream, args.data(), nullptr) != 0 ||
                    api().cu_event_record(context->compute_ready_event, context->stream) != 0 ||
                    api().cu_stream_wait_event(context->copy_stream, context->compute_ready_event, 0u) != 0 ||
                    api().cu_memcpy_dtoh_async(result.output.data(), d_out, result.output.size() * sizeof(float), context->copy_stream) != 0 ||
                    api().cu_stream_synchronize(context->copy_stream) != 0) {
                    result.error = "cuda-launch";
                    return;
                }
                result.success = true;
                result.used_host = false;
                result.used_opencl = false;
                return;
            }
            copy_in_us += measure_us([&]() {
                if (!alloc_and_copy_input(*context, "unary-in", d_in, input) ||
                    !ensure_buffer(*context, "unary-out", result.output.size() * sizeof(float), d_out)) {
                    result.error = "cuda-memory";
                }
            });
            if (!result.error.empty()) {
                result.error = "cuda-memory"; return;
            }
            args[0] = &d_in;
            args[1] = &d_out;
            const unsigned int block_x = kNativeTileSize;
            const unsigned int block_y = kNativeTileSize;
            const unsigned int grid_x = grid_items_x == 0 ? 1u : (grid_items_x + block_x - 1u) / block_x;
            const unsigned int grid_y = grid_items_y == 0 ? 1u : (grid_items_y + block_y - 1u) / block_y;
            compute_us += measure_us([&]() {
                if (api().cu_launch_kernel(kernel, grid_x, grid_y, 1, block_x, block_y, 1, 0u, context->stream, args.data(), nullptr) != 0 ||
                    api().cu_stream_synchronize(context->stream) != 0) {
                    result.error = "cuda-launch";
                }
            });
            if (!result.error.empty()) {
                return;
            }
            copy_out_us += measure_us([&]() {
                if (api().cu_memcpy_dtoh(result.output.data(), d_out, result.output.size() * sizeof(float)) != 0) {
                    result.error = "cuda-launch";
                }
            });
            if (!result.error.empty()) {
                result.error = "cuda-launch";
            } else { result.success = true; result.used_host = false; result.used_opencl = false; }
        });
        result.copy_runtime_us = copy_in_us + copy_out_us;
        result.compute_runtime_us = compute_us;
        return result;
    }

    BackendRunResult run_elementwise_native(const HardwareGraph& graph, const std::span<const float> lhs, const std::span<const float> rhs, const bool low_precision) const override {
        auto context = acquire_context(graph);
        if (context == nullptr) return failure("cuda-context");
        BackendRunResult result;
        result.output.resize(lhs.size(), 0.0f);
        double copy_in_us = 0.0;
        double copy_out_us = 0.0;
        double compute_us = 0.0;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock execution_lock(context->execution_mutex);
            if (!activate(*context)) { result.error = "cuda-activate"; return; }
            auto kernel = get_kernel(*context, "elementwise_map");
            if (kernel == nullptr) { result.error = "cuda-kernel"; return; }
            cu_device_ptr_t d_lhs = 0, d_rhs = 0, d_out = 0;
            if (can_split_dispatch(*context)) {
                if (!alloc_and_copy_input_async(*context, "lhs", d_lhs, lhs, context->copy_stream, 0u) ||
                    !alloc_and_copy_input_async(*context, "rhs", d_rhs, rhs, context->copy_stream, 1u) ||
                    !ensure_buffer(*context, "binary-out", result.output.size() * sizeof(float), d_out)) {
                    result.error = "cuda-memory";
                    return;
                }
                unsigned int count = static_cast<unsigned int>(lhs.size());
                int low = low_precision ? 1 : 0;
                void* args[] = {&d_lhs, &d_rhs, &d_out, &count, &low};
                const unsigned int block_x = kNativeReductionGroupSize;
                const unsigned int grid_x = count == 0 ? 1u : (count + block_x - 1u) / block_x;
                if (api().cu_event_record(context->copy_ready_event, context->copy_stream) != 0 ||
                    api().cu_stream_wait_event(context->stream, context->copy_ready_event, 0u) != 0 ||
                    api().cu_launch_kernel(kernel, grid_x, 1, 1, block_x, 1, 1, 0u, context->stream, args, nullptr) != 0 ||
                    api().cu_event_record(context->compute_ready_event, context->stream) != 0 ||
                    api().cu_stream_wait_event(context->copy_stream, context->compute_ready_event, 0u) != 0 ||
                    api().cu_memcpy_dtoh_async(result.output.data(), d_out, result.output.size() * sizeof(float), context->copy_stream) != 0 ||
                    api().cu_stream_synchronize(context->copy_stream) != 0) {
                    result.error = "cuda-launch";
                    return;
                }
                result.success = true;
                result.used_host = false;
                result.used_opencl = false;
                return;
            }
            copy_in_us += measure_us([&]() {
                if (!alloc_and_copy_input(*context, "lhs", d_lhs, lhs) ||
                    !alloc_and_copy_input(*context, "rhs", d_rhs, rhs) ||
                    !ensure_buffer(*context, "binary-out", result.output.size() * sizeof(float), d_out)) {
                    result.error = "cuda-memory";
                }
            });
            if (!result.error.empty()) { return; }
            unsigned int count = static_cast<unsigned int>(lhs.size());
            int low = low_precision ? 1 : 0;
            void* args[] = {&d_lhs, &d_rhs, &d_out, &count, &low};
            const unsigned int block_x = kNativeReductionGroupSize;
            const unsigned int grid_x = count == 0 ? 1u : (count + block_x - 1u) / block_x;
            compute_us += measure_us([&]() {
                if (api().cu_launch_kernel(kernel, grid_x, 1, 1, block_x, 1, 1, 0u, context->stream, args, nullptr) != 0 ||
                    api().cu_stream_synchronize(context->stream) != 0) {
                    result.error = "cuda-launch";
                }
            });
            if (!result.error.empty()) { return; }
            copy_out_us += measure_us([&]() {
                if (api().cu_memcpy_dtoh(result.output.data(), d_out, result.output.size() * sizeof(float)) != 0) {
                    result.error = "cuda-launch";
                }
            });
            if (result.error.empty()) { result.success = true; result.used_host = false; result.used_opencl = false; }
        });
        result.copy_runtime_us = copy_in_us + copy_out_us;
        result.compute_runtime_us = compute_us;
        return result;
    }

    BackendRunResult run_reduction_native(const HardwareGraph& graph, const std::span<const float> input, const bool low_precision) const override {
        auto context = acquire_context(graph);
        if (context == nullptr) return failure("cuda-context");
        BackendRunResult result;
        double copy_in_us = 0.0;
        double copy_out_us = 0.0;
        double compute_us = 0.0;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock execution_lock(context->execution_mutex);
            if (!activate(*context)) { result.error = "cuda-activate"; return; }
            auto kernel = get_kernel(*context, "reduce_sum");
            if (kernel == nullptr) { result.error = "cuda-kernel"; return; }
            const auto local = kNativeReductionGroupSize;
            const auto global = std::min<std::size_t>(std::max<std::size_t>(local, round_up<std::size_t>(input.size(), local)), local * 64u);
            const auto groups = global / local;
            std::vector<float> partials(groups, 0.0f);
            cu_device_ptr_t d_in = 0, d_partial = 0;
            if (can_split_dispatch(*context)) {
                if (!alloc_and_copy_input_async(*context, "reduce-in", d_in, input, context->copy_stream) ||
                    !ensure_buffer(*context, "reduce-partial", partials.size() * sizeof(float), d_partial)) {
                    result.error = "cuda-memory";
                    return;
                }
                unsigned int count = static_cast<unsigned int>(input.size());
                int low = low_precision ? 1 : 0;
                void* args[] = {&d_in, &d_partial, &count, &low};
                if (api().cu_event_record(context->copy_ready_event, context->copy_stream) != 0 ||
                    api().cu_stream_wait_event(context->stream, context->copy_ready_event, 0u) != 0 ||
                    api().cu_launch_kernel(kernel, static_cast<unsigned int>(groups), 1, 1, local, 1, 1, static_cast<unsigned int>(local * sizeof(float)), context->stream, args, nullptr) != 0 ||
                    api().cu_event_record(context->compute_ready_event, context->stream) != 0 ||
                    api().cu_stream_wait_event(context->copy_stream, context->compute_ready_event, 0u) != 0 ||
                    api().cu_memcpy_dtoh_async(partials.data(), d_partial, partials.size() * sizeof(float), context->copy_stream) != 0 ||
                    api().cu_stream_synchronize(context->copy_stream) != 0) {
                    result.error = "cuda-launch";
                    return;
                }
                for (const auto value : partials) {
                    result.scalar_output += value;
                }
                result.success = true;
                result.used_host = false;
                result.used_opencl = false;
                return;
            }
            copy_in_us += measure_us([&]() {
                if (!alloc_and_copy_input(*context, "reduce-in", d_in, input) ||
                    !ensure_buffer(*context, "reduce-partial", partials.size() * sizeof(float), d_partial)) {
                    result.error = "cuda-memory";
                }
            });
            if (!result.error.empty()) { return; }
            unsigned int count = static_cast<unsigned int>(input.size());
            int low = low_precision ? 1 : 0;
            void* args[] = {&d_in, &d_partial, &count, &low};
            compute_us += measure_us([&]() {
                if (api().cu_launch_kernel(kernel, static_cast<unsigned int>(groups), 1, 1, local, 1, 1, static_cast<unsigned int>(local * sizeof(float)), context->stream, args, nullptr) != 0 ||
                    api().cu_stream_synchronize(context->stream) != 0) {
                    result.error = "cuda-launch";
                }
            });
            if (!result.error.empty()) { return; }
            copy_out_us += measure_us([&]() {
                if (api().cu_memcpy_dtoh(partials.data(), d_partial, partials.size() * sizeof(float)) != 0) {
                    result.error = "cuda-launch";
                }
            });
            if (result.error.empty()) { for (const auto value : partials) result.scalar_output += value; result.success = true; result.used_host = false; result.used_opencl = false; }
        });
        result.copy_runtime_us = copy_in_us + copy_out_us;
        result.compute_runtime_us = compute_us;
        return result;
    }

    BackendRunResult run_matmul_native(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> lhs, const std::span<const float> rhs, const std::uint32_t rows, const std::uint32_t columns, const std::uint32_t depth, const bool low_precision) const override {
        auto context = acquire_context(graph);
        if (context == nullptr) return failure("cuda-context");
        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(rows) * columns, 0.0f);
        double copy_in_us = 0.0;
        double copy_out_us = 0.0;
        double compute_us = 0.0;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock execution_lock(context->execution_mutex);
            if (!activate(*context)) { result.error = "cuda-activate"; return; }
            auto kernel = get_kernel(*context, gpu_rhs_uses_transposed_layout(operation) ? "matmul_tiled_rhs_t" : "matmul_tiled");
            if (kernel == nullptr) { result.error = "cuda-kernel"; return; }
            cu_device_ptr_t d_lhs = 0, d_rhs = 0, d_out = 0;
            if (supports_split_streams() && context->copy_stream != nullptr &&
                context->copy_ready_event != nullptr && context->compute_ready_event != nullptr) {
                if (!alloc_and_copy_input_async(*context, "matmul-lhs", d_lhs, lhs, context->copy_stream, 0u) ||
                    !alloc_and_copy_input_async(*context, "matmul-rhs", d_rhs, rhs, context->copy_stream, 1u) ||
                    !ensure_buffer(*context, "matmul-out", result.output.size() * sizeof(float), d_out)) {
                    result.error = "cuda-memory";
                    return;
                }
                unsigned int row_count = rows, column_count = columns, depth_count = depth;
                int low = low_precision ? 1 : 0;
                void* args[] = {&d_lhs, &d_rhs, &d_out, &row_count, &column_count, &depth_count, &low};
                const unsigned int block_x = kNativeTileSize;
                const unsigned int block_y = kNativeTileSize;
                const unsigned int grid_x = column_count == 0 ? 1u : (column_count + block_x - 1u) / block_x;
                const unsigned int grid_y = row_count == 0 ? 1u : (row_count + block_y - 1u) / block_y;
                if (api().cu_event_record(context->copy_ready_event, context->copy_stream) != 0 ||
                    api().cu_stream_wait_event(context->stream, context->copy_ready_event, 0u) != 0 ||
                    api().cu_launch_kernel(kernel, grid_x, grid_y, 1, block_x, block_y, 1, 0u, context->stream, args, nullptr) != 0 ||
                    api().cu_event_record(context->compute_ready_event, context->stream) != 0 ||
                    api().cu_stream_wait_event(context->copy_stream, context->compute_ready_event, 0u) != 0 ||
                    api().cu_memcpy_dtoh_async(result.output.data(), d_out, result.output.size() * sizeof(float), context->copy_stream) != 0 ||
                    api().cu_stream_synchronize(context->copy_stream) != 0) {
                    result.error = "cuda-launch";
                    return;
                }
                result.success = true;
                result.used_host = false;
                result.used_opencl = false;
                return;
            }
            copy_in_us += measure_us([&]() {
                if (!alloc_and_copy_input(*context, "matmul-lhs", d_lhs, lhs) ||
                    !alloc_and_copy_input(*context, "matmul-rhs", d_rhs, rhs) ||
                    !ensure_buffer(*context, "matmul-out", result.output.size() * sizeof(float), d_out)) {
                    result.error = "cuda-memory";
                }
            });
            if (!result.error.empty()) { return; }
            unsigned int row_count = rows, column_count = columns, depth_count = depth;
            int low = low_precision ? 1 : 0;
            void* args[] = {&d_lhs, &d_rhs, &d_out, &row_count, &column_count, &depth_count, &low};
            const unsigned int block_x = kNativeTileSize;
            const unsigned int block_y = kNativeTileSize;
            const unsigned int grid_x = column_count == 0 ? 1u : (column_count + block_x - 1u) / block_x;
            const unsigned int grid_y = row_count == 0 ? 1u : (row_count + block_y - 1u) / block_y;
            compute_us += measure_us([&]() {
                if (api().cu_launch_kernel(kernel, grid_x, grid_y, 1, block_x, block_y, 1, 0u, context->stream, args, nullptr) != 0 ||
                    api().cu_stream_synchronize(context->stream) != 0) {
                    result.error = "cuda-launch";
                }
            });
            if (!result.error.empty()) { return; }
            copy_out_us += measure_us([&]() {
                if (api().cu_memcpy_dtoh(result.output.data(), d_out, result.output.size() * sizeof(float)) != 0) {
                    result.error = "cuda-launch";
                }
            });
            if (result.error.empty()) { result.success = true; result.used_host = false; result.used_opencl = false; }
        });
        result.copy_runtime_us = copy_in_us + copy_out_us;
        result.compute_runtime_us = compute_us;
        return result;
    }

    BackendRunResult run_conv3x3_native(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> input, const std::uint32_t height, const std::uint32_t width, const bool low_precision) const override {
        unsigned int h = height, w = width; int low = low_precision ? 1 : 0;
        return launch_unary_2d(
            graph,
            gpu_conv_uses_patch9_layout(operation) ? "conv3x3_valid_patch9" : "conv3x3_valid",
            input,
            static_cast<std::size_t>(height - 2u) * (width - 2u),
            width - 2u,
            height - 2u,
            {nullptr, nullptr, &h, &w, &low});
    }

    BackendRunResult run_resample_native(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> input, const std::uint32_t src_h, const std::uint32_t src_w, const std::uint32_t dst_h, const std::uint32_t dst_w, const std::uint32_t row_offset, const std::uint32_t row_count, const bool low_precision) const override {
        unsigned int src_height = src_h, src_width = src_w, dst_height = dst_h, dst_width = dst_w, offset = row_offset, rows = row_count; int low = low_precision ? 1 : 0;
        return launch_unary_2d(
            graph,
            gpu_resample_uses_packed6_layout(operation) ? "bilinear_resample_packed6" : "bilinear_resample",
            input,
            static_cast<std::size_t>(row_count) * dst_w,
            dst_w,
            row_count,
            {nullptr, nullptr, &src_height, &src_width, &dst_height, &dst_width, &offset, &rows, &low});
    }

    mutable std::mutex mutex_;
    mutable std::unordered_map<std::string, std::shared_ptr<Context>> contexts_;
};

class LevelZeroNativeBackend final : public NativeKernelBackendBase {
public:
    LevelZeroNativeBackend()
        : NativeKernelBackendBase(JakalBackendKind::level_zero, "level-zero") {}

private:
    using ze_result_t = std::int32_t;
    using ze_driver_handle_t = void*;
    using ze_device_handle_t = void*;
    using ze_context_handle_t = void*;
    using ze_command_queue_handle_t = void*;
    using ze_command_list_handle_t = void*;
    using ze_module_handle_t = void*;
    using ze_module_build_log_handle_t = void*;
    using ze_kernel_handle_t = void*;

    struct ze_context_desc_t { std::uint32_t stype; const void* pNext; std::uint32_t flags; };
    struct ze_command_queue_desc_t { std::uint32_t stype; const void* pNext; std::uint32_t ordinal; std::uint32_t index; std::uint32_t flags; std::uint32_t mode; std::uint32_t priority; };
    struct ze_command_list_desc_t { std::uint32_t stype; const void* pNext; std::uint32_t commandQueueGroupOrdinal; std::uint32_t flags; };
    struct ze_device_mem_alloc_desc_t { std::uint32_t stype; const void* pNext; std::uint32_t flags; std::uint32_t ordinal; };
    struct ze_host_mem_alloc_desc_t { std::uint32_t stype; const void* pNext; std::uint32_t flags; };
    struct ze_module_desc_t { std::uint32_t stype; const void* pNext; std::uint32_t format; std::size_t inputSize; const std::uint8_t* pInputModule; const char* pBuildFlags; const void* pConstants; };
    struct ze_kernel_desc_t { std::uint32_t stype; const void* pNext; std::uint32_t flags; const char* pKernelName; };
    struct ze_group_count_t { std::uint32_t groupCountX; std::uint32_t groupCountY; std::uint32_t groupCountZ; };

    static constexpr std::uint32_t ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0x0d;
    static constexpr std::uint32_t ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0x0e;
    static constexpr std::uint32_t ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC = 0x0f;
    static constexpr std::uint32_t ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x15;
    static constexpr std::uint32_t ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 0x16;
    static constexpr std::uint32_t ZE_STRUCTURE_TYPE_MODULE_DESC = 0x1b;
    static constexpr std::uint32_t ZE_STRUCTURE_TYPE_KERNEL_DESC = 0x1d;
    static constexpr std::uint32_t ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS = 2u;
    static constexpr std::uint32_t ZE_MODULE_FORMAT_IL_SPIRV = 0u;
    static constexpr std::uint32_t ZE_MODULE_FORMAT_NATIVE = 1u;

    using cl_int = std::int32_t;
    using cl_uint = std::uint32_t;
    using cl_ulong = std::uint64_t;
    using cl_bitfield = cl_ulong;
    using cl_device_type = cl_bitfield;
    using cl_platform_info = cl_uint;
    using cl_program_info = cl_uint;
    using cl_context_properties = intptr_t;
    using cl_platform_id = struct _cl_platform_id*;
    using cl_device_id = struct _cl_device_id*;
    using cl_context = struct _cl_context*;
    using cl_program = struct _cl_program*;

    static constexpr cl_int CL_SUCCESS = 0;
    static constexpr cl_device_type CL_DEVICE_TYPE_GPU = 1u << 2u;
    static constexpr cl_platform_info CL_PLATFORM_NAME = 0x0902;
    static constexpr cl_program_info CL_PROGRAM_BINARY_SIZES = 0x1165;
    static constexpr cl_program_info CL_PROGRAM_BINARIES = 0x1166;
    static constexpr cl_program_info CL_PROGRAM_IL = 0x1169;

    struct OpenClApi {
        using cl_get_platform_ids_fn = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
        using cl_get_device_ids_fn = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
        using cl_create_context_fn = cl_context (*)(const cl_context_properties*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, std::size_t, void*), void*, cl_int*);
        using cl_release_context_fn = cl_int (*)(cl_context);
        using cl_create_program_with_source_fn = cl_program (*)(cl_context, cl_uint, const char**, const std::size_t*, cl_int*);
        using cl_build_program_fn = cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*);
        using cl_get_program_info_fn = cl_int (*)(cl_program, cl_program_info, std::size_t, void*, std::size_t*);
        using cl_release_program_fn = cl_int (*)(cl_program);

        LibraryHandle library = nullptr;
        bool ready = false;
        cl_get_platform_ids_fn get_platform_ids = nullptr;
        cl_get_device_ids_fn get_device_ids = nullptr;
        cl_create_context_fn create_context = nullptr;
        cl_release_context_fn release_context = nullptr;
        cl_create_program_with_source_fn create_program_with_source = nullptr;
        cl_build_program_fn build_program = nullptr;
        cl_get_program_info_fn get_program_info = nullptr;
        cl_release_program_fn release_program = nullptr;

        OpenClApi() {
#if defined(_WIN32)
            library = load_library("OpenCL.dll");
#else
            library = load_library("libOpenCL.so");
            if (library == nullptr) library = load_library("libOpenCL.so.1");
#endif
            if (library == nullptr) return;
            get_platform_ids = reinterpret_cast<cl_get_platform_ids_fn>(load_symbol(library, "clGetPlatformIDs"));
            get_device_ids = reinterpret_cast<cl_get_device_ids_fn>(load_symbol(library, "clGetDeviceIDs"));
            create_context = reinterpret_cast<cl_create_context_fn>(load_symbol(library, "clCreateContext"));
            release_context = reinterpret_cast<cl_release_context_fn>(load_symbol(library, "clReleaseContext"));
            create_program_with_source = reinterpret_cast<cl_create_program_with_source_fn>(load_symbol(library, "clCreateProgramWithSource"));
            build_program = reinterpret_cast<cl_build_program_fn>(load_symbol(library, "clBuildProgram"));
            get_program_info = reinterpret_cast<cl_get_program_info_fn>(load_symbol(library, "clGetProgramInfo"));
            release_program = reinterpret_cast<cl_release_program_fn>(load_symbol(library, "clReleaseProgram"));
            ready = get_platform_ids != nullptr && get_device_ids != nullptr && create_context != nullptr && release_context != nullptr && create_program_with_source != nullptr && build_program != nullptr && get_program_info != nullptr && release_program != nullptr;
        }

        ~OpenClApi() { close_library(library); }
    };

    struct Api {
        using ze_init_fn = ze_result_t (*)(std::uint32_t);
        using ze_driver_get_fn = ze_result_t (*)(std::uint32_t*, ze_driver_handle_t*);
        using ze_device_get_fn = ze_result_t (*)(ze_driver_handle_t, std::uint32_t*, ze_device_handle_t*);
        using ze_context_create_fn = ze_result_t (*)(ze_driver_handle_t, const ze_context_desc_t*, ze_context_handle_t*);
        using ze_command_queue_create_fn = ze_result_t (*)(ze_context_handle_t, ze_device_handle_t, const ze_command_queue_desc_t*, ze_command_queue_handle_t*);
        using ze_command_queue_execute_fn = ze_result_t (*)(ze_command_queue_handle_t, std::uint32_t, ze_command_list_handle_t*, void*);
        using ze_command_queue_synchronize_fn = ze_result_t (*)(ze_command_queue_handle_t, std::uint64_t);
        using ze_command_list_create_fn = ze_result_t (*)(ze_context_handle_t, ze_device_handle_t, const ze_command_list_desc_t*, ze_command_list_handle_t*);
        using ze_command_list_destroy_fn = ze_result_t (*)(ze_command_list_handle_t);
        using ze_command_list_close_fn = ze_result_t (*)(ze_command_list_handle_t);
        using ze_command_list_append_launch_kernel_fn = ze_result_t (*)(ze_command_list_handle_t, ze_kernel_handle_t, const ze_group_count_t*, void*, std::uint32_t, void*);
        using ze_mem_alloc_shared_fn = ze_result_t (*)(ze_context_handle_t, const ze_device_mem_alloc_desc_t*, const ze_host_mem_alloc_desc_t*, std::size_t, std::size_t, ze_device_handle_t, void**);
        using ze_mem_free_fn = ze_result_t (*)(ze_context_handle_t, void*);
        using ze_module_create_fn = ze_result_t (*)(ze_context_handle_t, ze_device_handle_t, const ze_module_desc_t*, ze_module_handle_t*, ze_module_build_log_handle_t*);
        using ze_kernel_create_fn = ze_result_t (*)(ze_module_handle_t, const ze_kernel_desc_t*, ze_kernel_handle_t*);
        using ze_kernel_set_group_size_fn = ze_result_t (*)(ze_kernel_handle_t, std::uint32_t, std::uint32_t, std::uint32_t);
        using ze_kernel_set_argument_value_fn = ze_result_t (*)(ze_kernel_handle_t, std::uint32_t, std::size_t, const void*);

        LibraryHandle library = nullptr;
        bool ready = false;
        ze_init_fn ze_init = nullptr;
        ze_driver_get_fn ze_driver_get = nullptr;
        ze_device_get_fn ze_device_get = nullptr;
        ze_context_create_fn ze_context_create = nullptr;
        ze_command_queue_create_fn ze_command_queue_create = nullptr;
        ze_command_queue_execute_fn ze_command_queue_execute = nullptr;
        ze_command_queue_synchronize_fn ze_command_queue_synchronize = nullptr;
        ze_command_list_create_fn ze_command_list_create = nullptr;
        ze_command_list_destroy_fn ze_command_list_destroy = nullptr;
        ze_command_list_close_fn ze_command_list_close = nullptr;
        ze_command_list_append_launch_kernel_fn ze_command_list_append_launch_kernel = nullptr;
        ze_mem_alloc_shared_fn ze_mem_alloc_shared = nullptr;
        ze_mem_free_fn ze_mem_free = nullptr;
        ze_module_create_fn ze_module_create = nullptr;
        ze_kernel_create_fn ze_kernel_create = nullptr;
        ze_kernel_set_group_size_fn ze_kernel_set_group_size = nullptr;
        ze_kernel_set_argument_value_fn ze_kernel_set_argument_value = nullptr;

        Api() {
#if defined(_WIN32)
            library = load_library("ze_loader.dll");
#else
            library = load_library("libze_loader.so");
            if (library == nullptr) library = load_library("libze_loader.so.1");
#endif
            if (library == nullptr) return;
            ze_init = reinterpret_cast<ze_init_fn>(load_symbol(library, "zeInit"));
            ze_driver_get = reinterpret_cast<ze_driver_get_fn>(load_symbol(library, "zeDriverGet"));
            ze_device_get = reinterpret_cast<ze_device_get_fn>(load_symbol(library, "zeDeviceGet"));
            ze_context_create = reinterpret_cast<ze_context_create_fn>(load_symbol(library, "zeContextCreate"));
            ze_command_queue_create = reinterpret_cast<ze_command_queue_create_fn>(load_symbol(library, "zeCommandQueueCreate"));
            ze_command_queue_execute = reinterpret_cast<ze_command_queue_execute_fn>(load_symbol(library, "zeCommandQueueExecuteCommandLists"));
            ze_command_queue_synchronize = reinterpret_cast<ze_command_queue_synchronize_fn>(load_symbol(library, "zeCommandQueueSynchronize"));
            ze_command_list_create = reinterpret_cast<ze_command_list_create_fn>(load_symbol(library, "zeCommandListCreate"));
            ze_command_list_destroy = reinterpret_cast<ze_command_list_destroy_fn>(load_symbol(library, "zeCommandListDestroy"));
            ze_command_list_close = reinterpret_cast<ze_command_list_close_fn>(load_symbol(library, "zeCommandListClose"));
            ze_command_list_append_launch_kernel = reinterpret_cast<ze_command_list_append_launch_kernel_fn>(load_symbol(library, "zeCommandListAppendLaunchKernel"));
            ze_mem_alloc_shared = reinterpret_cast<ze_mem_alloc_shared_fn>(load_symbol(library, "zeMemAllocShared"));
            ze_mem_free = reinterpret_cast<ze_mem_free_fn>(load_symbol(library, "zeMemFree"));
            ze_module_create = reinterpret_cast<ze_module_create_fn>(load_symbol(library, "zeModuleCreate"));
            ze_kernel_create = reinterpret_cast<ze_kernel_create_fn>(load_symbol(library, "zeKernelCreate"));
            ze_kernel_set_group_size = reinterpret_cast<ze_kernel_set_group_size_fn>(load_symbol(library, "zeKernelSetGroupSize"));
            ze_kernel_set_argument_value = reinterpret_cast<ze_kernel_set_argument_value_fn>(load_symbol(library, "zeKernelSetArgumentValue"));
            ready = ze_init != nullptr && ze_driver_get != nullptr && ze_device_get != nullptr && ze_context_create != nullptr && ze_command_queue_create != nullptr && ze_command_queue_execute != nullptr && ze_command_queue_synchronize != nullptr && ze_command_list_create != nullptr && ze_command_list_destroy != nullptr && ze_command_list_close != nullptr && ze_command_list_append_launch_kernel != nullptr && ze_mem_alloc_shared != nullptr && ze_mem_free != nullptr && ze_module_create != nullptr && ze_kernel_create != nullptr && ze_kernel_set_group_size != nullptr && ze_kernel_set_argument_value != nullptr;
        }

        ~Api() { close_library(library); }
    };

    struct BinaryBlob { std::vector<std::uint8_t> bytes; std::uint32_t format = ZE_MODULE_FORMAT_NATIVE; };
    struct Context {
        ze_context_handle_t context = nullptr;
        ze_device_handle_t device = nullptr;
        ze_command_queue_handle_t queue = nullptr;
        ze_module_handle_t module = nullptr;
        std::unordered_map<std::string, ze_kernel_handle_t> kernels;
        std::unordered_map<std::string, void*> reusable_buffers;
        std::unordered_map<std::string, std::size_t> reusable_buffer_sizes;
        std::unordered_map<std::string, std::uint64_t> reusable_buffer_last_used;
        std::uint64_t reusable_buffer_bytes = 0u;
        std::uint64_t reusable_buffer_budget_bytes = 0u;
        std::uint64_t reusable_buffer_tick = 0u;
        std::string revision;
        std::mutex execution_mutex;
    };

    const Api& api() const { static const Api instance; return instance; }
    const OpenClApi& opencl_api() const { static const OpenClApi instance; return instance; }

    bool alloc_shared(const Context& context, const std::size_t bytes, void** ptr) const {
        ze_device_mem_alloc_desc_t device_desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0u, 0u};
        ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0u};
        return api().ze_mem_alloc_shared(context.context, &device_desc, &host_desc, bytes, sizeof(float), context.device, ptr) == 0;
    }

    void free_shared(const Context& context, void* ptr) const {
        if (ptr != nullptr) (void)api().ze_mem_free(context.context, ptr);
    }

    std::uint64_t reusable_buffer_budget_bytes(const HardwareGraph& graph) const {
        const auto summary = summarize_graph(graph);
        const auto capacity = summary.directly_attached_bytes > 0u
                                  ? summary.directly_attached_bytes
                                  : std::max(summary.addressable_bytes, summary.shared_host_bytes);
        return std::clamp<std::uint64_t>(capacity / 16u, 16ull * 1024ull * 1024ull, 256ull * 1024ull * 1024ull);
    }

    void touch_buffer(Context& context, const std::string& slot) const {
        context.reusable_buffer_last_used[slot] = ++context.reusable_buffer_tick;
    }

    void trim_reusable_buffers(Context& context, const std::string& preserve_slot, const std::size_t incoming_bytes) const {
        while (!context.reusable_buffers.empty() &&
               context.reusable_buffer_budget_bytes > 0u &&
               context.reusable_buffer_bytes + incoming_bytes > context.reusable_buffer_budget_bytes) {
            auto oldest = context.reusable_buffers.end();
            for (auto it = context.reusable_buffers.begin(); it != context.reusable_buffers.end(); ++it) {
                if (it->first == preserve_slot) {
                    continue;
                }
                if (oldest == context.reusable_buffers.end() ||
                    context.reusable_buffer_last_used[it->first] < context.reusable_buffer_last_used[oldest->first]) {
                    oldest = it;
                }
            }
            if (oldest == context.reusable_buffers.end()) {
                break;
            }
            context.reusable_buffer_bytes -= context.reusable_buffer_sizes[oldest->first];
            free_shared(context, oldest->second);
            context.reusable_buffer_sizes.erase(oldest->first);
            context.reusable_buffer_last_used.erase(oldest->first);
            context.reusable_buffers.erase(oldest);
        }
    }

    void invalidate_reusable_buffers(Context& context) const {
        for (const auto& [slot, ptr] : context.reusable_buffers) {
            (void)slot;
            free_shared(context, ptr);
        }
        context.reusable_buffers.clear();
        context.reusable_buffer_sizes.clear();
        context.reusable_buffer_last_used.clear();
        context.reusable_buffer_bytes = 0u;
        context.reusable_buffer_tick = 0u;
    }

    bool ensure_shared_buffer(Context& context, const std::string& slot, const std::size_t bytes, void** ptr) const {
        if (bytes == 0u) {
            *ptr = nullptr;
            return true;
        }
        auto& slot_ptr = context.reusable_buffers[slot];
        auto& slot_size = context.reusable_buffer_sizes[slot];
        if (slot_ptr == nullptr || slot_size < bytes) {
            if (slot_ptr != nullptr) {
                context.reusable_buffer_bytes -= slot_size;
            }
            free_shared(context, slot_ptr);
            slot_ptr = nullptr;
            slot_size = 0u;
            trim_reusable_buffers(context, slot, bytes);
            if (!alloc_shared(context, bytes, &slot_ptr)) {
                return false;
            }
            slot_size = bytes;
            context.reusable_buffer_bytes += bytes;
        }
        touch_buffer(context, slot);
        *ptr = slot_ptr;
        return true;
    }

    BinaryBlob compile_binary(const bool low_precision_variant) const {
        const auto variant_index = low_precision_variant ? 1u : 0u;
        std::scoped_lock lock(binary_mutex_);
        if (!cached_binaries_[variant_index].bytes.empty()) return cached_binaries_[variant_index];
        if (const auto ocloc = locate_ocloc(); ocloc.has_value()) {
            const auto temp_dir = std::filesystem::temp_directory_path() / "jakal-level-zero-ocloc";
            std::error_code ignore_error;
            std::filesystem::create_directories(temp_dir, ignore_error);
            const std::string variant_name = low_precision_variant ? "jakal_level_zero_native_lowp" : "jakal_level_zero_native_strict";
            const auto source_path = temp_dir / (variant_name + ".cl");
            const auto output_spv = temp_dir / (variant_name + ".spv");
            const auto output_bin = temp_dir / (variant_name + ".bin");
            const std::string source_text = low_precision_variant
                                                ? std::string(kOpenClFp16ProgramSource)
                                                : std::string("#define JAKAL_ALWAYS_LOW_PRECISION 0\n") + kOpenClProgramSource;
            if (write_text_file(source_path, source_text)) {
                std::string command =
#if defined(_WIN32)
                    "powershell -NoProfile -Command \"& '" + ocloc->string() + "' compile -file '" + source_path.string() +
                    "' -device xe-lp -output_no_suffix -output '" + variant_name +
                    "' -out_dir '" + temp_dir.string() + "'\"";
#else
                    "\"" + ocloc->string() + "\" compile -file \"" + source_path.string() +
                    "\" -device xe-lp -output_no_suffix -output \"" + variant_name +
                    "\" -out_dir \"" + temp_dir.string() + "\"";
#endif
                if (run_command(command) == 0) {
                    const auto spirv = read_binary_file(output_spv);
                    if (!spirv.empty()) {
                        cached_binaries_[variant_index].bytes = spirv;
                        cached_binaries_[variant_index].format = ZE_MODULE_FORMAT_IL_SPIRV;
                        return cached_binaries_[variant_index];
                    }
                    const auto bytes = read_binary_file(output_bin);
                    if (!bytes.empty()) {
                        cached_binaries_[variant_index].bytes = bytes;
                        cached_binaries_[variant_index].format = ZE_MODULE_FORMAT_NATIVE;
                        return cached_binaries_[variant_index];
                    }
                }
            }
        }
        if (!opencl_api().ready) return {};
        cl_uint platform_count = 0;
        if (opencl_api().get_platform_ids(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0) return {};
        std::vector<cl_platform_id> platforms(platform_count, nullptr);
        if (opencl_api().get_platform_ids(platform_count, platforms.data(), nullptr) != CL_SUCCESS) return {};
        cl_device_id device = nullptr;
        for (const auto platform : platforms) {
            cl_uint device_count = 0;
            if (opencl_api().get_device_ids(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count) != CL_SUCCESS || device_count == 0) continue;
            std::vector<cl_device_id> devices(device_count, nullptr);
            if (opencl_api().get_device_ids(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr) != CL_SUCCESS) continue;
            device = devices.front();
            break;
        }
        if (device == nullptr) return {};
        cl_int error = CL_SUCCESS;
        cl_context context = opencl_api().create_context(nullptr, 1, &device, nullptr, nullptr, &error);
        if (context == nullptr || error != CL_SUCCESS) return {};
        const char* source = kOpenClProgramSource;
        cl_program program = opencl_api().create_program_with_source(context, 1, &source, nullptr, &error);
        if (program == nullptr || error != CL_SUCCESS) { opencl_api().release_context(context); return {}; }
        if (opencl_api().build_program(program, 1, &device, "", nullptr, nullptr) != CL_SUCCESS) { opencl_api().release_program(program); opencl_api().release_context(context); return {}; }
        std::size_t il_size = 0;
        if (opencl_api().get_program_info(program, CL_PROGRAM_IL, 0, nullptr, &il_size) == CL_SUCCESS && il_size > 0) {
            cached_binaries_[variant_index].bytes.resize(il_size);
            if (opencl_api().get_program_info(program, CL_PROGRAM_IL, il_size, cached_binaries_[variant_index].bytes.data(), nullptr) == CL_SUCCESS) cached_binaries_[variant_index].format = ZE_MODULE_FORMAT_IL_SPIRV;
            else cached_binaries_[variant_index].bytes.clear();
        }
        if (cached_binaries_[variant_index].bytes.empty()) {
            std::size_t binary_size = 0;
            if (opencl_api().get_program_info(program, CL_PROGRAM_BINARY_SIZES, sizeof(std::size_t), &binary_size, nullptr) == CL_SUCCESS && binary_size > 0) {
                cached_binaries_[variant_index].bytes.resize(binary_size);
                unsigned char* binary_ptr = cached_binaries_[variant_index].bytes.data();
                if (opencl_api().get_program_info(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &binary_ptr, nullptr) == CL_SUCCESS) cached_binaries_[variant_index].format = ZE_MODULE_FORMAT_NATIVE;
                else cached_binaries_[variant_index].bytes.clear();
            }
        }
        opencl_api().release_program(program);
        opencl_api().release_context(context);
        return cached_binaries_[variant_index];
    }

    std::shared_ptr<Context> acquire_context(const HardwareGraph& graph, const bool low_precision_variant) const {
        std::scoped_lock lock(mutex_);
        last_error_.clear();
        if (!api().ready) { last_error_ = "level-zero-loader"; return {}; }
        if (api().ze_init(0u) != 0) { last_error_ = "level-zero-init"; return {}; }
        const auto context_key = graph.uid + (low_precision_variant ? ":lowp" : ":strict");
        const auto revision = structural_fingerprint(graph);
        if (const auto existing = contexts_.find(context_key); existing != contexts_.end()) {
            if (existing->second->revision != revision) {
                invalidate_reusable_buffers(*existing->second);
                existing->second->revision = revision;
                existing->second->reusable_buffer_budget_bytes = reusable_buffer_budget_bytes(graph);
            }
            return existing->second;
        }
        std::uint32_t driver_count = 0;
        if (api().ze_driver_get(&driver_count, nullptr) != 0 || driver_count == 0) { last_error_ = "level-zero-driver-enum"; return {}; }
        std::vector<ze_driver_handle_t> drivers(driver_count, nullptr);
        if (api().ze_driver_get(&driver_count, drivers.data()) != 0) { last_error_ = "level-zero-driver-list"; return {}; }
        std::uint32_t ordinal = 0;
        for (const auto driver : drivers) {
            std::uint32_t device_count = 0;
            if (api().ze_device_get(driver, &device_count, nullptr) != 0 || device_count == 0) continue;
            std::vector<ze_device_handle_t> devices(device_count, nullptr);
            if (api().ze_device_get(driver, &device_count, devices.data()) != 0) continue;
            for (const auto device : devices) {
                if (ordinal++ != graph.ordinal) continue;
                auto blob = compile_binary(low_precision_variant);
                if (blob.bytes.empty()) { last_error_ = "level-zero-binary"; return {}; }
                auto context = std::make_shared<Context>();
                context->device = device;
                ze_context_desc_t context_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0u};
                ze_command_queue_desc_t queue_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0u, 0u, 0u, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, 0u};
                ze_module_desc_t module_desc{ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr, blob.format, blob.bytes.size(), blob.bytes.data(), "", nullptr};
                ze_module_build_log_handle_t build_log = nullptr;
                if (api().ze_context_create(driver, &context_desc, &context->context) != 0) { last_error_ = "level-zero-context-create"; return {}; }
                if (api().ze_command_queue_create(context->context, device, &queue_desc, &context->queue) != 0) { last_error_ = "level-zero-queue-create"; return {}; }
                if (api().ze_module_create(context->context, device, &module_desc, &context->module, &build_log) != 0) { last_error_ = "level-zero-module-create"; return {}; }
                context->revision = revision;
                context->reusable_buffer_budget_bytes = reusable_buffer_budget_bytes(graph);
                contexts_.emplace(context_key, context);
                return context;
            }
        }
        last_error_ = "level-zero-device-match";
        return {};
    }

    ze_kernel_handle_t get_kernel(Context& context, const char* name) const {
        if (const auto existing = context.kernels.find(name); existing != context.kernels.end()) return existing->second;
        ze_kernel_desc_t desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0u, name};
        ze_kernel_handle_t kernel = nullptr;
        if (api().ze_kernel_create(context.module, &desc, &kernel) != 0 || kernel == nullptr) return nullptr;
        context.kernels.emplace(name, kernel);
        return kernel;
    }

    bool launch(Context& context, const char* name, const std::array<std::uint32_t, 3>& group_size, const std::array<std::uint32_t, 3>& group_count, const std::vector<std::pair<std::size_t, const void*>>& args, const std::vector<std::pair<std::uint32_t, std::size_t>>& local_args = {}) const {
        ze_command_list_desc_t list_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0u, 0u};
        ze_command_list_handle_t list = nullptr;
        if (api().ze_command_list_create(context.context, context.device, &list_desc, &list) != 0 || list == nullptr) return false;
        auto kernel = get_kernel(context, name);
        if (kernel == nullptr || api().ze_kernel_set_group_size(kernel, group_size[0], group_size[1], group_size[2]) != 0) { api().ze_command_list_destroy(list); return false; }
        for (std::uint32_t index = 0; index < args.size(); ++index) {
            if (api().ze_kernel_set_argument_value(kernel, index, args[index].first, args[index].second) != 0) { api().ze_command_list_destroy(list); return false; }
        }
        for (const auto& [index, bytes] : local_args) {
            if (api().ze_kernel_set_argument_value(kernel, index, bytes, nullptr) != 0) { api().ze_command_list_destroy(list); return false; }
        }
        ze_group_count_t dispatch{group_count[0], group_count[1], group_count[2]};
        const bool ok = api().ze_command_list_append_launch_kernel(list, kernel, &dispatch, nullptr, 0u, nullptr) == 0 && api().ze_command_list_close(list) == 0 && api().ze_command_queue_execute(context.queue, 1u, &list, nullptr) == 0 && api().ze_command_queue_synchronize(context.queue, std::numeric_limits<std::uint64_t>::max()) == 0;
        api().ze_command_list_destroy(list);
        return ok;
    }

    BackendRunResult run_elementwise_native(const HardwareGraph& graph, const std::span<const float> lhs, const std::span<const float> rhs, const bool low_precision) const override {
        auto context = acquire_context(graph, low_precision);
        if (context == nullptr) return failure(last_error_.empty() ? "level-zero-context" : last_error_);
        BackendRunResult result;
        result.output.resize(lhs.size(), 0.0f);
        double copy_in_us = 0.0;
        double copy_out_us = 0.0;
        double compute_us = 0.0;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock lock(context->execution_mutex);
            void* lhs_mem = nullptr; void* rhs_mem = nullptr; void* out_mem = nullptr;
            if (!ensure_shared_buffer(*context, "elementwise-lhs", lhs.size_bytes(), &lhs_mem) ||
                !ensure_shared_buffer(*context, "elementwise-rhs", rhs.size_bytes(), &rhs_mem) ||
                !ensure_shared_buffer(*context, "elementwise-out", result.output.size() * sizeof(float), &out_mem)) {
                result.error = "level-zero-memory"; return;
            }
            copy_in_us += measure_us([&]() {
                std::memcpy(lhs_mem, lhs.data(), lhs.size_bytes());
                std::memcpy(rhs_mem, rhs.data(), rhs.size_bytes());
            });
            unsigned int count = static_cast<unsigned int>(lhs.size()); int low = 0;
            compute_us += measure_us([&]() {
                if (!launch(*context, "elementwise_map", {256u,1u,1u}, {count == 0 ? 1u : (count + 255u) / 256u, 1u, 1u}, {{sizeof(void*), &lhs_mem}, {sizeof(void*), &rhs_mem}, {sizeof(void*), &out_mem}, {sizeof(unsigned int), &count}, {sizeof(int), &low}})) {
                    result.error = "level-zero-launch";
                }
            });
            if (result.error.empty()) {
                copy_out_us += measure_us([&]() {
                    std::memcpy(result.output.data(), out_mem, result.output.size() * sizeof(float));
                });
                result.success = true; result.used_host = false; result.used_opencl = false;
            }
        });
        result.copy_runtime_us = copy_in_us + copy_out_us;
        result.compute_runtime_us = compute_us;
        return result;
    }

    BackendRunResult run_reduction_native(const HardwareGraph& graph, const std::span<const float> input, const bool low_precision) const override {
        auto context = acquire_context(graph, low_precision);
        if (context == nullptr) return failure(last_error_.empty() ? "level-zero-context" : last_error_);
        BackendRunResult result;
        double copy_in_us = 0.0;
        double copy_out_us = 0.0;
        double compute_us = 0.0;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock lock(context->execution_mutex);
            const auto local = kNativeReductionGroupSize;
            const auto global = std::min<std::size_t>(std::max<std::size_t>(local, round_up<std::size_t>(input.size(), local)), local * 64u);
            const auto groups = static_cast<std::uint32_t>(global / local);
            std::vector<float> partials(groups, 0.0f);
            void* in_mem = nullptr; void* partial_mem = nullptr;
            if (!ensure_shared_buffer(*context, "reduce-in", input.size_bytes(), &in_mem) ||
                !ensure_shared_buffer(*context, "reduce-partial", partials.size() * sizeof(float), &partial_mem)) {
                result.error = "level-zero-memory"; return;
            }
            copy_in_us += measure_us([&]() {
                std::memcpy(in_mem, input.data(), input.size_bytes());
            });
            unsigned int count = static_cast<unsigned int>(input.size()); int low = 0;
            compute_us += measure_us([&]() {
                if (!launch(*context, "reduce_sum", {local,1u,1u}, {groups,1u,1u}, {{sizeof(void*), &in_mem}, {sizeof(void*), &partial_mem}, {sizeof(unsigned int), &count}, {sizeof(int), &low}}, {{4u, local * sizeof(float)}})) {
                    result.error = "level-zero-launch";
                }
            });
            if (result.error.empty()) {
                copy_out_us += measure_us([&]() {
                    std::memcpy(partials.data(), partial_mem, partials.size() * sizeof(float));
                });
                for (const auto value : partials) result.scalar_output += value; result.success = true; result.used_host = false; result.used_opencl = false;
            }
        });
        result.copy_runtime_us = copy_in_us + copy_out_us;
        result.compute_runtime_us = compute_us;
        return result;
    }

    BackendRunResult run_matmul_native(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> lhs, const std::span<const float> rhs, const std::uint32_t rows, const std::uint32_t columns, const std::uint32_t depth, const bool low_precision) const override {
        auto context = acquire_context(graph, low_precision);
        if (context == nullptr) return failure(last_error_.empty() ? "level-zero-context" : last_error_);
        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(rows) * columns, 0.0f);
        double copy_in_us = 0.0;
        double copy_out_us = 0.0;
        double compute_us = 0.0;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock lock(context->execution_mutex);
            void* lhs_mem = nullptr; void* rhs_mem = nullptr; void* out_mem = nullptr;
            if (!ensure_shared_buffer(*context, "matmul-lhs", lhs.size_bytes(), &lhs_mem) ||
                !ensure_shared_buffer(*context, "matmul-rhs", rhs.size_bytes(), &rhs_mem) ||
                !ensure_shared_buffer(*context, "matmul-out", result.output.size() * sizeof(float), &out_mem)) {
                result.error = "level-zero-memory"; return;
            }
            copy_in_us += measure_us([&]() {
                std::memcpy(lhs_mem, lhs.data(), lhs.size_bytes());
                std::memcpy(rhs_mem, rhs.data(), rhs.size_bytes());
            });
            unsigned int row_count = rows, column_count = columns, depth_count = depth; int low = 0;
            const char* kernel_name = gpu_rhs_uses_transposed_layout(operation) ? "matmul_tiled_rhs_t" : "matmul_tiled";
            compute_us += measure_us([&]() {
                if (!launch(*context, kernel_name, {kNativeTileSize,kNativeTileSize,1u}, {columns == 0 ? 1u : (columns + kNativeTileSize - 1u) / kNativeTileSize, rows == 0 ? 1u : (rows + kNativeTileSize - 1u) / kNativeTileSize, 1u}, {{sizeof(void*), &lhs_mem}, {sizeof(void*), &rhs_mem}, {sizeof(void*), &out_mem}, {sizeof(unsigned int), &row_count}, {sizeof(unsigned int), &column_count}, {sizeof(unsigned int), &depth_count}, {sizeof(int), &low}}, {{7u, kNativeTileSize * kNativeTileSize * sizeof(float)}, {8u, kNativeTileSize * kNativeTileSize * sizeof(float)}})) {
                    result.error = "level-zero-launch";
                }
            });
            if (result.error.empty()) {
                copy_out_us += measure_us([&]() {
                    std::memcpy(result.output.data(), out_mem, result.output.size() * sizeof(float));
                });
                result.success = true; result.used_host = false; result.used_opencl = false;
            }
        });
        result.copy_runtime_us = copy_in_us + copy_out_us;
        result.compute_runtime_us = compute_us;
        return result;
    }

    BackendRunResult run_conv3x3_native(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> input, const std::uint32_t height, const std::uint32_t width, const bool low_precision) const override {
        auto context = acquire_context(graph, low_precision);
        if (context == nullptr) return failure(last_error_.empty() ? "level-zero-context" : last_error_);
        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(height - 2u) * (width - 2u), 0.0f);
        double copy_in_us = 0.0;
        double copy_out_us = 0.0;
        double compute_us = 0.0;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock lock(context->execution_mutex);
            void* in_mem = nullptr;
            void* out_mem = nullptr;
            if (!ensure_shared_buffer(*context, "conv-in", input.size_bytes(), &in_mem) ||
                !ensure_shared_buffer(*context, "conv-out", result.output.size() * sizeof(float), &out_mem)) {
                result.error = "level-zero-memory";
                return;
            }
            copy_in_us += measure_us([&]() {
                std::memcpy(in_mem, input.data(), input.size_bytes());
            });
            unsigned int input_height = height;
            unsigned int input_width = width;
            int low = 0;
            const std::uint32_t out_width = width - 2u;
            const std::uint32_t out_height = height - 2u;
            compute_us += measure_us([&]() {
                if (!launch(
                        *context,
                        gpu_conv_uses_patch9_layout(operation) ? "conv3x3_valid_patch9" : "conv3x3_valid",
                        {kNativeTileSize, kNativeTileSize, 1u},
                        {
                            out_width == 0u ? 1u : (out_width + kNativeTileSize - 1u) / kNativeTileSize,
                            out_height == 0u ? 1u : (out_height + kNativeTileSize - 1u) / kNativeTileSize,
                            1u,
                        },
                        {
                            {sizeof(void*), &in_mem},
                            {sizeof(void*), &out_mem},
                            {sizeof(unsigned int), &input_height},
                            {sizeof(unsigned int), &input_width},
                            {sizeof(int), &low},
                        })) {
                    result.error = "level-zero-launch";
                }
            });
            if (result.error.empty()) {
                copy_out_us += measure_us([&]() {
                    std::memcpy(result.output.data(), out_mem, result.output.size() * sizeof(float));
                });
                result.success = true;
                result.used_host = false;
                result.used_opencl = false;
            }
        });
        result.copy_runtime_us = copy_in_us + copy_out_us;
        result.compute_runtime_us = compute_us;
        return result;
    }

    BackendRunResult run_resample_native(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> input, const std::uint32_t src_h, const std::uint32_t src_w, const std::uint32_t dst_h, const std::uint32_t dst_w, const std::uint32_t row_offset, const std::uint32_t row_count, const bool low_precision) const override {
        auto context = acquire_context(graph, low_precision);
        if (context == nullptr) return failure(last_error_.empty() ? "level-zero-context" : last_error_);
        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(row_count) * dst_w, 0.0f);
        double copy_in_us = 0.0;
        double copy_out_us = 0.0;
        double compute_us = 0.0;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock lock(context->execution_mutex);
            void* in_mem = nullptr;
            void* out_mem = nullptr;
            if (!ensure_shared_buffer(*context, "resample-in", input.size_bytes(), &in_mem) ||
                !ensure_shared_buffer(*context, "resample-out", result.output.size() * sizeof(float), &out_mem)) {
                result.error = "level-zero-memory";
                return;
            }
            copy_in_us += measure_us([&]() {
                std::memcpy(in_mem, input.data(), input.size_bytes());
            });
            unsigned int src_height = src_h;
            unsigned int src_width = src_w;
            unsigned int dst_height = dst_h;
            unsigned int dst_width = dst_w;
            unsigned int dst_row_offset = row_offset;
            unsigned int dst_row_count = row_count;
            int low = 0;
            compute_us += measure_us([&]() {
                if (!launch(
                        *context,
                        gpu_resample_uses_packed6_layout(operation) ? "bilinear_resample_packed6" : "bilinear_resample",
                        {kNativeTileSize, kNativeTileSize, 1u},
                        {
                            dst_w == 0u ? 1u : (dst_w + kNativeTileSize - 1u) / kNativeTileSize,
                            row_count == 0u ? 1u : (row_count + kNativeTileSize - 1u) / kNativeTileSize,
                            1u,
                        },
                        {
                            {sizeof(void*), &in_mem},
                            {sizeof(void*), &out_mem},
                            {sizeof(unsigned int), &src_height},
                            {sizeof(unsigned int), &src_width},
                            {sizeof(unsigned int), &dst_height},
                            {sizeof(unsigned int), &dst_width},
                            {sizeof(unsigned int), &dst_row_offset},
                            {sizeof(unsigned int), &dst_row_count},
                            {sizeof(int), &low},
                        })) {
                    result.error = "level-zero-launch";
                }
            });
            if (result.error.empty()) {
                copy_out_us += measure_us([&]() {
                    std::memcpy(result.output.data(), out_mem, result.output.size() * sizeof(float));
                });
                result.success = true;
                result.used_host = false;
                result.used_opencl = false;
            }
        });
        result.copy_runtime_us = copy_in_us + copy_out_us;
        result.compute_runtime_us = compute_us;
        return result;
    }

    mutable std::mutex mutex_;
    mutable std::unordered_map<std::string, std::shared_ptr<Context>> contexts_;
    mutable std::mutex binary_mutex_;
    mutable std::array<BinaryBlob, 2> cached_binaries_;
    mutable std::string last_error_;
};

class RocmNativeBackend final : public NativeKernelBackendBase {
public:
    RocmNativeBackend()
        : NativeKernelBackendBase(JakalBackendKind::rocm, "rocm") {}

private:
    using hip_result_t = int;
    using hip_stream_t = void*;
    using hip_event_t = void*;
    using hip_module_t = void*;
    using hip_function_t = void*;
    using hiprtc_result_t = int;
    using hiprtc_program_t = void*;

    struct Api {
        using hip_init_fn = hip_result_t (*)(unsigned int);
        using hip_set_device_fn = hip_result_t (*)(int);
        using hip_stream_create_fn = hip_result_t (*)(hip_stream_t*);
        using hip_stream_synchronize_fn = hip_result_t (*)(hip_stream_t);
        using hip_stream_destroy_fn = hip_result_t (*)(hip_stream_t);
        using hip_stream_wait_event_fn = hip_result_t (*)(hip_stream_t, hip_event_t, unsigned int);
        using hip_event_create_fn = hip_result_t (*)(hip_event_t*);
        using hip_event_record_fn = hip_result_t (*)(hip_event_t, hip_stream_t);
        using hip_event_destroy_fn = hip_result_t (*)(hip_event_t);
        using hip_malloc_fn = hip_result_t (*)(void**, std::size_t);
        using hip_free_fn = hip_result_t (*)(void*);
        using hip_memcpy_htod_fn = hip_result_t (*)(void*, const void*, std::size_t);
        using hip_memcpy_dtoh_fn = hip_result_t (*)(void*, const void*, std::size_t);
        using hip_memcpy_htod_async_fn = hip_result_t (*)(void*, const void*, std::size_t, hip_stream_t);
        using hip_memcpy_dtoh_async_fn = hip_result_t (*)(void*, const void*, std::size_t, hip_stream_t);
        using hip_module_load_data_fn = hip_result_t (*)(hip_module_t*, const void*);
        using hip_module_unload_fn = hip_result_t (*)(hip_module_t);
        using hip_module_get_function_fn = hip_result_t (*)(hip_function_t*, hip_module_t, const char*);
        using hip_module_launch_kernel_fn = hip_result_t (*)(hip_function_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, hip_stream_t, void**, void**);
        using hiprtc_create_program_fn = hiprtc_result_t (*)(hiprtc_program_t*, const char*, const char*, int, const char* const*, const char* const*);
        using hiprtc_compile_program_fn = hiprtc_result_t (*)(hiprtc_program_t, int, const char* const*);
        using hiprtc_get_code_size_fn = hiprtc_result_t (*)(hiprtc_program_t, std::size_t*);
        using hiprtc_get_code_fn = hiprtc_result_t (*)(hiprtc_program_t, char*);
        using hiprtc_destroy_program_fn = hiprtc_result_t (*)(hiprtc_program_t*);

        LibraryHandle hip_library = nullptr;
        LibraryHandle hiprtc_library = nullptr;
        bool ready = false;
        hip_init_fn hip_init = nullptr;
        hip_set_device_fn hip_set_device = nullptr;
        hip_stream_create_fn hip_stream_create = nullptr;
        hip_stream_synchronize_fn hip_stream_synchronize = nullptr;
        hip_stream_destroy_fn hip_stream_destroy = nullptr;
        hip_stream_wait_event_fn hip_stream_wait_event = nullptr;
        hip_event_create_fn hip_event_create = nullptr;
        hip_event_record_fn hip_event_record = nullptr;
        hip_event_destroy_fn hip_event_destroy = nullptr;
        hip_malloc_fn hip_malloc = nullptr;
        hip_free_fn hip_free = nullptr;
        hip_memcpy_htod_fn hip_memcpy_htod = nullptr;
        hip_memcpy_dtoh_fn hip_memcpy_dtoh = nullptr;
        hip_memcpy_htod_async_fn hip_memcpy_htod_async = nullptr;
        hip_memcpy_dtoh_async_fn hip_memcpy_dtoh_async = nullptr;
        hip_module_load_data_fn hip_module_load_data = nullptr;
        hip_module_unload_fn hip_module_unload = nullptr;
        hip_module_get_function_fn hip_module_get_function = nullptr;
        hip_module_launch_kernel_fn hip_module_launch_kernel = nullptr;
        hiprtc_create_program_fn hiprtc_create_program = nullptr;
        hiprtc_compile_program_fn hiprtc_compile_program = nullptr;
        hiprtc_get_code_size_fn hiprtc_get_code_size = nullptr;
        hiprtc_get_code_fn hiprtc_get_code = nullptr;
        hiprtc_destroy_program_fn hiprtc_destroy_program = nullptr;

        Api() {
#if defined(_WIN32)
            hip_library = load_library("amdhip64.dll");
            hiprtc_library = load_library("hiprtc0507.dll");
            if (hiprtc_library == nullptr) hiprtc_library = load_library("hiprtc0506.dll");
            if (hiprtc_library == nullptr) hiprtc_library = load_library("hiprtc.dll");
#else
            hip_library = load_library("libamdhip64.so");
            if (hip_library == nullptr) hip_library = load_library("libamdhip64.so.6");
            hiprtc_library = load_library("libhiprtc.so");
#endif
            if (hip_library == nullptr || hiprtc_library == nullptr) return;
            hip_init = reinterpret_cast<hip_init_fn>(load_symbol(hip_library, "hipInit"));
            hip_set_device = reinterpret_cast<hip_set_device_fn>(load_symbol(hip_library, "hipSetDevice"));
            hip_stream_create = reinterpret_cast<hip_stream_create_fn>(load_symbol(hip_library, "hipStreamCreate"));
            hip_stream_synchronize = reinterpret_cast<hip_stream_synchronize_fn>(load_symbol(hip_library, "hipStreamSynchronize"));
            hip_stream_destroy = reinterpret_cast<hip_stream_destroy_fn>(load_symbol(hip_library, "hipStreamDestroy"));
            hip_stream_wait_event = reinterpret_cast<hip_stream_wait_event_fn>(load_symbol(hip_library, "hipStreamWaitEvent"));
            hip_event_create = reinterpret_cast<hip_event_create_fn>(load_symbol(hip_library, "hipEventCreate"));
            hip_event_record = reinterpret_cast<hip_event_record_fn>(load_symbol(hip_library, "hipEventRecord"));
            hip_event_destroy = reinterpret_cast<hip_event_destroy_fn>(load_symbol(hip_library, "hipEventDestroy"));
            hip_malloc = reinterpret_cast<hip_malloc_fn>(load_symbol(hip_library, "hipMalloc"));
            hip_free = reinterpret_cast<hip_free_fn>(load_symbol(hip_library, "hipFree"));
            hip_memcpy_htod = reinterpret_cast<hip_memcpy_htod_fn>(load_symbol(hip_library, "hipMemcpyHtoD"));
            hip_memcpy_dtoh = reinterpret_cast<hip_memcpy_dtoh_fn>(load_symbol(hip_library, "hipMemcpyDtoH"));
            hip_memcpy_htod_async =
                reinterpret_cast<hip_memcpy_htod_async_fn>(load_symbol(hip_library, "hipMemcpyHtoDAsync"));
            hip_memcpy_dtoh_async =
                reinterpret_cast<hip_memcpy_dtoh_async_fn>(load_symbol(hip_library, "hipMemcpyDtoHAsync"));
            hip_module_load_data = reinterpret_cast<hip_module_load_data_fn>(load_symbol(hip_library, "hipModuleLoadData"));
            hip_module_unload = reinterpret_cast<hip_module_unload_fn>(load_symbol(hip_library, "hipModuleUnload"));
            hip_module_get_function = reinterpret_cast<hip_module_get_function_fn>(load_symbol(hip_library, "hipModuleGetFunction"));
            hip_module_launch_kernel = reinterpret_cast<hip_module_launch_kernel_fn>(load_symbol(hip_library, "hipModuleLaunchKernel"));
            hiprtc_create_program = reinterpret_cast<hiprtc_create_program_fn>(load_symbol(hiprtc_library, "hiprtcCreateProgram"));
            hiprtc_compile_program = reinterpret_cast<hiprtc_compile_program_fn>(load_symbol(hiprtc_library, "hiprtcCompileProgram"));
            hiprtc_get_code_size = reinterpret_cast<hiprtc_get_code_size_fn>(load_symbol(hiprtc_library, "hiprtcGetCodeSize"));
            hiprtc_get_code = reinterpret_cast<hiprtc_get_code_fn>(load_symbol(hiprtc_library, "hiprtcGetCode"));
            hiprtc_destroy_program = reinterpret_cast<hiprtc_destroy_program_fn>(load_symbol(hiprtc_library, "hiprtcDestroyProgram"));
            ready = hip_init != nullptr && hip_set_device != nullptr && hip_stream_create != nullptr && hip_stream_synchronize != nullptr && hip_stream_destroy != nullptr && hip_malloc != nullptr && hip_free != nullptr && hip_memcpy_htod != nullptr && hip_memcpy_dtoh != nullptr && hip_module_load_data != nullptr && hip_module_unload != nullptr && hip_module_get_function != nullptr && hip_module_launch_kernel != nullptr && hiprtc_create_program != nullptr && hiprtc_compile_program != nullptr && hiprtc_get_code_size != nullptr && hiprtc_get_code != nullptr && hiprtc_destroy_program != nullptr;
        }

        ~Api() { close_library(hiprtc_library); close_library(hip_library); }
    };

    struct Context {
        hip_stream_t stream = nullptr;
        hip_stream_t copy_stream = nullptr;
        hip_event_t copy_ready_event = nullptr;
        hip_event_t compute_ready_event = nullptr;
        hip_module_t module = nullptr;
        std::array<std::vector<float>, 2> staging_buffers;
        std::unordered_map<std::string, hip_function_t> kernels;
        std::unordered_map<std::string, void*> reusable_buffers;
        std::unordered_map<std::string, std::size_t> reusable_buffer_sizes;
        std::unordered_map<std::string, std::uint64_t> reusable_buffer_last_used;
        std::uint64_t reusable_buffer_bytes = 0u;
        std::uint64_t reusable_buffer_budget_bytes = 0u;
        std::uint64_t reusable_buffer_tick = 0u;
        std::string revision;
        std::mutex execution_mutex;
    };

    const Api& api() const { static const Api instance; return instance; }

    bool supports_split_streams() const {
        return api().hip_stream_wait_event != nullptr &&
               api().hip_event_create != nullptr &&
               api().hip_event_record != nullptr &&
               api().hip_event_destroy != nullptr &&
               api().hip_memcpy_htod_async != nullptr &&
               api().hip_memcpy_dtoh_async != nullptr;
    }

    bool can_split_dispatch(const Context& context) const {
        return supports_split_streams() &&
               context.copy_stream != nullptr &&
               context.copy_ready_event != nullptr &&
               context.compute_ready_event != nullptr;
    }

    void destroy_context(Context& context) const {
        invalidate_reusable_buffers(context);
        context.kernels.clear();
        if (context.module != nullptr) {
            (void)api().hip_module_unload(context.module);
            context.module = nullptr;
        }
        if (context.copy_ready_event != nullptr) {
            (void)api().hip_event_destroy(context.copy_ready_event);
            context.copy_ready_event = nullptr;
        }
        if (context.compute_ready_event != nullptr) {
            (void)api().hip_event_destroy(context.compute_ready_event);
            context.compute_ready_event = nullptr;
        }
        if (context.copy_stream != nullptr) {
            (void)api().hip_stream_destroy(context.copy_stream);
            context.copy_stream = nullptr;
        }
        if (context.stream != nullptr) {
            (void)api().hip_stream_destroy(context.stream);
            context.stream = nullptr;
        }
    }

    std::shared_ptr<Context> acquire_context(const HardwareGraph& graph) const {
        std::scoped_lock lock(mutex_);
        if (!api().ready || api().hip_init(0u) != 0 || api().hip_set_device(static_cast<int>(graph.ordinal)) != 0) return {};
        const auto revision = structural_fingerprint(graph);
        if (const auto existing = contexts_.find(graph.uid); existing != contexts_.end()) {
            if (existing->second->revision != revision) {
                invalidate_reusable_buffers(*existing->second);
                existing->second->revision = revision;
                existing->second->reusable_buffer_budget_bytes = reusable_buffer_budget_bytes(graph);
            }
            return existing->second;
        }
        auto context = std::make_shared<Context>();
        if (api().hip_stream_create(&context->stream) != 0) return {};
        if (supports_split_streams()) {
            if (api().hip_stream_create(&context->copy_stream) != 0 ||
                api().hip_event_create(&context->copy_ready_event) != 0 ||
                api().hip_event_create(&context->compute_ready_event) != 0) {
                destroy_context(*context);
                return {};
            }
        }
        if (!compile_module(*context)) {
            destroy_context(*context);
            return {};
        }
        context->revision = revision;
        context->reusable_buffer_budget_bytes = reusable_buffer_budget_bytes(graph);
        contexts_.emplace(graph.uid, context);
        return context;
    }

    bool compile_module(Context& context) const {
        hiprtc_program_t program = nullptr;
        if (api().hiprtc_create_program(&program, kCudaLikeProgramSource, "jakal_native.hip", 0, nullptr, nullptr) != 0) return false;
        if (api().hiprtc_compile_program(program, 0, nullptr) != 0) { api().hiprtc_destroy_program(&program); return false; }
        std::size_t code_size = 0;
        if (api().hiprtc_get_code_size(program, &code_size) != 0 || code_size == 0) { api().hiprtc_destroy_program(&program); return false; }
        std::string code(code_size, '\0');
        const bool ok = api().hiprtc_get_code(program, code.data()) == 0 && api().hip_module_load_data(&context.module, code.data()) == 0;
        api().hiprtc_destroy_program(&program);
        return ok;
    }

    hip_function_t get_kernel(Context& context, const char* name) const {
        if (const auto existing = context.kernels.find(name); existing != context.kernels.end()) return existing->second;
        hip_function_t kernel = nullptr;
        if (api().hip_module_get_function(&kernel, context.module, name) != 0 || kernel == nullptr) return nullptr;
        context.kernels.emplace(name, kernel);
        return kernel;
    }

    std::uint64_t reusable_buffer_budget_bytes(const HardwareGraph& graph) const {
        const auto summary = summarize_graph(graph);
        const auto capacity = summary.directly_attached_bytes > 0u
                                  ? summary.directly_attached_bytes
                                  : std::max(summary.addressable_bytes, summary.shared_host_bytes);
        return std::clamp<std::uint64_t>(capacity / 16u, 16ull * 1024ull * 1024ull, 256ull * 1024ull * 1024ull);
    }

    void touch_buffer(Context& context, const std::string& slot) const {
        context.reusable_buffer_last_used[slot] = ++context.reusable_buffer_tick;
    }

    void trim_reusable_buffers(Context& context, const std::string& preserve_slot, const std::size_t incoming_bytes) const {
        while (!context.reusable_buffers.empty() &&
               context.reusable_buffer_budget_bytes > 0u &&
               context.reusable_buffer_bytes + incoming_bytes > context.reusable_buffer_budget_bytes) {
            auto oldest = context.reusable_buffers.end();
            for (auto it = context.reusable_buffers.begin(); it != context.reusable_buffers.end(); ++it) {
                if (it->first == preserve_slot) {
                    continue;
                }
                if (oldest == context.reusable_buffers.end() ||
                    context.reusable_buffer_last_used[it->first] < context.reusable_buffer_last_used[oldest->first]) {
                    oldest = it;
                }
            }
            if (oldest == context.reusable_buffers.end()) {
                break;
            }
            context.reusable_buffer_bytes -= context.reusable_buffer_sizes[oldest->first];
            free_device(oldest->second);
            context.reusable_buffer_sizes.erase(oldest->first);
            context.reusable_buffer_last_used.erase(oldest->first);
            context.reusable_buffers.erase(oldest);
        }
    }

    void invalidate_reusable_buffers(Context& context) const {
        for (const auto& [slot, ptr] : context.reusable_buffers) {
            (void)slot;
            free_device(ptr);
        }
        context.reusable_buffers.clear();
        context.reusable_buffer_sizes.clear();
        context.reusable_buffer_last_used.clear();
        context.reusable_buffer_bytes = 0u;
        context.reusable_buffer_tick = 0u;
    }

    bool ensure_buffer(Context& context, const std::string& slot, const std::size_t bytes, void*& ptr) const {
        if (bytes == 0u) {
            ptr = nullptr;
            return true;
        }
        auto& slot_ptr = context.reusable_buffers[slot];
        auto& slot_size = context.reusable_buffer_sizes[slot];
        if (slot_ptr == nullptr || slot_size < bytes) {
            if (slot_ptr != nullptr) {
                context.reusable_buffer_bytes -= slot_size;
            }
            free_device(slot_ptr);
            slot_ptr = nullptr;
            slot_size = 0u;
            trim_reusable_buffers(context, slot, bytes);
            if (api().hip_malloc(&slot_ptr, bytes) != 0) {
                return false;
            }
            slot_size = bytes;
            context.reusable_buffer_bytes += bytes;
        }
        touch_buffer(context, slot);
        ptr = slot_ptr;
        return true;
    }

    bool alloc_and_copy_input(Context& context, const std::string& slot, void*& ptr, std::span<const float> input) const {
        if (!ensure_buffer(context, slot, input.size_bytes(), ptr)) return false;
        if (api().hip_memcpy_htod(ptr, input.data(), input.size_bytes()) != 0) { return false; }
        return true;
    }

    bool alloc_and_copy_input_async(
        Context& context,
        const std::string& slot,
        void*& ptr,
        std::span<const float> input,
        const hip_stream_t stream,
        const std::size_t staging_slot = 0u) const {
        if (!ensure_buffer(context, slot, input.size_bytes(), ptr)) return false;
        const void* source = input.data();
        if (!input.empty()) {
            auto& staging = context.staging_buffers[staging_slot % context.staging_buffers.size()];
            staging.resize(input.size());
            std::memcpy(staging.data(), input.data(), input.size_bytes());
            source = staging.data();
        }
        return api().hip_memcpy_htod_async(ptr, source, input.size_bytes(), stream) == 0;
    }

    void free_device(void* ptr) const { if (ptr != nullptr) (void)api().hip_free(ptr); }

    BackendRunResult launch_unary_2d(const HardwareGraph& graph, const char* kernel_name, const std::span<const float> input, const std::size_t output_count, const std::uint32_t grid_items_x, const std::uint32_t grid_items_y, std::vector<void*> args) const {
        auto context = acquire_context(graph);
        if (context == nullptr) return failure("rocm-context");
        BackendRunResult result;
        result.output.resize(output_count, 0.0f);
        double copy_in_us = 0.0;
        double copy_out_us = 0.0;
        double compute_us = 0.0;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock lock(context->execution_mutex);
            auto kernel = get_kernel(*context, kernel_name);
            if (kernel == nullptr) { result.error = "rocm-kernel"; return; }
            void* d_in = nullptr; void* d_out = nullptr;
            if (can_split_dispatch(*context)) {
                if (!alloc_and_copy_input_async(*context, "unary-in", d_in, input, context->copy_stream, 0u) ||
                    !ensure_buffer(*context, "unary-out", result.output.size() * sizeof(float), d_out)) {
                    result.error = "rocm-memory";
                    return;
                }
                args[0] = &d_in;
                args[1] = &d_out;
                const unsigned int block_x = kNativeTileSize;
                const unsigned int block_y = kNativeTileSize;
                const unsigned int grid_x = grid_items_x == 0 ? 1u : (grid_items_x + block_x - 1u) / block_x;
                const unsigned int grid_y = grid_items_y == 0 ? 1u : (grid_items_y + block_y - 1u) / block_y;
                if (api().hip_event_record(context->copy_ready_event, context->copy_stream) != 0 ||
                    api().hip_stream_wait_event(context->stream, context->copy_ready_event, 0u) != 0 ||
                    api().hip_module_launch_kernel(kernel, grid_x, grid_y, 1, block_x, block_y, 1, 0u, context->stream, args.data(), nullptr) != 0 ||
                    api().hip_event_record(context->compute_ready_event, context->stream) != 0 ||
                    api().hip_stream_wait_event(context->copy_stream, context->compute_ready_event, 0u) != 0 ||
                    api().hip_memcpy_dtoh_async(result.output.data(), d_out, result.output.size() * sizeof(float), context->copy_stream) != 0 ||
                    api().hip_stream_synchronize(context->copy_stream) != 0) {
                    result.error = "rocm-launch";
                    return;
                }
                result.success = true;
                result.used_host = false;
                result.used_opencl = false;
                return;
            }
            copy_in_us += measure_us([&]() {
                if (!alloc_and_copy_input(*context, "unary-in", d_in, input) ||
                    !ensure_buffer(*context, "unary-out", result.output.size() * sizeof(float), d_out)) {
                    result.error = "rocm-memory";
                }
            });
            if (!result.error.empty()) { return; }
            args[0] = &d_in;
            args[1] = &d_out;
            const unsigned int block_x = kNativeTileSize;
            const unsigned int block_y = kNativeTileSize;
            const unsigned int grid_x = grid_items_x == 0 ? 1u : (grid_items_x + block_x - 1u) / block_x;
            const unsigned int grid_y = grid_items_y == 0 ? 1u : (grid_items_y + block_y - 1u) / block_y;
            compute_us += measure_us([&]() {
                if (api().hip_module_launch_kernel(kernel, grid_x, grid_y, 1, block_x, block_y, 1, 0u, context->stream, args.data(), nullptr) != 0 ||
                    api().hip_stream_synchronize(context->stream) != 0) {
                    result.error = "rocm-launch";
                }
            });
            if (!result.error.empty()) { return; }
            copy_out_us += measure_us([&]() {
                if (api().hip_memcpy_dtoh(result.output.data(), d_out, result.output.size() * sizeof(float)) != 0) {
                    result.error = "rocm-launch";
                }
            });
            if (result.error.empty()) { result.success = true; result.used_host = false; result.used_opencl = false; }
        });
        result.copy_runtime_us = copy_in_us + copy_out_us;
        result.compute_runtime_us = compute_us;
        return result;
    }

    BackendRunResult launch_binary_2d(const HardwareGraph& graph, const char* kernel_name, const std::span<const float> lhs, const std::span<const float> rhs, const std::size_t output_count, const std::uint32_t grid_items_x, const std::uint32_t grid_items_y, std::vector<void*> args) const {
        auto context = acquire_context(graph);
        if (context == nullptr) return failure("rocm-context");
        BackendRunResult result;
        result.output.resize(output_count, 0.0f);
        double copy_in_us = 0.0;
        double copy_out_us = 0.0;
        double compute_us = 0.0;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock lock(context->execution_mutex);
            auto kernel = get_kernel(*context, kernel_name);
            if (kernel == nullptr) { result.error = "rocm-kernel"; return; }
            void* d_lhs = nullptr; void* d_rhs = nullptr; void* d_out = nullptr;
            if (can_split_dispatch(*context)) {
                if (!alloc_and_copy_input_async(*context, "binary-lhs", d_lhs, lhs, context->copy_stream, 0u) ||
                    !alloc_and_copy_input_async(*context, "binary-rhs", d_rhs, rhs, context->copy_stream, 1u) ||
                    !ensure_buffer(*context, "binary-out", result.output.size() * sizeof(float), d_out)) {
                    result.error = "rocm-memory";
                    return;
                }
                args[0] = &d_lhs;
                args[1] = &d_rhs;
                args[2] = &d_out;
                const unsigned int block_x = kNativeTileSize;
                const unsigned int block_y = kNativeTileSize;
                const unsigned int grid_x = grid_items_x == 0 ? 1u : (grid_items_x + block_x - 1u) / block_x;
                const unsigned int grid_y = grid_items_y == 0 ? 1u : (grid_items_y + block_y - 1u) / block_y;
                if (api().hip_event_record(context->copy_ready_event, context->copy_stream) != 0 ||
                    api().hip_stream_wait_event(context->stream, context->copy_ready_event, 0u) != 0 ||
                    api().hip_module_launch_kernel(kernel, grid_x, grid_y, 1, block_x, block_y, 1, 0u, context->stream, args.data(), nullptr) != 0 ||
                    api().hip_event_record(context->compute_ready_event, context->stream) != 0 ||
                    api().hip_stream_wait_event(context->copy_stream, context->compute_ready_event, 0u) != 0 ||
                    api().hip_memcpy_dtoh_async(result.output.data(), d_out, result.output.size() * sizeof(float), context->copy_stream) != 0 ||
                    api().hip_stream_synchronize(context->copy_stream) != 0) {
                    result.error = "rocm-launch";
                    return;
                }
                result.success = true;
                result.used_host = false;
                result.used_opencl = false;
                return;
            }
            copy_in_us += measure_us([&]() {
                if (!alloc_and_copy_input(*context, "binary-lhs", d_lhs, lhs) ||
                    !alloc_and_copy_input(*context, "binary-rhs", d_rhs, rhs) ||
                    !ensure_buffer(*context, "binary-out", result.output.size() * sizeof(float), d_out)) { result.error = "rocm-memory"; }
            });
            if (!result.error.empty()) { return; }
            args[0] = &d_lhs;
            args[1] = &d_rhs;
            args[2] = &d_out;
            const unsigned int block_x = kNativeTileSize;
            const unsigned int block_y = kNativeTileSize;
            const unsigned int grid_x = grid_items_x == 0 ? 1u : (grid_items_x + block_x - 1u) / block_x;
            const unsigned int grid_y = grid_items_y == 0 ? 1u : (grid_items_y + block_y - 1u) / block_y;
            compute_us += measure_us([&]() {
                if (api().hip_module_launch_kernel(kernel, grid_x, grid_y, 1, block_x, block_y, 1, 0u, context->stream, args.data(), nullptr) != 0 ||
                    api().hip_stream_synchronize(context->stream) != 0) {
                    result.error = "rocm-launch";
                }
            });
            if (!result.error.empty()) { return; }
            copy_out_us += measure_us([&]() {
                if (api().hip_memcpy_dtoh(result.output.data(), d_out, result.output.size() * sizeof(float)) != 0) {
                    result.error = "rocm-launch";
                }
            });
            if (result.error.empty()) { result.success = true; result.used_host = false; result.used_opencl = false; }
        });
        result.copy_runtime_us = copy_in_us + copy_out_us;
        result.compute_runtime_us = compute_us;
        return result;
    }

    BackendRunResult run_elementwise_native(const HardwareGraph& graph, const std::span<const float> lhs, const std::span<const float> rhs, const bool low_precision) const override {
        auto context = acquire_context(graph);
        if (context == nullptr) return failure("rocm-context");
        BackendRunResult result;
        result.output.resize(lhs.size(), 0.0f);
        double copy_in_us = 0.0;
        double copy_out_us = 0.0;
        double compute_us = 0.0;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock lock(context->execution_mutex);
            auto kernel = get_kernel(*context, "elementwise_map");
            if (kernel == nullptr) { result.error = "rocm-kernel"; return; }
            void* d_lhs = nullptr; void* d_rhs = nullptr; void* d_out = nullptr;
            if (can_split_dispatch(*context)) {
                if (!alloc_and_copy_input_async(*context, "elementwise-lhs", d_lhs, lhs, context->copy_stream, 0u) ||
                    !alloc_and_copy_input_async(*context, "elementwise-rhs", d_rhs, rhs, context->copy_stream, 1u) ||
                    !ensure_buffer(*context, "elementwise-out", result.output.size() * sizeof(float), d_out)) {
                    result.error = "rocm-memory";
                    return;
                }
                unsigned int count = static_cast<unsigned int>(lhs.size());
                int low = low_precision ? 1 : 0;
                void* args[] = {&d_lhs, &d_rhs, &d_out, &count, &low};
                const unsigned int block_x = kNativeReductionGroupSize;
                const unsigned int grid_x = count == 0 ? 1u : (count + block_x - 1u) / block_x;
                if (api().hip_event_record(context->copy_ready_event, context->copy_stream) != 0 ||
                    api().hip_stream_wait_event(context->stream, context->copy_ready_event, 0u) != 0 ||
                    api().hip_module_launch_kernel(kernel, grid_x, 1, 1, block_x, 1, 1, 0u, context->stream, args, nullptr) != 0 ||
                    api().hip_event_record(context->compute_ready_event, context->stream) != 0 ||
                    api().hip_stream_wait_event(context->copy_stream, context->compute_ready_event, 0u) != 0 ||
                    api().hip_memcpy_dtoh_async(result.output.data(), d_out, result.output.size() * sizeof(float), context->copy_stream) != 0 ||
                    api().hip_stream_synchronize(context->copy_stream) != 0) {
                    result.error = "rocm-launch";
                    return;
                }
                result.success = true;
                result.used_host = false;
                result.used_opencl = false;
                return;
            }
            if (!alloc_and_copy_input(*context, "elementwise-lhs", d_lhs, lhs) ||
                !alloc_and_copy_input(*context, "elementwise-rhs", d_rhs, rhs) ||
                !ensure_buffer(*context, "elementwise-out", result.output.size() * sizeof(float), d_out)) { result.error = "rocm-memory"; return; }
            unsigned int count = static_cast<unsigned int>(lhs.size()); int low = low_precision ? 1 : 0; void* args[] = {&d_lhs, &d_rhs, &d_out, &count, &low};
            const unsigned int block_x = kNativeReductionGroupSize; const unsigned int grid_x = count == 0 ? 1u : (count + block_x - 1u) / block_x;
            if (api().hip_module_launch_kernel(kernel, grid_x, 1, 1, block_x, 1, 1, 0u, context->stream, args, nullptr) != 0 || api().hip_stream_synchronize(context->stream) != 0 || api().hip_memcpy_dtoh(result.output.data(), d_out, result.output.size() * sizeof(float)) != 0) result.error = "rocm-launch";
            else { result.success = true; result.used_host = false; result.used_opencl = false; }
        });
        result.copy_runtime_us = copy_in_us + copy_out_us;
        result.compute_runtime_us = compute_us;
        return result;
    }

    BackendRunResult run_reduction_native(const HardwareGraph& graph, const std::span<const float> input, const bool low_precision) const override {
        auto context = acquire_context(graph);
        if (context == nullptr) return failure("rocm-context");
        BackendRunResult result;
        double copy_in_us = 0.0;
        double copy_out_us = 0.0;
        double compute_us = 0.0;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock lock(context->execution_mutex);
            auto kernel = get_kernel(*context, "reduce_sum");
            if (kernel == nullptr) { result.error = "rocm-kernel"; return; }
            const auto local = kNativeReductionGroupSize;
            const auto global = std::min<std::size_t>(std::max<std::size_t>(local, round_up<std::size_t>(input.size(), local)), local * 64u);
            const auto groups = global / local;
            std::vector<float> partials(groups, 0.0f);
            void* d_in = nullptr; void* d_partial = nullptr;
            if (can_split_dispatch(*context)) {
                if (!alloc_and_copy_input_async(*context, "reduce-in", d_in, input, context->copy_stream, 0u) ||
                    !ensure_buffer(*context, "reduce-partial", partials.size() * sizeof(float), d_partial)) {
                    result.error = "rocm-memory";
                    return;
                }
                unsigned int count = static_cast<unsigned int>(input.size());
                int low = low_precision ? 1 : 0;
                void* args[] = {&d_in, &d_partial, &count, &low};
                if (api().hip_event_record(context->copy_ready_event, context->copy_stream) != 0 ||
                    api().hip_stream_wait_event(context->stream, context->copy_ready_event, 0u) != 0 ||
                    api().hip_module_launch_kernel(kernel, static_cast<unsigned int>(groups), 1, 1, local, 1, 1, static_cast<unsigned int>(local * sizeof(float)), context->stream, args, nullptr) != 0 ||
                    api().hip_event_record(context->compute_ready_event, context->stream) != 0 ||
                    api().hip_stream_wait_event(context->copy_stream, context->compute_ready_event, 0u) != 0 ||
                    api().hip_memcpy_dtoh_async(partials.data(), d_partial, partials.size() * sizeof(float), context->copy_stream) != 0 ||
                    api().hip_stream_synchronize(context->copy_stream) != 0) {
                    result.error = "rocm-launch";
                    return;
                }
                for (const auto value : partials) {
                    result.scalar_output += value;
                }
                result.success = true;
                result.used_host = false;
                result.used_opencl = false;
                return;
            }
            if (!alloc_and_copy_input(*context, "reduce-in", d_in, input) ||
                !ensure_buffer(*context, "reduce-partial", partials.size() * sizeof(float), d_partial)) { result.error = "rocm-memory"; return; }
            unsigned int count = static_cast<unsigned int>(input.size()); int low = low_precision ? 1 : 0; void* args[] = {&d_in, &d_partial, &count, &low};
            if (api().hip_module_launch_kernel(kernel, static_cast<unsigned int>(groups), 1, 1, local, 1, 1, static_cast<unsigned int>(local * sizeof(float)), context->stream, args, nullptr) != 0 || api().hip_stream_synchronize(context->stream) != 0 || api().hip_memcpy_dtoh(partials.data(), d_partial, partials.size() * sizeof(float)) != 0) result.error = "rocm-launch";
            else { for (const auto value : partials) result.scalar_output += value; result.success = true; result.used_host = false; result.used_opencl = false; }
        });
        result.copy_runtime_us = copy_in_us + copy_out_us;
        result.compute_runtime_us = compute_us;
        return result;
    }

    BackendRunResult run_matmul_native(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> lhs, const std::span<const float> rhs, const std::uint32_t rows, const std::uint32_t columns, const std::uint32_t depth, const bool low_precision) const override {
        unsigned int row_count = rows, column_count = columns, depth_count = depth; int low = low_precision ? 1 : 0;
        return launch_binary_2d(graph, gpu_rhs_uses_transposed_layout(operation) ? "matmul_tiled_rhs_t" : "matmul_tiled", lhs, rhs, static_cast<std::size_t>(rows) * columns, columns, rows, {nullptr, nullptr, nullptr, &row_count, &column_count, &depth_count, &low});
    }

    BackendRunResult run_conv3x3_native(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> input, const std::uint32_t height, const std::uint32_t width, const bool low_precision) const override {
        unsigned int h = height, w = width; int low = low_precision ? 1 : 0;
        return launch_unary_2d(
            graph,
            gpu_conv_uses_patch9_layout(operation) ? "conv3x3_valid_patch9" : "conv3x3_valid",
            input,
            static_cast<std::size_t>(height - 2u) * (width - 2u),
            width - 2u,
            height - 2u,
            {nullptr, nullptr, &h, &w, &low});
    }

    BackendRunResult run_resample_native(const HardwareGraph& graph, const OperationSpec& operation, const std::span<const float> input, const std::uint32_t src_h, const std::uint32_t src_w, const std::uint32_t dst_h, const std::uint32_t dst_w, const std::uint32_t row_offset, const std::uint32_t row_count, const bool low_precision) const override {
        unsigned int src_height = src_h, src_width = src_w, dst_height = dst_h, dst_width = dst_w, offset = row_offset, rows = row_count; int low = low_precision ? 1 : 0;
        return launch_unary_2d(
            graph,
            gpu_resample_uses_packed6_layout(operation) ? "bilinear_resample_packed6" : "bilinear_resample",
            input,
            static_cast<std::size_t>(row_count) * dst_w,
            dst_w,
            row_count,
            {nullptr, nullptr, &src_height, &src_width, &dst_height, &dst_width, &offset, &rows, &low});
    }

    mutable std::mutex mutex_;
    mutable std::unordered_map<std::string, std::shared_ptr<Context>> contexts_;
};

}  // namespace

std::unique_ptr<IKernelBackend> make_native_gpu_kernel_backend(const JakalBackendKind backend) {
    switch (backend) {
    case JakalBackendKind::cuda:
        return std::make_unique<CudaNativeBackend>();
    case JakalBackendKind::level_zero:
        return std::make_unique<LevelZeroNativeBackend>();
    case JakalBackendKind::rocm:
        return std::make_unique<RocmNativeBackend>();
    case JakalBackendKind::vulkan_compute:
        return std::make_unique<FallbackNativeBackend>(backend, "vulkan");
    case JakalBackendKind::opencl:
    default:
        return std::make_unique<FallbackNativeBackend>(backend, "opencl");
    }
}

}  // namespace jakal::executors

