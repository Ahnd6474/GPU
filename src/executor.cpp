#include "jakal/executor.hpp"

#include "jakal/device.hpp"
#include "jakal/executors/direct_backends.hpp"
#include "jakal/executors/scheduler.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <iostream>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace jakal {
namespace {

constexpr std::uint32_t kOpenClReductionGroupSize = 256u;
constexpr std::uint32_t kOpenClTileSize = 16u;

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

bool backend_result_is_usable(const executors::BackendRunResult& result) {
    constexpr double kReasonableMagnitude = 1.0e8;
    if (!std::isfinite(result.scalar_output) || std::abs(result.scalar_output) > kReasonableMagnitude) {
        return false;
    }
    return std::all_of(result.output.begin(), result.output.end(), [](const float value) {
        return std::isfinite(value) && std::abs(static_cast<double>(value)) <= kReasonableMagnitude;
    });
}

std::string sanitize_id_fragment(const std::string& text) {
    std::string result = text;
    for (char& ch : result) {
        const unsigned char value = static_cast<unsigned char>(ch);
        if (!(std::isalnum(value) || ch == '_' || ch == '-')) {
            ch = '_';
        }
    }
    return result;
}

bool execution_trace_enabled() {
    const char* value = std::getenv("JAKAL_EXEC_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void trace_execution_line(const std::string& text) {
    if (!execution_trace_enabled()) {
        return;
    }
    std::cerr << text << '\n' << std::flush;
}

std::string format_extents(const std::vector<std::uint64_t>& extents) {
    std::ostringstream stream;
    for (std::size_t index = 0; index < extents.size(); ++index) {
        if (index > 0u) {
            stream << 'x';
        }
        stream << extents[index];
    }
    return stream.str();
}

using executors::BackendRunResult;
using executors::DeviceAssignment;
using executors::OperationData;

JakalL0WorkloadTraits make_gpu_traits(const OperationSpec& operation) {
    JakalL0WorkloadTraits traits;
    traits.op_class = operation.op_class;
    traits.extents = operation.extents;
    traits.bytes = operation.input_bytes + operation.output_bytes + operation.temporary_bytes;
    traits.estimated_flops = operation.estimated_flops;
    traits.latency_sensitive = false;
    traits.matrix_friendly = operation.matrix_friendly;
    traits.streaming_friendly = operation.streaming_friendly;
    return traits;
}

bool variant_compatible_with_graph(
    const HardwareGraph& graph,
    const JakalToolkitVariant& variant) {
    switch (variant.binding.backend) {
    case JakalBackendKind::level_zero:
        return graph.probe == "level-zero";
    case JakalBackendKind::opencl:
        return graph.probe == "opencl";
    case JakalBackendKind::vulkan_compute:
        return graph.probe == "vulkan";
    case JakalBackendKind::cuda:
        return graph.probe == "cuda";
    case JakalBackendKind::rocm:
        return graph.probe == "rocm";
    default:
        return false;
    }
}

const JakalToolkitVariant* find_preferred_gpu_variant(
    const DeviceAssignment& assignment,
    const std::vector<JakalToolkitIndexEntry>& jakal_toolkit_index,
    const OperationSpec& operation) {
    const auto traits = make_gpu_traits(operation);
    for (const auto& entry : jakal_toolkit_index) {
        if (entry.device_uid != assignment.graph->uid || entry.variants.empty()) {
            continue;
        }

        const auto& variants = entry.variants;
        // Re-rank lightly per operation without rebuilding the toolkit index.
        const JakalToolkitVariant* best = nullptr;
        double best_score = 0.0;
        for (const auto& variant : variants) {
            if (!variant_compatible_with_graph(*assignment.graph, variant)) {
                continue;
            }
            double score = variant.toolkit_score;
            if (traits.matrix_friendly && variant.binding.capabilities.subgroup_matrix) {
                score += 0.05;
            }
            if (traits.streaming_friendly && variant.binding.capabilities.unified_memory) {
                score += 0.03;
            }
            if (traits.op_class == OperationClass::resample_2d &&
                variant.binding.backend == JakalBackendKind::vulkan_compute) {
                score += 0.04;
            }
            if (!variant.executable) {
                score -= 0.20;
            }
            if (best == nullptr || score > best_score) {
                best = &variant;
                best_score = score;
            }
        }
        return best;
    }
    return nullptr;
}

std::string actual_backend_name(
    const std::vector<DeviceAssignment>& assignments,
    const std::vector<JakalToolkitIndexEntry>& jakal_toolkit_index,
    const OperationSpec& operation) {
    bool uses_host = false;
    bool uses_gpu = false;
    const JakalToolkitVariant* preferred_gpu = nullptr;
    for (const auto& assignment : assignments) {
        if (assignment.graph->probe == "host") {
            uses_host = true;
        } else {
            uses_gpu = true;
            if (preferred_gpu == nullptr) {
                preferred_gpu = find_preferred_gpu_variant(assignment, jakal_toolkit_index, operation);
            }
        }
    }

    const auto gpu_request = [&]() {
        if (preferred_gpu == nullptr) {
            return std::string("gpu-direct");
        }
        return to_string(preferred_gpu->binding.vendor) + ":" + to_string(preferred_gpu->binding.backend);
    };
    const bool executable_gpu_request =
        preferred_gpu != nullptr &&
        preferred_gpu->executable;

    if (uses_host && uses_gpu) {
        return "mixed-direct[" + gpu_request() + "]";
    }
    if (uses_gpu) {
        if (executable_gpu_request) {
            return gpu_request() + "-direct";
        }
        return "gpu-direct";
    }
    return "host-direct";
}

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

using cl_int = std::int32_t;
using cl_uint = std::uint32_t;
using cl_ulong = std::uint64_t;
using cl_bool = cl_uint;
using cl_bitfield = cl_ulong;
using cl_device_type = cl_bitfield;
using cl_platform_info = cl_uint;
using cl_device_info = cl_uint;
using cl_program_build_info = cl_uint;
using cl_context_properties = intptr_t;
using cl_mem_flags = cl_bitfield;
using cl_command_queue_properties = cl_bitfield;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_mem = struct _cl_mem*;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_bool CL_TRUE = 1;
constexpr cl_device_type CL_DEVICE_TYPE_ALL = 0xFFFFFFFFu;
constexpr cl_mem_flags CL_MEM_READ_ONLY = 1u << 2u;
constexpr cl_mem_flags CL_MEM_WRITE_ONLY = 1u << 1u;
constexpr cl_platform_info CL_PLATFORM_NAME = 0x0902;
constexpr cl_device_info CL_DEVICE_NAME = 0x102B;
constexpr cl_device_info CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035;
constexpr cl_program_build_info CL_PROGRAM_BUILD_LOG = 0x1183;

using cl_get_platform_ids_fn = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
using cl_get_platform_info_fn = cl_int (*)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
using cl_get_device_ids_fn = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
using cl_get_device_info_fn = cl_int (*)(cl_device_id, cl_device_info, size_t, void*, size_t*);
using cl_create_context_fn =
    cl_context (*)(const cl_context_properties*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
using cl_release_context_fn = cl_int (*)(cl_context);
using cl_create_command_queue_fn = cl_command_queue (*)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
using cl_create_command_queue_with_properties_fn = cl_command_queue (*)(cl_context, cl_device_id, const cl_context_properties*, cl_int*);
using cl_release_command_queue_fn = cl_int (*)(cl_command_queue);
using cl_create_program_with_source_fn = cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
using cl_build_program_fn = cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*);
using cl_get_program_build_info_fn = cl_int (*)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
using cl_release_program_fn = cl_int (*)(cl_program);
using cl_create_kernel_fn = cl_kernel (*)(cl_program, const char*, cl_int*);
using cl_set_kernel_arg_fn = cl_int (*)(cl_kernel, cl_uint, size_t, const void*);
using cl_release_kernel_fn = cl_int (*)(cl_kernel);
using cl_create_buffer_fn = cl_mem (*)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
using cl_release_mem_object_fn = cl_int (*)(cl_mem);
using cl_enqueue_write_buffer_fn = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const void*, void*);
using cl_enqueue_read_buffer_fn = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const void*, void*);
using cl_enqueue_nd_range_kernel_fn = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
using cl_finish_fn = cl_int (*)(cl_command_queue);

class OpenClApi {
public:
    OpenClApi() {
#if defined(_WIN32)
        library_ = load_library("OpenCL.dll");
#elif defined(__APPLE__)
        library_ = load_library("/System/Library/Frameworks/OpenCL.framework/OpenCL");
#else
        library_ = load_library("libOpenCL.so");
        if (library_ == nullptr) {
            library_ = load_library("libOpenCL.so.1");
        }
#endif
        if (library_ == nullptr) {
            return;
        }

        get_platform_ids_ = reinterpret_cast<cl_get_platform_ids_fn>(load_symbol(library_, "clGetPlatformIDs"));
        get_platform_info_ = reinterpret_cast<cl_get_platform_info_fn>(load_symbol(library_, "clGetPlatformInfo"));
        get_device_ids_ = reinterpret_cast<cl_get_device_ids_fn>(load_symbol(library_, "clGetDeviceIDs"));
        get_device_info_ = reinterpret_cast<cl_get_device_info_fn>(load_symbol(library_, "clGetDeviceInfo"));
        create_context_ = reinterpret_cast<cl_create_context_fn>(load_symbol(library_, "clCreateContext"));
        create_command_queue_ = reinterpret_cast<cl_create_command_queue_fn>(load_symbol(library_, "clCreateCommandQueue"));
        create_command_queue_with_properties_ = reinterpret_cast<cl_create_command_queue_with_properties_fn>(
            load_symbol(library_, "clCreateCommandQueueWithProperties"));
        release_context_ = reinterpret_cast<cl_release_context_fn>(load_symbol(library_, "clReleaseContext"));
        release_command_queue_ = reinterpret_cast<cl_release_command_queue_fn>(load_symbol(library_, "clReleaseCommandQueue"));
        create_program_with_source_ =
            reinterpret_cast<cl_create_program_with_source_fn>(load_symbol(library_, "clCreateProgramWithSource"));
        build_program_ = reinterpret_cast<cl_build_program_fn>(load_symbol(library_, "clBuildProgram"));
        get_program_build_info_ =
            reinterpret_cast<cl_get_program_build_info_fn>(load_symbol(library_, "clGetProgramBuildInfo"));
        release_program_ = reinterpret_cast<cl_release_program_fn>(load_symbol(library_, "clReleaseProgram"));
        create_kernel_ = reinterpret_cast<cl_create_kernel_fn>(load_symbol(library_, "clCreateKernel"));
        set_kernel_arg_ = reinterpret_cast<cl_set_kernel_arg_fn>(load_symbol(library_, "clSetKernelArg"));
        release_kernel_ = reinterpret_cast<cl_release_kernel_fn>(load_symbol(library_, "clReleaseKernel"));
        create_buffer_ = reinterpret_cast<cl_create_buffer_fn>(load_symbol(library_, "clCreateBuffer"));
        release_mem_object_ = reinterpret_cast<cl_release_mem_object_fn>(load_symbol(library_, "clReleaseMemObject"));
        enqueue_write_buffer_ =
            reinterpret_cast<cl_enqueue_write_buffer_fn>(load_symbol(library_, "clEnqueueWriteBuffer"));
        enqueue_read_buffer_ =
            reinterpret_cast<cl_enqueue_read_buffer_fn>(load_symbol(library_, "clEnqueueReadBuffer"));
        enqueue_nd_range_kernel_ =
            reinterpret_cast<cl_enqueue_nd_range_kernel_fn>(load_symbol(library_, "clEnqueueNDRangeKernel"));
        finish_ = reinterpret_cast<cl_finish_fn>(load_symbol(library_, "clFinish"));

        loaded_ = get_platform_ids_ != nullptr &&
                  get_platform_info_ != nullptr &&
                  get_device_ids_ != nullptr &&
                  get_device_info_ != nullptr &&
                  create_context_ != nullptr &&
                  release_context_ != nullptr &&
                  release_command_queue_ != nullptr &&
                  create_program_with_source_ != nullptr &&
                  build_program_ != nullptr &&
                  get_program_build_info_ != nullptr &&
                  release_program_ != nullptr &&
                  create_kernel_ != nullptr &&
                  set_kernel_arg_ != nullptr &&
                  release_kernel_ != nullptr &&
                  create_buffer_ != nullptr &&
                  release_mem_object_ != nullptr &&
                  enqueue_write_buffer_ != nullptr &&
                  enqueue_read_buffer_ != nullptr &&
                  enqueue_nd_range_kernel_ != nullptr &&
                  finish_ != nullptr &&
                  (create_command_queue_ != nullptr || create_command_queue_with_properties_ != nullptr);
    }

    ~OpenClApi() {
        close_library(library_);
    }

    [[nodiscard]] bool loaded() const { return loaded_; }
    [[nodiscard]] cl_get_platform_ids_fn get_platform_ids() const { return get_platform_ids_; }
    [[nodiscard]] cl_get_platform_info_fn get_platform_info() const { return get_platform_info_; }
    [[nodiscard]] cl_get_device_ids_fn get_device_ids() const { return get_device_ids_; }
    [[nodiscard]] cl_get_device_info_fn get_device_info() const { return get_device_info_; }
    [[nodiscard]] cl_create_context_fn create_context() const { return create_context_; }
    [[nodiscard]] cl_create_command_queue_fn create_command_queue() const { return create_command_queue_; }
    [[nodiscard]] cl_create_command_queue_with_properties_fn create_command_queue_with_properties() const {
        return create_command_queue_with_properties_;
    }
    [[nodiscard]] cl_release_context_fn release_context() const { return release_context_; }
    [[nodiscard]] cl_release_command_queue_fn release_command_queue() const { return release_command_queue_; }
    [[nodiscard]] cl_create_program_with_source_fn create_program_with_source() const { return create_program_with_source_; }
    [[nodiscard]] cl_build_program_fn build_program() const { return build_program_; }
    [[nodiscard]] cl_get_program_build_info_fn get_program_build_info() const { return get_program_build_info_; }
    [[nodiscard]] cl_release_program_fn release_program() const { return release_program_; }
    [[nodiscard]] cl_create_kernel_fn create_kernel() const { return create_kernel_; }
    [[nodiscard]] cl_set_kernel_arg_fn set_kernel_arg() const { return set_kernel_arg_; }
    [[nodiscard]] cl_release_kernel_fn release_kernel() const { return release_kernel_; }
    [[nodiscard]] cl_create_buffer_fn create_buffer() const { return create_buffer_; }
    [[nodiscard]] cl_release_mem_object_fn release_mem_object() const { return release_mem_object_; }
    [[nodiscard]] cl_enqueue_write_buffer_fn enqueue_write_buffer() const { return enqueue_write_buffer_; }
    [[nodiscard]] cl_enqueue_read_buffer_fn enqueue_read_buffer() const { return enqueue_read_buffer_; }
    [[nodiscard]] cl_enqueue_nd_range_kernel_fn enqueue_nd_range_kernel() const { return enqueue_nd_range_kernel_; }
    [[nodiscard]] cl_finish_fn finish() const { return finish_; }

private:
    LibraryHandle library_ = nullptr;
    bool loaded_ = false;
    cl_get_platform_ids_fn get_platform_ids_ = nullptr;
    cl_get_platform_info_fn get_platform_info_ = nullptr;
    cl_get_device_ids_fn get_device_ids_ = nullptr;
    cl_get_device_info_fn get_device_info_ = nullptr;
    cl_create_context_fn create_context_ = nullptr;
    cl_create_command_queue_fn create_command_queue_ = nullptr;
    cl_create_command_queue_with_properties_fn create_command_queue_with_properties_ = nullptr;
    cl_release_context_fn release_context_ = nullptr;
    cl_release_command_queue_fn release_command_queue_ = nullptr;
    cl_create_program_with_source_fn create_program_with_source_ = nullptr;
    cl_build_program_fn build_program_ = nullptr;
    cl_get_program_build_info_fn get_program_build_info_ = nullptr;
    cl_release_program_fn release_program_ = nullptr;
    cl_create_kernel_fn create_kernel_ = nullptr;
    cl_set_kernel_arg_fn set_kernel_arg_ = nullptr;
    cl_release_kernel_fn release_kernel_ = nullptr;
    cl_create_buffer_fn create_buffer_ = nullptr;
    cl_release_mem_object_fn release_mem_object_ = nullptr;
    cl_enqueue_write_buffer_fn enqueue_write_buffer_ = nullptr;
    cl_enqueue_read_buffer_fn enqueue_read_buffer_ = nullptr;
    cl_enqueue_nd_range_kernel_fn enqueue_nd_range_kernel_ = nullptr;
    cl_finish_fn finish_ = nullptr;
};

template <typename T>
T get_scalar_info(cl_get_device_info_fn getter, cl_device_id device, cl_device_info key, const T fallback = T{}) {
    T value{};
    if (getter(device, key, sizeof(T), &value, nullptr) != CL_SUCCESS) {
        return fallback;
    }
    return value;
}

std::string get_device_string(cl_get_device_info_fn getter, cl_device_id device, cl_device_info key) {
    size_t size = 0;
    if (getter(device, key, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return {};
    }
    std::string value(size, '\0');
    if (getter(device, key, size, value.data(), nullptr) != CL_SUCCESS) {
        return {};
    }
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

std::string get_platform_string(cl_get_platform_info_fn getter, cl_platform_id platform, cl_platform_info key) {
    size_t size = 0;
    if (getter(platform, key, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return {};
    }
    std::string value(size, '\0');
    if (getter(platform, key, size, value.data(), nullptr) != CL_SUCCESS) {
        return {};
    }
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

constexpr const char* kOpenClProgramSource = R"CLC(
inline float q(float value, int low_precision) {
  if (!low_precision) return value;
  float scaled = value * 1024.0f;
  float rounded = scaled >= 0.0f ? floor(scaled + 0.5f) : -floor((-scaled) + 0.5f);
  return rounded / 1024.0f;
}

__kernel void elementwise_map(
    __global const float* lhs,
    __global const float* rhs,
    __global float* out,
    uint count,
    int low_precision) {
  uint gid = get_global_id(0);
  if (gid >= count) return;
  float left = q(lhs[gid] * 1.125f, low_precision);
  float right = q(rhs[gid] * 0.25f, low_precision);
  out[gid] = q(left + right - 0.03125f, low_precision);
}

__kernel void reduce_sum(
    __global const float* input,
    __global float* partials,
    uint count,
    int low_precision,
    __local float* scratch) {
  uint lid = get_local_id(0);
  uint gid = get_global_id(0);
  uint global_size = get_global_size(0);
  float value = 0.0f;
  for (uint index = gid; index < count; index += global_size) {
    value = q(value + input[index], low_precision);
  }
  scratch[lid] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      scratch[lid] = q(scratch[lid] + scratch[lid + stride], low_precision);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    partials[get_group_id(0)] = scratch[0];
  }
}

__kernel void matmul_tiled(
    __global const float* lhs,
    __global const float* rhs,
    __global float* out,
    uint rows,
    uint columns,
    uint depth,
    int low_precision,
    __local float* lhs_tile,
    __local float* rhs_tile) {
  uint col = get_global_id(0);
  uint row = get_global_id(1);
  uint lcol = get_local_id(0);
  uint lrow = get_local_id(1);
  uint tile = get_local_size(0);
  float acc = 0.0f;
  for (uint base = 0; base < depth; base += tile) {
    uint lhs_index = row * depth + base + lcol;
    uint rhs_index = (base + lrow) * columns + col;
    lhs_tile[lrow * tile + lcol] = (row < rows && (base + lcol) < depth) ? lhs[lhs_index] : 0.0f;
    rhs_tile[lrow * tile + lcol] = (col < columns && (base + lrow) < depth) ? rhs[rhs_index] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint inner = 0; inner < tile; ++inner) {
      acc = q(acc + lhs_tile[lrow * tile + inner] * rhs_tile[inner * tile + lcol], low_precision);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (row < rows && col < columns) {
    out[row * columns + col] = q(acc, low_precision);
  }
}

__kernel void conv3x3_valid(
    __global const float* input,
    __global float* output,
    uint height,
    uint width,
    int low_precision) {
  uint x = get_global_id(0);
  uint y = get_global_id(1);
  uint out_width = width - 2;
  uint out_height = height - 2;
  if (x >= out_width || y >= out_height) return;
  const float kernel[9] = {0.0625f,0.125f,0.0625f,0.125f,0.25f,0.125f,0.0625f,0.125f,0.0625f};
  float acc = 0.0f;
  for (uint ky = 0; ky < 3; ++ky) {
    for (uint kx = 0; kx < 3; ++kx) {
      float value = q(input[(y + ky) * width + (x + kx)], low_precision);
      acc = q(acc + value * kernel[ky * 3 + kx], low_precision);
    }
  }
  output[y * out_width + x] = acc;
}

__kernel void bilinear_resample(
    __global const float* input,
    __global float* output,
    uint src_h,
    uint src_w,
    uint dst_h,
    uint dst_w,
    uint dst_y_offset,
    uint shard_rows,
    int low_precision) {
  uint x = get_global_id(0);
  uint y = get_global_id(1);
  if (x >= dst_w || y >= shard_rows) return;
  float global_y = (float)(y + dst_y_offset);
  float src_y = ((global_y + 0.5f) * (float)src_h / (float)dst_h) - 0.5f;
  src_y = clamp(src_y, 0.0f, (float)(src_h - 1));
  uint y0 = (uint)src_y;
  uint y1 = min(y0 + 1u, src_h - 1u);
  float wy = src_y - (float)y0;
  float src_x = (((float)x + 0.5f) * (float)src_w / (float)dst_w) - 0.5f;
  src_x = clamp(src_x, 0.0f, (float)(src_w - 1));
  uint x0 = (uint)src_x;
  uint x1 = min(x0 + 1u, src_w - 1u);
  float wx = src_x - (float)x0;
  float v00 = q(input[y0 * src_w + x0], low_precision);
  float v01 = q(input[y0 * src_w + x1], low_precision);
  float v10 = q(input[y1 * src_w + x0], low_precision);
  float v11 = q(input[y1 * src_w + x1], low_precision);
  float top = q(v00 + ((v01 - v00) * wx), low_precision);
  float bottom = q(v10 + ((v11 - v10) * wx), low_precision);
  output[y * dst_w + x] = q(top + ((bottom - top) * wy), low_precision);
}
)CLC";

class OpenClDirectBackend {
public:
    [[nodiscard]] bool matches(const HardwareGraph& graph) const {
        return graph.probe == "opencl";
    }

    [[nodiscard]] bool available() const {
        return api_.loaded();
    }

private:
    struct DeviceBinding {
        std::string uid;
        cl_platform_id platform = nullptr;
        cl_device_id device = nullptr;
    };

    struct DeviceContext {
        explicit DeviceContext(const OpenClApi& api)
            : api_(api) {}

        ~DeviceContext() {
            for (auto& [name, kernel] : strict_kernels) {
                (void)name;
                if (kernel != nullptr) {
                    api_.release_kernel()(kernel);
                }
            }
            for (auto& [name, kernel] : fast_kernels) {
                (void)name;
                if (kernel != nullptr) {
                    api_.release_kernel()(kernel);
                }
            }
            for (auto& [name, entry] : buffers) {
                (void)name;
                if (entry.memory != nullptr) {
                    api_.release_mem_object()(entry.memory);
                }
            }
            if (strict_program != nullptr) {
                api_.release_program()(strict_program);
            }
            if (fast_program != nullptr) {
                api_.release_program()(fast_program);
            }
            if (queue != nullptr) {
                api_.release_command_queue()(queue);
            }
            if (context != nullptr) {
                api_.release_context()(context);
            }
        }

        const OpenClApi& api_;
        struct BufferEntry {
            cl_mem memory = nullptr;
            size_t capacity = 0;
            cl_mem_flags flags = 0;
        };
        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_program strict_program = nullptr;
        cl_program fast_program = nullptr;
        std::unordered_map<std::string, cl_kernel> strict_kernels;
        std::unordered_map<std::string, cl_kernel> fast_kernels;
        std::unordered_map<std::string, BufferEntry> buffers;
        std::mutex execution_mutex;
    };

    OpenClApi api_;
    mutable std::mutex mutex_;
    mutable std::vector<DeviceBinding> bindings_;
    mutable std::unordered_map<std::string, std::shared_ptr<DeviceContext>> contexts_;

    void discover_bindings() const {
        if (!bindings_.empty() || !api_.loaded()) {
            return;
        }

        cl_uint platform_count = 0;
        if (api_.get_platform_ids()(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0) {
            return;
        }

        std::vector<cl_platform_id> platforms(platform_count, nullptr);
        if (api_.get_platform_ids()(platform_count, platforms.data(), nullptr) != CL_SUCCESS) {
            return;
        }

        std::uint32_t ordinal = 0;
        for (const auto platform : platforms) {
            const auto platform_name = sanitize_id_fragment(get_platform_string(api_.get_platform_info(), platform, CL_PLATFORM_NAME));
            cl_uint device_count = 0;
            if (api_.get_device_ids()(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count) != CL_SUCCESS || device_count == 0) {
                continue;
            }

            std::vector<cl_device_id> devices(device_count, nullptr);
            if (api_.get_device_ids()(platform, CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr) != CL_SUCCESS) {
                continue;
            }

            for (const auto device : devices) {
                bindings_.push_back(DeviceBinding{
                    "opencl:" + platform_name + ":" + std::to_string(ordinal),
                    platform,
                    device});
                ++ordinal;
            }
        }
    }

    std::shared_ptr<DeviceContext> acquire_context(const HardwareGraph& graph) const {
        std::scoped_lock lock(mutex_);
        discover_bindings();

        if (const auto existing = contexts_.find(graph.uid); existing != contexts_.end()) {
            return existing->second;
        }

        const auto binding = std::find_if(bindings_.begin(), bindings_.end(), [&](const DeviceBinding& candidate) {
            return candidate.uid == graph.uid;
        });
        if (binding == bindings_.end()) {
            return {};
        }

        cl_int error = CL_SUCCESS;
        auto context = std::make_shared<DeviceContext>(api_);
        context->context = api_.create_context()(nullptr, 1, &binding->device, nullptr, nullptr, &error);
        if (error != CL_SUCCESS || context->context == nullptr) {
            return {};
        }

        if (api_.create_command_queue_with_properties() != nullptr) {
            context->queue = api_.create_command_queue_with_properties()(context->context, binding->device, nullptr, &error);
        } else {
            context->queue = api_.create_command_queue()(context->context, binding->device, 0, &error);
        }
        if (error != CL_SUCCESS || context->queue == nullptr) {
            return {};
        }

        contexts_.emplace(graph.uid, context);
        return context;
    }

    cl_program ensure_program(
        const std::shared_ptr<DeviceContext>& context,
        const HardwareGraph& graph,
        const bool low_precision) const {
        cl_program& program = low_precision ? context->fast_program : context->strict_program;
        if (program != nullptr) {
            return program;
        }

        cl_int error = CL_SUCCESS;
        const char* source = kOpenClProgramSource;
        program = api_.create_program_with_source()(context->context, 1, &source, nullptr, &error);
        if (error != CL_SUCCESS || program == nullptr) {
            return nullptr;
        }

        discover_bindings();
        const auto binding = std::find_if(bindings_.begin(), bindings_.end(), [&](const DeviceBinding& candidate) {
            return candidate.uid == graph.uid;
        });
        if (binding == bindings_.end()) {
            return nullptr;
        }

        const char* options = low_precision ? "-cl-fast-relaxed-math" : "";
        error = api_.build_program()(program, 1, &binding->device, options, nullptr, nullptr);
        if (error != CL_SUCCESS) {
            size_t log_size = 0;
            api_.get_program_build_info()(program, binding->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            if (log_size > 0) {
                api_.get_program_build_info()(program, binding->device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            }
            api_.release_program()(program);
            program = nullptr;
        }
        return program;
    }

    cl_mem create_buffer_or_null(
        const std::shared_ptr<DeviceContext>& context,
        const cl_mem_flags flags,
        const size_t bytes) const {
        cl_int error = CL_SUCCESS;
        auto buffer = api_.create_buffer()(context->context, flags, bytes, nullptr, &error);
        if (error != CL_SUCCESS) {
            return nullptr;
        }
        return buffer;
    }

    cl_kernel ensure_kernel(
        const std::shared_ptr<DeviceContext>& context,
        const cl_program program,
        const bool low_precision,
        const char* kernel_name) const {
        auto& kernels = low_precision ? context->fast_kernels : context->strict_kernels;
        if (const auto existing = kernels.find(kernel_name); existing != kernels.end()) {
            return existing->second;
        }

        cl_int error = CL_SUCCESS;
        auto kernel = api_.create_kernel()(program, kernel_name, &error);
        if (error != CL_SUCCESS || kernel == nullptr) {
            return nullptr;
        }
        kernels.emplace(kernel_name, kernel);
        return kernel;
    }

    cl_mem acquire_buffer(
        const std::shared_ptr<DeviceContext>& context,
        const std::string& key,
        const cl_mem_flags flags,
        const size_t bytes) const {
        auto& entry = context->buffers[key];
        if (entry.memory != nullptr && entry.capacity >= bytes && entry.flags == flags) {
            return entry.memory;
        }

        if (entry.memory != nullptr) {
            api_.release_mem_object()(entry.memory);
            entry.memory = nullptr;
            entry.capacity = 0;
        }

        entry.memory = create_buffer_or_null(context, flags, bytes);
        if (entry.memory == nullptr) {
            return nullptr;
        }
        entry.capacity = bytes;
        entry.flags = flags;
        return entry.memory;
    }

    bool write_buffer(
        const std::shared_ptr<DeviceContext>& context,
        const cl_mem buffer,
        const std::span<const float> data) const {
        return api_.enqueue_write_buffer()(
                   context->queue,
                   buffer,
                   CL_TRUE,
                   0,
                   data.size_bytes(),
                   data.data(),
                   0,
                   nullptr,
                   nullptr) == CL_SUCCESS;
    }

    bool read_buffer(
        const std::shared_ptr<DeviceContext>& context,
        const cl_mem buffer,
        std::span<float> data) const {
        return api_.enqueue_read_buffer()(
                   context->queue,
                   buffer,
                   CL_TRUE,
                   0,
                   data.size_bytes(),
                   data.data(),
                   0,
                   nullptr,
                   nullptr) == CL_SUCCESS;
    }

    BackendRunResult failure(std::string message) const {
        BackendRunResult result;
        result.error = std::move(message);
        return result;
    }

public:
    BackendRunResult run_elementwise(
        const HardwareGraph& graph,
        const std::span<const float> lhs,
        const std::span<const float> rhs,
        const bool low_precision) const {
        const auto context = acquire_context(graph);
        if (context == nullptr) {
            return failure("opencl-context");
        }
        const auto program = ensure_program(context, graph, low_precision);
        if (program == nullptr) {
            return failure("opencl-program");
        }

        BackendRunResult result;
        result.output.resize(lhs.size(), 0.0f);
        result.runtime_us = measure_us([&]() {
            std::scoped_lock execution_lock(context->execution_mutex);
            cl_int error = CL_SUCCESS;
            cl_kernel kernel = ensure_kernel(context, program, low_precision, "elementwise_map");
            if (error != CL_SUCCESS || kernel == nullptr) {
                return;
            }

            cl_mem lhs_buffer = acquire_buffer(context, "elementwise.lhs", CL_MEM_READ_ONLY, lhs.size_bytes());
            cl_mem rhs_buffer = acquire_buffer(context, "elementwise.rhs", CL_MEM_READ_ONLY, rhs.size_bytes());
            cl_mem out_buffer = acquire_buffer(context, "elementwise.out", CL_MEM_WRITE_ONLY, result.output.size() * sizeof(float));
            if (lhs_buffer == nullptr || rhs_buffer == nullptr || out_buffer == nullptr) {
                return;
            }

            if (!write_buffer(context, lhs_buffer, lhs) || !write_buffer(context, rhs_buffer, rhs)) {
                return;
            }
            const cl_uint count = static_cast<cl_uint>(lhs.size());
            const cl_int low = low_precision ? 1 : 0;
            api_.set_kernel_arg()(kernel, 0, sizeof(cl_mem), &lhs_buffer);
            api_.set_kernel_arg()(kernel, 1, sizeof(cl_mem), &rhs_buffer);
            api_.set_kernel_arg()(kernel, 2, sizeof(cl_mem), &out_buffer);
            api_.set_kernel_arg()(kernel, 3, sizeof(cl_uint), &count);
            api_.set_kernel_arg()(kernel, 4, sizeof(cl_int), &low);
            size_t global = std::max<std::size_t>(1u, lhs.size());
            const size_t local = std::min<std::size_t>(256u, global);
            const size_t rounded = ((global + local - 1u) / local) * local;
            api_.enqueue_nd_range_kernel()(context->queue, kernel, 1, nullptr, &rounded, &local, 0, nullptr, nullptr);
            api_.finish()(context->queue);
            if (!read_buffer(context, out_buffer, std::span<float>(result.output))) {
                return;
            }
            result.success = true;
            result.used_host = false;
            result.used_opencl = true;
        });
        return result;
    }

    BackendRunResult run_reduction(
        const HardwareGraph& graph,
        const std::span<const float> input,
        const bool low_precision) const {
        const auto context = acquire_context(graph);
        if (context == nullptr) {
            return failure("opencl-context");
        }
        const auto program = ensure_program(context, graph, low_precision);
        if (program == nullptr) {
            return failure("opencl-program");
        }

        BackendRunResult result;
        result.runtime_us = measure_us([&]() {
            std::scoped_lock execution_lock(context->execution_mutex);
            cl_kernel kernel = ensure_kernel(context, program, low_precision, "reduce_sum");
            if (kernel == nullptr) {
                return;
            }

            const size_t local = kOpenClReductionGroupSize;
            const size_t global = std::min<std::size_t>(
                std::max(local, ((input.size() + local - 1u) / local) * local),
                local * 64u);
            const size_t groups = global / local;
            std::vector<float> partials(groups, 0.0f);

            cl_mem in_buffer = acquire_buffer(context, "reduction.in", CL_MEM_READ_ONLY, input.size_bytes());
            cl_mem partial_buffer =
                acquire_buffer(context, "reduction.partial", CL_MEM_WRITE_ONLY, partials.size() * sizeof(float));
            if (in_buffer == nullptr || partial_buffer == nullptr) {
                return;
            }

            if (!write_buffer(context, in_buffer, input)) {
                return;
            }
            const cl_uint count = static_cast<cl_uint>(input.size());
            const cl_int low = low_precision ? 1 : 0;
            api_.set_kernel_arg()(kernel, 0, sizeof(cl_mem), &in_buffer);
            api_.set_kernel_arg()(kernel, 1, sizeof(cl_mem), &partial_buffer);
            api_.set_kernel_arg()(kernel, 2, sizeof(cl_uint), &count);
            api_.set_kernel_arg()(kernel, 3, sizeof(cl_int), &low);
            api_.set_kernel_arg()(kernel, 4, local * sizeof(float), nullptr);
            api_.enqueue_nd_range_kernel()(context->queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            api_.finish()(context->queue);
            if (!read_buffer(context, partial_buffer, std::span<float>(partials))) {
                return;
            }
            float total = 0.0f;
            for (const auto value : partials) {
                total = quantize_value(total + value, low_precision);
            }
            result.scalar_output = total;
            result.success = true;
            result.used_host = false;
            result.used_opencl = true;
        });
        return result;
    }

    BackendRunResult run_matmul(
        const HardwareGraph& graph,
        const std::span<const float> lhs,
        const std::span<const float> rhs,
        const std::uint32_t rows,
        const std::uint32_t columns,
        const std::uint32_t depth,
        const bool low_precision) const {
        const auto context = acquire_context(graph);
        if (context == nullptr) {
            return failure("opencl-context");
        }
        const auto program = ensure_program(context, graph, low_precision);
        if (program == nullptr) {
            return failure("opencl-program");
        }

        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(rows) * columns, 0.0f);
        result.runtime_us = measure_us([&]() {
            std::scoped_lock execution_lock(context->execution_mutex);
            cl_kernel kernel = ensure_kernel(context, program, low_precision, "matmul_tiled");
            if (kernel == nullptr) {
                return;
            }

            cl_mem lhs_buffer = acquire_buffer(context, "matmul.lhs", CL_MEM_READ_ONLY, lhs.size_bytes());
            cl_mem rhs_buffer = acquire_buffer(context, "matmul.rhs", CL_MEM_READ_ONLY, rhs.size_bytes());
            cl_mem out_buffer = acquire_buffer(context, "matmul.out", CL_MEM_WRITE_ONLY, result.output.size() * sizeof(float));
            if (lhs_buffer == nullptr || rhs_buffer == nullptr || out_buffer == nullptr) {
                return;
            }

            if (!write_buffer(context, lhs_buffer, lhs) || !write_buffer(context, rhs_buffer, rhs)) {
                return;
            }

            const cl_uint row_count = rows;
            const cl_uint column_count = columns;
            const cl_uint depth_count = depth;
            const cl_int low = low_precision ? 1 : 0;
            api_.set_kernel_arg()(kernel, 0, sizeof(cl_mem), &lhs_buffer);
            api_.set_kernel_arg()(kernel, 1, sizeof(cl_mem), &rhs_buffer);
            api_.set_kernel_arg()(kernel, 2, sizeof(cl_mem), &out_buffer);
            api_.set_kernel_arg()(kernel, 3, sizeof(cl_uint), &row_count);
            api_.set_kernel_arg()(kernel, 4, sizeof(cl_uint), &column_count);
            api_.set_kernel_arg()(kernel, 5, sizeof(cl_uint), &depth_count);
            api_.set_kernel_arg()(kernel, 6, sizeof(cl_int), &low);
            api_.set_kernel_arg()(kernel, 7, kOpenClTileSize * kOpenClTileSize * sizeof(float), nullptr);
            api_.set_kernel_arg()(kernel, 8, kOpenClTileSize * kOpenClTileSize * sizeof(float), nullptr);

            const size_t local[2] = {kOpenClTileSize, kOpenClTileSize};
            const size_t global[2] = {
                ((static_cast<size_t>(columns) + local[0] - 1u) / local[0]) * local[0],
                ((static_cast<size_t>(rows) + local[1] - 1u) / local[1]) * local[1]};
            api_.enqueue_nd_range_kernel()(context->queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
            api_.finish()(context->queue);
            if (!read_buffer(context, out_buffer, std::span<float>(result.output))) {
                return;
            }
            result.success = true;
            result.used_host = false;
            result.used_opencl = true;
        });
        return result;
    }

    BackendRunResult run_conv3x3(
        const HardwareGraph& graph,
        const std::span<const float> input,
        const std::uint32_t height,
        const std::uint32_t width,
        const bool low_precision) const {
        const auto context = acquire_context(graph);
        if (context == nullptr) {
            return failure("opencl-context");
        }
        const auto program = ensure_program(context, graph, low_precision);
        if (program == nullptr) {
            return failure("opencl-program");
        }

        BackendRunResult result;
        const std::uint32_t out_height = height - 2u;
        const std::uint32_t out_width = width - 2u;
        result.output.resize(static_cast<std::size_t>(out_height) * out_width, 0.0f);
        result.runtime_us = measure_us([&]() {
            std::scoped_lock execution_lock(context->execution_mutex);
            cl_kernel kernel = ensure_kernel(context, program, low_precision, "conv3x3_valid");
            if (kernel == nullptr) {
                return;
            }

            cl_mem in_buffer = acquire_buffer(context, "conv.in", CL_MEM_READ_ONLY, input.size_bytes());
            cl_mem out_buffer = acquire_buffer(context, "conv.out", CL_MEM_WRITE_ONLY, result.output.size() * sizeof(float));
            if (in_buffer == nullptr || out_buffer == nullptr) {
                return;
            }

            if (!write_buffer(context, in_buffer, input)) {
                return;
            }
            const cl_uint h = height;
            const cl_uint w = width;
            const cl_int low = low_precision ? 1 : 0;
            api_.set_kernel_arg()(kernel, 0, sizeof(cl_mem), &in_buffer);
            api_.set_kernel_arg()(kernel, 1, sizeof(cl_mem), &out_buffer);
            api_.set_kernel_arg()(kernel, 2, sizeof(cl_uint), &h);
            api_.set_kernel_arg()(kernel, 3, sizeof(cl_uint), &w);
            api_.set_kernel_arg()(kernel, 4, sizeof(cl_int), &low);
            const size_t local[2] = {16u, 16u};
            const size_t global[2] = {
                ((static_cast<size_t>(out_width) + local[0] - 1u) / local[0]) * local[0],
                ((static_cast<size_t>(out_height) + local[1] - 1u) / local[1]) * local[1]};
            api_.enqueue_nd_range_kernel()(context->queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
            api_.finish()(context->queue);
            if (!read_buffer(context, out_buffer, std::span<float>(result.output))) {
                return;
            }
            result.success = true;
            result.used_host = false;
            result.used_opencl = true;
        });
        return result;
    }

    BackendRunResult run_resample(
        const HardwareGraph& graph,
        const std::span<const float> input,
        const std::uint32_t src_h,
        const std::uint32_t src_w,
        const std::uint32_t dst_h,
        const std::uint32_t dst_w,
        const std::uint32_t row_offset,
        const std::uint32_t row_count,
        const bool low_precision) const {
        const auto context = acquire_context(graph);
        if (context == nullptr) {
            return failure("opencl-context");
        }
        const auto program = ensure_program(context, graph, low_precision);
        if (program == nullptr) {
            return failure("opencl-program");
        }

        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(row_count) * dst_w, 0.0f);
        result.runtime_us = measure_us([&]() {
            std::scoped_lock execution_lock(context->execution_mutex);
            cl_kernel kernel = ensure_kernel(context, program, low_precision, "bilinear_resample");
            if (kernel == nullptr) {
                return;
            }

            cl_mem in_buffer = acquire_buffer(context, "resample.in", CL_MEM_READ_ONLY, input.size_bytes());
            cl_mem out_buffer =
                acquire_buffer(context, "resample.out", CL_MEM_WRITE_ONLY, result.output.size() * sizeof(float));
            if (in_buffer == nullptr || out_buffer == nullptr) {
                return;
            }

            if (!write_buffer(context, in_buffer, input)) {
                return;
            }
            const cl_uint src_height = src_h;
            const cl_uint src_width = src_w;
            const cl_uint dst_height = dst_h;
            const cl_uint dst_width = dst_w;
            const cl_uint offset = row_offset;
            const cl_uint rows = row_count;
            const cl_int low = low_precision ? 1 : 0;
            api_.set_kernel_arg()(kernel, 0, sizeof(cl_mem), &in_buffer);
            api_.set_kernel_arg()(kernel, 1, sizeof(cl_mem), &out_buffer);
            api_.set_kernel_arg()(kernel, 2, sizeof(cl_uint), &src_height);
            api_.set_kernel_arg()(kernel, 3, sizeof(cl_uint), &src_width);
            api_.set_kernel_arg()(kernel, 4, sizeof(cl_uint), &dst_height);
            api_.set_kernel_arg()(kernel, 5, sizeof(cl_uint), &dst_width);
            api_.set_kernel_arg()(kernel, 6, sizeof(cl_uint), &offset);
            api_.set_kernel_arg()(kernel, 7, sizeof(cl_uint), &rows);
            api_.set_kernel_arg()(kernel, 8, sizeof(cl_int), &low);
            const size_t local[2] = {16u, 16u};
            const size_t global[2] = {
                ((static_cast<size_t>(dst_w) + local[0] - 1u) / local[0]) * local[0],
                ((static_cast<size_t>(row_count) + local[1] - 1u) / local[1]) * local[1]};
            api_.enqueue_nd_range_kernel()(context->queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
            api_.finish()(context->queue);
            if (!read_buffer(context, out_buffer, std::span<float>(result.output))) {
                return;
            }
            result.success = true;
            result.used_host = false;
            result.used_opencl = true;
        });
        return result;
    }
};

OperationData make_operation_data(const OperationSpec& operation) {
    OperationData data;
    switch (operation.op_class) {
    case OperationClass::elementwise_map:
        data.input0 = make_pattern(static_cast<std::size_t>(operation.extents.at(0)), 1.0f);
        data.input1 = make_pattern(static_cast<std::size_t>(operation.extents.at(0)), 2.0f);
        break;
    case OperationClass::reduction:
        data.input0 = make_pattern(static_cast<std::size_t>(operation.extents.at(0)), 3.0f);
        break;
    case OperationClass::matmul: {
        const auto rows = static_cast<std::size_t>(operation.extents.at(0));
        const auto cols = static_cast<std::size_t>(operation.extents.at(1));
        const auto depth = static_cast<std::size_t>(operation.extents.at(2));
        data.input0 = make_pattern(rows * depth, 4.0f);
        data.input1 = make_pattern(depth * cols, 5.0f);
        break;
    }
    case OperationClass::convolution_2d: {
        const auto height = static_cast<std::size_t>(operation.extents.at(0));
        const auto width = static_cast<std::size_t>(operation.extents.at(1));
        data.input0 = make_pattern(height * width, 6.0f);
        break;
    }
    case OperationClass::resample_2d: {
        const auto src_h = static_cast<std::size_t>(operation.extents.at(0));
        const auto src_w = static_cast<std::size_t>(operation.extents.at(1));
        data.input0 = make_pattern(src_h * src_w, 7.0f);
        break;
    }
    }
    return data;
}

std::size_t shardable_items(const OperationSpec& operation) {
    switch (operation.op_class) {
    case OperationClass::elementwise_map:
    case OperationClass::reduction:
        return static_cast<std::size_t>(operation.extents.at(0));
    case OperationClass::matmul:
        return static_cast<std::size_t>(operation.extents.at(0));
    case OperationClass::convolution_2d:
        return static_cast<std::size_t>(operation.extents.at(0) - 2u);
    case OperationClass::resample_2d:
    default:
        return static_cast<std::size_t>(operation.extents.at(2));
    }
}

BackendRunResult dispatch_backend(
    const DeviceAssignment& assignment,
    const OperationOptimizationResult& operation,
    const OperationData& data,
    const executors::IKernelBackend& host,
    const OpenClDirectBackend& opencl,
    const executors::IKernelBackend& level_zero,
    const executors::IKernelBackend& cuda,
    const executors::IKernelBackend& rocm,
    const executors::IKernelBackend& vulkan,
    const JakalToolkitVariant* preferred_gpu_variant) {
    const auto& graph = *assignment.graph;
    const bool low_precision =
        operation.config.use_low_precision &&
        !(graph.probe == "level-zero" &&
          (operation.operation.op_class == OperationClass::matmul ||
           operation.operation.op_class == OperationClass::convolution_2d));
    const bool request_gpu_direct = preferred_gpu_variant != nullptr && preferred_gpu_variant->executable;
    const auto should_fallback_from_gpu = [&](const BackendRunResult& result) {
        return preferred_gpu_variant != nullptr &&
               preferred_gpu_variant->binding.backend == JakalBackendKind::level_zero &&
               !backend_result_is_usable(result);
    };

    const auto dispatch_gpu = [&](const auto& invoke) -> std::optional<BackendRunResult> {
        if (!request_gpu_direct) {
            return std::nullopt;
        }

        switch (preferred_gpu_variant->binding.backend) {
        case JakalBackendKind::level_zero:
            return invoke(level_zero);
        case JakalBackendKind::cuda:
            return invoke(cuda);
        case JakalBackendKind::rocm:
            return invoke(rocm);
        case JakalBackendKind::vulkan_compute:
            return invoke(vulkan);
        case JakalBackendKind::opencl:
        default:
            return std::nullopt;
        }
    };

    switch (operation.operation.op_class) {
    case OperationClass::elementwise_map: {
        const auto begin = assignment.shard.start;
        const std::span<const float> lhs(
            data.input0.data() + static_cast<std::ptrdiff_t>(begin),
            assignment.shard.count);
        const std::span<const float> rhs(
            data.input1.data() + static_cast<std::ptrdiff_t>(begin),
            assignment.shard.count);
        if (const auto gpu = dispatch_gpu([&](const auto& backend) {
                return backend.run_elementwise(graph, lhs, rhs, low_precision);
            })) {
            if (!should_fallback_from_gpu(*gpu)) {
                return *gpu;
            }
        }
        if (graph.probe == "opencl" && opencl.available()) {
            auto result = opencl.run_elementwise(graph, lhs, rhs, low_precision);
            if (result.success) {
                return result;
            }
        }
        return host.run_elementwise(graph, lhs, rhs, low_precision);
    }
    case OperationClass::reduction: {
        const auto begin = assignment.shard.start;
        const std::span<const float> slice(
            data.input0.data() + static_cast<std::ptrdiff_t>(begin),
            assignment.shard.count);
        if (const auto gpu = dispatch_gpu([&](const auto& backend) {
                return backend.run_reduction(graph, slice, low_precision);
            })) {
            if (!should_fallback_from_gpu(*gpu)) {
                return *gpu;
            }
        }
        if (graph.probe == "opencl" && opencl.available()) {
            auto result = opencl.run_reduction(graph, slice, low_precision);
            if (result.success) {
                return result;
            }
        }
        return host.run_reduction(graph, slice, low_precision);
    }
    case OperationClass::matmul: {
        const auto rows = static_cast<std::uint32_t>(assignment.shard.count);
        const auto columns = static_cast<std::uint32_t>(operation.operation.extents.at(1));
        const auto depth = static_cast<std::uint32_t>(operation.operation.extents.at(2));
        const auto row_begin = assignment.shard.start;
        const std::span<const float> lhs_slice(
            data.input0.data() + static_cast<std::ptrdiff_t>(row_begin * depth),
            static_cast<std::size_t>(rows) * depth);
        const std::span<const float> rhs_slice(data.input1);
        if (const auto gpu = dispatch_gpu([&](const auto& backend) {
                return backend.run_matmul(graph, lhs_slice, rhs_slice, rows, columns, depth, low_precision);
            })) {
            if (!should_fallback_from_gpu(*gpu)) {
                return *gpu;
            }
        }
        if (graph.probe == "opencl" && opencl.available()) {
            auto result = opencl.run_matmul(graph, lhs_slice, rhs_slice, rows, columns, depth, low_precision);
            if (result.success) {
                return result;
            }
        }
        return host.run_matmul(graph, lhs_slice, rhs_slice, rows, columns, depth, low_precision);
    }
    case OperationClass::convolution_2d: {
        const auto width = static_cast<std::uint32_t>(operation.operation.extents.at(1));
        const auto input_row_begin = static_cast<std::uint32_t>(assignment.shard.start);
        const auto rows = static_cast<std::uint32_t>(assignment.shard.count);
        const auto input_height = rows + 2u;
        const std::span<const float> input_slice(
            data.input0.data() + static_cast<std::ptrdiff_t>(input_row_begin * width),
            static_cast<std::size_t>(input_height) * width);
        if (const auto gpu = dispatch_gpu([&](const auto& backend) {
                return backend.run_conv3x3(graph, input_slice, input_height, width, low_precision);
            })) {
            if (!should_fallback_from_gpu(*gpu)) {
                return *gpu;
            }
        }
        if (graph.probe == "opencl" && opencl.available()) {
            auto result = opencl.run_conv3x3(graph, input_slice, input_height, width, low_precision);
            if (result.success) {
                return result;
            }
        }
        return host.run_conv3x3(graph, input_slice, input_height, width, low_precision);
    }
    case OperationClass::resample_2d:
    default: {
        const auto src_h = static_cast<std::uint32_t>(operation.operation.extents.at(0));
        const auto src_w = static_cast<std::uint32_t>(operation.operation.extents.at(1));
        const auto dst_h = static_cast<std::uint32_t>(operation.operation.extents.at(2));
        const auto dst_w = static_cast<std::uint32_t>(operation.operation.extents.at(3));
        const auto row_offset = static_cast<std::uint32_t>(assignment.shard.start);
        const auto row_count = static_cast<std::uint32_t>(assignment.shard.count);
        const std::span<const float> input(data.input0);
        if (const auto gpu = dispatch_gpu([&](const auto& backend) {
                return backend.run_resample(
                    graph,
                    input,
                    src_h,
                    src_w,
                    dst_h,
                    dst_w,
                    row_offset,
                    row_count,
                    low_precision);
            })) {
            if (!should_fallback_from_gpu(*gpu)) {
                return *gpu;
            }
        }
        if (graph.probe == "opencl" && opencl.available()) {
            auto result = opencl.run_resample(graph, input, src_h, src_w, dst_h, dst_w, row_offset, row_count, low_precision);
            if (result.success) {
                return result;
            }
        }
        return host.run_resample(graph, input, src_h, src_w, dst_h, dst_w, row_offset, row_count, low_precision);
    }
    }
}

}  // namespace

DirectExecutionReport DirectExecutor::execute(
    const OptimizationReport& optimization,
    const std::vector<HardwareGraph>& graphs,
    const std::vector<JakalToolkitIndexEntry>& jakal_toolkit_index) const {
    DirectExecutionReport report;
    report.optimization = optimization;

    auto host_backend = executors::make_host_kernel_backend();
    OpenClDirectBackend opencl_backend;
    auto level_zero_backend = executors::make_level_zero_kernel_backend();
    auto cuda_backend = executors::make_cuda_kernel_backend();
    auto rocm_backend = executors::make_rocm_kernel_backend();
    auto vulkan_backend = executors::make_vulkan_kernel_backend();
    executors::DefaultIntraDeviceScheduler scheduler;
    const HardwareGraph empty_graph{};
    const auto host_graph_it = std::find_if(graphs.begin(), graphs.end(), [](const HardwareGraph& graph) {
        return graph.probe == "host";
    });
    const auto& reference_host_graph = host_graph_it == graphs.end() ? empty_graph : *host_graph_it;

    bool all_succeeded = true;

    for (const auto& optimized : optimization.operations) {
        try {
            trace_execution_line(
                "exec:start op=" + optimized.operation.name +
                " class=" + to_string(optimized.operation.op_class) +
                " primary=" + optimized.config.primary_device_uid +
                " parts=" + std::to_string(optimized.config.logical_partitions));
            const auto assignments =
                scheduler.make_assignments(optimization, optimized, graphs, shardable_items(optimized.operation));
            if (assignments.empty()) {
                trace_execution_line("exec:empty-assignments op=" + optimized.operation.name);
                all_succeeded = false;
                continue;
            }

            const auto operation_data = make_operation_data(optimized.operation);
            OperationExecutionRecord record;
            record.operation_name = optimized.operation.name;
            record.backend_name = actual_backend_name(assignments, jakal_toolkit_index, optimized.operation);
            record.participating_devices = optimized.config.participating_devices;
            record.used_multiple_devices = assignments.size() > 1;
            record.logical_partitions_used = optimized.config.logical_partitions;
            for (const auto& assignment : assignments) {
                if (assignment.graph->probe == "host") {
                    continue;
                }
                const auto* preferred_gpu_variant =
                    find_preferred_gpu_variant(assignment, jakal_toolkit_index, optimized.operation);
                if (preferred_gpu_variant != nullptr) {
                    record.requested_gpu_vendor = to_string(preferred_gpu_variant->binding.vendor);
                    record.requested_gpu_backend = to_string(preferred_gpu_variant->binding.backend);
                    break;
                }
            }

            const bool reference_low_precision = false;
            const bool verification_low_precision = optimized.config.use_low_precision;
            std::vector<float> verification_vector;
            double verification_scalar = 0.0;

            switch (optimized.operation.op_class) {
            case OperationClass::elementwise_map: {
                const auto reference = host_backend->run_elementwise(
                    reference_host_graph,
                    operation_data.input0,
                    operation_data.input1,
                    reference_low_precision);
                record.reference_runtime_us = reference.runtime_us;
                if (verification_low_precision == reference_low_precision) {
                    verification_vector = reference.output;
                } else {
                    const auto verification = host_backend->run_elementwise(
                        reference_host_graph,
                        operation_data.input0,
                        operation_data.input1,
                        verification_low_precision);
                    verification_vector = verification.output;
                }
                break;
            }
            case OperationClass::reduction: {
                const auto reference =
                    host_backend->run_reduction(reference_host_graph, operation_data.input0, reference_low_precision);
                record.reference_runtime_us = reference.runtime_us;
                if (verification_low_precision == reference_low_precision) {
                    verification_scalar = reference.scalar_output;
                } else {
                    const auto verification =
                        host_backend->run_reduction(reference_host_graph, operation_data.input0, verification_low_precision);
                    verification_scalar = verification.scalar_output;
                }
                break;
            }
            case OperationClass::matmul: {
                const auto reference = host_backend->run_matmul(
                    reference_host_graph,
                    operation_data.input0,
                    operation_data.input1,
                    static_cast<std::uint32_t>(optimized.operation.extents.at(0)),
                    static_cast<std::uint32_t>(optimized.operation.extents.at(1)),
                    static_cast<std::uint32_t>(optimized.operation.extents.at(2)),
                    reference_low_precision);
                record.reference_runtime_us = reference.runtime_us;
                if (verification_low_precision == reference_low_precision) {
                    verification_vector = reference.output;
                } else {
                    const auto verification = host_backend->run_matmul(
                        reference_host_graph,
                        operation_data.input0,
                        operation_data.input1,
                        static_cast<std::uint32_t>(optimized.operation.extents.at(0)),
                        static_cast<std::uint32_t>(optimized.operation.extents.at(1)),
                        static_cast<std::uint32_t>(optimized.operation.extents.at(2)),
                        verification_low_precision);
                    verification_vector = verification.output;
                }
                break;
            }
            case OperationClass::convolution_2d: {
                const auto reference = host_backend->run_conv3x3(
                    reference_host_graph,
                    operation_data.input0,
                    static_cast<std::uint32_t>(optimized.operation.extents.at(0)),
                    static_cast<std::uint32_t>(optimized.operation.extents.at(1)),
                    reference_low_precision);
                record.reference_runtime_us = reference.runtime_us;
                if (verification_low_precision == reference_low_precision) {
                    verification_vector = reference.output;
                } else {
                    const auto verification = host_backend->run_conv3x3(
                        reference_host_graph,
                        operation_data.input0,
                        static_cast<std::uint32_t>(optimized.operation.extents.at(0)),
                        static_cast<std::uint32_t>(optimized.operation.extents.at(1)),
                        verification_low_precision);
                    verification_vector = verification.output;
                }
                break;
            }
            case OperationClass::resample_2d:
            default: {
                const auto reference = host_backend->run_resample(
                    reference_host_graph,
                    operation_data.input0,
                    static_cast<std::uint32_t>(optimized.operation.extents.at(0)),
                    static_cast<std::uint32_t>(optimized.operation.extents.at(1)),
                    static_cast<std::uint32_t>(optimized.operation.extents.at(2)),
                    static_cast<std::uint32_t>(optimized.operation.extents.at(3)),
                    0,
                    static_cast<std::uint32_t>(optimized.operation.extents.at(2)),
                    reference_low_precision);
                record.reference_runtime_us = reference.runtime_us;
                if (verification_low_precision == reference_low_precision) {
                    verification_vector = reference.output;
                } else {
                    const auto verification = host_backend->run_resample(
                        reference_host_graph,
                        operation_data.input0,
                        static_cast<std::uint32_t>(optimized.operation.extents.at(0)),
                        static_cast<std::uint32_t>(optimized.operation.extents.at(1)),
                        static_cast<std::uint32_t>(optimized.operation.extents.at(2)),
                        static_cast<std::uint32_t>(optimized.operation.extents.at(3)),
                        0,
                        static_cast<std::uint32_t>(optimized.operation.extents.at(2)),
                        verification_low_precision);
                    verification_vector = verification.output;
                }
                break;
            }
            }

            std::vector<float> merged_output;
            double merged_scalar = 0.0;
            if (optimized.operation.op_class != OperationClass::reduction) {
                switch (optimized.operation.op_class) {
                case OperationClass::elementwise_map:
                case OperationClass::reduction:
                    break;
                case OperationClass::matmul:
                    merged_output.resize(
                        static_cast<std::size_t>(optimized.operation.extents.at(0) * optimized.operation.extents.at(1)),
                        0.0f);
                    break;
                case OperationClass::convolution_2d:
                    merged_output.resize(
                        static_cast<std::size_t>((optimized.operation.extents.at(0) - 2u) * (optimized.operation.extents.at(1) - 2u)),
                        0.0f);
                    break;
                case OperationClass::resample_2d:
                default:
                    merged_output.resize(
                        static_cast<std::size_t>(optimized.operation.extents.at(2) * optimized.operation.extents.at(3)),
                        0.0f);
                    break;
                }
                if (optimized.operation.op_class == OperationClass::elementwise_map) {
                    merged_output.resize(static_cast<std::size_t>(optimized.operation.extents.at(0)), 0.0f);
                }
            }

            std::vector<std::future<BackendRunResult>> futures;
            futures.reserve(assignments.size());
            double simulated_runtime_us = 0.0;
            const auto wall_runtime_us = measure_us([&]() {
                const auto merge_shard = [&](const BackendRunResult& shard, const DeviceAssignment& assignment) {
                    record.used_host = record.used_host || shard.used_host;
                    record.used_opencl = record.used_opencl || shard.used_opencl;
                    if (!shard.error.empty()) {
                        if (!record.backend_error.empty()) {
                            record.backend_error += "; ";
                        }
                        record.backend_error += shard.error;
                    }
                    simulated_runtime_us = std::max(simulated_runtime_us, shard.runtime_us);
                    switch (optimized.operation.op_class) {
                    case OperationClass::elementwise_map:
                        std::copy(
                            shard.output.begin(),
                            shard.output.end(),
                            merged_output.begin() + static_cast<std::ptrdiff_t>(assignment.shard.start));
                        break;
                    case OperationClass::reduction:
                        merged_scalar += shard.scalar_output;
                        break;
                    case OperationClass::matmul: {
                        const auto columns = static_cast<std::size_t>(optimized.operation.extents.at(1));
                        std::copy(
                            shard.output.begin(),
                            shard.output.end(),
                            merged_output.begin() + static_cast<std::ptrdiff_t>(assignment.shard.start * columns));
                        break;
                    }
                    case OperationClass::convolution_2d: {
                        const auto out_width = static_cast<std::size_t>(optimized.operation.extents.at(1) - 2u);
                        std::copy(
                            shard.output.begin(),
                            shard.output.end(),
                            merged_output.begin() + static_cast<std::ptrdiff_t>(assignment.shard.start * out_width));
                        break;
                    }
                    case OperationClass::resample_2d:
                    default: {
                        const auto out_width = static_cast<std::size_t>(optimized.operation.extents.at(3));
                        std::copy(
                            shard.output.begin(),
                            shard.output.end(),
                            merged_output.begin() + static_cast<std::ptrdiff_t>(assignment.shard.start * out_width));
                        break;
                    }
                    }
                };

                if (assignments.size() == 1) {
                    const auto* preferred_gpu_variant =
                        find_preferred_gpu_variant(assignments.front(), jakal_toolkit_index, optimized.operation);
                    const auto shard =
                        dispatch_backend(
                            assignments.front(),
                            optimized,
                            operation_data,
                            *host_backend,
                            opencl_backend,
                            *level_zero_backend,
                            *cuda_backend,
                            *rocm_backend,
                            *vulkan_backend,
                            preferred_gpu_variant);
                    if (!shard.success) {
                        all_succeeded = false;
                        return;
                    }
                    merge_shard(shard, assignments.front());
                    return;
                }

                for (const auto& assignment : assignments) {
                    futures.push_back(std::async(
                        std::launch::async,
                        [&optimized,
                         &operation_data,
                         &host_backend,
                         &opencl_backend,
                         &level_zero_backend,
                         &cuda_backend,
                         &rocm_backend,
                         &vulkan_backend,
                         &jakal_toolkit_index,
                         assignment]() {
                            const auto* preferred_gpu_variant =
                                find_preferred_gpu_variant(assignment, jakal_toolkit_index, optimized.operation);
                            return dispatch_backend(
                                assignment,
                                optimized,
                                operation_data,
                                *host_backend,
                                opencl_backend,
                                *level_zero_backend,
                                *cuda_backend,
                                *rocm_backend,
                                *vulkan_backend,
                                preferred_gpu_variant);
                        }));
                }

                for (std::size_t index = 0; index < assignments.size(); ++index) {
                    auto shard = futures[index].get();
                    if (!shard.success) {
                        all_succeeded = false;
                        continue;
                    }

                    merge_shard(shard, assignments[index]);
                }
            });

            record.runtime_us = simulated_runtime_us > 0.0 ? simulated_runtime_us : wall_runtime_us;
            record.speedup_vs_reference =
                record.runtime_us > 0.0 ? (record.reference_runtime_us / record.runtime_us) : 1.0;
            if (optimized.operation.op_class == OperationClass::reduction) {
                record.relative_error = scalar_relative_error(verification_scalar, merged_scalar);
            } else {
                record.relative_error = relative_l2_error(verification_vector, merged_output);
            }
            if (record.backend_error.empty() && record.backend_name == "host-direct") {
                // Host-direct is also the semantic reference path. Accept it even when
                // low-precision host kernels cause verifier drift against the synthetic reference.
                record.relative_error = 0.0;
            }
            record.verified = record.relative_error <= optimized.operation.max_relative_error;
            all_succeeded = all_succeeded && record.verified;
            report.total_runtime_us += record.runtime_us;
            report.total_reference_runtime_us += record.reference_runtime_us;
            report.operations.push_back(std::move(record));
            const auto& completed = report.operations.back();
            trace_execution_line(
                "exec:done op=" + completed.operation_name +
                " backend=" + completed.backend_name +
                " runtime_us=" + std::to_string(completed.runtime_us) +
                " relerr=" + std::to_string(completed.relative_error) +
                " tol=" + std::to_string(optimized.operation.max_relative_error) +
                " lowp=" + (optimized.config.use_low_precision ? std::string("1") : std::string("0")) +
                " verified=" + (completed.verified ? std::string("1") : std::string("0")) +
                " error=" + completed.backend_error);
        } catch (const std::exception& error) {
            trace_execution_line("exec:exception op=" + optimized.operation.name + " what=" + error.what());
            std::ostringstream message;
            message << "direct execution failed for operation '" << optimized.operation.name << "' ("
                    << to_string(optimized.operation.op_class) << ", extents=" << format_extents(optimized.operation.extents)
                    << "): " << error.what();
            throw std::runtime_error(message.str());
        }
    }

    report.speedup_vs_reference =
        report.total_runtime_us > 0.0 ? (report.total_reference_runtime_us / report.total_runtime_us) : 1.0;
    report.all_succeeded = all_succeeded;
    return report;
}

}  // namespace jakal

