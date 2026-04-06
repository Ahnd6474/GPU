#include "gpu/executor.hpp"

#include "gpu/device.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
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

namespace gpu {
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

template <typename T>
T max_or_default(const T value, const T fallback) {
    return value == T{} ? fallback : value;
}

struct ShardRange {
    std::size_t start = 0;
    std::size_t count = 0;
};

struct DeviceAssignment {
    const HardwareGraph* graph = nullptr;
    double ratio = 0.0;
    ShardRange shard;
};

struct OperationData {
    std::vector<float> input0;
    std::vector<float> input1;
};

struct BackendRunResult {
    std::vector<float> output;
    double scalar_output = 0.0;
    double runtime_us = 0.0;
    bool success = false;
    std::string error;
};

std::vector<DeviceAssignment> make_assignments(
    const OptimizationReport& optimization,
    const OperationOptimizationResult& operation,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::size_t total_items) {
    std::unordered_map<std::string, double> ratios;
    double total_ratio = 0.0;

    for (const auto& allocation : optimization.placement.allocations) {
        if (std::find(
                operation.config.participating_devices.begin(),
                operation.config.participating_devices.end(),
                allocation.device.uid) != operation.config.participating_devices.end()) {
            ratios[allocation.device.uid] = allocation.ratio;
            total_ratio += allocation.ratio;
        }
    }

    if (total_ratio <= 0.0) {
        total_ratio = static_cast<double>(std::max<std::size_t>(operation.config.participating_devices.size(), 1u));
        for (const auto& uid : operation.config.participating_devices) {
            ratios[uid] = 1.0;
        }
    }

    std::vector<DeviceAssignment> assignments;
    assignments.reserve(operation.config.participating_devices.size());
    std::size_t consumed = 0;

    for (std::size_t index = 0; index < operation.config.participating_devices.size(); ++index) {
        const auto& uid = operation.config.participating_devices[index];
        const auto graph_it = graph_lookup.find(uid);
        if (graph_it == graph_lookup.end()) {
            continue;
        }

        double ratio = ratios.contains(uid) ? ratios.at(uid) / total_ratio : (1.0 / total_ratio);
        std::size_t count = 0;
        if (index + 1 == operation.config.participating_devices.size()) {
            count = total_items - consumed;
        } else {
            count = static_cast<std::size_t>(std::llround(static_cast<double>(total_items) * ratio));
            const std::size_t remaining = total_items - consumed;
            count = std::min(count, remaining);
        }

        assignments.push_back(DeviceAssignment{
            graph_it->second,
            ratio,
            {consumed, count}});
        consumed += count;
    }

    if (!assignments.empty() && consumed < total_items) {
        assignments.back().shard.count += total_items - consumed;
    }

    assignments.erase(
        std::remove_if(assignments.begin(), assignments.end(), [](const DeviceAssignment& assignment) {
            return assignment.graph == nullptr || assignment.shard.count == 0;
        }),
        assignments.end());

    return assignments;
}

std::string mixed_backend_name(const std::vector<DeviceAssignment>& assignments) {
    bool uses_host = false;
    bool uses_opencl = false;
    for (const auto& assignment : assignments) {
        if (assignment.graph->probe == "host") {
            uses_host = true;
        } else if (assignment.graph->probe == "opencl") {
            uses_opencl = true;
        }
    }
    if (uses_host && uses_opencl) {
        return "mixed-direct";
    }
    if (uses_opencl) {
        return "opencl-direct";
    }
    return "host-direct";
}

class HostDirectBackend {
public:
    [[nodiscard]] bool matches(const HardwareGraph& graph) const {
        return graph.probe == "host";
    }

    BackendRunResult run_elementwise(
        const std::vector<float>& lhs,
        const std::vector<float>& rhs,
        const bool low_precision) const {
        BackendRunResult result;
        result.output.resize(lhs.size(), 0.0f);
        result.runtime_us = measure_us([&]() {
            const std::size_t workers = std::max<std::size_t>(1u, std::thread::hardware_concurrency());
            const std::size_t chunk = std::max<std::size_t>(1u, lhs.size() / workers);
            std::vector<std::thread> threads;
            threads.reserve(workers);
            for (std::size_t worker = 0; worker < workers; ++worker) {
                const std::size_t begin = worker * chunk;
                if (begin >= lhs.size()) {
                    break;
                }
                const std::size_t end = worker + 1 == workers ? lhs.size() : std::min(lhs.size(), begin + chunk);
                threads.emplace_back([&, begin, end]() {
                    for (std::size_t index = begin; index < end; ++index) {
                        const float left = quantize_value(lhs[index] * 1.125f, low_precision);
                        const float right = quantize_value(rhs[index] * 0.25f, low_precision);
                        result.output[index] = quantize_value(left + right - 0.03125f, low_precision);
                    }
                });
            }
            for (auto& thread : threads) {
                thread.join();
            }
        });
        result.success = true;
        return result;
    }

    BackendRunResult run_reduction(const std::vector<float>& input, const bool low_precision) const {
        BackendRunResult result;
        result.runtime_us = measure_us([&]() {
            const std::size_t workers = std::max<std::size_t>(1u, std::thread::hardware_concurrency());
            const std::size_t chunk = std::max<std::size_t>(1u, input.size() / workers);
            std::vector<std::thread> threads;
            std::vector<float> partials(workers, 0.0f);
            threads.reserve(workers);
            for (std::size_t worker = 0; worker < workers; ++worker) {
                const std::size_t begin = worker * chunk;
                if (begin >= input.size()) {
                    break;
                }
                const std::size_t end = worker + 1 == workers ? input.size() : std::min(input.size(), begin + chunk);
                threads.emplace_back([&, worker, begin, end]() {
                    float partial = 0.0f;
                    for (std::size_t index = begin; index < end; ++index) {
                        partial = quantize_value(partial + input[index], low_precision);
                    }
                    partials[worker] = partial;
                });
            }
            for (auto& thread : threads) {
                thread.join();
            }
            float total = 0.0f;
            for (const auto partial : partials) {
                total = quantize_value(total + partial, low_precision);
            }
            result.scalar_output = total;
        });
        result.success = true;
        return result;
    }

    BackendRunResult run_matmul(
        const std::vector<float>& lhs,
        const std::vector<float>& rhs,
        const std::uint32_t rows,
        const std::uint32_t columns,
        const std::uint32_t depth,
        const bool low_precision) const {
        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(rows) * columns, 0.0f);
        result.runtime_us = measure_us([&]() {
            const std::size_t workers = std::max<std::size_t>(1u, std::thread::hardware_concurrency());
            const std::size_t chunk = std::max<std::size_t>(1u, rows / workers);
            std::vector<std::thread> threads;
            threads.reserve(workers);
            for (std::size_t worker = 0; worker < workers; ++worker) {
                const std::size_t begin = worker * chunk;
                if (begin >= rows) {
                    break;
                }
                const std::size_t end = worker + 1 == workers ? rows : std::min<std::size_t>(rows, begin + chunk);
                threads.emplace_back([&, begin, end]() {
                    for (std::size_t row = begin; row < end; ++row) {
                        for (std::uint32_t col = 0; col < columns; ++col) {
                            float acc = 0.0f;
                            for (std::uint32_t inner = 0; inner < depth; ++inner) {
                                const float left = quantize_value(lhs[row * depth + inner], low_precision);
                                const float right = quantize_value(rhs[inner * columns + col], low_precision);
                                acc = quantize_value(acc + (left * right), low_precision);
                            }
                            result.output[row * columns + col] = acc;
                        }
                    }
                });
            }
            for (auto& thread : threads) {
                thread.join();
            }
        });
        result.success = true;
        return result;
    }

    BackendRunResult run_conv3x3(
        const std::vector<float>& input,
        const std::uint32_t height,
        const std::uint32_t width,
        const bool low_precision) const {
        static constexpr std::array<float, 9> kernel{
            0.0625f, 0.125f, 0.0625f,
            0.125f, 0.25f, 0.125f,
            0.0625f, 0.125f, 0.0625f};

        const std::uint32_t out_height = height - 2u;
        const std::uint32_t out_width = width - 2u;
        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(out_height) * out_width, 0.0f);
        result.runtime_us = measure_us([&]() {
            for (std::uint32_t y = 1; y + 1 < height; ++y) {
                for (std::uint32_t x = 1; x + 1 < width; ++x) {
                    float acc = 0.0f;
                    for (std::uint32_t ky = 0; ky < 3; ++ky) {
                        for (std::uint32_t kx = 0; kx < 3; ++kx) {
                            const float value = quantize_value(
                                input[(y + ky - 1u) * width + (x + kx - 1u)],
                                low_precision);
                            acc = quantize_value(acc + (value * kernel[ky * 3u + kx]), low_precision);
                        }
                    }
                    result.output[(y - 1u) * out_width + (x - 1u)] = acc;
                }
            }
        });
        result.success = true;
        return result;
    }

    BackendRunResult run_resample(
        const std::vector<float>& input,
        const std::uint32_t src_h,
        const std::uint32_t src_w,
        const std::uint32_t dst_h,
        const std::uint32_t dst_w,
        const std::uint32_t row_offset,
        const std::uint32_t row_count,
        const bool low_precision) const {
        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(row_count) * dst_w, 0.0f);
        result.runtime_us = measure_us([&]() {
            for (std::uint32_t local_y = 0; local_y < row_count; ++local_y) {
                const std::uint32_t y = row_offset + local_y;
                const float src_y =
                    (static_cast<float>(y) + 0.5f) * static_cast<float>(src_h) / static_cast<float>(dst_h) - 0.5f;
                const float clamped_y = std::clamp(src_y, 0.0f, static_cast<float>(src_h - 1u));
                const auto y0 = static_cast<std::uint32_t>(clamped_y);
                const auto y1 = std::min(y0 + 1u, src_h - 1u);
                const float wy = clamped_y - static_cast<float>(y0);

                for (std::uint32_t x = 0; x < dst_w; ++x) {
                    const float src_x =
                        (static_cast<float>(x) + 0.5f) * static_cast<float>(src_w) / static_cast<float>(dst_w) - 0.5f;
                    const float clamped_x = std::clamp(src_x, 0.0f, static_cast<float>(src_w - 1u));
                    const auto x0 = static_cast<std::uint32_t>(clamped_x);
                    const auto x1 = std::min(x0 + 1u, src_w - 1u);
                    const float wx = clamped_x - static_cast<float>(x0);
                    const float v00 = quantize_value(input[y0 * src_w + x0], low_precision);
                    const float v01 = quantize_value(input[y0 * src_w + x1], low_precision);
                    const float v10 = quantize_value(input[y1 * src_w + x0], low_precision);
                    const float v11 = quantize_value(input[y1 * src_w + x1], low_precision);
                    const float top = quantize_value(v00 + ((v01 - v00) * wx), low_precision);
                    const float bottom = quantize_value(v10 + ((v11 - v10) * wx), low_precision);
                    result.output[local_y * dst_w + x] = quantize_value(top + ((bottom - top) * wy), low_precision);
                }
            }
        });
        result.success = true;
        return result;
    }
};

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
  return low_precision ? rint(value * 1024.0f) / 1024.0f : value;
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
        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_program strict_program = nullptr;
        cl_program fast_program = nullptr;
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

    BackendRunResult failure(std::string message) const {
        BackendRunResult result;
        result.error = std::move(message);
        return result;
    }

public:
    BackendRunResult run_elementwise(
        const HardwareGraph& graph,
        const std::vector<float>& lhs,
        const std::vector<float>& rhs,
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
            cl_int error = CL_SUCCESS;
            cl_kernel kernel = api_.create_kernel()(program, "elementwise_map", &error);
            if (error != CL_SUCCESS || kernel == nullptr) {
                return;
            }

            cl_mem lhs_buffer = create_buffer_or_null(context, CL_MEM_READ_ONLY, lhs.size() * sizeof(float));
            cl_mem rhs_buffer = create_buffer_or_null(context, CL_MEM_READ_ONLY, rhs.size() * sizeof(float));
            cl_mem out_buffer = create_buffer_or_null(context, CL_MEM_WRITE_ONLY, result.output.size() * sizeof(float));
            if (lhs_buffer == nullptr || rhs_buffer == nullptr || out_buffer == nullptr) {
                if (lhs_buffer) api_.release_mem_object()(lhs_buffer);
                if (rhs_buffer) api_.release_mem_object()(rhs_buffer);
                if (out_buffer) api_.release_mem_object()(out_buffer);
                api_.release_kernel()(kernel);
                return;
            }

            api_.enqueue_write_buffer()(context->queue, lhs_buffer, CL_TRUE, 0, lhs.size() * sizeof(float), lhs.data(), 0, nullptr, nullptr);
            api_.enqueue_write_buffer()(context->queue, rhs_buffer, CL_TRUE, 0, rhs.size() * sizeof(float), rhs.data(), 0, nullptr, nullptr);
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
            api_.enqueue_read_buffer()(context->queue, out_buffer, CL_TRUE, 0, result.output.size() * sizeof(float), result.output.data(), 0, nullptr, nullptr);
            api_.release_mem_object()(lhs_buffer);
            api_.release_mem_object()(rhs_buffer);
            api_.release_mem_object()(out_buffer);
            api_.release_kernel()(kernel);
            result.success = true;
        });
        return result;
    }

    BackendRunResult run_reduction(
        const HardwareGraph& graph,
        const std::vector<float>& input,
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
            cl_int error = CL_SUCCESS;
            cl_kernel kernel = api_.create_kernel()(program, "reduce_sum", &error);
            if (error != CL_SUCCESS || kernel == nullptr) {
                return;
            }

            const size_t local = kOpenClReductionGroupSize;
            const size_t global = std::min<std::size_t>(
                std::max(local, ((input.size() + local - 1u) / local) * local),
                local * 64u);
            const size_t groups = global / local;
            std::vector<float> partials(groups, 0.0f);

            cl_mem in_buffer = create_buffer_or_null(context, CL_MEM_READ_ONLY, input.size() * sizeof(float));
            cl_mem partial_buffer = create_buffer_or_null(context, CL_MEM_WRITE_ONLY, partials.size() * sizeof(float));
            if (in_buffer == nullptr || partial_buffer == nullptr) {
                if (in_buffer) api_.release_mem_object()(in_buffer);
                if (partial_buffer) api_.release_mem_object()(partial_buffer);
                api_.release_kernel()(kernel);
                return;
            }

            api_.enqueue_write_buffer()(context->queue, in_buffer, CL_TRUE, 0, input.size() * sizeof(float), input.data(), 0, nullptr, nullptr);
            const cl_uint count = static_cast<cl_uint>(input.size());
            const cl_int low = low_precision ? 1 : 0;
            api_.set_kernel_arg()(kernel, 0, sizeof(cl_mem), &in_buffer);
            api_.set_kernel_arg()(kernel, 1, sizeof(cl_mem), &partial_buffer);
            api_.set_kernel_arg()(kernel, 2, sizeof(cl_uint), &count);
            api_.set_kernel_arg()(kernel, 3, sizeof(cl_int), &low);
            api_.set_kernel_arg()(kernel, 4, local * sizeof(float), nullptr);
            api_.enqueue_nd_range_kernel()(context->queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            api_.finish()(context->queue);
            api_.enqueue_read_buffer()(context->queue, partial_buffer, CL_TRUE, 0, partials.size() * sizeof(float), partials.data(), 0, nullptr, nullptr);
            float total = 0.0f;
            for (const auto value : partials) {
                total = quantize_value(total + value, low_precision);
            }
            result.scalar_output = total;
            api_.release_mem_object()(in_buffer);
            api_.release_mem_object()(partial_buffer);
            api_.release_kernel()(kernel);
            result.success = true;
        });
        return result;
    }

    BackendRunResult run_matmul(
        const HardwareGraph& graph,
        const std::vector<float>& lhs,
        const std::vector<float>& rhs,
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
            cl_int error = CL_SUCCESS;
            cl_kernel kernel = api_.create_kernel()(program, "matmul_tiled", &error);
            if (error != CL_SUCCESS || kernel == nullptr) {
                return;
            }

            cl_mem lhs_buffer = create_buffer_or_null(context, CL_MEM_READ_ONLY, lhs.size() * sizeof(float));
            cl_mem rhs_buffer = create_buffer_or_null(context, CL_MEM_READ_ONLY, rhs.size() * sizeof(float));
            cl_mem out_buffer = create_buffer_or_null(context, CL_MEM_WRITE_ONLY, result.output.size() * sizeof(float));
            if (lhs_buffer == nullptr || rhs_buffer == nullptr || out_buffer == nullptr) {
                if (lhs_buffer) api_.release_mem_object()(lhs_buffer);
                if (rhs_buffer) api_.release_mem_object()(rhs_buffer);
                if (out_buffer) api_.release_mem_object()(out_buffer);
                api_.release_kernel()(kernel);
                return;
            }

            api_.enqueue_write_buffer()(context->queue, lhs_buffer, CL_TRUE, 0, lhs.size() * sizeof(float), lhs.data(), 0, nullptr, nullptr);
            api_.enqueue_write_buffer()(context->queue, rhs_buffer, CL_TRUE, 0, rhs.size() * sizeof(float), rhs.data(), 0, nullptr, nullptr);

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
            api_.enqueue_read_buffer()(context->queue, out_buffer, CL_TRUE, 0, result.output.size() * sizeof(float), result.output.data(), 0, nullptr, nullptr);
            api_.release_mem_object()(lhs_buffer);
            api_.release_mem_object()(rhs_buffer);
            api_.release_mem_object()(out_buffer);
            api_.release_kernel()(kernel);
            result.success = true;
        });
        return result;
    }

    BackendRunResult run_conv3x3(
        const HardwareGraph& graph,
        const std::vector<float>& input,
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
            cl_int error = CL_SUCCESS;
            cl_kernel kernel = api_.create_kernel()(program, "conv3x3_valid", &error);
            if (error != CL_SUCCESS || kernel == nullptr) {
                return;
            }

            cl_mem in_buffer = create_buffer_or_null(context, CL_MEM_READ_ONLY, input.size() * sizeof(float));
            cl_mem out_buffer = create_buffer_or_null(context, CL_MEM_WRITE_ONLY, result.output.size() * sizeof(float));
            if (in_buffer == nullptr || out_buffer == nullptr) {
                if (in_buffer) api_.release_mem_object()(in_buffer);
                if (out_buffer) api_.release_mem_object()(out_buffer);
                api_.release_kernel()(kernel);
                return;
            }

            api_.enqueue_write_buffer()(context->queue, in_buffer, CL_TRUE, 0, input.size() * sizeof(float), input.data(), 0, nullptr, nullptr);
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
            api_.enqueue_read_buffer()(context->queue, out_buffer, CL_TRUE, 0, result.output.size() * sizeof(float), result.output.data(), 0, nullptr, nullptr);
            api_.release_mem_object()(in_buffer);
            api_.release_mem_object()(out_buffer);
            api_.release_kernel()(kernel);
            result.success = true;
        });
        return result;
    }

    BackendRunResult run_resample(
        const HardwareGraph& graph,
        const std::vector<float>& input,
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
            cl_int error = CL_SUCCESS;
            cl_kernel kernel = api_.create_kernel()(program, "bilinear_resample", &error);
            if (error != CL_SUCCESS || kernel == nullptr) {
                return;
            }

            cl_mem in_buffer = create_buffer_or_null(context, CL_MEM_READ_ONLY, input.size() * sizeof(float));
            cl_mem out_buffer = create_buffer_or_null(context, CL_MEM_WRITE_ONLY, result.output.size() * sizeof(float));
            if (in_buffer == nullptr || out_buffer == nullptr) {
                if (in_buffer) api_.release_mem_object()(in_buffer);
                if (out_buffer) api_.release_mem_object()(out_buffer);
                api_.release_kernel()(kernel);
                return;
            }

            api_.enqueue_write_buffer()(context->queue, in_buffer, CL_TRUE, 0, input.size() * sizeof(float), input.data(), 0, nullptr, nullptr);
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
            api_.enqueue_read_buffer()(context->queue, out_buffer, CL_TRUE, 0, result.output.size() * sizeof(float), result.output.data(), 0, nullptr, nullptr);
            api_.release_mem_object()(in_buffer);
            api_.release_mem_object()(out_buffer);
            api_.release_kernel()(kernel);
            result.success = true;
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
    const HostDirectBackend& host,
    const OpenClDirectBackend& opencl) {
    const auto& graph = *assignment.graph;
    const bool low_precision = operation.config.use_low_precision;

    switch (operation.operation.op_class) {
    case OperationClass::elementwise_map: {
        const auto begin = assignment.shard.start;
        const auto end = begin + assignment.shard.count;
        std::vector<float> lhs(data.input0.begin() + static_cast<std::ptrdiff_t>(begin), data.input0.begin() + static_cast<std::ptrdiff_t>(end));
        std::vector<float> rhs(data.input1.begin() + static_cast<std::ptrdiff_t>(begin), data.input1.begin() + static_cast<std::ptrdiff_t>(end));
        if (graph.probe == "opencl" && opencl.available()) {
            auto result = opencl.run_elementwise(graph, lhs, rhs, low_precision);
            if (result.success) {
                return result;
            }
        }
        return host.run_elementwise(lhs, rhs, low_precision);
    }
    case OperationClass::reduction: {
        const auto begin = assignment.shard.start;
        const auto end = begin + assignment.shard.count;
        std::vector<float> slice(data.input0.begin() + static_cast<std::ptrdiff_t>(begin), data.input0.begin() + static_cast<std::ptrdiff_t>(end));
        if (graph.probe == "opencl" && opencl.available()) {
            auto result = opencl.run_reduction(graph, slice, low_precision);
            if (result.success) {
                return result;
            }
        }
        return host.run_reduction(slice, low_precision);
    }
    case OperationClass::matmul: {
        const auto rows = static_cast<std::uint32_t>(assignment.shard.count);
        const auto columns = static_cast<std::uint32_t>(operation.operation.extents.at(1));
        const auto depth = static_cast<std::uint32_t>(operation.operation.extents.at(2));
        const auto row_begin = assignment.shard.start;
        std::vector<float> lhs_slice(
            data.input0.begin() + static_cast<std::ptrdiff_t>(row_begin * depth),
            data.input0.begin() + static_cast<std::ptrdiff_t>((row_begin + rows) * depth));
        if (graph.probe == "opencl" && opencl.available()) {
            auto result = opencl.run_matmul(graph, lhs_slice, data.input1, rows, columns, depth, low_precision);
            if (result.success) {
                return result;
            }
        }
        return host.run_matmul(lhs_slice, data.input1, rows, columns, depth, low_precision);
    }
    case OperationClass::convolution_2d: {
        const auto width = static_cast<std::uint32_t>(operation.operation.extents.at(1));
        const auto input_row_begin = static_cast<std::uint32_t>(assignment.shard.start);
        const auto rows = static_cast<std::uint32_t>(assignment.shard.count);
        const auto input_height = rows + 2u;
        std::vector<float> input_slice(
            data.input0.begin() + static_cast<std::ptrdiff_t>(input_row_begin * width),
            data.input0.begin() + static_cast<std::ptrdiff_t>((input_row_begin + input_height) * width));
        if (graph.probe == "opencl" && opencl.available()) {
            auto result = opencl.run_conv3x3(graph, input_slice, input_height, width, low_precision);
            if (result.success) {
                return result;
            }
        }
        return host.run_conv3x3(input_slice, input_height, width, low_precision);
    }
    case OperationClass::resample_2d:
    default: {
        const auto src_h = static_cast<std::uint32_t>(operation.operation.extents.at(0));
        const auto src_w = static_cast<std::uint32_t>(operation.operation.extents.at(1));
        const auto dst_h = static_cast<std::uint32_t>(operation.operation.extents.at(2));
        const auto dst_w = static_cast<std::uint32_t>(operation.operation.extents.at(3));
        const auto row_offset = static_cast<std::uint32_t>(assignment.shard.start);
        const auto row_count = static_cast<std::uint32_t>(assignment.shard.count);
        if (graph.probe == "opencl" && opencl.available()) {
            auto result = opencl.run_resample(graph, data.input0, src_h, src_w, dst_h, dst_w, row_offset, row_count, low_precision);
            if (result.success) {
                return result;
            }
        }
        return host.run_resample(data.input0, src_h, src_w, dst_h, dst_w, row_offset, row_count, low_precision);
    }
    }
}

}  // namespace

DirectExecutionReport DirectExecutor::execute(
    const OptimizationReport& optimization,
    const std::vector<HardwareGraph>& graphs) const {
    DirectExecutionReport report;
    report.optimization = optimization;

    std::unordered_map<std::string, const HardwareGraph*> graph_lookup;
    graph_lookup.reserve(graphs.size());
    for (const auto& graph : graphs) {
        graph_lookup.emplace(graph.uid, &graph);
    }

    HostDirectBackend host_backend;
    OpenClDirectBackend opencl_backend;

    bool all_succeeded = true;

    for (const auto& optimized : optimization.operations) {
        const auto assignments =
            make_assignments(optimization, optimized, graph_lookup, shardable_items(optimized.operation));
        if (assignments.empty()) {
            all_succeeded = false;
            continue;
        }

        const auto operation_data = make_operation_data(optimized.operation);
        OperationExecutionRecord record;
        record.operation_name = optimized.operation.name;
        record.backend_name = mixed_backend_name(assignments);
        record.participating_devices = optimized.config.participating_devices;
        record.used_multiple_devices = assignments.size() > 1;

        for (const auto& assignment : assignments) {
            record.used_host = record.used_host || assignment.graph->probe == "host";
            record.used_opencl = record.used_opencl || assignment.graph->probe == "opencl";
        }

        const bool reference_low_precision = false;
        std::vector<float> reference_vector;
        double reference_scalar = 0.0;

        switch (optimized.operation.op_class) {
        case OperationClass::elementwise_map: {
            const auto reference =
                host_backend.run_elementwise(operation_data.input0, operation_data.input1, reference_low_precision);
            record.reference_runtime_us = reference.runtime_us;
            reference_vector = reference.output;
            break;
        }
        case OperationClass::reduction: {
            const auto reference = host_backend.run_reduction(operation_data.input0, reference_low_precision);
            record.reference_runtime_us = reference.runtime_us;
            reference_scalar = reference.scalar_output;
            break;
        }
        case OperationClass::matmul: {
            const auto reference = host_backend.run_matmul(
                operation_data.input0,
                operation_data.input1,
                static_cast<std::uint32_t>(optimized.operation.extents.at(0)),
                static_cast<std::uint32_t>(optimized.operation.extents.at(1)),
                static_cast<std::uint32_t>(optimized.operation.extents.at(2)),
                reference_low_precision);
            record.reference_runtime_us = reference.runtime_us;
            reference_vector = reference.output;
            break;
        }
        case OperationClass::convolution_2d: {
            const auto reference = host_backend.run_conv3x3(
                operation_data.input0,
                static_cast<std::uint32_t>(optimized.operation.extents.at(0)),
                static_cast<std::uint32_t>(optimized.operation.extents.at(1)),
                reference_low_precision);
            record.reference_runtime_us = reference.runtime_us;
            reference_vector = reference.output;
            break;
        }
        case OperationClass::resample_2d:
        default: {
            const auto reference = host_backend.run_resample(
                operation_data.input0,
                static_cast<std::uint32_t>(optimized.operation.extents.at(0)),
                static_cast<std::uint32_t>(optimized.operation.extents.at(1)),
                static_cast<std::uint32_t>(optimized.operation.extents.at(2)),
                static_cast<std::uint32_t>(optimized.operation.extents.at(3)),
                0,
                static_cast<std::uint32_t>(optimized.operation.extents.at(2)),
                reference_low_precision);
            record.reference_runtime_us = reference.runtime_us;
            reference_vector = reference.output;
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
        const auto runtime_us = measure_us([&]() {
            for (const auto& assignment : assignments) {
                futures.push_back(std::async(
                    std::launch::async,
                    [&optimized, &operation_data, &host_backend, &opencl_backend, assignment]() {
                        return dispatch_backend(assignment, optimized, operation_data, host_backend, opencl_backend);
                    }));
            }

            for (std::size_t index = 0; index < assignments.size(); ++index) {
                auto shard = futures[index].get();
                if (!shard.success) {
                    all_succeeded = false;
                    continue;
                }

                const auto& assignment = assignments[index];
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
            }
        });

        record.runtime_us = runtime_us;
        if (optimized.operation.op_class == OperationClass::reduction) {
            record.relative_error = scalar_relative_error(reference_scalar, merged_scalar);
        } else {
            record.relative_error = relative_l2_error(reference_vector, merged_output);
        }
        record.verified = record.relative_error <= optimized.operation.max_relative_error;
        all_succeeded = all_succeeded && record.verified;
        report.operations.push_back(std::move(record));
    }

    report.all_succeeded = all_succeeded;
    return report;
}

}  // namespace gpu
