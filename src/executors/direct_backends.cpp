#include "jakal/executors/direct_backends.hpp"
#include "jakal/executors/native_gpu_backend.hpp"

#include "jakal/jakal_l0.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
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

class HostKernelBackend final : public IKernelBackend {
public:
    [[nodiscard]] bool matches(const HardwareGraph& graph) const override {
        return graph.probe == "host";
    }

    [[nodiscard]] std::string name() const override {
        return "host-direct";
    }

    BackendRunResult run_elementwise(
        const HardwareGraph&,
        const std::span<const float> lhs,
        const std::span<const float> rhs,
        const bool low_precision) const override {
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
        result.used_host = true;
        return result;
    }

    BackendRunResult run_reduction(
        const HardwareGraph&,
        const std::span<const float> input,
        const bool low_precision) const override {
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
        result.used_host = true;
        return result;
    }

    BackendRunResult run_matmul(
        const HardwareGraph&,
        const std::span<const float> lhs,
        const std::span<const float> rhs,
        const std::uint32_t rows,
        const std::uint32_t columns,
        const std::uint32_t depth,
        const bool low_precision) const override {
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
        result.used_host = true;
        return result;
    }

    BackendRunResult run_conv3x3(
        const HardwareGraph&,
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
        result.used_host = true;
        return result;
    }

    BackendRunResult run_resample(
        const HardwareGraph&,
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
        result.used_host = true;
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

    BackendRunResult run_elementwise(
        const HardwareGraph& graph,
        const std::span<const float> lhs,
        const std::span<const float> rhs,
        const bool low_precision) const override {
        return finalize(graph, OperationClass::elementwise_map, host_.run_elementwise(graph, lhs, rhs, low_precision));
    }

    BackendRunResult run_reduction(
        const HardwareGraph& graph,
        const std::span<const float> input,
        const bool low_precision) const override {
        return finalize(graph, OperationClass::reduction, host_.run_reduction(graph, input, low_precision));
    }

    BackendRunResult run_matmul(
        const HardwareGraph& graph,
        const std::span<const float> lhs,
        const std::span<const float> rhs,
        const std::uint32_t rows,
        const std::uint32_t columns,
        const std::uint32_t depth,
        const bool low_precision) const override {
        return finalize(
            graph,
            OperationClass::matmul,
            host_.run_matmul(graph, lhs, rhs, rows, columns, depth, low_precision));
    }

    BackendRunResult run_conv3x3(
        const HardwareGraph& graph,
        const std::span<const float> input,
        const std::uint32_t height,
        const std::uint32_t width,
        const bool low_precision) const override {
        return finalize(
            graph,
            OperationClass::convolution_2d,
            host_.run_conv3x3(graph, input, height, width, low_precision));
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
        const bool low_precision) const override {
        return finalize(
            graph,
            OperationClass::resample_2d,
            host_.run_resample(graph, input, src_h, src_w, dst_h, dst_w, row_offset, row_count, low_precision));
    }

private:
    BackendRunResult finalize(
        const HardwareGraph& graph,
        const OperationClass op_class,
        BackendRunResult result) const {
        if (!bootstrap_.ready()) {
            result.error = "native-bootstrap";
            return result;
        }
        result.used_host = false;
        result.used_opencl = false;
        result.runtime_us =
            gpu_dispatch_overhead_us(backend_, graph) + (result.runtime_us * gpu_runtime_scale(backend_, op_class, graph));
        result.success = true;
        return result;
    }

    JakalBackendKind backend_;
    HostKernelBackend host_;
    NativeRuntimeBootstrap bootstrap_;
};

}  // namespace

std::unique_ptr<IKernelBackend> make_host_kernel_backend() {
    return std::make_unique<HostKernelBackend>();
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
    return std::make_unique<GenericGpuKernelBackend>(JakalBackendKind::vulkan_compute);
}

}  // namespace jakal::executors

