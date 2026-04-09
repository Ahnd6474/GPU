#pragma once

#include "jakal/executors/interfaces.hpp"
#include "jakal/jakal_l0.hpp"

#include <memory>
#include <string>

namespace jakal::executors {

std::unique_ptr<IKernelBackend> make_host_native_kernel_backend();
std::unique_ptr<IKernelBackend> make_host_kernel_backend();
std::unique_ptr<IKernelBackend> make_modeled_gpu_kernel_backend(JakalBackendKind backend);
std::unique_ptr<IKernelBackend> make_level_zero_kernel_backend();
std::unique_ptr<IKernelBackend> make_cuda_kernel_backend();
std::unique_ptr<IKernelBackend> make_rocm_kernel_backend();
std::unique_ptr<IKernelBackend> make_vulkan_kernel_backend();
[[nodiscard]] bool vulkan_direct_backend_available();
[[nodiscard]] std::string vulkan_direct_backend_status_detail();

}  // namespace jakal::executors

