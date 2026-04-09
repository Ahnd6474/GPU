#pragma once

#include "jakal/executors/interfaces.hpp"

#include <memory>

namespace jakal::executors {

std::unique_ptr<IKernelBackend> make_host_native_kernel_backend();
std::unique_ptr<IKernelBackend> make_host_kernel_backend();
std::unique_ptr<IKernelBackend> make_level_zero_kernel_backend();
std::unique_ptr<IKernelBackend> make_cuda_kernel_backend();
std::unique_ptr<IKernelBackend> make_rocm_kernel_backend();
std::unique_ptr<IKernelBackend> make_vulkan_kernel_backend();
[[nodiscard]] bool vulkan_direct_backend_available();

}  // namespace jakal::executors

