#pragma once

#include "jakal/executors/interfaces.hpp"
#include "jakal/jakal_l0.hpp"

#include <memory>

namespace jakal::executors {

std::unique_ptr<IKernelBackend> make_native_gpu_kernel_backend(JakalBackendKind backend);

}  // namespace jakal::executors

