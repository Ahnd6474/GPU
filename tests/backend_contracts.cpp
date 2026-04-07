#include "jakal/runtime.hpp"
#include "jakal/executors/native_gpu_backend.hpp"

#include <array>
#include <iostream>
#include <string>

namespace {

jakal::HardwareGraph make_graph(const std::string& probe, const std::string& uid) {
    jakal::HardwareGraph graph;
    graph.probe = probe;
    graph.uid = uid;
    graph.presentation_name = probe + "-device";
    return graph;
}

bool expect_backend_contract(
    const jakal::HardwareGraph& graph,
    const std::string& expected_backend_name) {
    if (jakal::runtime_backend_name_for_graph(graph) != expected_backend_name) {
        std::cerr << "unexpected backend name for probe " << graph.probe << '\n';
        return false;
    }

    constexpr std::array<jakal::OperationClass, 5> kAllOperationClasses = {
        jakal::OperationClass::elementwise_map,
        jakal::OperationClass::reduction,
        jakal::OperationClass::matmul,
        jakal::OperationClass::convolution_2d,
        jakal::OperationClass::resample_2d,
    };
    for (const auto op_class : kAllOperationClasses) {
        std::string reason;
        if (!jakal::runtime_backend_supports_operation(graph, op_class, &reason)) {
            std::cerr << "backend contract rejected op for probe " << graph.probe
                      << " reason=" << reason << '\n';
            return false;
        }
    }

    return true;
}

bool expect_native_backend_contract(
    const jakal::JakalBackendKind backend_kind,
    const jakal::HardwareGraph& graph,
    const std::string& expected_name) {
    auto backend = jakal::executors::make_native_gpu_kernel_backend(backend_kind);
    if (!backend) {
        std::cerr << "native backend factory returned null for " << graph.probe << '\n';
        return false;
    }
    if (backend->name() != expected_name) {
        std::cerr << "unexpected native backend name for " << graph.probe << '\n';
        return false;
    }
    if (!backend->matches(graph)) {
        std::cerr << "native backend did not match graph for " << graph.probe << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const auto opencl = make_graph("opencl", "gpu-opencl");
    const auto level_zero = make_graph("level-zero", "gpu-level-zero");
    const auto cuda = make_graph("cuda", "gpu-cuda");
    const auto rocm = make_graph("rocm", "gpu-rocm");

    if (!expect_backend_contract(opencl, "opencl-direct") ||
        !expect_backend_contract(level_zero, "level-zero-native") ||
        !expect_backend_contract(cuda, "cuda-native") ||
        !expect_backend_contract(rocm, "rocm-native")) {
        return 1;
    }

    if (!expect_native_backend_contract(jakal::JakalBackendKind::level_zero, level_zero, "level-zero-native") ||
        !expect_native_backend_contract(jakal::JakalBackendKind::cuda, cuda, "cuda-native") ||
        !expect_native_backend_contract(jakal::JakalBackendKind::rocm, rocm, "rocm-native")) {
        return 1;
    }

    std::string reason;
    const auto unknown = make_graph("mystery", "gpu-unknown");
    if (jakal::runtime_backend_supports_operation(unknown, jakal::OperationClass::matmul, &reason) ||
        reason.empty()) {
        std::cerr << "unknown backend should fail kernel coverage with a reason\n";
        return 1;
    }

    std::cout << "backend contracts ok\n";
    return 0;
}
