#pragma once

#include "jakal/device.hpp"
#include "jakal/execution.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace jakal::executors {

struct ShardRange {
    std::size_t start = 0;
    std::size_t count = 0;
};

struct DeviceAssignment {
    const HardwareGraph* graph = nullptr;
    double ratio = 0.0;
    ShardRange shard;
    std::uint32_t logical_partition_index = 0;
    std::uint32_t logical_partition_count = 1;
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
    bool used_host = true;
    bool used_opencl = false;
    std::string error;
};

class IKernelBackend {
public:
    virtual ~IKernelBackend() = default;

    [[nodiscard]] virtual bool matches(const HardwareGraph& graph) const = 0;
    [[nodiscard]] virtual std::string name() const = 0;

    virtual BackendRunResult run_elementwise(
        const HardwareGraph& graph,
        std::span<const float> lhs,
        std::span<const float> rhs,
        bool low_precision) const = 0;

    virtual BackendRunResult run_reduction(
        const HardwareGraph& graph,
        std::span<const float> input,
        bool low_precision) const = 0;

    virtual BackendRunResult run_matmul(
        const HardwareGraph& graph,
        std::span<const float> lhs,
        std::span<const float> rhs,
        std::uint32_t rows,
        std::uint32_t columns,
        std::uint32_t depth,
        bool low_precision) const = 0;

    virtual BackendRunResult run_conv3x3(
        const HardwareGraph& graph,
        std::span<const float> input,
        std::uint32_t height,
        std::uint32_t width,
        bool low_precision) const = 0;

    virtual BackendRunResult run_resample(
        const HardwareGraph& graph,
        std::span<const float> input,
        std::uint32_t src_h,
        std::uint32_t src_w,
        std::uint32_t dst_h,
        std::uint32_t dst_w,
        std::uint32_t row_offset,
        std::uint32_t row_count,
        bool low_precision) const = 0;
};

class IIntraDeviceScheduler {
public:
    virtual ~IIntraDeviceScheduler() = default;

    [[nodiscard]] virtual std::vector<DeviceAssignment> make_assignments(
        const OptimizationReport& optimization,
        const OperationOptimizationResult& operation,
        const std::vector<HardwareGraph>& graphs,
        std::size_t total_items) const = 0;
};

class IDeviceExecutor {
public:
    virtual ~IDeviceExecutor() = default;

    [[nodiscard]] virtual bool matches(const HardwareGraph& graph) const = 0;
    [[nodiscard]] virtual std::string name() const = 0;

    virtual BackendRunResult dispatch(
        const DeviceAssignment& assignment,
        const OperationOptimizationResult& operation,
        const OperationData& data) const = 0;
};

}  // namespace jakal::executors

