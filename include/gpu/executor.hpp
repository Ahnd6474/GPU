#pragma once

#include "gpu/execution.hpp"
#include "gpu/gpu_toolkit.hpp"

#include <string>
#include <vector>

namespace gpu {

struct OperationExecutionRecord {
    std::string operation_name;
    std::string backend_name;
    std::string requested_gpu_vendor;
    std::string requested_gpu_backend;
    std::vector<std::string> participating_devices;
    double runtime_us = 0.0;
    double reference_runtime_us = 0.0;
    double speedup_vs_reference = 1.0;
    double relative_error = 0.0;
    bool verified = false;
    bool used_host = false;
    bool used_opencl = false;
    bool used_multiple_devices = false;
    std::uint32_t logical_partitions_used = 1;
};

struct DirectExecutionReport {
    OptimizationReport optimization;
    std::vector<OperationExecutionRecord> operations;
    double total_runtime_us = 0.0;
    double total_reference_runtime_us = 0.0;
    double speedup_vs_reference = 1.0;
    bool all_succeeded = false;
};

class DirectExecutor {
public:
    [[nodiscard]] DirectExecutionReport execute(
        const OptimizationReport& optimization,
        const std::vector<HardwareGraph>& graphs,
        const std::vector<GpuToolkitIndexEntry>& gpu_toolkit_index) const;
};

}  // namespace gpu
