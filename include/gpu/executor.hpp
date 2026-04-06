#pragma once

#include "gpu/execution.hpp"

#include <string>
#include <vector>

namespace gpu {

struct OperationExecutionRecord {
    std::string operation_name;
    std::string backend_name;
    std::vector<std::string> participating_devices;
    double runtime_us = 0.0;
    double reference_runtime_us = 0.0;
    double relative_error = 0.0;
    bool verified = false;
    bool used_host = false;
    bool used_opencl = false;
    bool used_multiple_devices = false;
};

struct DirectExecutionReport {
    OptimizationReport optimization;
    std::vector<OperationExecutionRecord> operations;
    bool all_succeeded = false;
};

class DirectExecutor {
public:
    [[nodiscard]] DirectExecutionReport execute(
        const OptimizationReport& optimization,
        const std::vector<HardwareGraph>& graphs) const;
};

}  // namespace gpu
