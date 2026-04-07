#pragma once

#include "jakal/backend.hpp"
#include "jakal/execution.hpp"
#include "jakal/executor.hpp"
#include "jakal/jakal_toolkit.hpp"
#include "jakal/planner.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace jakal {

struct RuntimeOptions {
    bool enable_host_probe = true;
    bool enable_opencl_probe = true;
    bool enable_level_zero_probe = true;
    bool enable_cuda_probe = true;
    bool enable_rocm_probe = true;
    std::filesystem::path cache_path;
    std::filesystem::path execution_cache_path;
};

class Runtime {
public:
    explicit Runtime(RuntimeOptions options = {});

    void refresh_hardware();

    [[nodiscard]] const std::vector<HardwareGraph>& devices() const;
    [[nodiscard]] const std::vector<JakalToolkitIndexEntry>& jakal_toolkit_index() const;
    [[nodiscard]] ExecutionPlan plan(const WorkloadSpec& workload);
    [[nodiscard]] OptimizationReport optimize(const WorkloadSpec& workload);
    [[nodiscard]] DirectExecutionReport execute(const WorkloadSpec& workload);

private:
    [[nodiscard]] bool should_include_descriptor(const HardwareGraph& candidate) const;

    RuntimeOptions options_;
    Planner planner_;
    ExecutionOptimizer execution_optimizer_;
    DirectExecutor direct_executor_;
    JakalToolkit jakal_toolkit_;
    std::vector<std::unique_ptr<IDeviceProbe>> probes_;
    std::vector<HardwareGraph> devices_;
    std::vector<JakalToolkitIndexEntry> jakal_toolkit_index_;
};

}  // namespace jakal

