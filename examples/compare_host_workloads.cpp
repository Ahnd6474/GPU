#include "jakal/runtime.hpp"
#include "jakal/workloads.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

std::filesystem::path unique_temp_file(const std::string& stem) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (stem + "-" + std::to_string(nonce) + ".tsv");
}

double measure_ms(const auto& func) {
    const auto start = std::chrono::steady_clock::now();
    func();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

int main() {
    jakal::RuntimeOptions options;
    options.enable_opencl_probe = false;
    options.cache_path = unique_temp_file("gpu-compare-plan");
    options.execution_cache_path = unique_temp_file("gpu-compare-exec");

    jakal::Runtime runtime(options);
    const std::vector<jakal::WorkloadSpec> workloads{
        jakal::WorkloadSpec{
            "host-gaming-lite",
            jakal::WorkloadKind::gaming,
            "",
            192ull * 1024ull * 1024ull,
            24ull * 1024ull * 1024ull,
            1.2e11,
            1,
            true,
            true,
            false},
        jakal::WorkloadSpec{
            "host-inference-lite",
            jakal::WorkloadKind::inference,
            "",
            256ull * 1024ull * 1024ull,
            32ull * 1024ull * 1024ull,
            2.4e11,
            4,
            false,
            false,
            true},
        jakal::WorkloadSpec{
            "host-training-lite",
            jakal::WorkloadKind::training,
            "",
            320ull * 1024ull * 1024ull,
            48ull * 1024ull * 1024ull,
            3.2e11,
            8,
            false,
            false,
            true}};

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "profile\tpass\tkind\tops\twarm_ms\ttotal_runtime_us\treference_us\tspeedup\treadiness\tstability\n";
    for (const auto& workload : workloads) {
        jakal::DirectExecutionReport cold;
        const auto cold_ms = measure_ms([&]() {
            cold = runtime.execute(workload);
        });
        if (!cold.all_succeeded) {
            std::cerr << "Cold execution failed for " << workload.name << ".\n";
            return 1;
        }

        jakal::DirectExecutionReport warm;
        const auto warm_ms = measure_ms([&]() {
            warm = runtime.execute(workload);
        });
        if (!warm.all_succeeded) {
            std::cerr << "Warm execution failed for " << workload.name << ".\n";
            return 1;
        }

        std::cout << workload.name
                  << "\tcold"
                  << '\t' << jakal::to_string(workload.kind)
                  << '\t' << cold.operations.size()
                  << '\t' << cold_ms
                  << '\t' << cold.total_runtime_us
                  << '\t' << cold.total_reference_runtime_us
                  << '\t' << cold.speedup_vs_reference
                  << '\t' << cold.optimization.system_profile.readiness_score
                  << '\t' << cold.optimization.system_profile.stability_score
                  << '\n';

        std::cout << workload.name
                  << "\twarm"
                  << '\t' << jakal::to_string(workload.kind)
                  << '\t' << warm.operations.size()
                  << '\t' << warm_ms
                  << '\t' << warm.total_runtime_us
                  << '\t' << warm.total_reference_runtime_us
                  << '\t' << warm.speedup_vs_reference
                  << '\t' << warm.optimization.system_profile.readiness_score
                  << '\t' << warm.optimization.system_profile.stability_score
                  << '\n';
    }

    return 0;
}

