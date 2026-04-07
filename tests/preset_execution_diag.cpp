#include "jakal/runtime.hpp"
#include "jakal/workloads.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

std::filesystem::path unique_temp_file(const std::string& stem) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / (stem + "-" + std::to_string(nonce) + ".tsv");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        jakal::RuntimeOptions options;
        options.cache_path = unique_temp_file("jakal-preset-diag-plan");
        options.execution_cache_path = unique_temp_file("jakal-preset-diag-exec");

        std::string mode = argc > 1 ? argv[1] : "cpu-dl";
        std::cout << "init-runtime\n" << std::flush;
        jakal::Runtime runtime(options);
        std::cout << "runtime-ready devices=" << runtime.devices().size() << '\n' << std::flush;

        if (mode == "cpu-dl") {
            const auto presets = jakal::cpu_deep_learning_exploration_presets();
            for (const auto& preset : presets) {
                std::cout << "execute " << preset.workload.name << '\n' << std::flush;
                const auto report = runtime.execute(preset.workload);
                std::cout << "done " << preset.workload.name
                          << " success=" << report.all_succeeded
                          << " ops=" << report.operations.size()
                          << " total_us=" << report.total_runtime_us
                          << '\n'
                          << std::flush;
            }
        } else {
            const auto presets = jakal::canonical_workload_presets();
            for (const auto& preset : presets) {
                std::cout << "execute " << preset.workload.name << '\n' << std::flush;
                const auto report = runtime.execute(preset.workload);
                std::cout << "done " << preset.workload.name
                          << " success=" << report.all_succeeded
                          << " ops=" << report.operations.size()
                          << " total_us=" << report.total_runtime_us
                          << '\n'
                          << std::flush;
            }
        }

        std::cout << "diag-ok\n" << std::flush;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "exception: " << error.what() << '\n';
        return 1;
    }
}
