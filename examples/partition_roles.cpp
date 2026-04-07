#include "jakal/executor.hpp"
#include "jakal/runtime.hpp"
#include "jakal/workloads.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path unique_temp_file(const std::string& stem) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (stem + "-" + std::to_string(nonce) + ".tsv");
}

double mib(const std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

std::string join(const std::vector<std::string>& values) {
    if (values.empty()) {
        return "none";
    }
    std::ostringstream stream;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << values[index];
    }
    return stream.str();
}

bool is_host(const jakal::HardwareGraph& graph) {
    return graph.probe == "host";
}

bool is_small_matmul(const jakal::OperationSpec& operation) {
    if (operation.op_class != jakal::OperationClass::matmul || operation.extents.size() < 3) {
        return false;
    }
    const auto m = operation.extents[0];
    const auto n = operation.extents[1];
    const auto k = operation.extents[2];
    return (m * n * k) <= 2'500'000ull;
}

bool name_contains_any(const std::string& name, std::initializer_list<const char*> needles) {
    return std::any_of(needles.begin(), needles.end(), [&](const char* needle) {
        return name.find(needle) != std::string::npos;
    });
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

enum class OperationRole {
    control_memory,
    dense_compute,
};

enum class MicroStage {
    norm_gate,
    kv_cache,
    reduction,
    projection,
    sampling,
};

enum class PlacementMode {
    host_only,
    gpu_only,
    sharded,
};

struct StagePlacement {
    PlacementMode mode = PlacementMode::host_only;
    std::uint32_t logical_partitions = 1u;
};

OperationRole classify_role(const jakal::OperationSpec& operation) {
    const std::string name = lowercase(operation.name);
    if (name_contains_any(
            name,
            {"qkv", "context", "mlp-up", "mlp-down", "proj", "attention-qkv", "logits"})) {
        return OperationRole::dense_compute;
    }
    if (operation.op_class == jakal::OperationClass::convolution_2d ||
        operation.op_class == jakal::OperationClass::resample_2d) {
        return OperationRole::dense_compute;
    }
    if (operation.op_class == jakal::OperationClass::matmul && !is_small_matmul(operation)) {
        return OperationRole::dense_compute;
    }
    if (name_contains_any(
            name,
            {"kv", "cache", "sample", "norm", "dequant", "append", "scan", "evict", "writeback", "reduce"})) {
        return OperationRole::control_memory;
    }
    if (operation.op_class == jakal::OperationClass::elementwise_map ||
        operation.op_class == jakal::OperationClass::reduction) {
        return OperationRole::control_memory;
    }
    return OperationRole::control_memory;
}

MicroStage classify_micro_stage(const jakal::OperationSpec& operation) {
    const std::string name = lowercase(operation.name);
    if (name_contains_any(name, {"sample"})) {
        return MicroStage::sampling;
    }
    if (name_contains_any(name, {"qkv", "context", "mlp-up", "mlp-down", "proj", "logits"}) ||
        operation.op_class == jakal::OperationClass::matmul) {
        return MicroStage::projection;
    }
    if (name_contains_any(name, {"kv", "cache", "append", "scan", "evict", "writeback"})) {
        return MicroStage::kv_cache;
    }
    if (name_contains_any(name, {"reduce"}) || operation.op_class == jakal::OperationClass::reduction) {
        return MicroStage::reduction;
    }
    return MicroStage::norm_gate;
}

std::string to_string(const OperationRole role) {
    switch (role) {
    case OperationRole::dense_compute:
        return "dense";
    case OperationRole::control_memory:
    default:
        return "control";
    }
}

std::string to_string(const MicroStage stage) {
    switch (stage) {
    case MicroStage::projection:
        return "projection";
    case MicroStage::reduction:
        return "reduction";
    case MicroStage::kv_cache:
        return "kv-cache";
    case MicroStage::sampling:
        return "sampling";
    case MicroStage::norm_gate:
    default:
        return "norm-gate";
    }
}

std::string to_string(const PlacementMode mode) {
    switch (mode) {
    case PlacementMode::gpu_only:
        return "gpu";
    case PlacementMode::sharded:
        return "sharded";
    case PlacementMode::host_only:
    default:
        return "host";
    }
}

struct StrategySpec {
    std::string name;
    std::string description;
    double host_ratio = 0.0;
    double accelerator_ratio = 1.0;
    StagePlacement norm_gate;
    StagePlacement kv_cache;
    StagePlacement reduction;
    StagePlacement projection;
    StagePlacement sampling;
};

std::vector<StrategySpec> role_partition_strategies() {
    return {
        {
            "blind-sharded-2",
            "Baseline: every micro-stage is sharded across host+GPU with two logical partitions per device.",
            0.35,
            0.65,
            {PlacementMode::sharded, 2u},
            {PlacementMode::sharded, 2u},
            {PlacementMode::sharded, 2u},
            {PlacementMode::sharded, 2u},
            {PlacementMode::sharded, 2u},
        },
        {
            "role-split",
            "Host owns norm/gate, KV-cache, reduction, and sampling; GPU owns projection stages.",
            0.0,
            1.0,
            {PlacementMode::host_only, 1u},
            {PlacementMode::host_only, 1u},
            {PlacementMode::host_only, 1u},
            {PlacementMode::gpu_only, 2u},
            {PlacementMode::host_only, 1u},
        },
        {
            "reduce-on-gpu",
            "Keep norm/gate, KV-cache, and sampling on host, but move reduction stages to GPU with projections.",
            0.0,
            1.0,
            {PlacementMode::host_only, 1u},
            {PlacementMode::host_only, 1u},
            {PlacementMode::gpu_only, 2u},
            {PlacementMode::gpu_only, 2u},
            {PlacementMode::host_only, 1u},
        },
        {
            "reduce-sharded-2",
            "Keep norm/gate, KV-cache, and sampling on host; project on GPU; split reductions across host+GPU.",
            0.30,
            0.70,
            {PlacementMode::host_only, 1u},
            {PlacementMode::host_only, 1u},
            {PlacementMode::sharded, 2u},
            {PlacementMode::gpu_only, 2u},
            {PlacementMode::host_only, 1u},
        },
        {
            "projection-sharded-4",
            "Keep control stages on host, but cooperatively shard projection stages across host+GPU with four partitions.",
            0.25,
            0.75,
            {PlacementMode::host_only, 1u},
            {PlacementMode::host_only, 1u},
            {PlacementMode::host_only, 1u},
            {PlacementMode::sharded, 4u},
            {PlacementMode::host_only, 1u},
        }};
}

const jakal::CpuDeepLearningExplorationPreset* find_preset(
    const std::vector<jakal::CpuDeepLearningExplorationPreset>& presets,
    const std::optional<std::string>& requested) {
    if (!requested.has_value()) {
        const auto it = std::find_if(
            presets.begin(),
            presets.end(),
            [](const jakal::CpuDeepLearningExplorationPreset& preset) {
                return preset.workload.dataset_tag == "llm-decode-token-lite";
            });
        return it == presets.end() ? &presets.front() : &*it;
    }

    const auto it = std::find_if(
        presets.begin(),
        presets.end(),
        [&](const jakal::CpuDeepLearningExplorationPreset& preset) {
            return preset.workload.name == *requested || preset.workload.dataset_tag == *requested;
        });
    return it == presets.end() ? nullptr : &*it;
}

std::vector<jakal::PlanAllocation> make_strategy_allocations(
    const jakal::HardwareGraph& host,
    const jakal::HardwareGraph& accelerator,
    const StrategySpec& strategy) {
    std::vector<jakal::PlanAllocation> allocations;
    const double total = strategy.host_ratio + strategy.accelerator_ratio;
    const double host_ratio = total > 0.0 ? strategy.host_ratio / total : 0.0;
    const double accelerator_ratio = total > 0.0 ? strategy.accelerator_ratio / total : 1.0;
    allocations.push_back({host, host_ratio, host_ratio});
    allocations.push_back({accelerator, accelerator_ratio, accelerator_ratio});
    return allocations;
}

StagePlacement placement_for_stage(const StrategySpec& strategy, const MicroStage stage) {
    switch (stage) {
    case MicroStage::projection:
        return strategy.projection;
    case MicroStage::reduction:
        return strategy.reduction;
    case MicroStage::kv_cache:
        return strategy.kv_cache;
    case MicroStage::sampling:
        return strategy.sampling;
    case MicroStage::norm_gate:
    default:
        return strategy.norm_gate;
    }
}

void apply_strategy_to_operation(
    const StrategySpec& strategy,
    const jakal::HardwareGraph& host,
    const jakal::HardwareGraph& accelerator,
    jakal::OperationOptimizationResult& operation,
    const std::string& experiment_signature) {
    const auto role = classify_role(operation.operation);
    const auto stage = classify_micro_stage(operation.operation);
    const auto placement = placement_for_stage(strategy, stage);
    auto& config = operation.config;

    if (placement.mode == PlacementMode::host_only) {
        config.primary_device_uid = host.uid;
        config.participating_devices = {host.uid};
        config.strategy = jakal::ExecutionStrategy::single_device;
        config.logical_partitions = operation.operation.parallelizable ? placement.logical_partitions : 1u;
        config.overlap_transfers = false;
    } else if (placement.mode == PlacementMode::gpu_only) {
        config.primary_device_uid = accelerator.uid;
        config.participating_devices = {accelerator.uid};
        config.strategy = jakal::ExecutionStrategy::single_device;
        config.logical_partitions = operation.operation.parallelizable ? placement.logical_partitions : 1u;
        config.overlap_transfers = false;
    } else {
        config.primary_device_uid = accelerator.uid;
        config.participating_devices = {host.uid, accelerator.uid};
        config.strategy = jakal::ExecutionStrategy::sharded;
        config.logical_partitions = operation.operation.parallelizable ? placement.logical_partitions : 1u;
        config.overlap_transfers = true;
    }

    config.signature = experiment_signature + "|" + operation.operation.name + "|" +
                       to_string(role) + "|" + to_string(stage) + "|" + join(config.participating_devices) + "|" +
                       std::to_string(config.logical_partitions);
    operation.graph.signature = config.signature;
    operation.graph.participating_devices = config.participating_devices;
    operation.benchmark.optimizer_name = strategy.name;
}

jakal::OptimizationReport make_experiment_report(
    const jakal::OptimizationReport& baseline,
    const StrategySpec& strategy,
    const jakal::HardwareGraph& host,
    const jakal::HardwareGraph& accelerator) {
    auto report = baseline;
    report.signature = baseline.signature + "|partition-role|" + strategy.name;
    report.placement.allocations = make_strategy_allocations(host, accelerator, strategy);
    report.graph_optimization.optimizer_name = strategy.name;
    report.graph_optimization.total_logical_partitions = 0u;
    for (auto& operation : report.operations) {
        apply_strategy_to_operation(strategy, host, accelerator, operation, report.signature);
        report.graph_optimization.total_logical_partitions += operation.config.logical_partitions;
    }
    return report;
}

void print_report_summary(
    const StrategySpec& strategy,
    const jakal::DirectExecutionReport& report,
    const jakal::OptimizationReport& optimization) {
    std::size_t host_ops = 0;
    std::size_t gpu_ops = 0;
    std::size_t mixed_ops = 0;
    for (const auto& operation : report.operations) {
        if (operation.used_multiple_devices) {
            ++mixed_ops;
        } else if (operation.used_host) {
            ++host_ops;
        } else {
            ++gpu_ops;
        }
    }

    std::cout << "\n[" << strategy.name << "]\n"
              << "  " << strategy.description << '\n'
              << "  total_runtime_us=" << report.total_runtime_us
              << " reference_us=" << report.total_reference_runtime_us
              << " speedup=" << report.speedup_vs_reference
              << " host_ops=" << host_ops
              << " gpu_ops=" << gpu_ops
              << " mixed_ops=" << mixed_ops
              << " logical_parts=" << optimization.graph_optimization.total_logical_partitions
              << '\n';

    for (std::size_t index = 0; index < optimization.operations.size(); ++index) {
        const auto& optimized = optimization.operations[index];
        const auto& executed = report.operations[index];
        std::cout << "    " << optimized.operation.name
                  << " role=" << to_string(classify_role(optimized.operation))
                  << " stage=" << to_string(classify_micro_stage(optimized.operation))
                  << " devices=" << join(optimized.config.participating_devices)
                  << " parts=" << optimized.config.logical_partitions
                  << " backend=" << executed.backend_name
                  << " runtime_us=" << executed.runtime_us
                  << " speedup=" << executed.speedup_vs_reference
                  << " error=" << executed.relative_error;
        if (!executed.backend_error.empty()) {
            std::cout << " error_detail=" << executed.backend_error;
        }
        std::cout << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::optional<std::string> requested_preset;
    for (int index = 1; index < argc; ++index) {
        requested_preset = argv[index];
    }

    jakal::RuntimeOptions options;
    options.cache_path = unique_temp_file("jakal-role-plan");
    options.execution_cache_path = unique_temp_file("jakal-role-exec");
    jakal::Runtime runtime(options);

    const auto presets = jakal::cpu_deep_learning_exploration_presets();
    if (presets.empty()) {
        std::cerr << "No CPU deep-learning exploration presets available.\n";
        return 1;
    }
    const auto* preset = find_preset(presets, requested_preset);
    if (preset == nullptr) {
        std::cerr << "Unknown preset.\n";
        return 1;
    }

    const auto host_it = std::find_if(runtime.devices().begin(), runtime.devices().end(), is_host);
    const auto accelerator_it = std::find_if(runtime.devices().begin(), runtime.devices().end(), [](const jakal::HardwareGraph& graph) {
        return !is_host(graph);
    });
    if (host_it == runtime.devices().end() || accelerator_it == runtime.devices().end()) {
        std::cerr << "This experiment needs both host and at least one accelerator.\n";
        return 1;
    }

    const auto baseline = runtime.optimize(preset->workload);
    jakal::DirectExecutor executor;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Role-aware CPU/GPU partition experiment\n"
              << "Preset: " << preset->workload.name
              << " [" << preset->workload.dataset_tag << "]\n"
              << "Description: " << preset->description << '\n'
              << "CPU hypothesis: " << preset->cpu_hypothesis << '\n'
              << "Host: " << host_it->uid << " (" << host_it->presentation_name << ")\n"
              << "Accelerator: " << accelerator_it->uid << " (" << accelerator_it->presentation_name << ")\n";

    for (const auto& strategy : role_partition_strategies()) {
        auto experiment = make_experiment_report(baseline, strategy, *host_it, *accelerator_it);
        const auto executed = executor.execute(experiment, runtime.devices(), runtime.jakal_toolkit_index());
        if (!executed.all_succeeded) {
            std::cerr << "Execution failed for strategy " << strategy.name << ".\n";
            return 1;
        }
        print_report_summary(strategy, executed, experiment);
    }

    return 0;
}
