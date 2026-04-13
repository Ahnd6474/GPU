#include "jakal/runtime.hpp"
#include "jakal/executors/direct_backends.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <mutex>
#include <shared_mutex>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace jakal {
namespace {

bool runtime_regressed(
    const DirectExecutionReport& report,
    const double max_runtime_regression_ratio);

struct GraphWorkloadMetrics {
    std::size_t operation_count = 0;
    std::size_t matmul_count = 0;
    std::size_t reduction_count = 0;
    std::size_t elementwise_count = 0;
    std::size_t host_visible_tensor_count = 0;
    double total_flops = 0.0;
    double total_bytes = 0.0;
    double matmul_flops = 0.0;
    double reduction_flops = 0.0;
    double elementwise_flops = 0.0;
    double reduction_bytes = 0.0;
    double host_visible_bytes = 0.0;
    double small_op_ratio = 0.0;
    double matmul_flops_ratio = 0.0;
    double reduction_flops_ratio = 0.0;
    double host_visible_tensor_ratio = 0.0;
    double average_flops_per_operation = 0.0;
    double dispatch_bound_score = 0.0;
    bool has_terminal_reduction = false;
};

struct GraphAwareMetaPolicy {
    std::optional<PartitionStrategy> strategy_hint;
    double strategy_confidence = 0.0;
    std::string strategy_reason;
    bool disable_exploration = false;
    std::optional<ContinuousExecutionState> tuning_state;
    std::optional<std::uint32_t> tuning_graph_passes;
    std::uint32_t tuning_graph_rewrite_level = 1u;
    std::string tuning_reason;
};

struct RuntimeOptimizationContext {
    WorkloadSpec requested_workload;
    WorkloadSpec effective_workload;
    ExecutionTuningOverrides execution_tuning;
    std::string optimizer_budget_source = "heuristic";
    std::string meta_summary;
};

struct ResolvedOptimizerBudget {
    std::optional<std::uint32_t> budget_ms;
    std::string source = "heuristic";
};

struct TelemetryBudgetSignal {
    std::size_t samples = 0u;
    double average_speedup_vs_reference = 0.0;
    double average_transfer_overlap_ratio = 0.0;
    double average_copy_share = 0.0;
    double budget_exhaustion_ratio = 0.0;
    double average_queue_separation_ratio = 0.0;
};

struct UnifiedRuntimeSignal {
    TelemetryBudgetSignal signal;
    std::string source = "telemetry";
};

struct TelemetryBudgetCacheEntry {
    std::string kind;
    std::string phase;
    std::string shape_bucket;
    TelemetryBudgetSignal signal;
    std::uint32_t last_optimizer_budget_ms = 0u;
    std::uint64_t last_epoch = 0u;
};

struct TelemetryBudgetCacheState {
    std::unordered_map<std::string, TelemetryBudgetCacheEntry> entries;
    std::uint32_t delta_rows = 0u;
    std::chrono::steady_clock::time_point last_compaction = std::chrono::steady_clock::now();
    bool loaded = false;
};

class AsyncFileWriter final {
public:
    AsyncFileWriter()
        : worker_([this]() { run(); }) {}

    ~AsyncFileWriter() {
        {
            std::scoped_lock lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    AsyncFileWriter(const AsyncFileWriter&) = delete;
    AsyncFileWriter& operator=(const AsyncFileWriter&) = delete;

    void enqueue(std::function<void()> task) {
        {
            std::scoped_lock lock(mutex_);
            tasks_.push_back(std::move(task));
        }
        condition_.notify_one();
    }

private:
    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                condition_.wait(lock, [&]() { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) {
                    break;
                }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            if (task) {
                task();
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::function<void()>> tasks_;
    std::thread worker_;
    bool stopping_ = false;
};

AsyncFileWriter& telemetry_writer() {
    static AsyncFileWriter writer;
    return writer;
}

std::shared_mutex g_telemetry_budget_cache_mutex;
std::unordered_map<std::string, TelemetryBudgetCacheState> g_telemetry_budget_cache;

bool is_host_graph(const HardwareGraph& graph) {
    return graph.probe == "host";
}

double workload_host_exchange_ratio(const WorkloadSpec& workload) {
    const auto working_set = std::max<std::uint64_t>(workload.working_set_bytes, 1ull);
    return std::clamp(
        static_cast<double>(workload.host_exchange_bytes) / static_cast<double>(working_set),
        0.0,
        4.0);
}

bool has_partitionable_topology(const std::vector<HardwareGraph>& graphs) {
    bool saw_host = false;
    bool saw_accelerator = false;
    for (const auto& graph : graphs) {
        if (is_host_graph(graph)) {
            saw_host = true;
        } else {
            saw_accelerator = true;
        }
        if (saw_host && saw_accelerator) {
            return true;
        }
    }
    return false;
}

std::filesystem::path default_runtime_telemetry_path() {
    try {
        return std::filesystem::temp_directory_path() / "jakal_core_runtime_telemetry.tsv";
    } catch (const std::exception&) {
        return std::filesystem::path("jakal_core_runtime_telemetry.tsv");
    }
}

std::filesystem::path env_path(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return {};
    }
    return std::filesystem::path(value);
}

std::filesystem::path choose_writable_runtime_root() {
    if (const auto override_root = env_path("JAKAL_RUNTIME_HOME"); !override_root.empty()) {
        return override_root;
    }
#if defined(_WIN32)
    if (const auto local_app_data = env_path("LOCALAPPDATA"); !local_app_data.empty()) {
        return local_app_data / "Jakal-Core";
    }
    if (const auto app_data = env_path("APPDATA"); !app_data.empty()) {
        return app_data / "Jakal-Core";
    }
#else
    if (const auto xdg_state = env_path("XDG_STATE_HOME"); !xdg_state.empty()) {
        return xdg_state / "jakal-core";
    }
    if (const auto home = env_path("HOME"); !home.empty()) {
        return home / ".local" / "state" / "jakal-core";
    }
#endif
    try {
        return std::filesystem::temp_directory_path() / "jakal-core";
    } catch (const std::exception&) {
        return std::filesystem::path("jakal-core-state");
    }
}

std::filesystem::path choose_config_runtime_root() {
    if (const auto override_root = env_path("JAKAL_RUNTIME_HOME"); !override_root.empty()) {
        return override_root / "config";
    }
#if defined(_WIN32)
    if (const auto app_data = env_path("APPDATA"); !app_data.empty()) {
        return app_data / "Jakal-Core" / "config";
    }
#else
    if (const auto xdg_config = env_path("XDG_CONFIG_HOME"); !xdg_config.empty()) {
        return xdg_config / "jakal-core";
    }
    if (const auto home = env_path("HOME"); !home.empty()) {
        return home / ".config" / "jakal-core";
    }
#endif
    return choose_writable_runtime_root() / "config";
}

std::string trim_copy(std::string value) {
    auto not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

bool parse_ini_bool(const std::string& value, const bool fallback) {
    const auto normalized = trim_copy(value);
    if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on") {
        return true;
    }
    if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") {
        return false;
    }
    return fallback;
}

void apply_runtime_ini_overrides(const std::filesystem::path& path, RuntimeOptions& options) {
    std::ifstream input(path);
    if (!input.is_open()) {
        return;
    }

    std::string section;
    std::string line;
    while (std::getline(input, line)) {
        line = trim_copy(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            section = trim_copy(line.substr(1u, line.size() - 2u));
            continue;
        }
        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            continue;
        }
        const auto key = trim_copy(line.substr(0u, equals));
        const auto value = trim_copy(line.substr(equals + 1u));

        if (section == "runtime" && key == "home" && !value.empty()) {
            const auto root = std::filesystem::path(value);
            options.cache_path = root / "cache" / "planner-cache.tsv";
            options.execution_cache_path = root / "cache" / "execution-cache.tsv";
            options.product.observability.telemetry_path = root / "logs" / "runtime-telemetry.tsv";
        } else if (section == "paths") {
            if (key == "planner_cache" && !value.empty()) {
                options.cache_path = value;
            } else if (key == "execution_cache" && !value.empty()) {
                options.execution_cache_path = value;
            } else if (key == "telemetry" && !value.empty()) {
                options.product.observability.telemetry_path = value;
            }
        } else if (section == "backends") {
            if (key == "host") {
                options.enable_host_probe = parse_ini_bool(value, options.enable_host_probe);
            } else if (key == "opencl") {
                options.enable_opencl_probe = parse_ini_bool(value, options.enable_opencl_probe);
            } else if (key == "level_zero") {
                options.enable_level_zero_probe = parse_ini_bool(value, options.enable_level_zero_probe);
            } else if (key == "vulkan_probe") {
                options.enable_vulkan_probe = parse_ini_bool(value, options.enable_vulkan_probe);
            } else if (key == "vulkan_status") {
                options.enable_vulkan_status = parse_ini_bool(value, options.enable_vulkan_status);
            } else if (key == "cuda") {
                options.enable_cuda_probe = parse_ini_bool(value, options.enable_cuda_probe);
            } else if (key == "rocm") {
                options.enable_rocm_probe = parse_ini_bool(value, options.enable_rocm_probe);
            } else if (key == "prefer_level_zero_over_opencl") {
                options.prefer_level_zero_over_opencl =
                    parse_ini_bool(value, options.prefer_level_zero_over_opencl);
            }
        }
    }
}

std::filesystem::path telemetry_budget_cache_path(const std::filesystem::path& telemetry_path) {
    if (telemetry_path.empty()) {
        return {};
    }
    auto sidecar_path = telemetry_path;
    sidecar_path += ".budget.tsv";
    return sidecar_path;
}

std::filesystem::path telemetry_budget_delta_path(const std::filesystem::path& telemetry_path) {
    if (telemetry_path.empty()) {
        return {};
    }
    auto sidecar_path = telemetry_path;
    sidecar_path += ".budget.delta.tsv";
    return sidecar_path;
}

std::filesystem::path runtime_graph_family_cache_path(const std::filesystem::path& execution_cache_path) {
    if (execution_cache_path.empty()) {
        return {};
    }
    auto family_path = execution_cache_path;
    family_path += ".perf.family";
    return family_path;
}

std::size_t header_index_or(
    const std::unordered_map<std::string, std::size_t>& header_index,
    const std::string& key,
    const std::size_t fallback = std::numeric_limits<std::size_t>::max());
TelemetryBudgetSignal load_runtime_budget_signal(
    const std::filesystem::path& telemetry_path,
    const WorkloadSpec& workload);

std::vector<std::string> split_tab_fields(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) {
        fields.push_back(field);
    }
    return fields;
}

double parse_double_or(const std::string& value, const double fallback = 0.0) {
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        return fallback;
    }
}

std::uint32_t parse_uint32_or(const std::string& value, const std::uint32_t fallback = 0u) {
    try {
        return static_cast<std::uint32_t>(std::stoul(value));
    } catch (const std::exception&) {
        return fallback;
    }
}

std::uint64_t parse_uint64_or(const std::string& value, const std::uint64_t fallback = 0u) {
    try {
        return static_cast<std::uint64_t>(std::stoull(value));
    } catch (const std::exception&) {
        return fallback;
    }
}

void merge_budget_signal_sample(
    TelemetryBudgetSignal& signal,
    const double speedup_vs_reference,
    const double transfer_overlap_ratio,
    const double copy_share,
    const double budget_pressure,
    const double queue_separation_ratio,
    const double sample_weight = 1.0) {
    const auto weight = std::max(1.0, sample_weight);
    const auto next_samples = signal.samples + static_cast<std::size_t>(std::llround(weight));
    const auto update_average = [&](double& average, const double sample) {
        average += (sample - average) * (weight / static_cast<double>(next_samples));
    };
    update_average(signal.average_speedup_vs_reference, speedup_vs_reference);
    update_average(signal.average_transfer_overlap_ratio, transfer_overlap_ratio);
    update_average(signal.average_copy_share, copy_share);
    update_average(signal.budget_exhaustion_ratio, budget_pressure);
    update_average(signal.average_queue_separation_ratio, queue_separation_ratio);
    signal.samples = next_samples;
}

std::unordered_map<std::string, std::size_t> build_header_index(const std::string& header_line) {
    std::string normalized = header_line;
    if (!normalized.empty() && normalized.front() == '#') {
        normalized.erase(normalized.begin());
        if (!normalized.empty() && normalized.front() == ' ') {
            normalized.erase(normalized.begin());
        }
    }
    const auto fields = split_tab_fields(normalized);
    std::unordered_map<std::string, std::size_t> header_index;
    for (std::size_t index = 0; index < fields.size(); ++index) {
        header_index.emplace(fields[index], index);
    }
    return header_index;
}

bool read_tsv_header_line(std::ifstream& input, std::string& header_line) {
    while (std::getline(input, header_line)) {
        std::string normalized = header_line;
        if (!normalized.empty() && normalized.front() == '#') {
            normalized.erase(normalized.begin());
            if (!normalized.empty() && normalized.front() == ' ') {
                normalized.erase(normalized.begin());
            }
        }
        if (normalized.empty() || normalized.rfind("schema_version", 0u) == 0u) {
            continue;
        }
        return true;
    }
    return false;
}

std::string field_or_empty(const std::vector<std::string>& fields, const std::size_t index) {
    return index < fields.size() ? fields[index] : std::string();
}

std::string telemetry_budget_entry_key(
    const std::string& kind,
    const std::string& phase,
    const std::string& shape_bucket) {
    return kind + "|" + phase + "|" + shape_bucket;
}

void load_budget_snapshot_entries(
    const std::filesystem::path& path,
    std::unordered_map<std::string, TelemetryBudgetCacheEntry>& entries) {
    if (path.empty() || !std::filesystem::exists(path)) {
        return;
    }
    std::ifstream input(path);
    std::string header_line;
    if (!input.is_open() || !read_tsv_header_line(input, header_line)) {
        return;
    }
    const auto header_index = build_header_index(header_line);
    const auto kind_index = header_index_or(header_index, "kind");
    const auto phase_index = header_index_or(header_index, "phase");
    const auto bucket_index = header_index_or(header_index, "shape_bucket");
    const auto samples_index = header_index_or(header_index, "samples");
    const auto speedup_index = header_index_or(header_index, "average_speedup_vs_reference");
    const auto overlap_index = header_index_or(header_index, "average_transfer_overlap_ratio");
    const auto copy_share_index = header_index_or(header_index, "average_copy_share");
    const auto exhausted_index = header_index_or(header_index, "budget_exhaustion_ratio");
    const auto queue_index = header_index_or(header_index, "average_queue_separation_ratio");
    const auto budget_index = header_index_or(header_index, "last_optimizer_budget_ms");
    const auto epoch_index = header_index_or(header_index, "last_epoch");
    std::string line;
    while (std::getline(input, line)) {
        const auto fields = split_tab_fields(line);
        TelemetryBudgetCacheEntry entry;
        entry.kind = field_or_empty(fields, kind_index);
        entry.phase = field_or_empty(fields, phase_index);
        entry.shape_bucket = field_or_empty(fields, bucket_index);
        entry.signal.samples = parse_uint32_or(field_or_empty(fields, samples_index), 0u);
        entry.signal.average_speedup_vs_reference = parse_double_or(field_or_empty(fields, speedup_index), 0.0);
        entry.signal.average_transfer_overlap_ratio = parse_double_or(field_or_empty(fields, overlap_index), 0.0);
        entry.signal.average_copy_share = parse_double_or(field_or_empty(fields, copy_share_index), 0.0);
        entry.signal.budget_exhaustion_ratio = parse_double_or(field_or_empty(fields, exhausted_index), 0.0);
        entry.signal.average_queue_separation_ratio = parse_double_or(field_or_empty(fields, queue_index), 0.0);
        entry.last_optimizer_budget_ms = parse_uint32_or(field_or_empty(fields, budget_index), 0u);
        entry.last_epoch = parse_uint64_or(field_or_empty(fields, epoch_index), 0u);
        if (!entry.kind.empty() && !entry.phase.empty() && !entry.shape_bucket.empty()) {
            entries[telemetry_budget_entry_key(entry.kind, entry.phase, entry.shape_bucket)] = std::move(entry);
        }
    }
}

std::uint32_t load_budget_delta_entries(
    const std::filesystem::path& path,
    std::unordered_map<std::string, TelemetryBudgetCacheEntry>& entries) {
    if (path.empty() || !std::filesystem::exists(path)) {
        return 0u;
    }
    std::ifstream input(path);
    std::string header_line;
    if (!input.is_open() || !read_tsv_header_line(input, header_line)) {
        return 0u;
    }
    const auto header_index = build_header_index(header_line);
    const auto epoch_index = header_index_or(header_index, "epoch");
    const auto kind_index = header_index_or(header_index, "kind");
    const auto phase_index = header_index_or(header_index, "phase");
    const auto bucket_index = header_index_or(header_index, "shape_bucket");
    const auto speedup_index = header_index_or(header_index, "speedup_vs_reference");
    const auto overlap_index = header_index_or(header_index, "transfer_overlap_ratio");
    const auto copy_share_index = header_index_or(header_index, "copy_share");
    const auto exhausted_index = header_index_or(header_index, "budget_pressure");
    const auto queue_index = header_index_or(header_index, "queue_separation_ratio");
    const auto budget_index = header_index_or(header_index, "optimizer_budget_ms");
    std::uint32_t rows = 0u;
    std::string line;
    while (std::getline(input, line)) {
        const auto fields = split_tab_fields(line);
        const auto kind = field_or_empty(fields, kind_index);
        const auto phase = field_or_empty(fields, phase_index);
        const auto bucket = field_or_empty(fields, bucket_index);
        if (kind.empty() || phase.empty() || bucket.empty()) {
            continue;
        }
        auto& entry = entries[telemetry_budget_entry_key(kind, phase, bucket)];
        entry.kind = kind;
        entry.phase = phase;
        entry.shape_bucket = bucket;
        merge_budget_signal_sample(
            entry.signal,
            parse_double_or(field_or_empty(fields, speedup_index), 0.0),
            parse_double_or(field_or_empty(fields, overlap_index), 0.0),
            parse_double_or(field_or_empty(fields, copy_share_index), 0.0),
            parse_double_or(field_or_empty(fields, exhausted_index), 0.0),
            parse_double_or(field_or_empty(fields, queue_index), 0.0));
        entry.last_optimizer_budget_ms = parse_uint32_or(field_or_empty(fields, budget_index), 0u);
        entry.last_epoch = parse_uint64_or(field_or_empty(fields, epoch_index), 0u);
        ++rows;
    }
    return rows;
}

TelemetryBudgetCacheState& ensure_budget_cache_state_loaded(const std::filesystem::path& telemetry_path) {
    const auto key = telemetry_path.string();
    auto& state = g_telemetry_budget_cache[key];
    if (state.loaded) {
        return state;
    }
    state.entries.clear();
    load_budget_snapshot_entries(telemetry_budget_cache_path(telemetry_path), state.entries);
    state.delta_rows = load_budget_delta_entries(telemetry_budget_delta_path(telemetry_path), state.entries);
    state.last_compaction = std::chrono::steady_clock::now();
    state.loaded = true;
    return state;
}

void persist_budget_cache_snapshot(
    const std::filesystem::path& telemetry_path,
    TelemetryBudgetCacheState& state) {
    const auto cache_path = telemetry_budget_cache_path(telemetry_path);
    if (cache_path.empty()) {
        return;
    }
    const auto parent = cache_path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
    std::ofstream output(cache_path, std::ios::trunc);
    if (!output.is_open()) {
        return;
    }
    output << "# schema_version\t" << kRuntimeTelemetryBudgetSchemaVersion << '\n';
    output << "# kind\tphase\tshape_bucket\tsamples\taverage_speedup_vs_reference\taverage_transfer_overlap_ratio\taverage_copy_share\tbudget_exhaustion_ratio\taverage_queue_separation_ratio\tlast_optimizer_budget_ms\tlast_epoch\n";
    for (const auto& [key, entry] : state.entries) {
        (void)key;
        output << entry.kind << '\t'
               << entry.phase << '\t'
               << entry.shape_bucket << '\t'
               << entry.signal.samples << '\t'
               << entry.signal.average_speedup_vs_reference << '\t'
               << entry.signal.average_transfer_overlap_ratio << '\t'
               << entry.signal.average_copy_share << '\t'
               << entry.signal.budget_exhaustion_ratio << '\t'
               << entry.signal.average_queue_separation_ratio << '\t'
               << entry.last_optimizer_budget_ms << '\t'
               << entry.last_epoch << '\n';
    }
    output.close();
    std::error_code ec;
    std::filesystem::remove(telemetry_budget_delta_path(telemetry_path), ec);
    state.delta_rows = 0u;
    state.last_compaction = std::chrono::steady_clock::now();
}

void append_tsv_line(
    const std::filesystem::path& path,
    const std::string& header,
    const std::string& line) {
    if (path.empty()) {
        return;
    }
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
    const bool write_header = !std::filesystem::exists(path);
    std::ofstream output(path, std::ios::app);
    if (!output.is_open()) {
        return;
    }
    if (write_header && !header.empty()) {
        output << header;
    }
    output << line;
}

void persist_budget_cache_snapshot_copy(
    const std::filesystem::path& telemetry_path,
    const std::unordered_map<std::string, TelemetryBudgetCacheEntry>& entries) {
    const auto cache_path = telemetry_budget_cache_path(telemetry_path);
    if (cache_path.empty()) {
        return;
    }
    const auto parent = cache_path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
    std::ofstream output(cache_path, std::ios::trunc);
    if (!output.is_open()) {
        return;
    }
    output << "# kind\tphase\tshape_bucket\tsamples\taverage_speedup_vs_reference\taverage_transfer_overlap_ratio\taverage_copy_share\tbudget_exhaustion_ratio\taverage_queue_separation_ratio\tlast_optimizer_budget_ms\tlast_epoch\n";
    for (const auto& [key, entry] : entries) {
        (void)key;
        output << entry.kind << '\t'
               << entry.phase << '\t'
               << entry.shape_bucket << '\t'
               << entry.signal.samples << '\t'
               << entry.signal.average_speedup_vs_reference << '\t'
               << entry.signal.average_transfer_overlap_ratio << '\t'
               << entry.signal.average_copy_share << '\t'
               << entry.signal.budget_exhaustion_ratio << '\t'
               << entry.signal.average_queue_separation_ratio << '\t'
               << entry.last_optimizer_budget_ms << '\t'
               << entry.last_epoch << '\n';
    }
    output.close();
    std::error_code ec;
    std::filesystem::remove(telemetry_budget_delta_path(telemetry_path), ec);
}

std::string runtime_shape_bucket_for(const OperationSpec& operation) {
    std::ostringstream stream;
    stream << to_string(operation.op_class);
    for (const auto extent : operation.extents) {
        std::uint64_t bucket = 1u;
        while (bucket < extent) {
            bucket <<= 1u;
        }
        stream << ':' << bucket;
    }
    const auto bytes_bucket = std::max<std::uint64_t>(1ull, operation.input_bytes / (4ull * 1024ull * 1024ull));
    stream << "|b" << bytes_bucket
           << "|cpuv:" << operation.cpu_vectorized
           << "|gput:" << operation.gpu_tensorized
           << "|cpu.in:" << operation.cpu_input_layout
           << "|cpu.w:" << operation.cpu_weight_layout
           << "|cpu.out:" << operation.cpu_output_layout
           << "|gpu.in:" << operation.gpu_input_layout
           << "|gpu.w:" << operation.gpu_weight_layout
           << "|gpu.out:" << operation.gpu_output_layout
           << "|cpu.pack:" << operation.cpu_pack_weights
           << "|gpu.pack:" << operation.gpu_pack_weights
           << "|cpu.preT:" << operation.cpu_pretranspose_rhs
           << "|gpu.preT:" << operation.gpu_pretranspose_rhs
           << "|cpu.u" << std::max(operation.cpu_micro_kernel_unroll, 1u)
           << "|cpu.tm" << std::max(operation.cpu_tile_m, 1u)
           << "|cpu.tn" << std::max(operation.cpu_tile_n, 1u)
           << "|cpu.tk" << std::max(operation.cpu_tile_k, 1u)
           << "|cpu.chunk" << std::max(operation.cpu_parallel_chunk, 1u)
           << "|gpu.u" << std::max(operation.gpu_micro_kernel_unroll, 1u);
    for (const auto& fused : operation.fused_operation_names) {
        stream << "|f:" << fused;
    }
    return stream.str();
}

TelemetryBudgetSignal load_graph_family_budget_signal(
    const std::filesystem::path& execution_cache_path,
    const WorkloadGraph* workload_graph) {
    TelemetryBudgetSignal signal;
    if (workload_graph == nullptr || workload_graph->operations.empty()) {
        return signal;
    }

    const auto family_path = runtime_graph_family_cache_path(execution_cache_path);
    if (family_path.empty() || !std::filesystem::exists(family_path)) {
        return signal;
    }

    std::unordered_set<std::string> target_buckets;
    for (const auto& operation : workload_graph->operations) {
        target_buckets.insert(runtime_shape_bucket_for(operation));
    }

    std::ifstream input(family_path);
    if (!input.is_open()) {
        return signal;
    }

    std::string header_line;
    if (!read_tsv_header_line(input, header_line)) {
        return signal;
    }
    const auto header_index = build_header_index(header_line);
    const auto bucket_index =
        header_index_or(header_index, "shape_bucket", std::numeric_limits<std::size_t>::max());
    const auto observations_index =
        header_index_or(header_index, "observations", std::numeric_limits<std::size_t>::max());
    const auto speedup_index =
        header_index_or(header_index, "avg_reward", std::numeric_limits<std::size_t>::max());
    const auto overlap_index =
        header_index_or(header_index, "avg_transfer_overlap_ratio", std::numeric_limits<std::size_t>::max());
    const auto copy_share_index =
        header_index_or(header_index, "avg_copy_share", std::numeric_limits<std::size_t>::max());
    const auto budget_index =
        header_index_or(header_index, "avg_budget_pressure", std::numeric_limits<std::size_t>::max());
    const auto queue_index =
        header_index_or(header_index, "avg_queue_separation_ratio", std::numeric_limits<std::size_t>::max());
    if (bucket_index == std::numeric_limits<std::size_t>::max() ||
        observations_index == std::numeric_limits<std::size_t>::max()) {
        return signal;
    }

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const auto fields = split_tab_fields(line);
        const auto bucket = field_or_empty(fields, bucket_index);
        if (target_buckets.find(bucket) == target_buckets.end()) {
            continue;
        }
        const auto observations = parse_uint32_or(field_or_empty(fields, observations_index), 0u);
        if (observations == 0u) {
            continue;
        }
        const double reward = parse_double_or(field_or_empty(fields, speedup_index), 0.0);
        const double implied_speedup = std::clamp(std::exp(reward), 0.25, 4.0);
        merge_budget_signal_sample(
            signal,
            implied_speedup,
            parse_double_or(field_or_empty(fields, overlap_index), 0.0),
            parse_double_or(field_or_empty(fields, copy_share_index), 0.0),
            parse_double_or(field_or_empty(fields, budget_index), 0.0),
            parse_double_or(field_or_empty(fields, queue_index), 0.0),
            static_cast<double>(observations));
    }

    return signal;
}

void seed_budget_cache_from_signal(
    const std::filesystem::path& telemetry_path,
    const WorkloadSpec& workload,
    const TelemetryBudgetSignal& signal) {
    if (telemetry_path.empty() || signal.samples == 0u) {
        return;
    }
    std::unique_lock lock(g_telemetry_budget_cache_mutex);
    auto& state = ensure_budget_cache_state_loaded(telemetry_path);
    const auto key = telemetry_budget_entry_key(
        to_string(workload.kind),
        to_string(canonical_workload_phase(workload)),
        canonical_workload_shape_bucket(workload));
    auto& entry = state.entries[key];
    if (entry.signal.samples >= signal.samples) {
        return;
    }
    entry.kind = to_string(workload.kind);
    entry.phase = to_string(canonical_workload_phase(workload));
    entry.shape_bucket = canonical_workload_shape_bucket(workload);
    entry.signal = signal;
}

UnifiedRuntimeSignal load_unified_runtime_signal(
    const std::filesystem::path& telemetry_path,
    const std::filesystem::path& execution_cache_path,
    const WorkloadSpec& workload,
    const WorkloadGraph* workload_graph) {
    UnifiedRuntimeSignal unified;
    unified.signal = load_runtime_budget_signal(telemetry_path, workload);
    const auto family_signal = load_graph_family_budget_signal(execution_cache_path, workload_graph);
    if (unified.signal.samples == 0u && family_signal.samples > 0u) {
        unified.signal = family_signal;
        unified.source = "graph-family";
        seed_budget_cache_from_signal(telemetry_path, workload, family_signal);
    } else if (unified.signal.samples > 0u && family_signal.samples > 0u) {
        merge_budget_signal_sample(
            unified.signal,
            family_signal.average_speedup_vs_reference,
            family_signal.average_transfer_overlap_ratio,
            family_signal.average_copy_share,
            family_signal.budget_exhaustion_ratio,
            family_signal.average_queue_separation_ratio,
            static_cast<double>(family_signal.samples));
        unified.source = "telemetry+family";
        seed_budget_cache_from_signal(telemetry_path, workload, unified.signal);
    }
    return unified;
}

std::size_t header_index_or(
    const std::unordered_map<std::string, std::size_t>& header_index,
    const std::string& key,
    const std::size_t fallback) {
    if (const auto it = header_index.find(key); it != header_index.end()) {
        return it->second;
    }
    return fallback;
}

TelemetryBudgetSignal load_runtime_budget_signal(
    const std::filesystem::path& telemetry_path,
    const WorkloadSpec& workload) {
    TelemetryBudgetSignal signal;
    const auto expected_kind = to_string(workload.kind);
    const auto expected_phase = to_string(canonical_workload_phase(workload));
    const auto expected_bucket = canonical_workload_shape_bucket(workload);

    if (!telemetry_path.empty()) {
        const auto cache_key = telemetry_path.string();
        {
            std::shared_lock lock(g_telemetry_budget_cache_mutex);
            const auto state_it = g_telemetry_budget_cache.find(cache_key);
            if (state_it != g_telemetry_budget_cache.end() && state_it->second.loaded) {
                const auto entry_it = state_it->second.entries.find(
                    telemetry_budget_entry_key(expected_kind, expected_phase, expected_bucket));
                if (entry_it != state_it->second.entries.end()) {
                    return entry_it->second.signal;
                }
            }
        }
        std::unique_lock lock(g_telemetry_budget_cache_mutex);
        auto& state = ensure_budget_cache_state_loaded(telemetry_path);
        const auto it = state.entries.find(telemetry_budget_entry_key(expected_kind, expected_phase, expected_bucket));
        if (it != state.entries.end()) {
            return it->second.signal;
        }
    }

    if (telemetry_path.empty() || !std::filesystem::exists(telemetry_path)) {
        return signal;
    }

    std::ifstream input(telemetry_path);
    if (!input.is_open()) {
        return signal;
    }

    std::string header_line;
    if (!read_tsv_header_line(input, header_line)) {
        return signal;
    }
    if (!header_line.empty() && header_line.front() == '#') {
        header_line.erase(header_line.begin());
        if (!header_line.empty() && header_line.front() == ' ') {
            header_line.erase(header_line.begin());
        }
    }
    const auto header_fields = split_tab_fields(header_line);
    std::unordered_map<std::string, std::size_t> header_index;
    for (std::size_t index = 0; index < header_fields.size(); ++index) {
        header_index.emplace(header_fields[index], index);
    }

    const auto kind_index = header_index_or(header_index, "kind");
    const auto phase_index = header_index_or(header_index, "phase");
    const auto bucket_index = header_index_or(header_index, "shape_bucket");
    const auto executed_index = header_index_or(header_index, "executed");
    const auto speedup_index = header_index_or(header_index, "speedup_vs_reference");
    const auto overlap_index = header_index_or(header_index, "transfer_overlap_ratio");
    const auto copy_index = header_index_or(header_index, "copy_runtime_us");
    const auto compute_index = header_index_or(header_index, "compute_runtime_us");
    const auto budget_index = header_index_or(header_index, "optimizer_budget_ms");
    const auto exhausted_index = header_index_or(header_index, "budget_exhausted");
    if (kind_index == std::numeric_limits<std::size_t>::max() ||
        phase_index == std::numeric_limits<std::size_t>::max() ||
        bucket_index == std::numeric_limits<std::size_t>::max() ||
        executed_index == std::numeric_limits<std::size_t>::max()) {
        return signal;
    }

    std::string line;
    while (std::getline(input, line)) {
        const auto fields = split_tab_fields(line);
        const auto field_at = [&](const std::size_t index) -> std::string {
            return index < fields.size() ? fields[index] : std::string();
        };
        if (field_at(kind_index) != expected_kind ||
            field_at(phase_index) != expected_phase ||
            field_at(bucket_index) != expected_bucket ||
            parse_uint32_or(field_at(executed_index), 0u) == 0u) {
            continue;
        }

        const double copy_runtime_us = parse_double_or(field_at(copy_index), 0.0);
        const double compute_runtime_us = parse_double_or(field_at(compute_index), 0.0);
        const double total_copy_compute = std::max(copy_runtime_us + compute_runtime_us, 1.0);
        merge_budget_signal_sample(
            signal,
            parse_double_or(field_at(speedup_index), 0.0),
            parse_double_or(field_at(overlap_index), 0.0),
            copy_runtime_us / total_copy_compute,
            parse_uint32_or(field_at(exhausted_index), 0u) > 0u ? 1.0 : 0.0,
            parse_double_or(field_at(header_index_or(header_index, "copy_overlap_ratio")), 0.0));
        (void)budget_index;
    }

    return signal;
}

void update_runtime_budget_cache(
    const std::filesystem::path& telemetry_path,
    const std::uint64_t epoch,
    const WorkloadSpec& workload,
    const ManagedExecutionReport& report) {
    if (telemetry_path.empty() || !report.executed) {
        return;
    }

    const auto delta_path = telemetry_budget_delta_path(telemetry_path);
    if (delta_path.empty()) {
        return;
    }

    const std::string expected_kind = to_string(workload.kind);
    const std::string expected_phase = to_string(canonical_workload_phase(workload));
    const std::string expected_bucket = canonical_workload_shape_bucket(workload);
    const double copy_runtime_us = report.execution.total_copy_runtime_us;
    const double compute_runtime_us = report.execution.total_compute_runtime_us;
    const double total_copy_compute = std::max(copy_runtime_us + compute_runtime_us, 1.0);
    const double sample_copy_share = copy_runtime_us / total_copy_compute;
    const double sample_queue_separation = report.execution.queue_separation_ratio;
    const double sample_overlap_ratio = report.execution.transfer_overlap_ratio;
    const double sample_speedup = report.execution.speedup_vs_reference;
    const double sample_budget_exhaustion =
        report.execution.optimization.graph_optimization.budget_exhausted ? 1.0 : 0.0;

    std::uint32_t last_optimizer_budget_ms = 0u;
    bool persist_snapshot = false;
    std::unordered_map<std::string, TelemetryBudgetCacheEntry> snapshot_entries;
    {
        std::unique_lock lock(g_telemetry_budget_cache_mutex);
        auto& state = ensure_budget_cache_state_loaded(telemetry_path);
        const auto key = telemetry_budget_entry_key(expected_kind, expected_phase, expected_bucket);
        auto& entry = state.entries[key];
        entry.kind = expected_kind;
        entry.phase = expected_phase;
        entry.shape_bucket = expected_bucket;
        merge_budget_signal_sample(
            entry.signal,
            sample_speedup,
            sample_overlap_ratio,
            sample_copy_share,
            sample_budget_exhaustion,
            sample_queue_separation);
        entry.last_optimizer_budget_ms = report.execution.optimization.graph_optimization.time_budget_ms;
        entry.last_epoch = epoch;
        last_optimizer_budget_ms = entry.last_optimizer_budget_ms;
        state.delta_rows += 1u;

        const auto now = std::chrono::steady_clock::now();
        const bool compact_by_rows = state.delta_rows >= 8u;
        const bool compact_by_age =
            state.delta_rows >= 2u &&
            (now - state.last_compaction) >= std::chrono::seconds(30);
        const bool compact_by_cardinality =
            state.entries.size() >= 32u && state.delta_rows >= 4u;
        persist_snapshot =
            compact_by_rows || compact_by_age || compact_by_cardinality ||
            !std::filesystem::exists(telemetry_budget_cache_path(telemetry_path));
        if (persist_snapshot) {
            snapshot_entries = state.entries;
            state.delta_rows = 0u;
            state.last_compaction = now;
        }
    }

    const std::string delta_header =
        "# schema_version\t" + std::to_string(kRuntimeTelemetryBudgetSchemaVersion) + "\n"
        "# epoch\tkind\tphase\tshape_bucket\tspeedup_vs_reference\ttransfer_overlap_ratio\tcopy_share\tbudget_pressure\tqueue_separation_ratio\toptimizer_budget_ms\n";
    std::ostringstream delta_line;
    delta_line << epoch << '\t'
               << expected_kind << '\t'
               << expected_phase << '\t'
               << expected_bucket << '\t'
               << sample_speedup << '\t'
               << sample_overlap_ratio << '\t'
               << sample_copy_share << '\t'
               << sample_budget_exhaustion << '\t'
               << sample_queue_separation << '\t'
               << last_optimizer_budget_ms << '\n';

    telemetry_writer().enqueue([telemetry_path, delta_path, delta_header, delta_payload = delta_line.str(), persist_snapshot, snapshot_entries = std::move(snapshot_entries)]() mutable {
        if (persist_snapshot) {
            persist_budget_cache_snapshot_copy(telemetry_path, snapshot_entries);
            return;
        }
        append_tsv_line(delta_path, delta_header, delta_payload);
    });
}

std::uint32_t clamp_runtime_optimizer_budget(
    const WorkloadSpec& workload,
    const std::uint32_t budget_ms) {
    const auto phase = canonical_workload_phase(workload);
    if (workload.kind == WorkloadKind::training) {
        return std::clamp<std::uint32_t>(budget_ms, 100u, 400u);
    }
    if (workload.kind == WorkloadKind::custom) {
        return std::clamp<std::uint32_t>(budget_ms, 80u, 320u);
    }
    if (workload.kind == WorkloadKind::inference &&
        workload.latency_sensitive &&
        (phase == WorkloadPhase::decode || phase == WorkloadPhase::prefill)) {
        return std::clamp<std::uint32_t>(budget_ms, 15u, 80u);
    }
    if (workload.kind == WorkloadKind::gaming || workload.kind == WorkloadKind::image) {
        return std::clamp<std::uint32_t>(budget_ms, 20u, 90u);
    }
    if (workload.latency_sensitive) {
        return std::clamp<std::uint32_t>(budget_ms, 25u, 120u);
    }
    return std::clamp<std::uint32_t>(budget_ms, 40u, 160u);
}

bool has_manual_execution_tuning(const ExecutionTuningOverrides& tuning) {
    return tuning.initial_state_override.has_value() ||
           tuning.graph_optimization_passes_override.has_value() ||
           tuning.optimizer_wall_time_budget_ms.has_value() ||
           tuning.graph_rewrite_level > 1u ||
           tuning.validation_tier != ValidationTier::adaptive;
}

ValidationTier choose_runtime_validation_tier(
    const WorkloadSpec& workload,
    const WorkloadGraph* workload_graph,
    const ValidationTier configured_tier) {
    if (configured_tier != ValidationTier::adaptive) {
        return configured_tier;
    }

    const auto phase = canonical_workload_phase(workload);
    const std::size_t operation_count = workload_graph == nullptr ? 0u : workload_graph->operations.size();
    if (workload.kind == WorkloadKind::training || workload.kind == WorkloadKind::custom) {
        return ValidationTier::full;
    }
    if (workload.kind == WorkloadKind::inference &&
        workload.latency_sensitive &&
        (phase == WorkloadPhase::decode || phase == WorkloadPhase::prefill || workload.batch_size <= 4u)) {
        return ValidationTier::minimal;
    }
    if (workload.latency_sensitive && operation_count > 0u && operation_count <= 3u) {
        return ValidationTier::minimal;
    }
    if (workload.kind == WorkloadKind::gaming || workload.kind == WorkloadKind::image) {
        return ValidationTier::minimal;
    }
    return ValidationTier::adaptive;
}

ResolvedOptimizerBudget choose_runtime_optimizer_budget_ms(
    const WorkloadSpec& workload,
    const WorkloadGraph* workload_graph,
    const std::optional<std::uint32_t>& configured_budget_ms,
    const std::filesystem::path& telemetry_path,
    const std::filesystem::path& execution_cache_path) {
    ResolvedOptimizerBudget resolved;
    if (configured_budget_ms.has_value()) {
        resolved.budget_ms = configured_budget_ms;
        resolved.source = "configured";
        return resolved;
    }

    const auto phase = canonical_workload_phase(workload);
    const std::size_t operation_count = workload_graph == nullptr ? 0u : workload_graph->operations.size();
    std::uint32_t budget_ms = 90u;
    if (workload.kind == WorkloadKind::training) {
        budget_ms = 250u;
    } else if (workload.kind == WorkloadKind::custom) {
        budget_ms = 180u;
    } else if (
        workload.kind == WorkloadKind::inference &&
        workload.latency_sensitive &&
        (phase == WorkloadPhase::decode || phase == WorkloadPhase::prefill)) {
        budget_ms = operation_count > 6u ? 35u : 25u;
    } else if (workload.kind == WorkloadKind::gaming || workload.kind == WorkloadKind::image) {
        budget_ms = 40u;
    } else if (workload.latency_sensitive) {
        budget_ms = 55u;
    }

    const auto unified_signal =
        load_unified_runtime_signal(telemetry_path, execution_cache_path, workload, workload_graph);
    const auto& signal = unified_signal.signal;
    const auto& signal_source = unified_signal.source;
    if (signal.samples >= 2u) {
        int delta_ms = 0;
        if (signal.budget_exhaustion_ratio >= 0.30) {
            delta_ms += workload.latency_sensitive ? 12 : 24;
            resolved.source = signal_source + "-expand";
        }
        if (signal.average_copy_share >= 0.30 && signal.average_transfer_overlap_ratio <= 0.25) {
            delta_ms += workload.latency_sensitive ? 8 : 16;
            resolved.source = resolved.source == "heuristic" ? signal_source + "-overlap" : resolved.source;
        }
        if (signal.average_queue_separation_ratio >= 0.45 && signal.average_transfer_overlap_ratio >= 0.45) {
            delta_ms -= workload.latency_sensitive ? 3 : 6;
            if (resolved.source == "heuristic") {
                resolved.source = signal_source + "-queue";
            }
        }
        if (signal.budget_exhaustion_ratio <= 0.05 &&
            signal.average_speedup_vs_reference >= 1.20 &&
            signal.average_copy_share <= 0.18 &&
            signal.average_transfer_overlap_ratio >= 0.50) {
            delta_ms -= workload.latency_sensitive ? 5 : 10;
            if (resolved.source == "heuristic") {
                resolved.source = signal_source + "-shrink";
            }
        }
        budget_ms = clamp_runtime_optimizer_budget(
            workload,
            static_cast<std::uint32_t>(std::max(1, static_cast<int>(budget_ms) + delta_ms)));
    }

    resolved.budget_ms = budget_ms;
    return resolved;
}

ContinuousExecutionState tuning_profile_state(const std::string& profile_name) {
    ContinuousExecutionState state;
    if (profile_name == "host-latency") {
        state.queue_depth_raw = -0.8;
        state.stage_raw = -0.6;
        state.tile_raw = -0.1;
        state.overlap_raw = -0.7;
        state.partition_raw = -1.2;
        state.precision_raw = -0.4;
        state.single_device_logit = 1.6;
        state.sharded_logit = -1.5;
        state.streaming_logit = -0.2;
        state.overlapped_logit = 0.0;
        return state;
    }
    if (profile_name == "hybrid-balanced") {
        state.queue_depth_raw = 0.0;
        state.stage_raw = 0.1;
        state.tile_raw = 0.3;
        state.overlap_raw = -0.1;
        state.partition_raw = -0.5;
        state.precision_raw = 0.1;
        state.single_device_logit = 0.9;
        state.sharded_logit = -0.2;
        state.streaming_logit = 0.2;
        state.overlapped_logit = 0.3;
        return state;
    }
    if (profile_name == "accelerator-throughput") {
        state.queue_depth_raw = 0.7;
        state.stage_raw = 0.6;
        state.tile_raw = 0.9;
        state.overlap_raw = 0.5;
        state.partition_raw = 0.5;
        state.precision_raw = 0.5;
        state.single_device_logit = 0.1;
        state.sharded_logit = 0.8;
        state.streaming_logit = 0.1;
        state.overlapped_logit = 0.8;
        return state;
    }

    state.queue_depth_raw = -0.1;
    state.stage_raw = -0.2;
    state.tile_raw = 0.5;
    state.overlap_raw = -0.2;
    state.partition_raw = 0.2;
    state.precision_raw = 0.2;
    state.single_device_logit = 0.4;
    state.sharded_logit = 0.7;
    state.streaming_logit = 0.0;
    state.overlapped_logit = 0.2;
    return state;
}

GraphWorkloadMetrics analyze_workload_graph(const WorkloadSpec& workload, const WorkloadGraph& workload_graph) {
    GraphWorkloadMetrics metrics;
    metrics.operation_count = workload_graph.operations.size();
    metrics.host_visible_tensor_count = std::count_if(
        workload_graph.tensors.begin(),
        workload_graph.tensors.end(),
        [](const WorkloadTensor& tensor) {
            return tensor.host_visible;
        });
    for (const auto& tensor : workload_graph.tensors) {
        if (tensor.host_visible) {
            metrics.host_visible_bytes += static_cast<double>(tensor.bytes);
        }
    }

    for (std::size_t index = 0; index < workload_graph.operations.size(); ++index) {
        const auto& operation = workload_graph.operations[index];
        const double bytes = static_cast<double>(
            std::max<std::uint64_t>(
                operation.input_bytes + operation.output_bytes + operation.temporary_bytes,
                1ull));
        const double flops = std::max(operation.estimated_flops, 0.0);
        metrics.total_bytes += bytes;
        metrics.total_flops += flops;

        switch (operation.op_class) {
        case OperationClass::matmul:
            ++metrics.matmul_count;
            metrics.matmul_flops += flops;
            break;
        case OperationClass::reduction:
            ++metrics.reduction_count;
            metrics.reduction_flops += flops;
            metrics.reduction_bytes += bytes;
            if (index + 2u >= workload_graph.operations.size()) {
                metrics.has_terminal_reduction = true;
            }
            break;
        case OperationClass::elementwise_map:
            ++metrics.elementwise_count;
            metrics.elementwise_flops += flops;
            break;
        default:
            break;
        }
    }

    const double op_count = static_cast<double>(std::max<std::size_t>(metrics.operation_count, 1u));
    const double total_flops = std::max(metrics.total_flops, 1.0);
    const double total_bytes = std::max(metrics.total_bytes, 1.0);
    metrics.small_op_ratio =
        static_cast<double>(metrics.elementwise_count + metrics.reduction_count) / op_count;
    metrics.matmul_flops_ratio = metrics.matmul_flops / total_flops;
    metrics.reduction_flops_ratio = metrics.reduction_flops / total_flops;
    metrics.host_visible_tensor_ratio =
        static_cast<double>(metrics.host_visible_tensor_count) /
        static_cast<double>(std::max<std::size_t>(workload_graph.tensors.size(), 1u));
    metrics.average_flops_per_operation = metrics.total_flops / op_count;

    double dispatch_bound_score = 0.0;
    dispatch_bound_score += std::clamp(metrics.small_op_ratio * 0.42, 0.0, 0.32);
    dispatch_bound_score += std::clamp(workload_host_exchange_ratio(workload) * 0.18, 0.0, 0.18);
    dispatch_bound_score += std::clamp(metrics.host_visible_bytes / total_bytes, 0.0, 0.14);
    if (workload.latency_sensitive) {
        dispatch_bound_score += workload.batch_size <= 4u ? 0.10 : 0.06;
    }
    if (workload.prefer_unified_memory) {
        dispatch_bound_score += 0.10;
    }
    if (metrics.average_flops_per_operation < 3.0e6) {
        dispatch_bound_score += 0.12;
    } else if (metrics.average_flops_per_operation < 1.0e7) {
        dispatch_bound_score += 0.06;
    }
    if (metrics.matmul_flops_ratio > 0.75 && canonical_workload_phase(workload) == WorkloadPhase::decode) {
        dispatch_bound_score -= 0.10;
    }
    metrics.dispatch_bound_score = std::clamp(dispatch_bound_score, 0.0, 1.0);
    return metrics;
}

GraphAwareMetaPolicy derive_graph_aware_meta_policy(
    const WorkloadSpec& workload,
    const WorkloadGraph& workload_graph) {
    GraphAwareMetaPolicy policy;
    if (workload_graph.operations.empty()) {
        return policy;
    }

    const auto metrics = analyze_workload_graph(workload, workload_graph);
    const auto phase = canonical_workload_phase(workload);
    const bool decode_projection_dominant =
        workload.matrix_friendly &&
        (phase == WorkloadPhase::decode || phase == WorkloadPhase::prefill) &&
        metrics.matmul_flops_ratio >= 0.40;
    const bool cooperative_auto =
        workload.latency_sensitive &&
        phase != WorkloadPhase::decode &&
        phase != WorkloadPhase::prefill &&
        !decode_projection_dominant &&
        ((workload.prefer_unified_memory && metrics.small_op_ratio >= 0.50) ||
         (workload.kind == WorkloadKind::tensor &&
          metrics.small_op_ratio >= 0.55 &&
          workload_host_exchange_ratio(workload) >= 0.30));
    const bool projection_sharded =
        workload.matrix_friendly &&
        !cooperative_auto &&
        (decode_projection_dominant ||
         (metrics.matmul_flops_ratio >= 0.55 &&
          (phase == WorkloadPhase::decode || phase == WorkloadPhase::prefill || workload.latency_sensitive)));
    const bool reduce_on_gpu =
        workload.matrix_friendly &&
        metrics.has_terminal_reduction &&
        !cooperative_auto &&
        !workload.latency_sensitive &&
        metrics.matmul_flops_ratio >= 0.35;

    std::ostringstream summary;
    if (projection_sharded) {
        policy.strategy_hint = PartitionStrategy::projection_sharded;
        policy.strategy_confidence =
            phase == WorkloadPhase::decode ? 0.72 : (metrics.dispatch_bound_score < 0.35 ? 0.64 : 0.58);
        policy.disable_exploration = true;
        summary << "graph-aware heuristic projection_sharded matmul=" << metrics.matmul_flops_ratio
                << " dispatch=" << metrics.dispatch_bound_score
                << " tail_reduce=" << (metrics.has_terminal_reduction ? "yes" : "no");
    } else if (reduce_on_gpu) {
        policy.strategy_hint = PartitionStrategy::reduce_on_gpu;
        policy.strategy_confidence = metrics.reduction_count >= 2u ? 0.70 : 0.62;
        policy.disable_exploration = true;
        summary << "graph-aware heuristic reduce_on_gpu matmul=" << metrics.matmul_flops_ratio
                << " reductions=" << metrics.reduction_count
                << " dispatch=" << metrics.dispatch_bound_score;
    } else if (cooperative_auto) {
        policy.disable_exploration = true;
        summary << "graph-aware heuristic kept auto_balanced cooperative dispatch=" << metrics.dispatch_bound_score
                << " small_ops=" << metrics.small_op_ratio;
    } else {
        summary << "graph-aware heuristic kept auto_balanced balanced dispatch=" << metrics.dispatch_bound_score
                << " matmul=" << metrics.matmul_flops_ratio;
    }
    policy.strategy_reason = summary.str();

    const auto tuning_target = workload.partition_strategy != PartitionStrategy::auto_balanced
                                   ? workload.partition_strategy
                                   : policy.strategy_hint.value_or(PartitionStrategy::auto_balanced);
    if (cooperative_auto) {
        policy.tuning_state = tuning_profile_state("cooperative-split");
        policy.tuning_graph_rewrite_level = 2u;
        policy.tuning_graph_passes = 2u;
        policy.tuning_reason = "cooperative-split";
        return policy;
    }

    if (tuning_target == PartitionStrategy::projection_sharded ||
        tuning_target == PartitionStrategy::reduce_on_gpu) {
        auto state = tuning_profile_state("accelerator-throughput");
        if (tuning_target == PartitionStrategy::projection_sharded &&
            metrics.operation_count <= 6u &&
            workload.estimated_flops > 0.0 &&
            workload.estimated_flops < 3.0e7) {
            state.sharded_logit = 0.35;
        }
        if (tuning_target == PartitionStrategy::reduce_on_gpu &&
            metrics.has_terminal_reduction &&
            metrics.reduction_count >= 2u) {
            state.single_device_logit = -0.35;
        }
        policy.tuning_state = state;
        policy.tuning_graph_rewrite_level = 2u;
        policy.tuning_graph_passes = 4u;
        policy.tuning_reason = "accelerator-throughput";
        return policy;
    }

    if (workload.matrix_friendly && workload_graph.operations.size() >= 4u) {
        policy.tuning_state = tuning_profile_state("hybrid-balanced");
        policy.tuning_graph_rewrite_level = 2u;
        policy.tuning_graph_passes = 2u;
        policy.tuning_reason = "hybrid-balanced";
    }
    return policy;
}

RuntimeOptimizationContext resolve_runtime_optimization_context(
    const RuntimeOptions& options,
    const WorkloadSpec& workload,
    const WorkloadGraph* workload_graph) {
    RuntimeOptimizationContext context;
    context.requested_workload = workload;
    if (options.optimization.forced_partition_strategy.has_value()) {
        context.requested_workload.partition_strategy = *options.optimization.forced_partition_strategy;
    }
    context.effective_workload = context.requested_workload;
    context.execution_tuning = options.optimization.execution;
    context.execution_tuning.validation_tier = choose_runtime_validation_tier(
        context.requested_workload,
        workload_graph,
        options.optimization.execution.validation_tier);
    const auto telemetry_path =
        options.product.observability.telemetry_path.empty()
            ? default_runtime_telemetry_path()
            : options.product.observability.telemetry_path;
    const auto resolved_budget = choose_runtime_optimizer_budget_ms(
        context.requested_workload,
        workload_graph,
        options.optimization.execution.optimizer_wall_time_budget_ms,
        telemetry_path,
        options.execution_cache_path.empty()
            ? ExecutionOptimizer::default_cache_path()
            : options.execution_cache_path);
    context.execution_tuning.optimizer_wall_time_budget_ms = resolved_budget.budget_ms;
    context.optimizer_budget_source = resolved_budget.source;

    if (workload_graph == nullptr) {
        return context;
    }

    const auto meta = derive_graph_aware_meta_policy(context.requested_workload, *workload_graph);
    std::string seeded_tuning_reason;
    const auto unified_signal = load_unified_runtime_signal(
        telemetry_path,
        options.execution_cache_path.empty()
            ? ExecutionOptimizer::default_cache_path()
            : options.execution_cache_path,
        context.requested_workload,
        workload_graph);
    if (options.optimization.enable_graph_aware_strategy_hints &&
        !context.requested_workload.disable_heuristic_partition_hint &&
        !options.optimization.forced_partition_strategy.has_value() &&
        context.requested_workload.partition_strategy == PartitionStrategy::auto_balanced &&
        meta.strategy_hint.has_value()) {
        context.effective_workload.heuristic_partition_hint = meta.strategy_hint;
        context.effective_workload.heuristic_partition_hint_confidence = meta.strategy_confidence;
        context.effective_workload.heuristic_partition_hint_reason = meta.strategy_reason;
    }
    if (meta.disable_exploration) {
        context.effective_workload.disable_strategy_exploration = true;
    }

    if (options.optimization.enable_graph_aware_execution_tuning &&
        !context.requested_workload.disable_automatic_execution_tuning &&
        !has_manual_execution_tuning(options.optimization.execution)) {
        if (meta.tuning_state.has_value()) {
            context.execution_tuning.initial_state_override = meta.tuning_state;
        }
        if (meta.tuning_graph_passes.has_value()) {
            context.execution_tuning.graph_optimization_passes_override = meta.tuning_graph_passes;
        }
        context.execution_tuning.graph_rewrite_level =
            std::max(context.execution_tuning.graph_rewrite_level, meta.tuning_graph_rewrite_level);

        if (unified_signal.signal.samples >= 3u) {
            if (unified_signal.signal.average_copy_share >= 0.24 &&
                unified_signal.signal.average_transfer_overlap_ratio <= 0.30) {
                context.execution_tuning.graph_rewrite_level =
                    std::max<std::uint32_t>(context.execution_tuning.graph_rewrite_level, 5u);
                if (!context.execution_tuning.graph_optimization_passes_override.has_value()) {
                    context.execution_tuning.graph_optimization_passes_override = 3u;
                }
                seeded_tuning_reason = unified_signal.source + "-fusion-seed";
            }
            if (unified_signal.signal.average_speedup_vs_reference >= 1.08 &&
                unified_signal.signal.budget_exhaustion_ratio <= 0.15) {
                context.effective_workload.disable_strategy_exploration = true;
                if (!context.execution_tuning.initial_state_override.has_value()) {
                    context.execution_tuning.initial_state_override =
                        tuning_profile_state(
                            unified_signal.signal.average_queue_separation_ratio >= 0.35
                                ? "accelerator-throughput"
                                : "hybrid-balanced");
                }
                if (!context.execution_tuning.graph_optimization_passes_override.has_value()) {
                    context.execution_tuning.graph_optimization_passes_override =
                        workload_graph->operations.size() <= 3u ? 1u : 2u;
                } else {
                    context.execution_tuning.graph_optimization_passes_override =
                        std::min<std::uint32_t>(
                            *context.execution_tuning.graph_optimization_passes_override,
                            workload_graph->operations.size() <= 3u ? 1u : 2u);
                }
                if (seeded_tuning_reason.empty()) {
                    seeded_tuning_reason = unified_signal.source + "-steady-seed";
                } else {
                    seeded_tuning_reason += "+";
                    seeded_tuning_reason += unified_signal.source;
                    seeded_tuning_reason += "-steady-seed";
                }
            }
        }
    }

    if (!meta.strategy_reason.empty() || !meta.tuning_reason.empty() || !seeded_tuning_reason.empty()) {
        std::ostringstream summary;
        if (!meta.strategy_reason.empty()) {
            summary << meta.strategy_reason;
        }
        if (!meta.tuning_reason.empty()) {
            if (summary.tellp() > 0) {
                summary << "; ";
            }
            summary << "execution tuning " << meta.tuning_reason
                    << " rewrite=" << context.execution_tuning.graph_rewrite_level
                    << " passes=";
            if (context.execution_tuning.graph_optimization_passes_override.has_value()) {
                summary << *context.execution_tuning.graph_optimization_passes_override;
            } else {
                summary << "auto";
            }
            summary << " budget=";
            if (context.execution_tuning.optimizer_wall_time_budget_ms.has_value()) {
                summary << *context.execution_tuning.optimizer_wall_time_budget_ms << "ms"
                        << '[' << context.optimizer_budget_source << ']';
            } else {
                summary << "auto";
            }
        }
        if (!seeded_tuning_reason.empty()) {
            if (summary.tellp() > 0) {
                summary << "; ";
            }
            summary << "cache tuning " << seeded_tuning_reason
                    << " rewrite=" << context.execution_tuning.graph_rewrite_level
                    << " passes=";
            if (context.execution_tuning.graph_optimization_passes_override.has_value()) {
                summary << *context.execution_tuning.graph_optimization_passes_override;
            } else {
                summary << "auto";
            }
        }
        context.meta_summary = summary.str();
    } else if (context.execution_tuning.optimizer_wall_time_budget_ms.has_value() &&
               context.optimizer_budget_source != "heuristic") {
        std::ostringstream summary;
        summary << "execution tuning budget="
                << *context.execution_tuning.optimizer_wall_time_budget_ms
                << "ms[" << context.optimizer_budget_source << ']';
        context.meta_summary = summary.str();
    }

    return context;
}

std::vector<ExecutionFeedbackRecord> make_feedback_records(const DirectExecutionReport& report) {
    std::vector<ExecutionFeedbackRecord> feedback;
    feedback.reserve(report.operations.size());
    for (const auto& operation : report.operations) {
        const auto total_copy_compute = std::max(operation.copy_runtime_us + operation.compute_runtime_us, 1.0);
        const auto optimized_it = std::find_if(
            report.optimization.operations.begin(),
            report.optimization.operations.end(),
            [&](const OperationOptimizationResult& optimized) {
                return optimized.operation.name == operation.operation_name;
            });
        double staging_hit_rate = 0.0;
        double cross_device_sync_cost_us = operation.synchronize_runtime_us;
        double residency_pressure = 0.0;
        double kv_host_residency_ratio = 0.0;
        if (optimized_it != report.optimization.operations.end()) {
            const auto& graph = optimized_it->graph;
            const auto staging_node_count = std::count_if(
                graph.nodes.begin(),
                graph.nodes.end(),
                [](const ExecutionNode& node) {
                    return node.label.find("staging") != std::string::npos;
                });
            const auto cross_device_transfer_count = std::count_if(
                graph.transfer_schedule.begin(),
                graph.transfer_schedule.end(),
                [](const TransferScheduleEntry& entry) { return entry.cross_device; });
            staging_hit_rate = std::clamp(
                operation.transfer_overlap_ratio +
                    (staging_node_count > 0 ? 0.20 : 0.0) +
                    (cross_device_transfer_count > 0 ? 0.10 : 0.0),
                0.0,
                1.0);
            cross_device_sync_cost_us +=
                graph.predicted_transfer_latency_us * (1.0 - std::clamp(operation.transfer_overlap_ratio, 0.0, 0.95));
            residency_pressure = graph.predicted_memory_pressure;

            std::uint32_t kv_entries = 0u;
            std::uint32_t kv_host_entries = 0u;
            for (const auto& entry : graph.residency_plan) {
                const bool kv_entry =
                    entry.tensor_id.find("kv") != std::string::npos ||
                    entry.tensor_id.find("cache") != std::string::npos;
                if (!kv_entry) {
                    continue;
                }
                ++kv_entries;
                if (entry.device_uid == "host" || entry.device_uid.rfind("host:", 0) == 0) {
                    ++kv_host_entries;
                }
            }
            kv_host_residency_ratio =
                kv_entries == 0u ? 0.0 : static_cast<double>(kv_host_entries) / static_cast<double>(kv_entries);
        }
        feedback.push_back(ExecutionFeedbackRecord{
            operation.operation_name,
            operation.backend_name,
            operation.participating_devices,
            operation.runtime_us,
            operation.reference_runtime_us,
            operation.relative_error,
            operation.verified,
            operation.used_host,
            operation.used_opencl,
            operation.used_multiple_devices,
            operation.logical_partitions_used,
            operation.copy_runtime_us / total_copy_compute,
            operation.transfer_overlap_ratio,
            report.optimization.graph_optimization.budget_exhausted ? 1.0 : 0.0,
            operation.queue_separation_ratio,
            staging_hit_rate,
            cross_device_sync_cost_us,
            residency_pressure,
            kv_host_residency_ratio,
            operation.dispatch_count,
            operation.event_wait_count});
    }
    return feedback;
}

double total_runtime_us(const DirectExecutionReport& report) {
    return std::accumulate(
        report.operations.begin(),
        report.operations.end(),
        0.0,
        [](const double total, const OperationExecutionRecord& operation) {
            return total + operation.runtime_us;
        });
}

bool should_retry_execution(const DirectExecutionReport& report) {
    if (!report.all_succeeded) {
        return true;
    }

    return std::any_of(report.operations.begin(), report.operations.end(), [](const OperationExecutionRecord& operation) {
        return !operation.used_host &&
               operation.reference_runtime_us > 0.0 &&
               operation.runtime_us > (operation.reference_runtime_us * 1.10);
    });
}

bool selection_changed(const OptimizationReport& left, const OptimizationReport& right) {
    if (left.operations.size() != right.operations.size()) {
        return true;
    }

    std::unordered_map<std::string, std::string> left_by_operation;
    left_by_operation.reserve(left.operations.size());
    for (const auto& operation : left.operations) {
        left_by_operation.emplace(operation.operation.name, operation.config.signature);
    }

    for (const auto& operation : right.operations) {
        const auto it = left_by_operation.find(operation.operation.name);
        if (it == left_by_operation.end() || it->second != operation.config.signature) {
            return true;
        }
    }

    return false;
}

double head_runtime_us(const WorkloadSpec& workload, const DirectExecutionReport& report) {
    if (report.operations.empty()) {
        return std::max(report.total_runtime_us, 0.0);
    }
    if (!workload.latency_sensitive) {
        return std::max(report.total_runtime_us, 0.0);
    }

    const std::size_t lead_operations =
        std::min<std::size_t>(3u, std::max<std::size_t>(1u, (report.operations.size() + 2u) / 3u));
    return std::accumulate(
        report.operations.begin(),
        report.operations.begin() + static_cast<std::ptrdiff_t>(lead_operations),
        0.0,
        [](const double total, const OperationExecutionRecord& operation) {
            return total + std::max(operation.runtime_us, 0.0);
        });
}

struct LayoutCacheDescriptor {
    std::string materialization_kind;
    std::string backend_hint;
    std::string source_tensor_id;
    std::uint64_t bytes = 0;
};

struct PackedLayoutBlobHeader {
    std::array<char, 8> magic{'J', 'A', 'K', 'P', 'A', 'C', 'K', 'D'};
    std::uint32_t version = 2u;
    std::uint32_t materialization_hash = 0u;
    std::uint32_t backend_cache_hash = 0u;
    std::uint64_t source_offset = 0u;
    std::uint64_t source_bytes = 0u;
    std::uint64_t payload_bytes = 0u;
    std::int64_t source_mtime_ticks = 0;
};

struct SpillArtifactHeader {
    std::array<char, 8> magic{'J', 'A', 'K', 'S', 'P', 'I', 'L', 'L'};
    std::uint32_t version = 1u;
    std::uint32_t tensor_hash = 0u;
    std::uint32_t device_hash = 0u;
    std::uint32_t operation_hash = 0u;
    std::uint64_t declared_bytes = 0u;
    std::uint64_t operation_index = 0u;
};

std::uint32_t stable_text_hash(const std::string& text) {
    std::uint32_t hash = 2166136261u;
    for (const unsigned char ch : text) {
        hash ^= static_cast<std::uint32_t>(ch);
        hash *= 16777619u;
    }
    return hash;
}

std::string sanitize_cache_component(std::string value) {
    for (auto& ch : value) {
        const bool keep =
            std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '.' || ch == '-' || ch == '_';
        if (!keep) {
            ch = '_';
        }
    }
    if (value.empty()) {
        value = "unnamed";
    }
    return value;
}

std::int64_t file_mtime_ticks(const std::filesystem::path& path) {
    std::error_code ec;
    const auto write_time = std::filesystem::last_write_time(path, ec);
    if (ec) {
        return 0;
    }
    return static_cast<std::int64_t>(write_time.time_since_epoch().count());
}

std::string file_metadata_fingerprint(const std::filesystem::path& path) {
    std::error_code ec;
    const auto size = std::filesystem::exists(path, ec) ? std::filesystem::file_size(path, ec) : 0u;
    return sanitize_cache_component(path.filename().string()) + ":" + std::to_string(size) + ":" +
           std::to_string(file_mtime_ticks(path));
}

std::vector<std::filesystem::path> existing_binary_paths(std::initializer_list<std::filesystem::path> candidates) {
    std::vector<std::filesystem::path> paths;
    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec)) {
            paths.push_back(candidate);
        }
    }
    return paths;
}

std::string join_fingerprints(const std::vector<std::filesystem::path>& paths) {
    if (paths.empty()) {
        return "unresolved";
    }
    std::ostringstream stream;
    bool first = true;
    for (const auto& path : paths) {
        if (!first) {
            stream << ";";
        }
        first = false;
        stream << file_metadata_fingerprint(path);
    }
    return stream.str();
}

std::string backend_binary_cache_tag(const HardwareGraph& graph) {
    std::ostringstream declared_versions;
    if (!graph.driver_version.empty()) {
        declared_versions << "drv=" << sanitize_cache_component(graph.driver_version);
    }
    if (!graph.runtime_version.empty()) {
        if (declared_versions.tellp() > 0) {
            declared_versions << ";";
        }
        declared_versions << "rt=" << sanitize_cache_component(graph.runtime_version);
    }
    if (!graph.compiler_version.empty()) {
        if (declared_versions.tellp() > 0) {
            declared_versions << ";";
        }
        declared_versions << "cc=" << sanitize_cache_component(graph.compiler_version);
    }
    const auto version_prefix = declared_versions.str();
    if (graph.probe == "host") {
        return "host:" + structural_fingerprint(graph) + ":" + version_prefix;
    }
#if defined(_WIN32)
    if (graph.probe == "level-zero") {
        const auto paths = existing_binary_paths({
            "C:\\Windows\\System32\\ze_loader.dll",
            "C:\\Windows\\System32\\ze_intel_gpu64.dll",
            "C:\\Windows\\System32\\igdrcl64.dll",
            "C:\\Program Files (x86)\\Intel\\oneAPI\\2025.1\\bin\\ocloc.exe",
            "C:\\Program Files (x86)\\Intel\\oneAPI\\latest\\bin\\ocloc.exe"});
        return "level-zero:" + structural_fingerprint(graph) + ":" + version_prefix + ":" + join_fingerprints(paths);
    }
    if (graph.probe == "cuda") {
        const auto paths = existing_binary_paths({
            "C:\\Windows\\System32\\nvcuda.dll",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2\\bin\\nvrtc64_130_0.dll",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2\\bin\\nvrtc-builtins64_130.dll"});
        return "cuda:" + structural_fingerprint(graph) + ":" + version_prefix + ":" + join_fingerprints(paths);
    }
    if (graph.probe == "rocm") {
        const auto paths = existing_binary_paths({
            "C:\\Windows\\System32\\amdhip64.dll",
            "C:\\Windows\\System32\\hiprtc.dll",
            "C:\\Windows\\System32\\hiprtc0507.dll"});
        return "rocm:" + structural_fingerprint(graph) + ":" + version_prefix + ":" + join_fingerprints(paths);
    }
    if (graph.probe == "opencl") {
        const auto paths = existing_binary_paths({
            "C:\\Windows\\System32\\OpenCL.dll",
            "C:\\Windows\\System32\\igdrcl64.dll"});
        return "opencl:" + structural_fingerprint(graph) + ":" + version_prefix + ":" + join_fingerprints(paths);
    }
#else
    if (graph.probe == "level-zero") {
        const auto paths = existing_binary_paths({
            "/usr/lib/libze_loader.so",
            "/usr/lib/x86_64-linux-gnu/libze_loader.so",
            "/opt/intel/oneapi/compiler/latest/bin/ocloc"});
        return "level-zero:" + structural_fingerprint(graph) + ":" + version_prefix + ":" + join_fingerprints(paths);
    }
    if (graph.probe == "cuda") {
        const auto paths = existing_binary_paths({
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/local/cuda/lib64/libnvrtc.so"});
        return "cuda:" + structural_fingerprint(graph) + ":" + version_prefix + ":" + join_fingerprints(paths);
    }
    if (graph.probe == "rocm") {
        const auto paths = existing_binary_paths({
            "/opt/rocm/lib/libamdhip64.so",
            "/opt/rocm/lib/libhiprtc.so"});
        return "rocm:" + structural_fingerprint(graph) + ":" + version_prefix + ":" + join_fingerprints(paths);
    }
    if (graph.probe == "opencl") {
        const auto paths = existing_binary_paths({
            "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1"});
        return "opencl:" + structural_fingerprint(graph) + ":" + version_prefix + ":" + join_fingerprints(paths);
    }
#endif
    return graph.probe + ":" + structural_fingerprint(graph) + ":" + version_prefix;
}

std::vector<std::uint8_t> read_asset_window(
    const std::filesystem::path& path,
    const std::uint64_t file_offset,
    const std::uint64_t bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        return {};
    }
    input.seekg(0, std::ios::end);
    const auto file_size = static_cast<std::uint64_t>(std::max<std::streamoff>(input.tellg(), 0));
    if (file_offset >= file_size) {
        return {};
    }
    const auto available = file_size - file_offset;
    const auto window_bytes = bytes == 0u ? available : std::min(bytes, available);
    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(window_bytes), 0u);
    input.seekg(static_cast<std::streamoff>(file_offset), std::ios::beg);
    input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    const auto read_bytes = static_cast<std::size_t>(std::max<std::streamsize>(input.gcount(), 0));
    buffer.resize(read_bytes);
    return buffer;
}

std::vector<std::uint8_t> build_packed_layout_payload(
    const std::vector<std::uint8_t>& source,
    const LayoutCacheDescriptor& descriptor) {
    if (source.empty() || descriptor.bytes == 0u) {
        return {};
    }
    std::vector<std::uint8_t> payload(static_cast<std::size_t>(descriptor.bytes), 0u);
    if (descriptor.materialization_kind.find("packed-rhs") != std::string::npos) {
        constexpr std::size_t kBlock = 16u;
        const std::size_t source_size = source.size();
        for (std::size_t dst = 0; dst < payload.size(); ++dst) {
            const std::size_t block_base = (dst / kBlock) * kBlock;
            const std::size_t lane = dst % kBlock;
            const std::size_t src = (lane * kBlock + block_base / kBlock) % source_size;
            payload[dst] = source[src];
        }
        return payload;
    }
    if (descriptor.materialization_kind.find("conv-patch9") != std::string::npos) {
        for (std::size_t dst = 0; dst < payload.size(); ++dst) {
            payload[dst] = source[(dst / 9u) % source.size()];
        }
        return payload;
    }
    for (std::size_t dst = 0; dst < payload.size(); ++dst) {
        payload[dst] = source[dst % source.size()];
    }
    return payload;
}

std::filesystem::path runtime_layout_cache_root(const RuntimeOptions& options) {
    auto base = options.cache_path.empty() ? Planner::default_cache_path() : options.cache_path;
    if (base.has_extension()) {
        const auto parent = base.parent_path();
        const auto stem = sanitize_cache_component(base.stem().string());
        return (parent.empty() ? std::filesystem::path(".") : parent) / (stem + "-packed-layouts");
    }
    return base / "packed-layouts";
}

std::filesystem::path runtime_spill_artifact_root(const RuntimeInstallPaths& paths) {
    return paths.cache_dir / "spill-artifacts";
}

std::filesystem::path packed_layout_blob_path(
    const std::filesystem::path& root,
    const AssetPrefetchEntry& entry) {
    return root /
           (sanitize_cache_component(entry.source_asset_id.empty() ? entry.asset_id : entry.source_asset_id) + "-" +
            sanitize_cache_component(entry.tensor_id.empty() ? "global" : entry.tensor_id) + "-" +
            sanitize_cache_component(entry.materialization_kind) + "-" +
            sanitize_cache_component(entry.device_uid.empty() ? "host" : entry.device_uid) + "-" +
            std::to_string(stable_text_hash(entry.backend_cache_tag)) + ".jpkd");
}

bool packed_layout_blob_matches(
    const std::filesystem::path& path,
    const AssetPrefetchEntry& entry,
    const std::int64_t source_mtime) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        return false;
    }
    PackedLayoutBlobHeader header;
    input.read(reinterpret_cast<char*>(&header), static_cast<std::streamsize>(sizeof(header)));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(header))) {
        return false;
    }
    return header.magic == PackedLayoutBlobHeader{}.magic &&
           header.version == 2u &&
           header.materialization_hash == stable_text_hash(entry.materialization_kind + "|" + entry.backend_hint) &&
           header.backend_cache_hash == stable_text_hash(entry.backend_cache_tag) &&
           header.source_offset == entry.file_offset &&
           header.source_bytes == entry.bytes &&
           header.payload_bytes == entry.bytes &&
           header.source_mtime_ticks == source_mtime;
}

bool write_packed_layout_blob(
    const std::filesystem::path& path,
    const AssetPrefetchEntry& entry,
    const std::int64_t source_mtime,
    const std::vector<std::uint8_t>& payload) {
    PackedLayoutBlobHeader header;
    header.materialization_hash = stable_text_hash(entry.materialization_kind + "|" + entry.backend_hint);
    header.backend_cache_hash = stable_text_hash(entry.backend_cache_tag);
    header.source_offset = entry.file_offset;
    header.source_bytes = entry.bytes;
    header.payload_bytes = static_cast<std::uint64_t>(payload.size());
    header.source_mtime_ticks = source_mtime;

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        return false;
    }
    output.write(reinterpret_cast<const char*>(&header), static_cast<std::streamsize>(sizeof(header)));
    output.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    return output.good();
}

bool should_persist_packed_layout_blob(const AssetPrefetchEntry& entry) {
    if (!entry.derived_cache) {
        return false;
    }
    return entry.materialization_kind.find("packed-rhs") != std::string::npos ||
           entry.materialization_kind.find("conv-patch9") != std::string::npos;
}

std::filesystem::path spill_artifact_path(
    const std::filesystem::path& root,
    const ResidencyAction& action) {
    return root /
           (sanitize_cache_component(action.tensor_id.empty() ? "tensor" : action.tensor_id) + "-" +
            sanitize_cache_component(action.device_uid.empty() ? "host" : action.device_uid) + "-" +
            std::to_string(action.operation_index) + ".jspill");
}

bool spill_artifact_matches(const std::filesystem::path& path, const ResidencyAction& action) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        return false;
    }
    SpillArtifactHeader header;
    input.read(reinterpret_cast<char*>(&header), static_cast<std::streamsize>(sizeof(header)));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(header))) {
        return false;
    }
    return header.magic == SpillArtifactHeader{}.magic &&
           header.version == 1u &&
           header.tensor_hash == stable_text_hash(action.tensor_id) &&
           header.device_hash == stable_text_hash(action.device_uid) &&
           header.operation_hash == stable_text_hash(action.trigger_operation_name) &&
           header.declared_bytes == action.bytes &&
           header.operation_index == action.operation_index;
}

bool write_spill_artifact(const std::filesystem::path& path, const ResidencyAction& action) {
    SpillArtifactHeader header;
    header.tensor_hash = stable_text_hash(action.tensor_id);
    header.device_hash = stable_text_hash(action.device_uid);
    header.operation_hash = stable_text_hash(action.trigger_operation_name);
    header.declared_bytes = action.bytes;
    header.operation_index = action.operation_index;

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        return false;
    }
    output.write(reinterpret_cast<const char*>(&header), static_cast<std::streamsize>(sizeof(header)));
    return output.good();
}

void materialize_packed_layout_blobs(
    const RuntimeOptions& options,
    AssetPrefetchReport& report) {
    const auto root = runtime_layout_cache_root(options);
    std::error_code ec;
    std::filesystem::create_directories(root, ec);

    for (auto& entry : report.entries) {
        if (!should_persist_packed_layout_blob(entry) || entry.path.empty()) {
            continue;
        }
        const auto source_path = entry.path;
        const auto source_mtime = file_mtime_ticks(source_path);
        const auto cache_path = packed_layout_blob_path(root, entry);
        if (packed_layout_blob_matches(cache_path, entry, source_mtime)) {
            entry.path = cache_path;
            entry.exists_on_disk = true;
            continue;
        }

        const auto source = read_asset_window(source_path, entry.file_offset, 0u);
        const auto payload = build_packed_layout_payload(
            source,
            LayoutCacheDescriptor{
                entry.materialization_kind,
                entry.backend_hint,
                entry.tensor_id,
                entry.bytes});
        if (payload.size() != entry.bytes || payload.empty()) {
            continue;
        }
        if (write_packed_layout_blob(cache_path, entry, source_mtime, payload)) {
            entry.path = cache_path;
            entry.exists_on_disk = true;
        }
    }
}

std::optional<LayoutCacheDescriptor> describe_layout_cache(
    const OperationSpec& operation,
    const WorkloadGraph& workload_graph,
    const bool gpu_target) {
    const auto find_tensor_bytes = [&](const std::string& tensor_id) -> std::uint64_t {
        const auto it = std::find_if(
            workload_graph.tensors.begin(),
            workload_graph.tensors.end(),
            [&](const WorkloadTensor& tensor) { return tensor.id == tensor_id; });
        return it == workload_graph.tensors.end() ? 0u : it->bytes;
    };

    switch (operation.op_class) {
    case OperationClass::matmul:
        if ((gpu_target && !(operation.gpu_pack_weights || operation.gpu_pretranspose_rhs)) ||
            (!gpu_target && !(operation.cpu_pack_weights || operation.cpu_pretranspose_rhs)) ||
            operation.input_tensor_ids.size() < 2u) {
            return std::nullopt;
        }
        return LayoutCacheDescriptor{
            gpu_target ? "gpu-packed-rhs" : "cpu-packed-rhs",
            gpu_target ? "gpu" : "cpu",
            operation.input_tensor_ids[1],
            find_tensor_bytes(operation.input_tensor_ids[1])};
    case OperationClass::convolution_2d:
        if ((gpu_target && operation.gpu_input_layout.find("conv-patch9") == std::string::npos) ||
            (!gpu_target && operation.cpu_input_layout.find("conv-patch9") == std::string::npos) ||
            operation.input_tensor_ids.empty() || operation.extents.size() < 2u) {
            return std::nullopt;
        }
        return LayoutCacheDescriptor{
            gpu_target ? "gpu-conv-patch9" : "cpu-conv-patch9",
            gpu_target ? "gpu" : "cpu",
            operation.input_tensor_ids.front(),
            operation.output_bytes * 9u};
    case OperationClass::resample_2d:
        if ((gpu_target && operation.gpu_input_layout.find("resample-packed6") == std::string::npos) ||
            (!gpu_target && operation.cpu_input_layout.find("resample-packed6") == std::string::npos) ||
            operation.input_tensor_ids.empty() || operation.extents.size() < 4u) {
            return std::nullopt;
        }
        return LayoutCacheDescriptor{
            gpu_target ? "gpu-resample-packed6" : "cpu-resample-packed6",
            gpu_target ? "gpu" : "cpu",
            operation.input_tensor_ids.front(),
            operation.output_bytes * 6u};
    default:
        return std::nullopt;
    }
}

double successful_operation_ratio(const DirectExecutionReport& report) {
    if (report.operations.empty()) {
        return report.all_succeeded ? 1.0 : 0.0;
    }

    const auto successful = static_cast<double>(std::count_if(
        report.operations.begin(),
        report.operations.end(),
        [](const OperationExecutionRecord& operation) {
            return operation.backend_error.empty() && operation.verified;
        }));
    return successful / static_cast<double>(report.operations.size());
}

StrategyFeedbackSample make_strategy_feedback_sample(
    const WorkloadSpec& requested_workload,
    const DirectExecutionReport& report,
    const PartitionStrategy strategy,
    const PlanStrategySource strategy_source,
    const double planned_confidence,
    const bool rolled_back_to_auto,
    const double max_runtime_regression_ratio) {
    const bool regressed = runtime_regressed(report, max_runtime_regression_ratio);
    return StrategyFeedbackSample{
        strategy,
        report.total_runtime_us,
        head_runtime_us(requested_workload, report),
        report.speedup_vs_reference,
        successful_operation_ratio(report),
        report.all_succeeded,
        strategy_source,
        planned_confidence,
        rolled_back_to_auto,
        regressed};
}

void record_partition_strategy_feedback(
    Planner& planner,
    const WorkloadSpec& requested_workload,
    const std::vector<HardwareGraph>& graphs,
    const StrategyFeedbackSample& feedback) {
    planner.ingest_strategy_feedback(
        requested_workload,
        graphs,
        feedback);
}

bool runtime_regressed(
    const DirectExecutionReport& report,
    const double max_runtime_regression_ratio) {
    return report.total_reference_runtime_us > 0.0 &&
           report.total_runtime_us > (report.total_reference_runtime_us * max_runtime_regression_ratio);
}

std::uint64_t effective_capacity_bytes(
    const HardwareGraph& graph,
    const RuntimeMemoryPolicy& policy) {
    const auto summary = summarize_graph(graph);
    std::uint64_t capacity = 0u;
    if (graph.probe == "host") {
        capacity = summary.addressable_bytes == 0u ? summary.shared_host_bytes : summary.addressable_bytes;
    } else {
        capacity = summary.directly_attached_bytes;
        if (policy.allow_host_spill && (summary.unified_address_space || summary.coherent_with_host)) {
            capacity += summary.shared_host_bytes;
        }
        if (capacity == 0u) {
            capacity = summary.addressable_bytes;
        }
    }
    const double reserve_ratio = graph.probe == "host" ? policy.host_reserve_ratio : policy.accelerator_reserve_ratio;
    return static_cast<std::uint64_t>(std::max(0.0, static_cast<double>(capacity) * std::max(0.0, 1.0 - reserve_ratio)));
}

const HardwareGraph* find_graph_by_uid(
    const std::vector<HardwareGraph>& graphs,
    const std::string& uid) {
    const auto it = std::find_if(graphs.begin(), graphs.end(), [&](const HardwareGraph& graph) {
        return graph.uid == uid;
    });
    return it == graphs.end() ? nullptr : &(*it);
}

std::string lowercase_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool looks_like_intel_opencl_graph(const HardwareGraph& graph) {
    if (graph.probe != "opencl") {
        return false;
    }
    const auto haystack = lowercase_copy(graph.presentation_name + " " + graph.uid);
    return haystack.find("intel") != std::string::npos;
}

bool has_level_zero_graph(const std::vector<HardwareGraph>& graphs) {
    return std::any_of(graphs.begin(), graphs.end(), [](const HardwareGraph& graph) {
        return graph.probe == "level-zero";
    });
}

std::string backend_name_for_graph(const HardwareGraph& graph) {
    if (graph.probe == "host") {
        return "host-native";
    }
    if (graph.probe == "level-zero") {
        return "level-zero-native";
    }
    if (graph.probe == "cuda") {
        return "cuda-native";
    }
    if (graph.probe == "rocm") {
        return "rocm-native";
    }
    if (graph.probe == "vulkan") {
        return executors::vulkan_direct_backend_available() ? "vulkan-direct" : "vulkan-modeled";
    }
    if (graph.probe == "opencl") {
        return "opencl-direct";
    }
    return graph.probe + "-direct";
}

std::optional<JakalBackendKind> backend_kind_for_probe(const std::string& probe) {
    if (probe == "opencl") {
        return JakalBackendKind::opencl;
    }
    if (probe == "level-zero") {
        return JakalBackendKind::level_zero;
    }
    if (probe == "vulkan") {
        return JakalBackendKind::vulkan_compute;
    }
    if (probe == "cuda") {
        return JakalBackendKind::cuda;
    }
    if (probe == "rocm") {
        return JakalBackendKind::rocm;
    }
    return std::nullopt;
}

bool backend_supports_operation(
    const HardwareGraph& graph,
    const OperationClass op_class,
    std::string* reason = nullptr) {
    if (graph.probe == "host" || graph.probe == "host-emulated") {
        return true;
    }
    if (const auto backend_kind = backend_kind_for_probe(graph.probe)) {
        if (*backend_kind == JakalBackendKind::vulkan_compute &&
            !executors::vulkan_direct_backend_available()) {
            if (reason != nullptr) {
                *reason = "vulkan direct backend unavailable";
            }
            return false;
        }
        if (backend_kind_supports_operation(*backend_kind, op_class)) {
            return true;
        }
        if (reason != nullptr) {
            *reason =
                *backend_kind == JakalBackendKind::vulkan_compute
                    ? "vulkan direct backend lacks kernel coverage for the requested operation"
                    : "backend lacks direct kernel coverage";
        }
        return false;
    }
    if (reason != nullptr) {
        *reason = "unknown backend kernel coverage";
    }
    return false;
}

double planner_risk_score(
    const WorkloadSpec& workload,
    const ExecutionPlan& planning,
    const OptimizationReport& optimization,
    const MemoryPreflightReport& memory,
    const KernelCoverageReport& kernel_coverage,
    const ResidencySequenceReport& residency) {
    double score = 0.0;
    if (planning.strategy_source == PlanStrategySource::exploration) {
        score += 0.35;
    } else if (planning.strategy_source == PlanStrategySource::family_learning) {
        score += 0.18;
    }
    if (!planning.loaded_from_cache) {
        score += 0.05;
    }
    if (workload.latency_sensitive) {
        score += 0.05;
    }
    if (optimization.workload_host_exchange_bytes >= (64ull * 1024ull * 1024ull)) {
        score += 0.10;
    }
    if (memory.requires_spill) {
        score += 0.25;
    }
    if (memory.peak_pressure_ratio >= 0.90) {
        score += 0.10;
    }
    if (residency.spill_bytes > 0u) {
        score += 0.15;
    }
    if (residency.forced_spill_count > 0u) {
        score += 0.25;
    }
    if (!kernel_coverage.all_supported) {
        score += 0.30;
    }
    return std::clamp(score, 0.0, 1.0);
}

}  // namespace

std::string to_string(const RuntimeBackendStatusCode code) {
    switch (code) {
    case RuntimeBackendStatusCode::disabled:
        return "disabled";
    case RuntimeBackendStatusCode::unavailable:
        return "unavailable";
    case RuntimeBackendStatusCode::no_devices:
        return "no-devices";
    case RuntimeBackendStatusCode::ready_direct:
        return "ready-direct";
    case RuntimeBackendStatusCode::ready_modeled:
        return "ready-modeled";
    }
    return "unavailable";
}

RuntimeInstallPaths resolve_runtime_install_paths(const std::filesystem::path& install_root) {
    RuntimeInstallPaths paths;
    paths.install_root =
        install_root.empty() ? env_path("JAKAL_INSTALL_ROOT") : install_root;
    paths.writable_root = choose_writable_runtime_root();
    paths.config_dir = choose_config_runtime_root();
    paths.cache_dir = paths.writable_root / "cache";
    paths.logs_dir = paths.writable_root / "logs";
    paths.telemetry_path = paths.logs_dir / "runtime-telemetry.tsv";
    paths.planner_cache_path = paths.cache_dir / "planner-cache.tsv";
    paths.execution_cache_path = paths.cache_dir / "execution-cache.tsv";
    paths.python_dir = paths.install_root.empty() ? (paths.writable_root / "python") : (paths.install_root / "python");
    return paths;
}

RuntimeOptions make_runtime_options_for_install(const std::filesystem::path& install_root) {
    RuntimeOptions options;
    const auto paths = resolve_runtime_install_paths(install_root);
    options.install_root = paths.install_root;
    options.cache_path = paths.planner_cache_path;
    options.execution_cache_path = paths.execution_cache_path;
    options.product.observability.telemetry_path = paths.telemetry_path;
    const auto config_path = paths.config_dir / "jakal-runtime-config.ini";
    apply_runtime_ini_overrides(config_path, options);
    return options;
}

std::string runtime_backend_cache_tag_for_graph(const HardwareGraph& graph) {
    return backend_binary_cache_tag(graph);
}

std::string runtime_backend_name_for_graph(const HardwareGraph& graph) {
    return backend_name_for_graph(graph);
}

bool runtime_backend_supports_operation(
    const HardwareGraph& graph,
    const OperationClass op_class,
    std::string* reason) {
    return backend_supports_operation(graph, op_class, reason);
}

Runtime::Runtime(RuntimeOptions options)
    : options_([&options]() {
          auto normalized = std::move(options);
          const auto paths = resolve_runtime_install_paths(normalized.install_root);
          normalized.install_root = paths.install_root;
          if (normalized.cache_path.empty()) {
              normalized.cache_path = paths.planner_cache_path;
          }
          if (normalized.execution_cache_path.empty()) {
              normalized.execution_cache_path = paths.execution_cache_path;
          }
          if (normalized.product.observability.telemetry_path.empty()) {
              normalized.product.observability.telemetry_path = paths.telemetry_path;
          }
          return normalized;
      }()),
      install_paths_(resolve_runtime_install_paths(options_.install_root)),
      planner_(options_.cache_path),
      execution_optimizer_(options_.execution_cache_path) {
    if (options_.enable_host_probe) {
        probes_.push_back(make_host_probe());
    }
    if (options_.enable_opencl_probe) {
        probes_.push_back(make_opencl_probe());
    }
    if (options_.enable_level_zero_probe) {
        probes_.push_back(make_level_zero_probe());
    }
    if (options_.enable_vulkan_probe) {
        probes_.push_back(make_vulkan_probe());
    }
    if (options_.enable_cuda_probe) {
        probes_.push_back(make_cuda_probe());
    }
    if (options_.enable_rocm_probe) {
        probes_.push_back(make_rocm_probe());
    }
    rebuild_backend_statuses();
    if (options_.eager_hardware_refresh) {
        refresh_hardware();
    }
}

const RuntimeOptions& Runtime::options() const {
    return options_;
}

const RuntimeInstallPaths& Runtime::install_paths() const {
    return install_paths_;
}

const std::vector<RuntimeBackendStatus>& Runtime::backend_statuses() const {
    return backend_statuses_;
}

void Runtime::rebuild_backend_statuses() {
    backend_statuses_.clear();

    auto append_backend = [&](const std::string& backend_name,
                              const bool enabled,
                              const bool direct_available,
                              const bool modeled_fallback,
                              const std::string& detail) {
        std::size_t device_count = 0u;
        for (const auto& graph : devices_) {
            if (graph.probe == backend_name) {
                RuntimeBackendStatus entry;
                entry.backend_name = backend_name;
                entry.device_uid = graph.uid;
                entry.enabled = enabled;
                entry.available = true;
                entry.direct_execution = direct_available;
                entry.modeled_fallback = modeled_fallback;
                entry.code = direct_available ? RuntimeBackendStatusCode::ready_direct
                                              : RuntimeBackendStatusCode::ready_modeled;
                entry.detail = detail.empty() ? runtime_backend_name_for_graph(graph) : detail;
                backend_statuses_.push_back(std::move(entry));
                ++device_count;
            }
        }
        if (device_count == 0u) {
            RuntimeBackendStatus entry;
            entry.backend_name = backend_name;
            entry.enabled = enabled;
            entry.available = false;
            entry.direct_execution = direct_available;
            entry.modeled_fallback = modeled_fallback;
            entry.code = !enabled ? RuntimeBackendStatusCode::disabled
                                  : (direct_available ? RuntimeBackendStatusCode::unavailable
                                                      : RuntimeBackendStatusCode::no_devices);
            entry.detail = detail;
            backend_statuses_.push_back(std::move(entry));
        }
    };

    append_backend("host", options_.enable_host_probe, true, false, options_.enable_host_probe ? "host probe enabled" : "host probe disabled");
    append_backend("opencl", options_.enable_opencl_probe, true, false, options_.enable_opencl_probe ? "opencl probe active" : "opencl probe disabled");
    append_backend("level-zero", options_.enable_level_zero_probe, true, false, options_.enable_level_zero_probe ? "level-zero probe active" : "level-zero probe disabled");
    append_backend("cuda", options_.enable_cuda_probe, true, false, options_.enable_cuda_probe ? "cuda probe active" : "cuda probe disabled");
    append_backend("rocm", options_.enable_rocm_probe, true, false, options_.enable_rocm_probe ? "rocm probe active" : "rocm probe disabled");

    RuntimeBackendStatus vulkan;
    vulkan.backend_name = "vulkan";
    vulkan.enabled = options_.enable_vulkan_status;
    vulkan.available = executors::vulkan_direct_backend_available();
    vulkan.direct_execution = vulkan.available;
    vulkan.modeled_fallback = !vulkan.available;
    vulkan.code = !vulkan.enabled
        ? RuntimeBackendStatusCode::disabled
        : (vulkan.available ? RuntimeBackendStatusCode::ready_direct : RuntimeBackendStatusCode::ready_modeled);
    vulkan.detail = executors::vulkan_direct_backend_status_detail();
    backend_statuses_.push_back(std::move(vulkan));
}

void Runtime::refresh_hardware() {
    devices_.clear();
    jakal_toolkit_index_.clear();
    std::vector<HardwareGraph> discovered;

    for (auto& probe : probes_) {
        if (!probe->available()) {
            continue;
        }

        for (auto& graph : probe->discover_hardware()) {
            if (should_include_descriptor(graph)) {
                discovered.push_back(std::move(graph));
            }
        }
    }

    const bool shadow_intel_opencl =
        options_.prefer_level_zero_over_opencl && options_.enable_level_zero_probe && has_level_zero_graph(discovered);
    for (auto& graph : discovered) {
        if (shadow_intel_opencl && looks_like_intel_opencl_graph(graph)) {
            continue;
        }
        devices_.push_back(std::move(graph));
    }
    std::sort(devices_.begin(), devices_.end(), [](const HardwareGraph& left, const HardwareGraph& right) {
        const auto left_summary = summarize_graph(left);
        const auto right_summary = summarize_graph(right);

        if (left_summary.execution_objects != right_summary.execution_objects) {
            return left_summary.execution_objects > right_summary.execution_objects;
        }
        if (left_summary.addressable_bytes != right_summary.addressable_bytes) {
            return left_summary.addressable_bytes > right_summary.addressable_bytes;
        }
        if (left_summary.host_read_gbps != right_summary.host_read_gbps) {
            return left_summary.host_read_gbps > right_summary.host_read_gbps;
        }
        return structural_fingerprint(left) < structural_fingerprint(right);
    });

    jakal_toolkit_index_ = jakal_toolkit_.build_index(devices_);
    rebuild_backend_statuses();
    hardware_refreshed_ = true;
}

void Runtime::ensure_hardware_refreshed() const {
    if (!hardware_refreshed_) {
        const_cast<Runtime*>(this)->refresh_hardware();
    }
}

const std::vector<HardwareGraph>& Runtime::devices() const {
    ensure_hardware_refreshed();
    return devices_;
}

const std::vector<JakalToolkitIndexEntry>& Runtime::jakal_toolkit_index() const {
    ensure_hardware_refreshed();
    return jakal_toolkit_index_;
}

ExecutionPlan Runtime::plan(const WorkloadSpec& workload) {
    ensure_hardware_refreshed();

    const auto workload_graph = default_workload_graph(workload);
    const auto context = resolve_runtime_optimization_context(options_, workload, &workload_graph);
    return planner_.build_plan(context.effective_workload, devices_);
}

OptimizationReport Runtime::optimize(const WorkloadSpec& workload) {
    ensure_hardware_refreshed();

    const auto workload_graph = default_workload_graph(workload);
    const auto context = resolve_runtime_optimization_context(options_, workload, &workload_graph);
    const auto placement = planner_.build_plan(context.effective_workload, devices_);
    auto report = execution_optimizer_.optimize(
        context.effective_workload,
        placement,
        devices_,
        &workload_graph,
        &context.execution_tuning);
    return report;
}

OptimizationReport Runtime::optimize(const WorkloadSpec& workload, const WorkloadGraph& workload_graph) {
    ensure_hardware_refreshed();

    const auto context = resolve_runtime_optimization_context(options_, workload, &workload_graph);
    const auto placement = planner_.build_plan(context.effective_workload, devices_);
    auto report = execution_optimizer_.optimize(
        context.effective_workload,
        placement,
        devices_,
        &workload_graph,
        &context.execution_tuning);
    return report;
}

DirectExecutionReport Runtime::execute_with_feedback(
    const WorkloadSpec& workload,
    const OptimizationReport& optimization,
    const WorkloadGraph* workload_graph_override) {
    auto initial_report = direct_executor_.execute(optimization, devices_, jakal_toolkit_index_);
    auto initial_feedback = make_feedback_records(initial_report);
    execution_optimizer_.ingest_execution_feedback(
        initial_report.optimization,
        initial_feedback,
        devices_);

    if (!should_retry_execution(initial_report)) {
        return initial_report;
    }

    const auto refined_optimization =
        workload_graph_override == nullptr ? optimize(workload) : optimize(workload, *workload_graph_override);
    if (!selection_changed(initial_report.optimization, refined_optimization)) {
        return initial_report;
    }

    auto refined_report = direct_executor_.execute(refined_optimization, devices_, jakal_toolkit_index_);
    auto refined_feedback = make_feedback_records(refined_report);
    execution_optimizer_.ingest_execution_feedback(
        refined_report.optimization,
        refined_feedback,
        devices_);

    if (!refined_report.all_succeeded) {
        return initial_report;
    }
    if (!initial_report.all_succeeded) {
        return refined_report;
    }
    if (total_runtime_us(refined_report) < (total_runtime_us(initial_report) * 0.95)) {
        return refined_report;
    }
    return initial_report;
}

DirectExecutionReport Runtime::execute(const WorkloadSpec& workload) {
    return execute_managed(workload).execution;
}

ManagedExecutionReport Runtime::execute_managed(const WorkloadSpec& workload) {
    return execute_managed(workload, default_workload_graph(workload));
}

ManagedExecutionReport Runtime::execute_managed(const WorkloadSpec& workload, const WorkloadGraph& workload_graph) {
    ensure_hardware_refreshed();

    ++execution_epoch_;

    const auto context = resolve_runtime_optimization_context(options_, workload, &workload_graph);
    const auto requested_workload = context.requested_workload;
    ManagedExecutionReport managed;
    managed.telemetry_path = telemetry_path();
    managed.safety.requested_strategy = requested_workload.partition_strategy;

    std::ostringstream safety_summary;
    auto effective_workload = context.effective_workload;
    if (requested_workload.partition_strategy != PartitionStrategy::auto_balanced &&
        is_strategy_blacklisted(requested_workload, requested_workload.partition_strategy)) {
        effective_workload.partition_strategy = PartitionStrategy::auto_balanced;
        effective_workload.heuristic_partition_hint.reset();
        effective_workload.heuristic_partition_hint_confidence = 0.0;
        effective_workload.heuristic_partition_hint_reason.clear();
        effective_workload.disable_heuristic_partition_hint = true;
        effective_workload.disable_automatic_execution_tuning = true;
        effective_workload.disable_strategy_exploration = true;
        managed.safety.blacklisted_before_run = true;
        safety_summary << "strategy blacklisted -> auto";
    }

    auto update_planner_diagnostics = [&](const ExecutionPlan& plan) {
        managed.planning = plan;
        managed.safety.planner_strategy_source = plan.strategy_source;
        managed.safety.planner_confidence = plan.strategy_confidence;
    };

    auto planned = planner_.build_plan(effective_workload, devices_);
    update_planner_diagnostics(planned);
    auto optimization = execution_optimizer_.optimize(
        effective_workload,
        planned,
        devices_,
        &workload_graph,
        &context.execution_tuning);
    auto refresh_execution_reports = [&](const DirectExecutionReport* execution_report) {
        managed.kernel_coverage = build_kernel_coverage(optimization);
        managed.residency_sequence = build_residency_sequence(optimization);
        managed.tensor_allocator = build_tensor_allocator(managed.residency_sequence);
        managed.spill_artifacts = materialize_spill_artifacts(managed.residency_sequence);
        managed.backend_buffer_bindings = build_backend_buffer_bindings(
            execution_report,
            managed.tensor_allocator,
            managed.residency_sequence,
            managed.spill_artifacts);
        managed.executed_residency_movements = build_executed_residency_movements(
            execution_report,
            managed.residency_sequence,
            managed.spill_artifacts);
        managed.memory_preflight.predicted_spill_bytes = managed.residency_sequence.spill_bytes;
        managed.memory_preflight.predicted_reload_bytes = managed.residency_sequence.reload_bytes;
        managed.memory_preflight.forced_spill_count = managed.residency_sequence.forced_spill_count;
        managed.memory_preflight.requires_spill =
            managed.memory_preflight.requires_spill || managed.residency_sequence.spill_bytes > 0u;
    };
    auto refresh_regression_summary = [&]() {
        const auto summary = execution_optimizer_.persisted_regression_summary(optimization.signature);
        managed.safety.persisted_regression_events =
            std::max(managed.safety.persisted_regression_events, summary.max_regression_events);
        managed.safety.persisted_worst_slowdown =
            std::max(managed.safety.persisted_worst_slowdown, summary.worst_slowdown_vs_reference);
    };
    managed.safety.selected_strategy = optimization.partition_strategy;
    managed.memory_preflight = build_memory_preflight(optimization);
    refresh_execution_reports(nullptr);
    refresh_regression_summary();
    managed.safety.planner_risk_score = planner_risk_score(
        effective_workload,
        managed.planning,
        optimization,
        managed.memory_preflight,
        managed.kernel_coverage,
        managed.residency_sequence);

    auto force_auto_if_needed = [&](const char* reason) {
        if (optimization.partition_strategy == PartitionStrategy::auto_balanced) {
            return;
        }
        auto fallback_workload = requested_workload;
        fallback_workload.partition_strategy = PartitionStrategy::auto_balanced;
        fallback_workload.heuristic_partition_hint.reset();
        fallback_workload.heuristic_partition_hint_confidence = 0.0;
        fallback_workload.heuristic_partition_hint_reason.clear();
        fallback_workload.disable_heuristic_partition_hint = true;
        fallback_workload.disable_automatic_execution_tuning = true;
        fallback_workload.disable_strategy_exploration = true;
        auto fallback_plan = planner_.build_plan(fallback_workload, devices_);
        auto fallback_optimization = execution_optimizer_.optimize(
            fallback_workload,
            fallback_plan,
            devices_,
            &workload_graph,
            &context.execution_tuning);
        auto fallback_memory = build_memory_preflight(fallback_optimization);
        if (fallback_memory.safe_to_run || !managed.memory_preflight.safe_to_run) {
            effective_workload = fallback_workload;
            planned = std::move(fallback_plan);
            update_planner_diagnostics(planned);
            optimization = std::move(fallback_optimization);
            managed.memory_preflight = std::move(fallback_memory);
            refresh_execution_reports(nullptr);
            refresh_regression_summary();
            managed.safety.planner_risk_score = planner_risk_score(
                effective_workload,
                managed.planning,
                optimization,
                managed.memory_preflight,
                managed.kernel_coverage,
                managed.residency_sequence);
            managed.safety.memory_forced_auto = true;
            if (safety_summary.tellp() > 0) {
                safety_summary << "; ";
            }
            safety_summary << reason;
        }
    };

    if (optimization.partition_strategy != PartitionStrategy::auto_balanced &&
        is_strategy_blacklisted(requested_workload, optimization.partition_strategy)) {
        force_auto_if_needed("selected strategy blacklisted -> auto");
    }

    if (!managed.memory_preflight.safe_to_run) {
        force_auto_if_needed("memory preflight forced auto");
    }
    if (!managed.kernel_coverage.all_supported) {
        force_auto_if_needed("kernel coverage forced auto");
        managed.kernel_coverage.forced_auto = managed.safety.memory_forced_auto;
    }
    if (options_.product.safety.enable_planner_risk_gate &&
        managed.planning.resolved_partition_strategy != PartitionStrategy::auto_balanced &&
        managed.safety.planner_confidence < options_.product.safety.minimum_planner_confidence &&
        managed.safety.planner_risk_score >= options_.product.safety.planner_risk_gate) {
        force_auto_if_needed("planner risk gate forced auto");
        managed.safety.planner_forced_auto = true;
    }
    if (options_.product.safety.enable_persisted_regression_gate &&
        optimization.partition_strategy != PartitionStrategy::auto_balanced &&
        managed.safety.persisted_regression_events >=
            options_.product.safety.persisted_regression_gate_after_events &&
        managed.safety.persisted_worst_slowdown >=
            options_.product.safety.persisted_regression_slowdown_ratio) {
        const auto regression_gate_events = managed.safety.persisted_regression_events;
        const auto regression_gate_worst = managed.safety.persisted_worst_slowdown;
        force_auto_if_needed("persisted regression gate forced auto");
        managed.safety.regression_gate_forced_auto = true;
        managed.safety.persisted_regression_events =
            std::max(managed.safety.persisted_regression_events, regression_gate_events);
        managed.safety.persisted_worst_slowdown =
            std::max(managed.safety.persisted_worst_slowdown, regression_gate_worst);
    }

    managed.safety.selected_strategy = optimization.partition_strategy;
    if (!managed.memory_preflight.safe_to_run && options_.product.memory.enforce_preflight) {
        managed.safety.blocked_by_memory = true;
        managed.safety.final_strategy = optimization.partition_strategy;
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << managed.memory_preflight.summary;
        managed.safety.summary = safety_summary.str();
        persist_telemetry(requested_workload, managed);
        return managed;
    }

    managed.execution = execute_with_feedback(effective_workload, optimization, &workload_graph);
    managed.executed = true;
    refresh_execution_reports(&managed.execution);
    refresh_regression_summary();
    managed.safety.final_strategy = managed.execution.optimization.partition_strategy;
    auto planner_feedback = make_strategy_feedback_sample(
        effective_workload,
        managed.execution,
        managed.safety.selected_strategy,
        managed.planning.strategy_source,
        managed.planning.strategy_confidence,
        false,
        options_.product.safety.max_runtime_regression_ratio);

    const bool explicit_strategy = managed.execution.optimization.partition_strategy != PartitionStrategy::auto_balanced;
    const bool canary_triggered =
        explicit_strategy &&
        options_.product.safety.enable_canary &&
        (!managed.execution.all_succeeded ||
         runtime_regressed(managed.execution, options_.product.safety.max_runtime_regression_ratio));

    if (canary_triggered) {
        managed.safety.canary_triggered = true;
        record_strategy_failure(requested_workload, managed.execution.optimization.partition_strategy);
        planner_feedback.runtime_regressed =
            runtime_regressed(managed.execution, options_.product.safety.max_runtime_regression_ratio);
        if (options_.product.safety.enable_strategy_rollback) {
            auto fallback_workload = requested_workload;
            fallback_workload.partition_strategy = PartitionStrategy::auto_balanced;
            fallback_workload.heuristic_partition_hint.reset();
            fallback_workload.heuristic_partition_hint_confidence = 0.0;
            fallback_workload.heuristic_partition_hint_reason.clear();
            fallback_workload.disable_heuristic_partition_hint = true;
            fallback_workload.disable_automatic_execution_tuning = true;
            fallback_workload.disable_strategy_exploration = true;
            auto fallback_plan = planner_.build_plan(fallback_workload, devices_);
            auto fallback_optimization = execution_optimizer_.optimize(
                fallback_workload,
                fallback_plan,
                devices_,
                &workload_graph,
                &context.execution_tuning);
            auto fallback_memory = build_memory_preflight(fallback_optimization);
            if (fallback_memory.safe_to_run || !options_.product.memory.enforce_preflight) {
                auto fallback_execution = execute_with_feedback(fallback_workload, fallback_optimization, &workload_graph);
                if (fallback_execution.all_succeeded &&
                    (!managed.execution.all_succeeded ||
                     total_runtime_us(fallback_execution) <= total_runtime_us(managed.execution))) {
                    update_planner_diagnostics(fallback_plan);
                    managed.execution = std::move(fallback_execution);
                    managed.memory_preflight = std::move(fallback_memory);
                    optimization = std::move(fallback_optimization);
                    refresh_execution_reports(&managed.execution);
                    refresh_regression_summary();
                    managed.safety.planner_risk_score = planner_risk_score(
                        fallback_workload,
                        managed.planning,
                        optimization,
                        managed.memory_preflight,
                        managed.kernel_coverage,
                        managed.residency_sequence);
                    managed.safety.rolled_back_to_auto = true;
                    planner_feedback.rolled_back_to_auto = true;
                    managed.safety.final_strategy = managed.execution.optimization.partition_strategy;
                    if (safety_summary.tellp() > 0) {
                        safety_summary << "; ";
                    }
                    safety_summary << "rolled back to auto";
                }
            }
        }
    } else if (explicit_strategy) {
        record_strategy_success(requested_workload, managed.execution.optimization.partition_strategy);
    }

    if (managed.executed) {
        record_partition_strategy_feedback(planner_, requested_workload, devices_, planner_feedback);
    }

    if (managed.memory_preflight.summary.size() > 0) {
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << managed.memory_preflight.summary;
    }
    if (!context.meta_summary.empty()) {
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << "meta=" << context.meta_summary;
    }
    if (!managed.planning.strategy_reason.empty()) {
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << "planner=" << to_string(managed.planning.strategy_source)
                       << " conf=" << managed.planning.strategy_confidence
                       << " reason=" << managed.planning.strategy_reason;
    }
    if (!managed.residency_sequence.summary.empty()) {
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << managed.residency_sequence.summary;
    }
    if (!managed.tensor_allocator.summary.empty()) {
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << managed.tensor_allocator.summary;
    }
    if (!managed.spill_artifacts.summary.empty()) {
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << managed.spill_artifacts.summary;
    }
    if (!managed.backend_buffer_bindings.summary.empty()) {
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << managed.backend_buffer_bindings.summary;
    }
    if (!managed.executed_residency_movements.summary.empty()) {
        if (safety_summary.tellp() > 0) {
            safety_summary << "; ";
        }
        safety_summary << managed.executed_residency_movements.summary;
    }
    managed.safety.summary = safety_summary.str();
    persist_telemetry(requested_workload, managed);
    return managed;
}

ManagedExecutionReport Runtime::execute_manifest(const std::filesystem::path& manifest_path) {
    const auto manifest = load_workload_manifest(manifest_path);
    ManagedExecutionReport report;
    report.telemetry_path = telemetry_path();
    if (!manifest.assets.empty()) {
        const auto missing_required = std::find_if(
            manifest.assets.begin(),
            manifest.assets.end(),
            [](const WorkloadAsset& asset) {
                return asset.preload_required && (!asset.path.empty()) && !std::filesystem::exists(asset.path);
            });
        if (missing_required != manifest.assets.end()) {
            report.asset_prefetch.missing_required_assets = true;
            report.safety.summary = "required asset missing: " + missing_required->id;
            persist_telemetry(manifest.workload, report);
            return report;
        }
    }

    report = manifest.has_graph ? execute_managed(manifest.workload, manifest.graph) : execute_managed(manifest.workload);
    if (manifest.has_graph) {
        report.asset_prefetch = build_asset_prefetch(manifest, report.execution.optimization);
        materialize_packed_layout_blobs(options_, report.asset_prefetch);
        if (report.asset_prefetch.missing_required_assets && report.executed) {
            report.executed = false;
            report.safety.summary += report.safety.summary.empty() ? "" : "; ";
            report.safety.summary += report.asset_prefetch.summary;
        }
    }
    return report;
}

std::filesystem::path Runtime::telemetry_path() const {
    if (!options_.product.observability.telemetry_path.empty()) {
        return options_.product.observability.telemetry_path;
    }
    try {
        return std::filesystem::temp_directory_path() / "jakal_core_runtime_telemetry.tsv";
    } catch (const std::exception&) {
        return std::filesystem::path("jakal_core_runtime_telemetry.tsv");
    }
}

std::string Runtime::strategy_safety_key(
    const WorkloadSpec& workload,
    const PartitionStrategy strategy) const {
    std::ostringstream stream;
    stream << workload.name << '|'
           << to_string(workload.kind) << '|'
           << workload.dataset_tag << '|'
           << to_string(canonical_workload_phase(workload)) << '|'
           << canonical_workload_shape_bucket(workload) << '|'
           << to_string(strategy);
    return stream.str();
}

bool Runtime::is_strategy_blacklisted(
    const WorkloadSpec& workload,
    const PartitionStrategy strategy) const {
    if (strategy == PartitionStrategy::auto_balanced) {
        return false;
    }
    const auto it = strategy_blacklist_until_epoch_.find(strategy_safety_key(workload, strategy));
    return it != strategy_blacklist_until_epoch_.end() && it->second > execution_epoch_;
}

void Runtime::record_strategy_failure(const WorkloadSpec& workload, const PartitionStrategy strategy) {
    if (strategy == PartitionStrategy::auto_balanced) {
        return;
    }
    const auto key = strategy_safety_key(workload, strategy);
    const auto failures = ++strategy_failure_counts_[key];
    if (failures >= options_.product.safety.blacklist_after_failures) {
        strategy_blacklist_until_epoch_[key] = execution_epoch_ + options_.product.safety.blacklist_cooldown_epochs;
        strategy_failure_counts_[key] = 0u;
    }
}

void Runtime::record_strategy_success(const WorkloadSpec& workload, const PartitionStrategy strategy) {
    if (strategy == PartitionStrategy::auto_balanced) {
        return;
    }
    const auto key = strategy_safety_key(workload, strategy);
    strategy_failure_counts_.erase(key);
    strategy_blacklist_until_epoch_.erase(key);
}

MemoryPreflightReport Runtime::build_memory_preflight(const OptimizationReport& optimization) const {
    MemoryPreflightReport report;
    if (devices_.empty()) {
        report.safe_to_run = false;
        report.summary = "no devices available";
        return report;
    }

    std::unordered_map<std::string, DeviceMemoryReservation> reservations;
    reservations.reserve(devices_.size());
    for (const auto& graph : devices_) {
        DeviceMemoryReservation reservation;
        reservation.device_uid = graph.uid;
        reservation.host = graph.probe == "host";
        reservation.effective_capacity_bytes = effective_capacity_bytes(graph, options_.product.memory);
        reservations.emplace(graph.uid, reservation);
    }

    std::unordered_map<std::string, std::uint64_t> persistent_seen;
    std::unordered_map<std::string, std::uint64_t> transient_seen;
    for (const auto& result : optimization.operations) {
        for (const auto& entry : result.graph.residency_plan) {
            const auto key = entry.device_uid + "|" + entry.tensor_id;
            auto& target = entry.persistent ? persistent_seen[key] : transient_seen[key];
            target = std::max(target, entry.bytes);
        }
    }

    for (const auto& [key, bytes] : persistent_seen) {
        const auto delimiter = key.find('|');
        const auto device_uid = key.substr(0u, delimiter);
        if (const auto it = reservations.find(device_uid); it != reservations.end()) {
            it->second.persistent_bytes += bytes;
        }
        report.aggregate_persistent_bytes += bytes;
    }

    for (const auto& [key, bytes] : transient_seen) {
        const auto delimiter = key.find('|');
        const auto device_uid = key.substr(0u, delimiter);
        if (const auto it = reservations.find(device_uid); it != reservations.end()) {
            it->second.transient_bytes += bytes;
        }
        report.aggregate_transient_bytes += bytes;
    }

    if (!optimization.placement.allocations.empty()) {
        for (const auto& allocation : optimization.placement.allocations) {
            if (const auto it = reservations.find(allocation.device.uid); it != reservations.end()) {
                it->second.transient_bytes += static_cast<std::uint64_t>(
                    std::round(static_cast<double>(optimization.workload_working_set_bytes) * allocation.ratio));
                it->second.transient_bytes += static_cast<std::uint64_t>(
                    std::round(static_cast<double>(optimization.workload_host_exchange_bytes) * allocation.ratio * 0.5));
            }
        }
        report.aggregate_transient_bytes += optimization.workload_working_set_bytes;
        report.aggregate_transient_bytes += optimization.workload_host_exchange_bytes / 2u;
    }

    report.pinned_host_visible_bytes = std::accumulate(
        optimization.workload_graph.tensors.begin(),
        optimization.workload_graph.tensors.end(),
        std::uint64_t{0},
        [](const std::uint64_t total, const WorkloadTensor& tensor) {
            return total + (tensor.host_visible ? tensor.bytes : 0ull);
        });

    std::ostringstream summary;
    for (auto& [device_uid, reservation] : reservations) {
        reservation.reserved_bytes = reservation.persistent_bytes + reservation.transient_bytes;
        reservation.pressure_ratio =
            reservation.effective_capacity_bytes == 0u
                ? (reservation.reserved_bytes > 0u ? 1.0 : 0.0)
                : static_cast<double>(reservation.reserved_bytes) /
                      static_cast<double>(std::max<std::uint64_t>(1u, reservation.effective_capacity_bytes));
        report.peak_pressure_ratio = std::max(report.peak_pressure_ratio, reservation.pressure_ratio);
        if (reservation.pressure_ratio > options_.product.memory.max_pressure_ratio) {
            report.requires_spill = true;
            if (reservation.host || !options_.product.memory.allow_host_spill || reservation.pressure_ratio > 1.15) {
                report.safe_to_run = false;
            }
        }
        report.devices.push_back(reservation);
        if (reservation.pressure_ratio > options_.product.memory.max_pressure_ratio) {
            if (summary.tellp() > 0) {
                summary << "; ";
            }
            summary << device_uid << " pressure=" << reservation.pressure_ratio;
        }
    }

    if (!report.safe_to_run && report.devices.empty()) {
        report.summary = "no devices available for memory preflight";
    } else if (report.safe_to_run && report.requires_spill) {
        report.summary = "memory spill expected";
    } else if (!report.safe_to_run && summary.tellp() > 0) {
        report.summary = summary.str();
    }
    return report;
}

ResidencySequenceReport Runtime::build_residency_sequence(const OptimizationReport& optimization) const {
    struct SequencedTensor {
        std::string tensor_id;
        std::string device_uid;
        std::uint32_t tensor_index = kInvalidExecutionIndex;
        std::uint32_t device_index = kInvalidExecutionIndex;
        std::uint64_t bytes = 0;
        std::uint32_t first = 0;
        std::uint32_t last = 0;
        bool persistent = false;
    };

    ResidencySequenceReport report;
    if (optimization.workload_graph.operations.empty()) {
        return report;
    }
    const auto compiled_workload = compile_workload_graph(optimization.workload_graph);
    report.indexed_tensors.reserve(optimization.workload_graph.tensors.size());
    for (const auto& tensor : optimization.workload_graph.tensors) {
        report.indexed_tensors.push_back(tensor.id);
    }
    report.indexed_operations.reserve(optimization.workload_graph.operations.size());
    for (const auto& operation : optimization.workload_graph.operations) {
        report.indexed_operations.push_back(operation.name);
    }

    std::unordered_map<std::string, TensorLifetime> lifetimes_by_id;
    lifetimes_by_id.reserve(optimization.workload_graph.lifetimes.size());
    for (const auto& lifetime : optimization.workload_graph.lifetimes) {
        lifetimes_by_id.emplace(lifetime.tensor_id, lifetime);
    }

    std::unordered_map<std::string, std::uint32_t> device_indices;
    auto device_index_for = [&](const std::string& device_uid) {
        if (const auto it = device_indices.find(device_uid); it != device_indices.end()) {
            return it->second;
        }
        const auto index = static_cast<std::uint32_t>(report.indexed_devices.size());
        report.indexed_devices.push_back(device_uid);
        device_indices.emplace(device_uid, index);
        return index;
    };

    std::unordered_map<std::string, std::vector<SequencedTensor>> tensors_by_device;
    for (const auto& optimized : optimization.operations) {
        for (const auto& entry : optimized.graph.residency_plan) {
            auto& device_tensors = tensors_by_device[entry.device_uid];
            const auto duplicate = std::find_if(device_tensors.begin(), device_tensors.end(), [&](const SequencedTensor& tensor) {
                return tensor.tensor_index == entry.tensor_index;
            });
            const auto lifetime_it = lifetimes_by_id.find(entry.tensor_id);
            const auto first = lifetime_it == lifetimes_by_id.end() ? 0u : lifetime_it->second.first_operation_index;
            const auto last = lifetime_it == lifetimes_by_id.end()
                                  ? static_cast<std::uint32_t>(optimization.workload_graph.operations.size() - 1u)
                                  : lifetime_it->second.last_operation_index;
            if (duplicate == device_tensors.end()) {
                device_tensors.push_back(SequencedTensor{
                    entry.tensor_id,
                    entry.device_uid,
                    entry.tensor_index,
                    device_index_for(entry.device_uid),
                    entry.bytes,
                    first,
                    last,
                    entry.persistent});
            } else {
                duplicate->bytes = std::max(duplicate->bytes, entry.bytes);
                duplicate->first = std::min(duplicate->first, first);
                duplicate->last = std::max(duplicate->last, last);
                duplicate->persistent = duplicate->persistent || entry.persistent;
            }
        }
    }

    std::ostringstream summary;
    for (const auto& [device_uid, tensors] : tensors_by_device) {
        const auto* graph = find_graph_by_uid(devices_, device_uid);
        const auto effective_capacity =
            graph == nullptr ? 0u : effective_capacity_bytes(*graph, options_.product.memory);
        const auto safe_capacity = static_cast<std::uint64_t>(
            static_cast<double>(effective_capacity) * options_.product.memory.max_pressure_ratio);

        std::unordered_map<std::string, SequencedTensor> live_tensors;
        std::unordered_set<std::string> spilled_tensors;
        std::uint64_t live_bytes = 0u;

        for (std::uint32_t operation_index = 0u;
             operation_index < optimization.workload_graph.operations.size();
             ++operation_index) {
            const auto& operation = optimization.workload_graph.operations[operation_index];
            std::unordered_set<std::uint32_t> current_needed_indices;
            if (operation_index < compiled_workload.operations.size()) {
                const auto& compiled_operation = compiled_workload.operations[operation_index];
                current_needed_indices.insert(
                    compiled_operation.input_tensor_indices.begin(),
                    compiled_operation.input_tensor_indices.end());
                current_needed_indices.insert(
                    compiled_operation.output_tensor_indices.begin(),
                    compiled_operation.output_tensor_indices.end());
                current_needed_indices.insert(
                    compiled_operation.temporary_tensor_indices.begin(),
                    compiled_operation.temporary_tensor_indices.end());
            }

            for (const auto& tensor : tensors) {
                if (tensor.first > operation_index || tensor.last < operation_index) {
                    continue;
                }
                if (live_tensors.find(tensor.tensor_id) != live_tensors.end()) {
                    continue;
                }
                const bool spilled = spilled_tensors.find(tensor.tensor_id) != spilled_tensors.end();
                const bool needed_now = current_needed_indices.find(tensor.tensor_index) != current_needed_indices.end() ||
                                        tensor.first == operation_index || tensor.persistent;
                if (!needed_now) {
                    continue;
                }
                report.actions.push_back(ResidencyAction{
                    spilled ? ResidencyActionKind::reload : ResidencyActionKind::prefetch,
                    tensor.tensor_id,
                    device_uid,
                    operation.name,
                    tensor.tensor_index,
                    tensor.device_index,
                    operation_index,
                    tensor.bytes,
                    tensor.persistent});
                live_tensors.emplace(tensor.tensor_id, tensor);
                live_bytes += tensor.bytes;
                if (spilled) {
                    report.reload_bytes += tensor.bytes;
                    spilled_tensors.erase(tensor.tensor_id);
                }
            }

            report.peak_live_bytes = std::max(report.peak_live_bytes, live_bytes);
            while (safe_capacity > 0u && live_bytes > safe_capacity) {
                auto candidate_it = live_tensors.end();
                for (auto it = live_tensors.begin(); it != live_tensors.end(); ++it) {
                    if (it->second.persistent ||
                        current_needed_indices.find(it->second.tensor_index) != current_needed_indices.end()) {
                        continue;
                    }
                    if (candidate_it == live_tensors.end() || it->second.last > candidate_it->second.last) {
                        candidate_it = it;
                    }
                }
                if (candidate_it == live_tensors.end()) {
                    ++report.forced_spill_count;
                    report.viable_without_spill = false;
                    break;
                }
                report.actions.push_back(ResidencyAction{
                    ResidencyActionKind::spill,
                    candidate_it->second.tensor_id,
                    device_uid,
                    operation.name,
                    candidate_it->second.tensor_index,
                    candidate_it->second.device_index,
                    operation_index,
                    candidate_it->second.bytes,
                    candidate_it->second.persistent});
                report.spill_bytes += candidate_it->second.bytes;
                live_bytes -= candidate_it->second.bytes;
                spilled_tensors.insert(candidate_it->second.tensor_id);
                live_tensors.erase(candidate_it);
            }

            for (auto it = live_tensors.begin(); it != live_tensors.end();) {
                if (!it->second.persistent && it->second.last == operation_index) {
                    report.actions.push_back(ResidencyAction{
                        ResidencyActionKind::evict,
                        it->second.tensor_id,
                        device_uid,
                        operation.name,
                        it->second.tensor_index,
                        it->second.device_index,
                        operation_index,
                        it->second.bytes,
                        false});
                    live_bytes -= it->second.bytes;
                    it = live_tensors.erase(it);
                } else {
                    ++it;
                }
            }
        }

        if (report.forced_spill_count > 0u) {
            if (summary.tellp() > 0) {
                summary << "; ";
            }
            summary << device_uid << " forced_spills=" << report.forced_spill_count;
        }
    }

    if (report.spill_bytes > 0u && summary.tellp() == 0) {
        summary << "spill=" << report.spill_bytes << " reload=" << report.reload_bytes;
    }
    report.summary = summary.str();
    return report;
}

TensorAllocatorReport Runtime::build_tensor_allocator(const ResidencySequenceReport& residency_sequence) const {
    struct FreeBlock {
        std::uint32_t block_id = 0u;
        std::uint64_t offset_bytes = 0u;
        std::uint64_t bytes = 0u;
    };
    struct LiveBlock {
        std::uint32_t block_id = 0u;
        std::uint64_t offset_bytes = 0u;
        std::uint64_t bytes = 0u;
        bool persistent = false;
    };
    struct DeviceAllocatorState {
        std::vector<FreeBlock> free_blocks;
        std::unordered_map<std::string, LiveBlock> live_blocks;
        std::uint64_t next_offset_bytes = 0u;
        std::uint64_t live_bytes = 0u;
        std::uint32_t next_block_id = 1u;
    };

    TensorAllocatorReport report;
    report.indexed_tensors = residency_sequence.indexed_tensors;
    report.indexed_devices = residency_sequence.indexed_devices;
    report.indexed_operations = residency_sequence.indexed_operations;
    if (residency_sequence.actions.empty()) {
        return report;
    }

    std::unordered_map<std::string, DeviceAllocatorState> states;
    const auto coalesce_free_blocks = [](std::vector<FreeBlock>& free_blocks) {
        if (free_blocks.size() < 2u) {
            return;
        }
        std::sort(free_blocks.begin(), free_blocks.end(), [](const FreeBlock& left, const FreeBlock& right) {
            if (left.offset_bytes != right.offset_bytes) {
                return left.offset_bytes < right.offset_bytes;
            }
            return left.block_id < right.block_id;
        });
        std::vector<FreeBlock> merged;
        merged.reserve(free_blocks.size());
        merged.push_back(free_blocks.front());
        for (std::size_t index = 1; index < free_blocks.size(); ++index) {
            auto& current = merged.back();
            const auto& next = free_blocks[index];
            if (current.offset_bytes + current.bytes == next.offset_bytes) {
                current.bytes += next.bytes;
                current.block_id = std::min(current.block_id, next.block_id);
                continue;
            }
            merged.push_back(next);
        }
        free_blocks = std::move(merged);
    };

    for (const auto& action : residency_sequence.actions) {
        auto& state = states[action.device_uid];
        switch (action.kind) {
        case ResidencyActionKind::prefetch:
        case ResidencyActionKind::reload: {
            if (state.live_blocks.find(action.tensor_id) != state.live_blocks.end()) {
                break;
            }

            auto best_fit = state.free_blocks.end();
            for (auto it = state.free_blocks.begin(); it != state.free_blocks.end(); ++it) {
                if (it->bytes < action.bytes) {
                    continue;
                }
                if (best_fit == state.free_blocks.end() || it->bytes < best_fit->bytes) {
                    best_fit = it;
                }
            }

            TensorAllocationEvent event;
            event.kind =
                best_fit == state.free_blocks.end() ? TensorAllocationEventKind::allocate : TensorAllocationEventKind::reuse;
            event.residency_cause = action.kind;
            event.tensor_id = action.tensor_id;
            event.device_uid = action.device_uid;
            event.trigger_operation_name = action.trigger_operation_name;
            event.tensor_index = action.tensor_index;
            event.device_index = action.device_index;
            event.operation_index = action.operation_index;
            event.bytes = action.bytes;
            event.persistent = action.persistent;

            if (best_fit == state.free_blocks.end()) {
                event.block_id = state.next_block_id++;
                event.offset_bytes = state.next_offset_bytes;
                state.next_offset_bytes += action.bytes;
                report.peak_reserved_bytes = std::max(report.peak_reserved_bytes, state.next_offset_bytes);
                ++report.allocation_count;
            } else {
                const auto reused = *best_fit;
                state.free_blocks.erase(best_fit);
                event.block_id = reused.block_id;
                event.offset_bytes = reused.offset_bytes;
                if (reused.bytes > action.bytes) {
                    state.free_blocks.push_back(FreeBlock{
                        state.next_block_id++,
                        reused.offset_bytes + action.bytes,
                        reused.bytes - action.bytes});
                    coalesce_free_blocks(state.free_blocks);
                }
                ++report.reuse_count;
            }

            state.live_blocks.emplace(
                action.tensor_id,
                LiveBlock{event.block_id, event.offset_bytes, action.bytes, action.persistent});
            state.live_bytes += action.bytes;
            report.peak_live_bytes = std::max(report.peak_live_bytes, state.live_bytes);
            report.events.push_back(std::move(event));
            break;
        }
        case ResidencyActionKind::spill:
        case ResidencyActionKind::evict: {
            const auto live_it = state.live_blocks.find(action.tensor_id);
            if (live_it == state.live_blocks.end()) {
                break;
            }

            report.events.push_back(TensorAllocationEvent{
                TensorAllocationEventKind::release,
                action.kind,
                action.tensor_id,
                action.device_uid,
                action.trigger_operation_name,
                action.tensor_index,
                action.device_index,
                action.operation_index,
                live_it->second.block_id,
                live_it->second.offset_bytes,
                live_it->second.bytes,
                live_it->second.persistent});
            state.live_bytes -= live_it->second.bytes;
            state.free_blocks.push_back(FreeBlock{
                live_it->second.block_id,
                live_it->second.offset_bytes,
                live_it->second.bytes});
            state.live_blocks.erase(live_it);
            coalesce_free_blocks(state.free_blocks);
            break;
        }
        }
    }

    if (!report.events.empty()) {
        std::ostringstream summary;
        summary << "allocator peak_live=" << report.peak_live_bytes
                << " peak_reserved=" << report.peak_reserved_bytes
                << " allocations=" << report.allocation_count
                << " reuse=" << report.reuse_count;
        report.summary = summary.str();
    }
    return report;
}

SpillArtifactReport Runtime::materialize_spill_artifacts(const ResidencySequenceReport& residency_sequence) const {
    SpillArtifactReport report;
    if (residency_sequence.actions.empty()) {
        return report;
    }

    const auto root = runtime_spill_artifact_root(install_paths_);
    std::error_code ec;
    std::filesystem::create_directories(root, ec);

    std::unordered_map<std::string, std::filesystem::path> latest_spill_by_tensor_device;
    latest_spill_by_tensor_device.reserve(residency_sequence.actions.size());
    const auto tensor_device_key = [](const ResidencyAction& action) {
        return action.tensor_id + "|" + action.device_uid;
    };

    for (const auto& action : residency_sequence.actions) {
        if (action.kind != ResidencyActionKind::spill && action.kind != ResidencyActionKind::reload) {
            continue;
        }

        SpillArtifactEntry entry;
        entry.kind = action.kind;
        entry.tensor_id = action.tensor_id;
        entry.device_uid = action.device_uid;
        entry.trigger_operation_name = action.trigger_operation_name;
        entry.tensor_index = action.tensor_index;
        entry.device_index = action.device_index;
        entry.operation_index = action.operation_index;
        entry.bytes = action.bytes;

        const auto key = tensor_device_key(action);
        if (action.kind == ResidencyActionKind::spill) {
            const auto path = spill_artifact_path(root, action);
            if (spill_artifact_matches(path, action) || write_spill_artifact(path, action)) {
                entry.path = path;
                entry.exists_on_disk = true;
                report.materialized_spill_bytes += action.bytes;
                latest_spill_by_tensor_device[key] = path;
            }
        } else if (const auto it = latest_spill_by_tensor_device.find(key); it != latest_spill_by_tensor_device.end()) {
            entry.path = it->second;
            entry.exists_on_disk = std::filesystem::exists(entry.path, ec);
            if (entry.exists_on_disk) {
                report.materialized_reload_bytes += action.bytes;
            }
        }
        report.entries.push_back(std::move(entry));
    }

    if (!report.entries.empty()) {
        std::ostringstream summary;
        summary << "spill_artifacts=" << report.materialized_spill_bytes
                << " reload_refs=" << report.materialized_reload_bytes;
        report.summary = summary.str();
    }
    return report;
}

BackendBufferBindingReport Runtime::build_backend_buffer_bindings(
    const DirectExecutionReport* execution,
    const TensorAllocatorReport& tensor_allocator,
    const ResidencySequenceReport& residency_sequence,
    const SpillArtifactReport& spill_artifacts) const {
    BackendBufferBindingReport report;
    std::unordered_map<std::string, BackendBufferBindingEntry> entries_by_key;
    entries_by_key.reserve(std::max<std::size_t>(residency_sequence.indexed_devices.size(), 1u));

    const auto make_key = [](const std::string& device_uid, const std::string& pool_id) {
        return device_uid + "|" + (pool_id.empty() ? std::string("runtime-local") : pool_id);
    };

    const auto ensure_entry = [&](const std::string& device_uid, const std::string& pool_id = std::string()) -> BackendBufferBindingEntry& {
        const auto key = make_key(device_uid, pool_id);
        auto [it, inserted] = entries_by_key.try_emplace(key);
        auto& entry = it->second;
        if (inserted) {
            entry.device_uid = device_uid;
            entry.pool_id = pool_id;
            if (const auto* graph = find_graph_by_uid(devices_, device_uid); graph != nullptr) {
                entry.backend_name = backend_name_for_graph(*graph);
                entry.ownership_scope = graph->probe == "host" ? "host-shared" : "runtime-local";
            } else {
                entry.backend_name = "runtime-local";
            }
            if (entry.pool_id.empty()) {
                entry.pool_id = "runtime-local:" + device_uid;
            }
        }
        return entry;
    };

    for (const auto& device_uid : residency_sequence.indexed_devices) {
        ensure_entry(device_uid);
    }

    std::unordered_map<std::string, std::uint64_t> live_bytes_by_device;
    for (const auto& event : tensor_allocator.events) {
        auto& entry = ensure_entry(event.device_uid);
        auto& live_bytes = live_bytes_by_device[event.device_uid];
        switch (event.kind) {
        case TensorAllocationEventKind::allocate:
        case TensorAllocationEventKind::reuse:
            live_bytes += event.bytes;
            entry.planned_peak_bytes = std::max(entry.planned_peak_bytes, live_bytes);
            entry.reserved_bytes = std::max(entry.reserved_bytes, entry.planned_peak_bytes);
            break;
        case TensorAllocationEventKind::release:
            live_bytes = live_bytes > event.bytes ? (live_bytes - event.bytes) : 0u;
            break;
        }
    }

    for (const auto& action : residency_sequence.actions) {
        auto& entry = ensure_entry(action.device_uid);
        if (action.kind == ResidencyActionKind::spill) {
            entry.spill_bytes += action.bytes;
        } else if (action.kind == ResidencyActionKind::reload) {
            entry.reload_bytes += action.bytes;
        }
    }

    for (const auto& artifact : spill_artifacts.entries) {
        auto& entry = ensure_entry(artifact.device_uid);
        if (artifact.kind == ResidencyActionKind::spill && artifact.exists_on_disk) {
            entry.materialized_spill_bytes += artifact.bytes;
            entry.uses_runtime_spill_artifacts = true;
        } else if (artifact.kind == ResidencyActionKind::reload && artifact.exists_on_disk) {
            entry.materialized_reload_bytes += artifact.bytes;
            entry.uses_runtime_spill_artifacts = true;
        }
    }

    if (execution != nullptr) {
        for (const auto& operation : execution->operations) {
            for (const auto& device_uid : operation.participating_devices) {
                auto& entry = ensure_entry(device_uid);
                entry.direct_execution_operation_count += 1u;
                entry.direct_execution_active = true;
            }
            for (const auto& binding : operation.buffer_pool_bindings) {
                auto& entry = ensure_entry(binding.device_uid, binding.pool_id);
                entry.backend_name = binding.backend_name;
                entry.ownership_scope = binding.backend_name.rfind("host", 0) == 0 ? "host-shared" : "backend-owned";
                entry.resource_tag = binding.resource_tag;
                entry.reserved_bytes = std::max(entry.reserved_bytes, binding.reserved_bytes);
                entry.persistent_resource_reuse_hits =
                    std::max(entry.persistent_resource_reuse_hits, binding.reuse_hits);
                entry.direct_execution_operation_count += 1u;
                entry.direct_execution_active = true;
                auto& fallback_entry = ensure_entry(binding.device_uid);
                fallback_entry.direct_execution_active = true;
                fallback_entry.direct_execution_operation_count += 1u;
            }
            if (operation.buffer_pool_bindings.empty() && operation.backend_name.rfind("host-native", 0) != 0) {
                for (const auto& device_uid : operation.participating_devices) {
                    auto& entry = ensure_entry(device_uid);
                    entry.persistent_resource_reuse_hits += operation.persistent_resource_reuse_hits;
                    if (entry.ownership_scope != "host-shared") {
                        entry.ownership_scope = "backend-owned";
                    }
                }
            }
        }
    }

    std::vector<std::string> ordered_keys;
    ordered_keys.reserve(entries_by_key.size());
    for (const auto& [key, _] : entries_by_key) {
        ordered_keys.push_back(key);
    }
    std::sort(ordered_keys.begin(), ordered_keys.end());

    std::ostringstream summary;
    for (const auto& key : ordered_keys) {
        auto entry = std::move(entries_by_key.at(key));
        if (entry.ownership_scope == "backend-owned") {
            report.backend_owned_peak_bytes += std::max(entry.reserved_bytes, entry.planned_peak_bytes);
        } else {
            report.runtime_local_peak_bytes += std::max(entry.reserved_bytes, entry.planned_peak_bytes);
        }
        report.total_persistent_resource_reuse_hits += entry.persistent_resource_reuse_hits;
        if (summary.tellp() > 0) {
            summary << "; ";
        }
        summary << entry.device_uid
                << " owner=" << entry.ownership_scope
                << " reserved=" << entry.reserved_bytes
                << " reuse=" << entry.persistent_resource_reuse_hits;
        if (!entry.pool_id.empty()) {
            summary << " pool=" << entry.pool_id;
        }
        if (entry.uses_runtime_spill_artifacts) {
            summary << " spill-artifacts";
        }
        report.entries.push_back(std::move(entry));
    }
    report.summary = summary.str();
    return report;
}

ExecutedResidencyMovementReport Runtime::build_executed_residency_movements(
    const DirectExecutionReport* execution,
    const ResidencySequenceReport& residency_sequence,
    const SpillArtifactReport& spill_artifacts) const {
    ExecutedResidencyMovementReport report;
    if (execution == nullptr) {
        return report;
    }

    std::unordered_map<std::string, std::uint32_t> operation_index_by_name;
    operation_index_by_name.reserve(residency_sequence.indexed_operations.size());
    for (std::uint32_t index = 0u; index < residency_sequence.indexed_operations.size(); ++index) {
        operation_index_by_name.emplace(residency_sequence.indexed_operations[index], index);
    }

    std::vector<bool> action_consumed(residency_sequence.actions.size(), false);
    const auto find_matching_action = [&](const std::string& operation_name,
                                          const std::string& device_uid,
                                          const std::string& movement_kind) -> const ResidencyAction* {
        for (std::size_t index = 0; index < residency_sequence.actions.size(); ++index) {
            const auto& action = residency_sequence.actions[index];
            if (action_consumed[index] ||
                action.trigger_operation_name != operation_name ||
                action.device_uid != device_uid) {
                continue;
            }
            const bool h2d_match =
                movement_kind == "h2d" &&
                (action.kind == ResidencyActionKind::prefetch || action.kind == ResidencyActionKind::reload);
            const bool d2h_match =
                movement_kind == "d2h" &&
                (action.kind == ResidencyActionKind::spill || action.kind == ResidencyActionKind::evict);
            if (!h2d_match && !d2h_match) {
                continue;
            }
            action_consumed[index] = true;
            return &action;
        }
        return nullptr;
    };

    const auto find_spill_artifact = [&](const ResidencyAction* action) -> const SpillArtifactEntry* {
        if (action == nullptr) {
            return nullptr;
        }
        return std::find_if(
            spill_artifacts.entries.begin(),
            spill_artifacts.entries.end(),
            [&](const SpillArtifactEntry& artifact) {
                return artifact.kind == action->kind &&
                       artifact.device_uid == action->device_uid &&
                       artifact.trigger_operation_name == action->trigger_operation_name &&
                       artifact.tensor_id == action->tensor_id &&
                       artifact.exists_on_disk;
            }) == spill_artifacts.entries.end()
                   ? nullptr
                   : &(*std::find_if(
                         spill_artifacts.entries.begin(),
                         spill_artifacts.entries.end(),
                         [&](const SpillArtifactEntry& artifact) {
                             return artifact.kind == action->kind &&
                                    artifact.device_uid == action->device_uid &&
                                    artifact.trigger_operation_name == action->trigger_operation_name &&
                                    artifact.tensor_id == action->tensor_id &&
                                    artifact.exists_on_disk;
                         }));
    };

    for (const auto& operation : execution->operations) {
        const auto operation_index_it = operation_index_by_name.find(operation.operation_name);
        const auto operation_index =
            operation_index_it == operation_index_by_name.end() ? 0u : operation_index_it->second;
        for (const auto& transfer : operation.transfer_records) {
            auto entry = ExecutedResidencyMovementEntry{};
            entry.kind = transfer.movement_kind;
            entry.device_uid = transfer.device_uid;
            entry.backend_name = transfer.backend_name;
            entry.trigger_operation_name = operation.operation_name;
            entry.pool_id = transfer.pool_id;
            entry.operation_index = operation_index;
            entry.bytes = transfer.bytes;
            entry.runtime_us = transfer.runtime_us;
            entry.from_direct_execution = true;

            const auto* action = find_matching_action(operation.operation_name, transfer.device_uid, transfer.movement_kind);
            if (action != nullptr) {
                switch (action->kind) {
                case ResidencyActionKind::prefetch:
                    entry.kind = "prefetch";
                    report.executed_h2d_bytes += transfer.bytes;
                    break;
                case ResidencyActionKind::reload:
                    entry.kind = "reload";
                    report.executed_reload_bytes += transfer.bytes;
                    break;
                case ResidencyActionKind::spill:
                    entry.kind = "spill";
                    report.executed_spill_bytes += transfer.bytes;
                    break;
                case ResidencyActionKind::evict:
                    entry.kind = "evict";
                    report.executed_d2h_bytes += transfer.bytes;
                    break;
                }
                entry.tensor_id = action->tensor_id;
                if (const auto* artifact = find_spill_artifact(action); artifact != nullptr) {
                    entry.spill_artifact_path = artifact->path;
                    entry.from_spill_artifact = true;
                }
            } else if (transfer.movement_kind == "h2d") {
                report.executed_h2d_bytes += transfer.bytes;
            } else if (transfer.movement_kind == "d2h") {
                report.executed_d2h_bytes += transfer.bytes;
            }

            report.total_transfer_runtime_us += transfer.runtime_us;
            report.entries.push_back(std::move(entry));
        }
    }

    if (!report.entries.empty()) {
        std::ostringstream summary;
        summary << "executed_h2d=" << report.executed_h2d_bytes
                << " executed_d2h=" << report.executed_d2h_bytes
                << " executed_spill=" << report.executed_spill_bytes
                << " executed_reload=" << report.executed_reload_bytes
                << " transfer_us=" << report.total_transfer_runtime_us;
        report.summary = summary.str();
    }
    return report;
}

KernelCoverageReport Runtime::build_kernel_coverage(const OptimizationReport& optimization) const {
    KernelCoverageReport report;
    std::ostringstream summary;
    for (const auto& optimized : optimization.operations) {
        const auto* graph = find_graph_by_uid(devices_, optimized.config.primary_device_uid);
        if (graph == nullptr) {
            continue;
        }
        std::string reason;
        if (!backend_supports_operation(*graph, optimized.operation.op_class, &reason)) {
            report.all_supported = false;
            report.issues.push_back(KernelCoverageIssue{
                optimized.operation.name,
                graph->uid,
                backend_name_for_graph(*graph),
                optimized.operation.op_class,
                false,
                reason});
            if (summary.tellp() > 0) {
                summary << "; ";
            }
            summary << optimized.operation.name << '@' << graph->probe << ": " << reason;
        }
    }
    report.summary = summary.str();
    return report;
}

AssetPrefetchReport Runtime::build_asset_prefetch(
    const WorkloadManifest& manifest,
    const OptimizationReport& optimization) const {
    const auto host_graph_it = std::find_if(devices_.begin(), devices_.end(), [](const HardwareGraph& graph) {
        return graph.probe == "host";
    });
    const std::string canonical_host_uid = host_graph_it == devices_.end() ? std::string() : host_graph_it->uid;
    const auto canonicalize_device_uid = [&](std::string device_uid) {
        if ((device_uid == "host" || device_uid.rfind("host:", 0) == 0) && !canonical_host_uid.empty()) {
            return canonical_host_uid;
        }
        return device_uid;
    };
    const auto queue_hint_for_asset = [&](const WorkloadAsset& asset, const std::string& device_uid) {
        const auto canonical_device_uid = canonicalize_device_uid(device_uid);
        const bool target_is_host =
            canonical_device_uid.empty() ? false : (canonical_device_uid == canonical_host_uid);
        if (canonical_device_uid.empty() || target_is_host || asset.preferred_residency == "host") {
            return std::string("host_io");
        }
        if (asset.preferred_residency == "device" || asset.preferred_residency == "accelerator") {
            return std::string("host_to_device");
        }
        return std::string("host_to_device");
    };

    AssetPrefetchReport report;
    if (manifest.assets.empty()) {
        return report;
    }

    std::unordered_set<std::string> seen_prefetch_entries;
    const auto append_prefetch_entry = [&](AssetPrefetchEntry entry) {
        std::ostringstream key;
        const bool collapse_host_resident_raw_asset =
            !entry.derived_cache && entry.target_residency == "host";
        key << entry.asset_id << '\n'
            << entry.source_asset_id << '\n'
            << entry.path.string() << '\n'
            << entry.tensor_id << '\n'
            << (collapse_host_resident_raw_asset ? std::string("host-resident") : entry.device_uid) << '\n'
            << entry.file_offset << '\n'
            << entry.bytes << '\n'
            << entry.queue_hint << '\n'
            << entry.target_residency << '\n'
            << entry.materialization_kind << '\n'
            << entry.backend_hint << '\n'
            << entry.backend_cache_tag << '\n'
            << entry.derived_cache;
        if (!seen_prefetch_entries.insert(key.str()).second) {
            return;
        }
        report.total_prefetch_bytes += entry.bytes;
        if (entry.derived_cache) {
            report.total_layout_cache_bytes += entry.bytes;
        }
        if (entry.queue_hint == "host_to_device") {
            report.total_host_to_device_bytes += entry.bytes;
        } else {
            report.total_host_io_bytes += entry.bytes;
        }
        report.entries.push_back(std::move(entry));
    };

    std::unordered_map<std::string, std::vector<std::string>> tensor_devices;
    for (const auto& optimized : optimization.operations) {
        for (const auto& entry : optimized.graph.residency_plan) {
            if (!entry.persistent) {
                continue;
            }
            auto& devices = tensor_devices[entry.tensor_id];
            const auto canonical_device_uid = canonicalize_device_uid(entry.device_uid);
            if (std::find(devices.begin(), devices.end(), canonical_device_uid) == devices.end()) {
                devices.push_back(canonical_device_uid);
            }
        }
    }

    std::unordered_map<std::string, std::vector<const WorkloadAsset*>> assets_by_tensor;
    for (const auto& asset : manifest.assets) {
        for (const auto& tensor_id : asset.tensor_ids) {
            assets_by_tensor[tensor_id].push_back(&asset);
        }
    }

    std::ostringstream summary;
    for (const auto& asset : manifest.assets) {
        const bool exists = asset.path.empty() ? false : std::filesystem::exists(asset.path);
        if (asset.preload_required && !exists) {
            report.missing_required_assets = true;
            if (summary.tellp() > 0) {
                summary << "; ";
            }
            summary << "missing asset " << asset.id;
        }
        if (asset.tensor_ids.empty()) {
            const auto queue_hint = queue_hint_for_asset(asset, std::string());
            append_prefetch_entry(AssetPrefetchEntry{
                asset.id,
                asset.id,
                asset.path,
                std::string(),
                std::string(),
                asset.file_offset,
                asset.bytes,
                queue_hint,
                asset.preferred_residency,
                "raw",
                "any",
                "raw",
                exists,
                asset.preload_required,
                asset.persistent,
                asset.host_visible,
                !asset.host_visible || queue_hint != "host_io",
                false});
            continue;
        }
        const auto per_tensor_bytes = asset.bytes == 0u
                                          ? 0u
                                          : std::max<std::uint64_t>(
                                                1u,
                                                asset.bytes / static_cast<std::uint64_t>(asset.tensor_ids.size()));
        for (const auto& tensor_id : asset.tensor_ids) {
            if (asset.preferred_residency == "host") {
                const auto queue_hint = queue_hint_for_asset(asset, canonical_host_uid);
                append_prefetch_entry(AssetPrefetchEntry{
                    asset.id,
                    asset.id,
                    asset.path,
                    tensor_id,
                    canonical_host_uid,
                    asset.file_offset,
                    per_tensor_bytes,
                    queue_hint,
                    asset.preferred_residency,
                    "raw",
                    "any",
                    "raw",
                    exists,
                    asset.preload_required,
                    asset.persistent,
                    asset.host_visible,
                    !asset.host_visible || queue_hint != "host_io",
                    false});
                continue;
            }
            const auto devices_it = tensor_devices.find(tensor_id);
            if (devices_it == tensor_devices.end() || devices_it->second.empty()) {
                const auto queue_hint = queue_hint_for_asset(asset, std::string());
                append_prefetch_entry(AssetPrefetchEntry{
                    asset.id,
                    asset.id,
                    asset.path,
                    tensor_id,
                    std::string(),
                    asset.file_offset,
                    per_tensor_bytes,
                    queue_hint,
                    asset.preferred_residency,
                    "raw",
                    "any",
                    "raw",
                    exists,
                    asset.preload_required,
                    asset.persistent,
                    asset.host_visible,
                    !asset.host_visible || queue_hint != "host_io",
                    false});
                continue;
            }
            for (const auto& device_uid : devices_it->second) {
                const auto queue_hint = queue_hint_for_asset(asset, device_uid);
                const auto canonical_device_uid = canonicalize_device_uid(device_uid);
                append_prefetch_entry(AssetPrefetchEntry{
                    asset.id,
                    asset.id,
                    asset.path,
                    tensor_id,
                    canonical_device_uid,
                    asset.file_offset,
                    per_tensor_bytes,
                    queue_hint,
                    asset.preferred_residency,
                    "raw",
                    "any",
                    "raw",
                    exists,
                    asset.preload_required,
                    asset.persistent,
                    asset.host_visible,
                    !asset.host_visible || queue_hint != "host_io",
                    false});
            }
        }
    }

    // Imported model sources currently report raw external blobs only.
    // Native manifest assets still surface derived layout-cache materializations.
    if (manifest.imported) {
        if (!report.missing_required_assets) {
            summary << (summary.tellp() > 0 ? "; " : "")
                    << "prefetch=" << report.total_prefetch_bytes
                    << " host_io=" << report.total_host_io_bytes
                    << " h2d=" << report.total_host_to_device_bytes
                    << " layout_cache=" << report.total_layout_cache_bytes;
        }
        report.summary = summary.str();
        return report;
    }

    std::unordered_set<std::string> seen_layout_caches;
    for (const auto& optimized : optimization.operations) {
        for (const bool gpu_target : {false, true}) {
            const auto descriptor = describe_layout_cache(optimized.operation, optimization.workload_graph, gpu_target);
            if (!descriptor.has_value()) {
                continue;
            }
            const auto assets_it = assets_by_tensor.find(descriptor->source_tensor_id);
            if (assets_it == assets_by_tensor.end()) {
                continue;
            }
            for (const auto* source_asset : assets_it->second) {
                if (source_asset == nullptr || !source_asset->persistent) {
                    continue;
                }
                std::vector<std::string> target_devices;
                if (gpu_target) {
                    const auto tensor_devices_it = tensor_devices.find(descriptor->source_tensor_id);
                    if (tensor_devices_it != tensor_devices.end()) {
                        for (const auto& device_uid : tensor_devices_it->second) {
                            const auto* target_graph = find_graph_by_uid(devices_, device_uid);
                            if (target_graph != nullptr && target_graph->probe != "host") {
                                target_devices.push_back(device_uid);
                            }
                        }
                    }
                } else {
                    target_devices.push_back(canonical_host_uid);
                }
                if (target_devices.empty()) {
                    continue;
                }

                for (const auto& device_uid : target_devices) {
                    const std::string cache_id =
                        source_asset->id + "#" + optimized.operation.name + "#" + descriptor->materialization_kind +
                        "#" + (device_uid.empty() ? std::string("host") : device_uid);
                    if (!seen_layout_caches.insert(cache_id).second) {
                        continue;
                    }
                    const auto* target_graph = device_uid.empty() ? nullptr : find_graph_by_uid(devices_, device_uid);
                    const std::string backend_hint =
                        gpu_target ? (target_graph == nullptr ? std::string("gpu") : target_graph->probe) : "host";
                    const std::string backend_cache_tag =
                        gpu_target && target_graph != nullptr ? runtime_backend_cache_tag_for_graph(*target_graph) : "host";
                    append_prefetch_entry(AssetPrefetchEntry{
                        cache_id,
                        source_asset->id,
                        source_asset->path,
                        descriptor->source_tensor_id,
                        device_uid,
                        source_asset->file_offset,
                        descriptor->bytes,
                        gpu_target ? std::string("host_to_device") : std::string("host_io"),
                        gpu_target ? std::string("device") : std::string("host"),
                        descriptor->materialization_kind,
                        backend_hint,
                        backend_cache_tag,
                        source_asset->path.empty() ? false : std::filesystem::exists(source_asset->path),
                        source_asset->preload_required,
                        true,
                        !gpu_target,
                        gpu_target,
                        true});
                }
            }
        }
    }
    if (!report.missing_required_assets) {
        summary << (summary.tellp() > 0 ? "; " : "")
                << "prefetch=" << report.total_prefetch_bytes
                << " host_io=" << report.total_host_io_bytes
                << " h2d=" << report.total_host_to_device_bytes
                << " layout_cache=" << report.total_layout_cache_bytes;
    }
    report.summary = summary.str();
    return report;
}

void Runtime::persist_telemetry(
    const WorkloadSpec& workload,
    const ManagedExecutionReport& report) const {
    if (!options_.product.observability.persist_telemetry) {
        return;
    }

    const auto path = telemetry_path();
    const std::string header =
        "# telemetry_schema_version\tepoch\tworkload\tkind\tphase\tshape_bucket\trequested_strategy\tselected_strategy\tfinal_strategy\tplanner_source\tplanner_confidence\tplanner_risk\texecuted\tall_succeeded\tblocked_by_memory\trolled_back_to_auto\tblacklisted_before_run\tregression_gate_forced_auto\tpersisted_regression_events\tpersisted_worst_slowdown\tpeak_pressure_ratio\tspill_bytes\treload_bytes\tforced_spills\tprefetch_bytes\thost_io_bytes\th2d_bytes\ttotal_runtime_us\tspeedup_vs_reference\tcopy_runtime_us\tcompute_runtime_us\tcopy_overlap_ratio\ttransfer_us\toverlapped_transfer_us\ttransfer_overlap_gain_us\ttransfer_overlap_ratio\toptimizer_budget_ms\tbudget_exhausted\tallocator_peak_live_bytes\tallocator_peak_reserved_bytes\tallocator_reuse_count\tspill_artifact_bytes\treload_artifact_bytes\tbackend_owned_peak_bytes\tbackend_resource_reuse_hits\texecuted_h2d_bytes\texecuted_d2h_bytes\texecuted_spill_bytes\texecuted_reload_bytes\texecuted_transfer_us\tsummary\n";

    std::ostringstream line;
    line << kRuntimeTelemetrySchemaVersion << '\t'
         << execution_epoch_ << '\t'
         << workload.name << '\t'
         << to_string(workload.kind) << '\t'
         << to_string(canonical_workload_phase(workload)) << '\t'
         << canonical_workload_shape_bucket(workload) << '\t'
         << to_string(report.safety.requested_strategy) << '\t'
         << to_string(report.safety.selected_strategy) << '\t'
         << to_string(report.safety.final_strategy) << '\t'
         << to_string(report.safety.planner_strategy_source) << '\t'
         << report.safety.planner_confidence << '\t'
         << report.safety.planner_risk_score << '\t'
         << (report.executed ? 1 : 0) << '\t'
         << (report.executed && report.execution.all_succeeded ? 1 : 0) << '\t'
         << (report.safety.blocked_by_memory ? 1 : 0) << '\t'
         << (report.safety.rolled_back_to_auto ? 1 : 0) << '\t'
         << (report.safety.blacklisted_before_run ? 1 : 0) << '\t'
         << (report.safety.regression_gate_forced_auto ? 1 : 0) << '\t'
         << report.safety.persisted_regression_events << '\t'
         << report.safety.persisted_worst_slowdown << '\t'
         << report.memory_preflight.peak_pressure_ratio << '\t'
         << report.residency_sequence.spill_bytes << '\t'
         << report.residency_sequence.reload_bytes << '\t'
         << report.residency_sequence.forced_spill_count << '\t'
         << report.asset_prefetch.total_prefetch_bytes << '\t'
         << report.asset_prefetch.total_host_io_bytes << '\t'
         << report.asset_prefetch.total_host_to_device_bytes << '\t'
         << (report.executed ? report.execution.total_runtime_us : 0.0) << '\t'
         << (report.executed ? report.execution.speedup_vs_reference : 0.0) << '\t'
         << (report.executed ? report.execution.total_copy_runtime_us : 0.0) << '\t'
         << (report.executed ? report.execution.total_compute_runtime_us : 0.0) << '\t'
         << (report.executed ? report.execution.copy_overlap_ratio : 0.0) << '\t'
         << (report.executed ? report.execution.total_predicted_transfer_runtime_us : 0.0) << '\t'
         << (report.executed ? report.execution.total_overlapped_transfer_runtime_us : 0.0) << '\t'
         << (report.executed ? report.execution.total_transfer_overlap_gain_us : 0.0) << '\t'
         << (report.executed ? report.execution.transfer_overlap_ratio : 0.0) << '\t'
         << (report.executed ? report.execution.optimization.graph_optimization.time_budget_ms : 0u) << '\t'
         << (report.executed && report.execution.optimization.graph_optimization.budget_exhausted ? 1 : 0) << '\t'
         << report.tensor_allocator.peak_live_bytes << '\t'
         << report.tensor_allocator.peak_reserved_bytes << '\t'
         << report.tensor_allocator.reuse_count << '\t'
         << report.spill_artifacts.materialized_spill_bytes << '\t'
         << report.spill_artifacts.materialized_reload_bytes << '\t'
         << report.backend_buffer_bindings.backend_owned_peak_bytes << '\t'
         << report.backend_buffer_bindings.total_persistent_resource_reuse_hits << '\t'
         << report.executed_residency_movements.executed_h2d_bytes << '\t'
         << report.executed_residency_movements.executed_d2h_bytes << '\t'
         << report.executed_residency_movements.executed_spill_bytes << '\t'
         << report.executed_residency_movements.executed_reload_bytes << '\t'
         << report.executed_residency_movements.total_transfer_runtime_us << '\t'
         << report.safety.summary << '\n';

    if (options_.product.observability.async_telemetry_flush) {
        telemetry_writer().enqueue([path, header, payload = line.str()]() {
            append_tsv_line(path, header, payload);
        });
    } else {
        append_tsv_line(path, header, line.str());
    }

    update_runtime_budget_cache(path, execution_epoch_, workload, report);
}

bool Runtime::should_include_descriptor(const HardwareGraph& candidate) const {
    return std::none_of(devices_.begin(), devices_.end(), [&](const HardwareGraph& existing) {
        if (existing.uid == candidate.uid) {
            return true;
        }

        const bool same_name = existing.presentation_name == candidate.presentation_name;
        const bool same_probe_shape = structural_fingerprint(existing) == structural_fingerprint(candidate);
        return same_name && same_probe_shape;
    });
}

}  // namespace jakal

