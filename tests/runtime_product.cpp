#include "jakal/runtime.hpp"
#include "jakal/workloads.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

std::filesystem::path unique_temp_file(const std::string& stem, const std::string& extension) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / (stem + "-" + std::to_string(nonce) + extension);
}

void write_manifest(
    const std::filesystem::path& path,
    const std::uint64_t working_set_bytes,
    const std::uint64_t tensor_bytes,
    const bool include_graph = true,
    const std::filesystem::path& weight_asset_path = {}) {
    std::ofstream output(path, std::ios::trunc);
    output << "[workload]\n";
    output << "name=manifest-exec\n";
    output << "kind=inference\n";
    output << "dataset_tag=manifest-exec-lite\n";
    output << "phase=decode\n";
    output << "working_set_bytes=" << working_set_bytes << "\n";
    output << "host_exchange_bytes=1048576\n";
    output << "estimated_flops=12000000\n";
    output << "batch_size=1\n";
    output << "latency_sensitive=true\n";
    output << "prefer_unified_memory=true\n";
    output << "matrix_friendly=false\n\n";

    if (!weight_asset_path.empty()) {
        output << "[asset]\n";
        output << "id=weights-shard\n";
        output << "path=" << weight_asset_path.string() << "\n";
        output << "tensor_ids=weights\n";
        output << "preload_required=true\n";
        output << "persistent=true\n";
        output << "host_visible=true\n\n";
    }

    if (!include_graph) {
        return;
    }

    output << "[tensor]\n";
    output << "id=input\n";
    output << "bytes=" << tensor_bytes << "\n";
    output << "consumers=normalize\n";
    output << "host_visible=true\n\n";

    output << "[tensor]\n";
    output << "id=weights\n";
    output << "bytes=" << tensor_bytes << "\n";
    output << "consumers=normalize\n";
    output << "persistent=true\n";
    output << "host_visible=true\n\n";

    output << "[tensor]\n";
    output << "id=hidden\n";
    output << "bytes=" << tensor_bytes << "\n";
    output << "producer=normalize\n";
    output << "consumers=score\n\n";

    output << "[tensor]\n";
    output << "id=score-out\n";
    output << "bytes=4\n";
    output << "producer=score\n\n";

    output << "[operation]\n";
    output << "name=normalize\n";
    output << "class=elementwise_map\n";
    output << "extents=4096\n";
    output << "input_bytes=" << tensor_bytes * 2ull << "\n";
    output << "output_bytes=" << tensor_bytes << "\n";
    output << "estimated_flops=8192\n";
    output << "parallelizable=true\n";
    output << "streaming_friendly=true\n";
    output << "inputs=input,weights\n";
    output << "outputs=hidden\n\n";

    output << "[operation]\n";
    output << "name=score\n";
    output << "class=reduction\n";
    output << "extents=4096\n";
    output << "input_bytes=" << tensor_bytes << "\n";
    output << "output_bytes=4\n";
    output << "estimated_flops=4096\n";
    output << "parallelizable=true\n";
    output << "reduction_like=true\n";
    output << "inputs=hidden\n";
    output << "outputs=score-out\n";
}

}  // namespace

int main() {
    try {
        const auto manifest_path = unique_temp_file("runtime-product-graph", ".workload");
        const auto runtime_manifest_path = unique_temp_file("runtime-product-runtime", ".workload");
        const auto blocked_manifest_path = unique_temp_file("runtime-product-blocked", ".workload");
        const auto missing_asset_manifest_path = unique_temp_file("runtime-product-missing-asset", ".workload");
        const auto weight_asset_path = unique_temp_file("runtime-product-weights", ".bin");
        const auto missing_weight_asset_path = unique_temp_file("runtime-product-weights-missing", ".bin");

        {
            std::ofstream weights(weight_asset_path, std::ios::binary | std::ios::trunc);
            std::string payload(16ull * 1024ull, '\x5a');
            weights.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        }

        write_manifest(manifest_path, 8ull * 1024ull * 1024ull, 16ull * 1024ull, true, weight_asset_path);
        write_manifest(runtime_manifest_path, 8ull * 1024ull * 1024ull, 16ull * 1024ull, false);
        write_manifest(blocked_manifest_path, 1ull << 50u, 4ull * 1024ull * 1024ull * 1024ull, true);
        write_manifest(missing_asset_manifest_path, 8ull * 1024ull * 1024ull, 16ull * 1024ull, true, missing_weight_asset_path);

        const auto manifest = jakal::load_workload_manifest(manifest_path);
        if (!manifest.has_graph || manifest.graph.operations.size() != 2u || manifest.graph.tensors.size() != 4u) {
            std::cerr << "manifest graph parsing failed\n";
            return 1;
        }
        if (manifest.assets.size() != 1u || manifest.assets.front().bytes != 16ull * 1024ull) {
            std::cerr << "manifest asset parsing failed\n";
            return 1;
        }

        jakal::RuntimeOptions options;
        options.enable_host_probe = true;
        options.enable_opencl_probe = false;
        options.enable_level_zero_probe = false;
        options.enable_cuda_probe = false;
        options.enable_rocm_probe = false;
        options.product.observability.telemetry_path = unique_temp_file("runtime-product-managed", ".telemetry.tsv");

        jakal::Runtime runtime(options);
        const auto optimized = runtime.optimize(manifest.workload, manifest.graph);
        if (optimized.operations.empty()) {
            std::cerr << "custom graph optimize produced no operations\n";
            return 1;
        }
        const auto manifest_managed = runtime.execute_manifest(manifest_path);
        if (!manifest_managed.executed) {
            std::cerr << "managed execute did not run custom manifest path\n";
            return 1;
        }
        if (manifest_managed.execution.optimization.operations.size() != manifest.graph.operations.size()) {
            std::cerr << "managed execute optimized unexpected custom graph operation count\n";
            return 1;
        }
        if (manifest_managed.asset_prefetch.entries.empty() || manifest_managed.asset_prefetch.total_prefetch_bytes != 16ull * 1024ull) {
            std::cerr << "managed execute did not produce asset prefetch plan\n";
            return 1;
        }
        if (manifest_managed.asset_prefetch.total_host_io_bytes != 16ull * 1024ull ||
            manifest_managed.asset_prefetch.total_host_to_device_bytes != 0u) {
            std::cerr << "managed execute asset queue accounting mismatch\n";
            return 1;
        }
        if (manifest_managed.asset_prefetch.entries.front().queue_hint != "host_io" ||
            manifest_managed.asset_prefetch.entries.front().target_residency != "auto") {
            std::cerr << "managed execute asset queue hint mismatch\n";
            return 1;
        }
        if (!manifest_managed.kernel_coverage.all_supported) {
            std::cerr << "host-only managed path should have full kernel coverage\n";
            return 1;
        }
        if (manifest_managed.planning.resolved_partition_strategy != manifest_managed.execution.optimization.partition_strategy ||
            manifest_managed.safety.planner_confidence < 0.0 ||
            manifest_managed.safety.planner_strategy_source == jakal::PlanStrategySource::exploration) {
            std::cerr << "managed execute planner diagnostics mismatch\n";
            return 1;
        }
        if (manifest_managed.residency_sequence.actions.empty()) {
            std::cerr << "managed execute did not emit residency sequence\n";
            return 1;
        }

        const auto runtime_managed = runtime.execute_manifest(runtime_manifest_path);
        if (!runtime_managed.executed) {
            std::cerr << "managed execute did not run spec-only manifest path\n";
            return 1;
        }
        if (runtime_managed.execution.optimization.operations.empty()) {
            std::cerr << "spec-only manifest execute produced no optimized operations\n";
            return 1;
        }

        jakal::RuntimeOptions blocked_options = options;
        blocked_options.product.memory.max_pressure_ratio = 0.01;
        blocked_options.product.memory.enforce_preflight = true;
        blocked_options.product.observability.telemetry_path = unique_temp_file("runtime-product-blocked", ".telemetry.tsv");
        jakal::Runtime blocked_runtime(blocked_options);
        const auto blocked = blocked_runtime.execute_manifest(blocked_manifest_path);
        if (blocked.executed || !blocked.safety.blocked_by_memory) {
            std::cerr << "memory safety gate did not block oversized workload\n";
            return 1;
        }
        if (!std::filesystem::exists(blocked.telemetry_path)) {
            std::cerr << "blocked managed path did not emit telemetry\n";
            return 1;
        }
        if (blocked.memory_preflight.devices.empty()) {
            std::cerr << "memory preflight did not capture device reservations\n";
            return 1;
        }
        if (blocked.residency_sequence.actions.empty()) {
            std::cerr << "blocked managed path did not emit residency sequence\n";
            return 1;
        }
        if (blocked.memory_preflight.predicted_spill_bytes != blocked.residency_sequence.spill_bytes ||
            blocked.memory_preflight.predicted_reload_bytes != blocked.residency_sequence.reload_bytes ||
            blocked.memory_preflight.forced_spill_count != blocked.residency_sequence.forced_spill_count) {
            std::cerr << "memory preflight and residency sequence drifted\n";
            return 1;
        }
        if (blocked.memory_preflight.predicted_spill_bytes == 0u && blocked.memory_preflight.forced_spill_count == 0u) {
            std::cerr << "blocked managed path did not predict spill pressure\n";
            return 1;
        }

        const auto missing_asset = runtime.execute_manifest(missing_asset_manifest_path);
        if (missing_asset.executed || !missing_asset.asset_prefetch.missing_required_assets) {
            std::cerr << "missing required asset was not blocked\n";
            return 1;
        }

        std::error_code ec;
        std::filesystem::remove(manifest_path, ec);
        std::filesystem::remove(runtime_manifest_path, ec);
        std::filesystem::remove(blocked_manifest_path, ec);
        std::filesystem::remove(missing_asset_manifest_path, ec);
        std::filesystem::remove(weight_asset_path, ec);
        std::filesystem::remove(manifest_managed.telemetry_path, ec);
        std::filesystem::remove(runtime_managed.telemetry_path, ec);
        std::filesystem::remove(blocked.telemetry_path, ec);
        std::filesystem::remove(missing_asset.telemetry_path, ec);

        std::cout << "runtime product path ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "exception: " << error.what() << '\n';
        return 1;
    }
}
