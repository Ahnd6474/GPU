#include "jakal/c_api.h"

#include "jakal/runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <new>
#include <optional>
#include <string>

struct jakal_core_runtime {
    jakal::Runtime runtime;
    std::string last_error;
    std::optional<jakal::ManagedExecutionReport> last_managed_execution;
};

namespace {

void copy_string(const std::string& source, char* destination, const std::size_t capacity) {
    if (destination == nullptr || capacity == 0) {
        return;
    }

    const auto length = std::min(source.size(), capacity - 1);
    std::memcpy(destination, source.data(), length);
    destination[length] = '\0';
}

void fill_device_info(const jakal::HardwareGraph& graph, jakal_core_device_info* out_device) {
    if (out_device == nullptr) {
        return;
    }

    const auto summary = jakal::summarize_graph(graph);
    copy_string(graph.uid, out_device->uid, sizeof(out_device->uid));
    copy_string(graph.probe, out_device->probe, sizeof(out_device->probe));
    copy_string(graph.presentation_name, out_device->presentation_name, sizeof(out_device->presentation_name));
    out_device->ordinal = graph.ordinal;
    out_device->execution_objects = summary.execution_objects;
    out_device->lanes_per_object = summary.lanes_per_object;
    out_device->resident_contexts = summary.resident_contexts;
    out_device->matrix_units = summary.matrix_units;
    out_device->clock_mhz = summary.clock_mhz;
    out_device->queue_slots = summary.queue_slots;
    out_device->numa_domain = summary.numa_domain;
    out_device->native_vector_bits = summary.native_vector_bits;
    out_device->addressable_memory_bytes = summary.addressable_bytes;
    out_device->directly_attached_memory_bytes = summary.directly_attached_bytes;
    out_device->shared_host_memory_bytes = summary.shared_host_bytes;
    out_device->local_scratch_bytes = summary.local_scratch_bytes;
    out_device->cache_bytes = summary.cache_bytes;
    out_device->host_read_gbps = summary.host_read_gbps;
    out_device->host_write_gbps = summary.host_write_gbps;
    out_device->dispatch_latency_us = summary.dispatch_latency_us;
    out_device->synchronization_latency_us = summary.synchronization_latency_us;
    out_device->node_count = static_cast<unsigned long long>(graph.nodes.size());
    out_device->edge_count = static_cast<unsigned long long>(graph.edges.size());
    out_device->coherent_with_host = summary.coherent_with_host ? 1 : 0;
    out_device->unified_address_space = summary.unified_address_space ? 1 : 0;
    out_device->host_visible = summary.host_visible ? 1 : 0;
    out_device->supports_asynchronous_dispatch = summary.supports_asynchronous_dispatch ? 1 : 0;
    out_device->supports_fp64 = summary.supports_fp64 ? 1 : 0;
    out_device->supports_fp32 = summary.supports_fp32 ? 1 : 0;
    out_device->supports_fp16 = summary.supports_fp16 ? 1 : 0;
    out_device->supports_bf16 = summary.supports_bf16 ? 1 : 0;
    out_device->supports_int8 = summary.supports_int8 ? 1 : 0;
}

void fill_graph_node_info(const jakal::HardwareObjectNode& node, jakal_core_graph_node_info* out_node) {
    if (out_node == nullptr) {
        return;
    }

    copy_string(node.id, out_node->id, sizeof(out_node->id));
    copy_string(node.label, out_node->label, sizeof(out_node->label));
    copy_string(node.parent_id, out_node->parent_id, sizeof(out_node->parent_id));
    copy_string(jakal::to_string(node.domain), out_node->domain, sizeof(out_node->domain));
    copy_string(jakal::to_string(node.role), out_node->role, sizeof(out_node->role));
    copy_string(jakal::to_string(node.resolution), out_node->resolution, sizeof(out_node->resolution));
    out_node->ordinal = node.ordinal;
    out_node->execution_width = node.compute.execution_width;
    out_node->resident_contexts = node.compute.resident_contexts;
    out_node->matrix_engines = node.compute.matrix_engines;
    out_node->clock_mhz = node.compute.clock_mhz;
    out_node->native_vector_bits = node.compute.native_vector_bits;
    out_node->queue_slots = node.control.queue_slots;
    out_node->numa_domain = node.control.numa_domain;
    out_node->capacity_bytes = node.storage.capacity_bytes;
    out_node->directly_attached_bytes = node.storage.directly_attached_bytes;
    out_node->shared_host_bytes = node.storage.shared_host_bytes;
    out_node->read_bandwidth_gbps = node.transfer.read_bandwidth_gbps;
    out_node->write_bandwidth_gbps = node.transfer.write_bandwidth_gbps;
    out_node->dispatch_latency_us = node.transfer.dispatch_latency_us;
    out_node->synchronization_latency_us = node.transfer.synchronization_latency_us;
    out_node->coherent_with_host = node.storage.coherent_with_host ? 1 : 0;
    out_node->unified_address_space = node.storage.unified_address_space ? 1 : 0;
    out_node->host_visible = node.control.host_visible ? 1 : 0;
    out_node->supports_asynchronous_dispatch = node.control.supports_asynchronous_dispatch ? 1 : 0;
    out_node->supports_fp64 = node.compute.supports_fp64 ? 1 : 0;
    out_node->supports_fp32 = node.compute.supports_fp32 ? 1 : 0;
    out_node->supports_fp16 = node.compute.supports_fp16 ? 1 : 0;
    out_node->supports_bf16 = node.compute.supports_bf16 ? 1 : 0;
    out_node->supports_int8 = node.compute.supports_int8 ? 1 : 0;
}

void fill_graph_edge_info(const jakal::HardwareGraphEdge& edge, jakal_core_graph_edge_info* out_edge) {
    if (out_edge == nullptr) {
        return;
    }

    copy_string(edge.source_id, out_edge->source_id, sizeof(out_edge->source_id));
    copy_string(edge.target_id, out_edge->target_id, sizeof(out_edge->target_id));
    copy_string(jakal::to_string(edge.semantics), out_edge->semantics, sizeof(out_edge->semantics));
    out_edge->directed = edge.directed ? 1 : 0;
    out_edge->weight = edge.weight;
    out_edge->bandwidth_gbps = edge.bandwidth_gbps;
    out_edge->latency_us = edge.latency_us;
}

void set_last_error(jakal_core_runtime_t* runtime, const std::string& message) {
    if (runtime != nullptr) {
        runtime->last_error = message;
    }
}

void clear_last_error(jakal_core_runtime_t* runtime) {
    if (runtime != nullptr) {
        runtime->last_error.clear();
    }
}

void fill_optimization_info(const jakal::OptimizationReport& report, jakal_core_optimization_info* out_optimization) {
    if (out_optimization == nullptr) {
        return;
    }

    copy_string(report.signature, out_optimization->signature, sizeof(out_optimization->signature));
    copy_string(jakal::to_string(report.workload_kind), out_optimization->workload_kind, sizeof(out_optimization->workload_kind));
    copy_string(report.dataset_tag, out_optimization->dataset_tag, sizeof(out_optimization->dataset_tag));
    out_optimization->operation_count = static_cast<unsigned long long>(report.operations.size());
    out_optimization->tensor_count = static_cast<unsigned long long>(report.workload_graph.tensors.size());
    out_optimization->dependency_count = static_cast<unsigned long long>(report.workload_graph.dependencies.size());
    out_optimization->readiness_score = report.system_profile.readiness_score;
    out_optimization->stability_score = report.system_profile.stability_score;
    out_optimization->sustained_slowdown = report.system_profile.sustained_slowdown;
    out_optimization->loaded_from_cache = report.loaded_from_cache ? 1 : 0;
}

void fill_operation_optimization_info(
    const jakal::OperationOptimizationResult& operation,
    jakal_core_operation_optimization_info* out_operation) {
    if (out_operation == nullptr) {
        return;
    }

    copy_string(operation.operation.name, out_operation->operation_name, sizeof(out_operation->operation_name));
    copy_string(jakal::to_string(operation.config.strategy), out_operation->strategy, sizeof(out_operation->strategy));
    copy_string(operation.config.primary_device_uid, out_operation->primary_device_uid, sizeof(out_operation->primary_device_uid));
    out_operation->logical_partitions = operation.config.logical_partitions;
    out_operation->participating_device_count = static_cast<unsigned int>(operation.config.participating_devices.size());
    out_operation->predicted_latency_us = operation.graph.predicted_latency_us;
    out_operation->predicted_speedup_vs_reference = operation.graph.predicted_speedup_vs_reference;
    out_operation->predicted_transfer_latency_us = operation.graph.predicted_transfer_latency_us;
    out_operation->predicted_memory_pressure = operation.graph.predicted_memory_pressure;
    out_operation->peak_resident_bytes = operation.graph.peak_resident_bytes;
    out_operation->target_error_tolerance = operation.config.target_error_tolerance;
    out_operation->use_low_precision = operation.config.use_low_precision ? 1 : 0;
}

void fill_execution_info(const jakal::DirectExecutionReport& report, jakal_core_execution_info* out_execution) {
    if (out_execution == nullptr) {
        return;
    }

    copy_string(report.optimization.signature, out_execution->signature, sizeof(out_execution->signature));
    out_execution->operation_count = static_cast<unsigned long long>(report.operations.size());
    out_execution->total_runtime_us = report.total_runtime_us;
    out_execution->total_reference_runtime_us = report.total_reference_runtime_us;
    out_execution->speedup_vs_reference = report.speedup_vs_reference;
    out_execution->all_succeeded = report.all_succeeded ? 1 : 0;
}

void fill_execution_operation_info(
    const jakal::OperationExecutionRecord& operation,
    jakal_core_execution_operation_info* out_operation) {
    if (out_operation == nullptr) {
        return;
    }

    copy_string(operation.operation_name, out_operation->operation_name, sizeof(out_operation->operation_name));
    copy_string(operation.backend_name, out_operation->backend_name, sizeof(out_operation->backend_name));
    copy_string(operation.requested_gpu_vendor, out_operation->requested_gpu_vendor, sizeof(out_operation->requested_gpu_vendor));
    copy_string(operation.requested_gpu_backend, out_operation->requested_gpu_backend, sizeof(out_operation->requested_gpu_backend));
    out_operation->runtime_us = operation.runtime_us;
    out_operation->reference_runtime_us = operation.reference_runtime_us;
    out_operation->speedup_vs_reference = operation.speedup_vs_reference;
    out_operation->relative_error = operation.relative_error;
    out_operation->verified = operation.verified ? 1 : 0;
    out_operation->used_host = operation.used_host ? 1 : 0;
    out_operation->used_opencl = operation.used_opencl ? 1 : 0;
    out_operation->used_multiple_devices = operation.used_multiple_devices ? 1 : 0;
    out_operation->logical_partitions_used = operation.logical_partitions_used;
}

void fill_backend_buffer_binding_info(
    const jakal::BackendBufferBindingEntry& binding,
    jakal_core_backend_buffer_binding_info* out_binding) {
    if (out_binding == nullptr) {
        return;
    }

    copy_string(binding.device_uid, out_binding->device_uid, sizeof(out_binding->device_uid));
    copy_string(binding.backend_name, out_binding->backend_name, sizeof(out_binding->backend_name));
    copy_string(binding.ownership_scope, out_binding->ownership_scope, sizeof(out_binding->ownership_scope));
    copy_string(binding.pool_id, out_binding->pool_id, sizeof(out_binding->pool_id));
    copy_string(binding.resource_tag, out_binding->resource_tag, sizeof(out_binding->resource_tag));
    out_binding->planned_peak_bytes = binding.planned_peak_bytes;
    out_binding->reserved_bytes = binding.reserved_bytes;
    out_binding->spill_bytes = binding.spill_bytes;
    out_binding->reload_bytes = binding.reload_bytes;
    out_binding->direct_execution_operation_count = binding.direct_execution_operation_count;
    out_binding->persistent_resource_reuse_hits = binding.persistent_resource_reuse_hits;
    out_binding->direct_execution_active = binding.direct_execution_active ? 1 : 0;
    out_binding->uses_runtime_spill_artifacts = binding.uses_runtime_spill_artifacts ? 1 : 0;
}

void fill_residency_movement_info(
    const jakal::ExecutedResidencyMovementEntry& movement,
    jakal_core_residency_movement_info* out_movement) {
    if (out_movement == nullptr) {
        return;
    }

    copy_string(movement.kind, out_movement->kind, sizeof(out_movement->kind));
    copy_string(movement.tensor_id, out_movement->tensor_id, sizeof(out_movement->tensor_id));
    copy_string(movement.device_uid, out_movement->device_uid, sizeof(out_movement->device_uid));
    copy_string(movement.backend_name, out_movement->backend_name, sizeof(out_movement->backend_name));
    copy_string(
        movement.trigger_operation_name,
        out_movement->trigger_operation_name,
        sizeof(out_movement->trigger_operation_name));
    copy_string(movement.pool_id, out_movement->pool_id, sizeof(out_movement->pool_id));
    copy_string(
        movement.spill_artifact_path.string(),
        out_movement->spill_artifact_path,
        sizeof(out_movement->spill_artifact_path));
    out_movement->operation_index = movement.operation_index;
    out_movement->bytes = movement.bytes;
    out_movement->runtime_us = movement.runtime_us;
    out_movement->from_direct_execution = movement.from_direct_execution ? 1 : 0;
    out_movement->from_spill_artifact = movement.from_spill_artifact ? 1 : 0;
}

void fill_runtime_paths(
    const jakal::RuntimeInstallPaths& paths,
    jakal_core_runtime_paths* out_paths) {
    if (out_paths == nullptr) {
        return;
    }

    copy_string(paths.install_root.string(), out_paths->install_root, sizeof(out_paths->install_root));
    copy_string(paths.writable_root.string(), out_paths->writable_root, sizeof(out_paths->writable_root));
    copy_string(paths.config_dir.string(), out_paths->config_dir, sizeof(out_paths->config_dir));
    copy_string(paths.cache_dir.string(), out_paths->cache_dir, sizeof(out_paths->cache_dir));
    copy_string(paths.logs_dir.string(), out_paths->logs_dir, sizeof(out_paths->logs_dir));
    copy_string(paths.telemetry_path.string(), out_paths->telemetry_path, sizeof(out_paths->telemetry_path));
    copy_string(
        paths.planner_cache_path.string(),
        out_paths->planner_cache_path,
        sizeof(out_paths->planner_cache_path));
    copy_string(
        paths.execution_cache_path.string(),
        out_paths->execution_cache_path,
        sizeof(out_paths->execution_cache_path));
    copy_string(paths.python_dir.string(), out_paths->python_dir, sizeof(out_paths->python_dir));
}

void fill_backend_status_info(
    const jakal::RuntimeBackendStatus& status,
    jakal_core_backend_status_info* out_status) {
    if (out_status == nullptr) {
        return;
    }

    copy_string(status.backend_name, out_status->backend_name, sizeof(out_status->backend_name));
    copy_string(status.device_uid, out_status->device_uid, sizeof(out_status->device_uid));
    copy_string(jakal::to_string(status.code), out_status->code, sizeof(out_status->code));
    copy_string(status.detail, out_status->detail, sizeof(out_status->detail));
    out_status->enabled = status.enabled ? 1 : 0;
    out_status->available = status.available ? 1 : 0;
    out_status->direct_execution = status.direct_execution ? 1 : 0;
    out_status->modeled_fallback = status.modeled_fallback ? 1 : 0;
}

jakal::WorkloadKind parse_workload_kind(const char* kind) {
    if (kind == nullptr) {
        return jakal::WorkloadKind::custom;
    }

    const std::string value(kind);
    if (value == "inference") {
        return jakal::WorkloadKind::inference;
    }
    if (value == "image") {
        return jakal::WorkloadKind::image;
    }
    if (value == "tensor") {
        return jakal::WorkloadKind::tensor;
    }
    if (value == "gaming") {
        return jakal::WorkloadKind::gaming;
    }
    if (value == "training") {
        return jakal::WorkloadKind::training;
    }
    return jakal::WorkloadKind::custom;
}

jakal::WorkloadPhase parse_workload_phase(const char* phase) {
    if (phase == nullptr) {
        return jakal::WorkloadPhase::unknown;
    }
    const std::string value(phase);
    if (value == "decode") {
        return jakal::WorkloadPhase::decode;
    }
    if (value == "prefill") {
        return jakal::WorkloadPhase::prefill;
    }
    if (value == "cache_update") {
        return jakal::WorkloadPhase::cache_update;
    }
    if (value == "dequantize") {
        return jakal::WorkloadPhase::dequantize;
    }
    if (value == "training_step") {
        return jakal::WorkloadPhase::training_step;
    }
    return jakal::WorkloadPhase::unknown;
}

jakal::WorkloadSpec convert_workload(const jakal_core_workload_spec& workload) {
    return jakal::WorkloadSpec{
        workload.name == nullptr ? std::string("unnamed") : std::string(workload.name),
        parse_workload_kind(workload.kind),
        workload.dataset_tag == nullptr ? std::string{} : std::string(workload.dataset_tag),
        workload.working_set_bytes,
        workload.host_exchange_bytes,
        workload.estimated_flops,
        workload.batch_size,
        workload.latency_sensitive != 0,
        workload.prefer_unified_memory != 0,
        workload.matrix_friendly != 0,
        jakal::PartitionStrategy::auto_balanced,
        parse_workload_phase(workload.phase),
        workload.shape_bucket == nullptr ? std::string{} : std::string(workload.shape_bucket)};
}

jakal::RuntimeOptions convert_runtime_options(const jakal_core_runtime_options& options) {
    auto runtime_options = jakal::make_runtime_options_for_install(
        options.install_root == nullptr ? std::filesystem::path{} : std::filesystem::path(options.install_root));
    runtime_options.enable_host_probe = options.enable_host_probe != 0;
    runtime_options.enable_opencl_probe = options.enable_opencl_probe != 0;
    runtime_options.enable_level_zero_probe = options.enable_level_zero_probe != 0;
    runtime_options.enable_vulkan_probe = options.enable_vulkan_probe != 0;
    runtime_options.enable_vulkan_status = options.enable_vulkan_status != 0;
    runtime_options.enable_cuda_probe = options.enable_cuda_probe != 0;
    runtime_options.enable_rocm_probe = options.enable_rocm_probe != 0;
    runtime_options.prefer_level_zero_over_opencl = options.prefer_level_zero_over_opencl != 0;
    runtime_options.eager_hardware_refresh = options.eager_hardware_refresh != 0;
    if (options.cache_path != nullptr && options.cache_path[0] != '\0') {
        runtime_options.cache_path = options.cache_path;
    }
    if (options.execution_cache_path != nullptr && options.execution_cache_path[0] != '\0') {
        runtime_options.execution_cache_path = options.execution_cache_path;
    }
    if (options.telemetry_path != nullptr && options.telemetry_path[0] != '\0') {
        runtime_options.product.observability.telemetry_path = options.telemetry_path;
    }
    return runtime_options;
}

}  // namespace

unsigned int jakal_core_c_api_abi_version(void) {
    return JAKAL_CORE_C_API_ABI_VERSION;
}

unsigned int jakal_core_runtime_telemetry_schema_version(void) {
    return JAKAL_CORE_RUNTIME_TELEMETRY_SCHEMA_VERSION;
}

unsigned int jakal_core_execution_performance_cache_schema_version(void) {
    return JAKAL_CORE_EXECUTION_PERFORMANCE_CACHE_SCHEMA_VERSION;
}

jakal_core_runtime_t* jakal_core_runtime_create(void) {
    try {
        return new jakal_core_runtime_t{jakal::Runtime{}};
    } catch (const std::exception&) {
        return nullptr;
    }
}

jakal_core_runtime_t* jakal_core_runtime_create_with_options(const jakal_core_runtime_options* options) {
    try {
        if (options == nullptr) {
            return new jakal_core_runtime_t{jakal::Runtime{}};
        }
        return new jakal_core_runtime_t{jakal::Runtime{convert_runtime_options(*options)}};
    } catch (const std::exception&) {
        return nullptr;
    }
}

void jakal_core_runtime_destroy(jakal_core_runtime_t* runtime) {
    delete runtime;
}

int jakal_core_runtime_refresh(jakal_core_runtime_t* runtime) {
    if (runtime == nullptr) {
        return -1;
    }

    try {
        clear_last_error(runtime);
        runtime->runtime.refresh_hardware();
        return 0;
    } catch (const std::exception& error) {
        set_last_error(runtime, error.what());
        return -2;
    }
}

int jakal_core_runtime_get_last_error(const jakal_core_runtime_t* runtime, char* buffer, size_t capacity) {
    if (runtime == nullptr || buffer == nullptr || capacity == 0u) {
        return -1;
    }
    copy_string(runtime->last_error, buffer, capacity);
    return 0;
}

int jakal_core_runtime_get_install_paths(const jakal_core_runtime_t* runtime, jakal_core_runtime_paths* out_paths) {
    if (runtime == nullptr || out_paths == nullptr) {
        return -1;
    }
    fill_runtime_paths(runtime->runtime.install_paths(), out_paths);
    return 0;
}

size_t jakal_core_runtime_backend_status_count(const jakal_core_runtime_t* runtime) {
    if (runtime == nullptr) {
        return 0u;
    }
    return runtime->runtime.backend_statuses().size();
}

int jakal_core_runtime_get_backend_status(
    const jakal_core_runtime_t* runtime,
    size_t index,
    jakal_core_backend_status_info* out_status) {
    if (runtime == nullptr || out_status == nullptr) {
        return -1;
    }
    const auto& statuses = runtime->runtime.backend_statuses();
    if (index >= statuses.size()) {
        return -2;
    }
    fill_backend_status_info(statuses[index], out_status);
    return 0;
}

size_t jakal_core_runtime_device_count(const jakal_core_runtime_t* runtime) {
    if (runtime == nullptr) {
        return 0;
    }

    return runtime->runtime.devices().size();
}

int jakal_core_runtime_get_device(const jakal_core_runtime_t* runtime, size_t index, jakal_core_device_info* out_device) {
    if (runtime == nullptr || out_device == nullptr) {
        return -1;
    }

    const auto& devices = runtime->runtime.devices();
    if (index >= devices.size()) {
        return -2;
    }

    fill_device_info(devices[index], out_device);
    return 0;
}

size_t jakal_core_runtime_graph_node_count(const jakal_core_runtime_t* runtime, size_t device_index) {
    if (runtime == nullptr) {
        return 0;
    }

    const auto& devices = runtime->runtime.devices();
    if (device_index >= devices.size()) {
        return 0;
    }

    return devices[device_index].nodes.size();
}

int jakal_core_runtime_get_graph_node(
    const jakal_core_runtime_t* runtime,
    size_t device_index,
    size_t node_index,
    jakal_core_graph_node_info* out_node) {
    if (runtime == nullptr || out_node == nullptr) {
        return -1;
    }

    const auto& devices = runtime->runtime.devices();
    if (device_index >= devices.size()) {
        return -2;
    }

    const auto& nodes = devices[device_index].nodes;
    if (node_index >= nodes.size()) {
        return -3;
    }

    fill_graph_node_info(nodes[node_index], out_node);
    return 0;
}

size_t jakal_core_runtime_graph_edge_count(const jakal_core_runtime_t* runtime, size_t device_index) {
    if (runtime == nullptr) {
        return 0;
    }

    const auto& devices = runtime->runtime.devices();
    if (device_index >= devices.size()) {
        return 0;
    }

    return devices[device_index].edges.size();
}

int jakal_core_runtime_get_graph_edge(
    const jakal_core_runtime_t* runtime,
    size_t device_index,
    size_t edge_index,
    jakal_core_graph_edge_info* out_edge) {
    if (runtime == nullptr || out_edge == nullptr) {
        return -1;
    }

    const auto& devices = runtime->runtime.devices();
    if (device_index >= devices.size()) {
        return -2;
    }

    const auto& edges = devices[device_index].edges;
    if (edge_index >= edges.size()) {
        return -3;
    }

    fill_graph_edge_info(edges[edge_index], out_edge);
    return 0;
}

int jakal_core_runtime_plan(
    jakal_core_runtime_t* runtime,
    const jakal_core_workload_spec* workload,
    jakal_core_plan_entry* entries,
    size_t capacity,
    size_t* out_count,
    int* out_loaded_from_cache) {
    if (runtime == nullptr || workload == nullptr || out_count == nullptr) {
        return -1;
    }

    try {
        clear_last_error(runtime);
        const auto cpp_workload = convert_workload(*workload);
        const auto plan = runtime->runtime.plan(cpp_workload);
        *out_count = plan.allocations.size();

        if (out_loaded_from_cache != nullptr) {
            *out_loaded_from_cache = plan.loaded_from_cache ? 1 : 0;
        }

        if (entries == nullptr || capacity < plan.allocations.size()) {
            return -2;
        }

        for (size_t index = 0; index < plan.allocations.size(); ++index) {
            fill_device_info(plan.allocations[index].device, &entries[index].device);
            entries[index].ratio = plan.allocations[index].ratio;
            entries[index].score = plan.allocations[index].score;
        }
        return 0;
    } catch (const std::exception& error) {
        set_last_error(runtime, error.what());
        return -3;
    }
}

int jakal_core_runtime_optimize(
    jakal_core_runtime_t* runtime,
    const jakal_core_workload_spec* workload,
    jakal_core_optimization_info* out_optimization,
    jakal_core_operation_optimization_info* operations,
    size_t capacity,
    size_t* out_count) {
    if (runtime == nullptr || workload == nullptr || out_count == nullptr) {
        return -1;
    }

    try {
        clear_last_error(runtime);
        runtime->last_managed_execution.reset();
        const auto report = runtime->runtime.optimize(convert_workload(*workload));
        *out_count = report.operations.size();
        fill_optimization_info(report, out_optimization);

        if (operations == nullptr || capacity < report.operations.size()) {
            return -2;
        }

        for (size_t index = 0; index < report.operations.size(); ++index) {
            fill_operation_optimization_info(report.operations[index], &operations[index]);
        }
        return 0;
    } catch (const std::exception& error) {
        set_last_error(runtime, error.what());
        return -3;
    }
}

int jakal_core_runtime_execute(
    jakal_core_runtime_t* runtime,
    const jakal_core_workload_spec* workload,
    jakal_core_execution_info* out_execution,
    jakal_core_execution_operation_info* operations,
    size_t capacity,
    size_t* out_count) {
    if (runtime == nullptr || workload == nullptr || out_count == nullptr) {
        return -1;
    }

    try {
        clear_last_error(runtime);
        runtime->last_managed_execution.reset();
        const auto report = runtime->runtime.execute(convert_workload(*workload));
        *out_count = report.operations.size();
        fill_execution_info(report, out_execution);

        if (operations == nullptr || capacity < report.operations.size()) {
            return -2;
        }

        for (size_t index = 0; index < report.operations.size(); ++index) {
            fill_execution_operation_info(report.operations[index], &operations[index]);
        }
        return 0;
    } catch (const std::exception& error) {
        runtime->last_managed_execution.reset();
        set_last_error(runtime, error.what());
        return -3;
    }
}

int jakal_core_runtime_execute_manifest(
    jakal_core_runtime_t* runtime,
    const char* manifest_path,
    jakal_core_execution_info* out_execution,
    jakal_core_execution_operation_info* operations,
    size_t capacity,
    size_t* out_count) {
    if (runtime == nullptr || manifest_path == nullptr || out_count == nullptr) {
        return -1;
    }

    try {
        clear_last_error(runtime);
        runtime->last_managed_execution = runtime->runtime.execute_manifest(manifest_path);
        const auto& report = *runtime->last_managed_execution;
        *out_count = report.execution.operations.size();
        fill_execution_info(report.execution, out_execution);

        if (operations == nullptr || capacity < report.execution.operations.size()) {
            return -2;
        }

        for (size_t index = 0; index < report.execution.operations.size(); ++index) {
            fill_execution_operation_info(report.execution.operations[index], &operations[index]);
        }
        return 0;
    } catch (const std::exception& error) {
        runtime->last_managed_execution.reset();
        set_last_error(runtime, error.what());
        return -3;
    }
}

size_t jakal_core_runtime_last_backend_buffer_binding_count(const jakal_core_runtime_t* runtime) {
    if (runtime == nullptr || !runtime->last_managed_execution.has_value()) {
        return 0u;
    }
    return runtime->last_managed_execution->backend_buffer_bindings.entries.size();
}

int jakal_core_runtime_get_last_backend_buffer_binding(
    const jakal_core_runtime_t* runtime,
    size_t index,
    jakal_core_backend_buffer_binding_info* out_binding) {
    if (runtime == nullptr || out_binding == nullptr) {
        return -1;
    }
    if (!runtime->last_managed_execution.has_value()) {
        return -2;
    }
    const auto& entries = runtime->last_managed_execution->backend_buffer_bindings.entries;
    if (index >= entries.size()) {
        return -3;
    }
    fill_backend_buffer_binding_info(entries[index], out_binding);
    return 0;
}

size_t jakal_core_runtime_last_residency_movement_count(const jakal_core_runtime_t* runtime) {
    if (runtime == nullptr || !runtime->last_managed_execution.has_value()) {
        return 0u;
    }
    return runtime->last_managed_execution->executed_residency_movements.entries.size();
}

int jakal_core_runtime_get_last_residency_movement(
    const jakal_core_runtime_t* runtime,
    size_t index,
    jakal_core_residency_movement_info* out_movement) {
    if (runtime == nullptr || out_movement == nullptr) {
        return -1;
    }
    if (!runtime->last_managed_execution.has_value()) {
        return -2;
    }
    const auto& entries = runtime->last_managed_execution->executed_residency_movements.entries;
    if (index >= entries.size()) {
        return -3;
    }
    fill_residency_movement_info(entries[index], out_movement);
    return 0;
}


