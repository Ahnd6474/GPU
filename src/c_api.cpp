#include "jakal/c_api.h"

#include "jakal/runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <new>
#include <string>

struct jakal_core_runtime {
    jakal::Runtime runtime;
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

}  // namespace

jakal_core_runtime_t* jakal_core_runtime_create(void) {
    try {
        return new jakal_core_runtime_t{jakal::Runtime{}};
    } catch (const std::bad_alloc&) {
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

    runtime->runtime.refresh_hardware();
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

    const jakal::WorkloadSpec cpp_workload{
        workload->name == nullptr ? std::string("unnamed") : std::string(workload->name),
        parse_workload_kind(workload->kind),
        "",
        workload->working_set_bytes,
        workload->host_exchange_bytes,
        workload->estimated_flops,
        workload->batch_size,
        workload->latency_sensitive != 0,
        workload->prefer_unified_memory != 0,
        workload->matrix_friendly != 0};

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

    const jakal::WorkloadSpec cpp_workload{
        workload->name == nullptr ? std::string("unnamed") : std::string(workload->name),
        parse_workload_kind(workload->kind),
        "",
        workload->working_set_bytes,
        workload->host_exchange_bytes,
        workload->estimated_flops,
        workload->batch_size,
        workload->latency_sensitive != 0,
        workload->prefer_unified_memory != 0,
        workload->matrix_friendly != 0};

    const auto report = runtime->runtime.optimize(cpp_workload);
    *out_count = report.operations.size();
    fill_optimization_info(report, out_optimization);

    if (operations == nullptr || capacity < report.operations.size()) {
        return -2;
    }

    for (size_t index = 0; index < report.operations.size(); ++index) {
        fill_operation_optimization_info(report.operations[index], &operations[index]);
    }

    return 0;
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

    const jakal::WorkloadSpec cpp_workload{
        workload->name == nullptr ? std::string("unnamed") : std::string(workload->name),
        parse_workload_kind(workload->kind),
        "",
        workload->working_set_bytes,
        workload->host_exchange_bytes,
        workload->estimated_flops,
        workload->batch_size,
        workload->latency_sensitive != 0,
        workload->prefer_unified_memory != 0,
        workload->matrix_friendly != 0};

    const auto report = runtime->runtime.execute(cpp_workload);
    *out_count = report.operations.size();
    fill_execution_info(report, out_execution);

    if (operations == nullptr || capacity < report.operations.size()) {
        return -2;
    }

    for (size_t index = 0; index < report.operations.size(); ++index) {
        fill_execution_operation_info(report.operations[index], &operations[index]);
    }

    return 0;
}


