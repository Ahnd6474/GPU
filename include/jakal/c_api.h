#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jakal_core_runtime jakal_core_runtime_t;

typedef struct jakal_core_device_info {
    char uid[128];
    char probe[32];
    char presentation_name[128];
    unsigned int ordinal;
    unsigned int execution_objects;
    unsigned int lanes_per_object;
    unsigned int resident_contexts;
    unsigned int matrix_units;
    unsigned int clock_mhz;
    unsigned int queue_slots;
    unsigned int numa_domain;
    unsigned int native_vector_bits;
    unsigned long long addressable_memory_bytes;
    unsigned long long directly_attached_memory_bytes;
    unsigned long long shared_host_memory_bytes;
    unsigned long long local_scratch_bytes;
    unsigned long long cache_bytes;
    double host_read_gbps;
    double host_write_gbps;
    double dispatch_latency_us;
    double synchronization_latency_us;
    unsigned long long node_count;
    unsigned long long edge_count;
    int coherent_with_host;
    int unified_address_space;
    int host_visible;
    int supports_asynchronous_dispatch;
    int supports_fp64;
    int supports_fp32;
    int supports_fp16;
    int supports_bf16;
    int supports_int8;
} jakal_core_device_info;

typedef struct jakal_core_graph_node_info {
    char id[128];
    char label[128];
    char parent_id[128];
    char domain[32];
    char role[32];
    char resolution[32];
    unsigned int ordinal;
    unsigned int execution_width;
    unsigned int resident_contexts;
    unsigned int matrix_engines;
    unsigned int clock_mhz;
    unsigned int native_vector_bits;
    unsigned int queue_slots;
    unsigned int numa_domain;
    unsigned long long capacity_bytes;
    unsigned long long directly_attached_bytes;
    unsigned long long shared_host_bytes;
    double read_bandwidth_gbps;
    double write_bandwidth_gbps;
    double dispatch_latency_us;
    double synchronization_latency_us;
    int coherent_with_host;
    int unified_address_space;
    int host_visible;
    int supports_asynchronous_dispatch;
    int supports_fp64;
    int supports_fp32;
    int supports_fp16;
    int supports_bf16;
    int supports_int8;
} jakal_core_graph_node_info;

typedef struct jakal_core_graph_edge_info {
    char source_id[128];
    char target_id[128];
    char semantics[32];
    int directed;
    double weight;
    double bandwidth_gbps;
    double latency_us;
} jakal_core_graph_edge_info;

typedef struct jakal_core_workload_spec {
    const char* name;
    const char* kind;
    unsigned long long working_set_bytes;
    unsigned long long host_exchange_bytes;
    double estimated_flops;
    unsigned int batch_size;
    int latency_sensitive;
    int prefer_unified_memory;
    int matrix_friendly;
} jakal_core_workload_spec;

typedef struct jakal_core_plan_entry {
    jakal_core_device_info device;
    double ratio;
    double score;
} jakal_core_plan_entry;

typedef struct jakal_core_optimization_info {
    char signature[128];
    char workload_kind[32];
    char dataset_tag[128];
    unsigned long long operation_count;
    unsigned long long tensor_count;
    unsigned long long dependency_count;
    double readiness_score;
    double stability_score;
    double sustained_slowdown;
    int loaded_from_cache;
} jakal_core_optimization_info;

typedef struct jakal_core_operation_optimization_info {
    char operation_name[128];
    char strategy[32];
    char primary_device_uid[128];
    unsigned int logical_partitions;
    unsigned int participating_device_count;
    double predicted_latency_us;
    double predicted_speedup_vs_reference;
    double predicted_transfer_latency_us;
    double predicted_memory_pressure;
    unsigned long long peak_resident_bytes;
    double target_error_tolerance;
    int use_low_precision;
} jakal_core_operation_optimization_info;

typedef struct jakal_core_execution_info {
    char signature[128];
    unsigned long long operation_count;
    double total_runtime_us;
    double total_reference_runtime_us;
    double speedup_vs_reference;
    int all_succeeded;
} jakal_core_execution_info;

typedef struct jakal_core_execution_operation_info {
    char operation_name[128];
    char backend_name[64];
    char requested_gpu_vendor[32];
    char requested_gpu_backend[32];
    double runtime_us;
    double reference_runtime_us;
    double speedup_vs_reference;
    double relative_error;
    int verified;
    int used_host;
    int used_opencl;
    int used_multiple_devices;
    unsigned int logical_partitions_used;
} jakal_core_execution_operation_info;

jakal_core_runtime_t* jakal_core_runtime_create(void);
void jakal_core_runtime_destroy(jakal_core_runtime_t* runtime);
int jakal_core_runtime_refresh(jakal_core_runtime_t* runtime);
size_t jakal_core_runtime_device_count(const jakal_core_runtime_t* runtime);
int jakal_core_runtime_get_device(const jakal_core_runtime_t* runtime, size_t index, jakal_core_device_info* out_device);
size_t jakal_core_runtime_graph_node_count(const jakal_core_runtime_t* runtime, size_t device_index);
int jakal_core_runtime_get_graph_node(
    const jakal_core_runtime_t* runtime,
    size_t device_index,
    size_t node_index,
    jakal_core_graph_node_info* out_node);
size_t jakal_core_runtime_graph_edge_count(const jakal_core_runtime_t* runtime, size_t device_index);
int jakal_core_runtime_get_graph_edge(
    const jakal_core_runtime_t* runtime,
    size_t device_index,
    size_t edge_index,
    jakal_core_graph_edge_info* out_edge);
int jakal_core_runtime_plan(
    jakal_core_runtime_t* runtime,
    const jakal_core_workload_spec* workload,
    jakal_core_plan_entry* entries,
    size_t capacity,
    size_t* out_count,
    int* out_loaded_from_cache);
int jakal_core_runtime_optimize(
    jakal_core_runtime_t* runtime,
    const jakal_core_workload_spec* workload,
    jakal_core_optimization_info* out_optimization,
    jakal_core_operation_optimization_info* operations,
    size_t capacity,
    size_t* out_count);
int jakal_core_runtime_execute(
    jakal_core_runtime_t* runtime,
    const jakal_core_workload_spec* workload,
    jakal_core_execution_info* out_execution,
    jakal_core_execution_operation_info* operations,
    size_t capacity,
    size_t* out_count);

#ifdef __cplusplus
}
#endif

