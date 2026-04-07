#include "jakal/workloads.hpp"

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <Windows.h>
#include <DirectML.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

struct CliOptions {
    std::filesystem::path manifest_path;
    std::string ollama_model;
    std::uint32_t passes = 3;
    bool show_operations = false;
};

struct ScopedEvent {
    HANDLE handle = nullptr;
    ScopedEvent() : handle(CreateEventW(nullptr, FALSE, FALSE, nullptr)) {}
    ~ScopedEvent() {
        if (handle != nullptr) {
            CloseHandle(handle);
        }
    }
};

struct DmlContext {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;
    ComPtr<ID3D12Fence> fence;
    ScopedEvent fence_event;
    UINT64 fence_value = 1;
    ComPtr<IDMLDevice> dml_device;
    ComPtr<IDMLCommandRecorder> recorder;
};

void throw_if_failed(const HRESULT hr, const char* message) {
    if (FAILED(hr)) {
        std::ostringstream stream;
        stream << message << " hr=0x" << std::hex << static_cast<unsigned long>(hr);
        throw std::runtime_error(stream.str());
    }
}

std::filesystem::path unique_temp_file(const std::string& stem, const std::string& extension) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / (stem + "-" + std::to_string(nonce) + extension);
}

void print_usage() {
    std::cout << "Usage: jakal_directml_manifest_bench <path-to-gguf-or-onnx> | --ollama-model <model>"
              << " [--passes N] [--show-ops]\n";
}

#define JAKAL_POPEN _popen
#define JAKAL_PCLOSE _pclose

std::string trim_copy(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1u);
}

std::string unquote_copy(const std::string& value) {
    if (value.size() >= 2u && value.front() == '"' && value.back() == '"') {
        return value.substr(1u, value.size() - 2u);
    }
    return value;
}

std::optional<std::filesystem::path> resolve_ollama_blob_path(const std::string& model_name) {
    const std::string command = "ollama show " + model_name + " --modelfile 2>NUL";
    FILE* pipe = JAKAL_POPEN(command.c_str(), "r");
    if (pipe == nullptr) {
        return std::nullopt;
    }

    std::string output;
    std::array<char, 512> buffer{};
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    const int exit_code = JAKAL_PCLOSE(pipe);
    if (exit_code != 0) {
        return std::nullopt;
    }

    std::istringstream stream(output);
    std::string line;
    while (std::getline(stream, line)) {
        line = trim_copy(line);
        if (!line.starts_with("FROM ")) {
            continue;
        }
        const auto path = std::filesystem::path(unquote_copy(trim_copy(line.substr(5u))));
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return std::nullopt;
}

std::optional<CliOptions> parse_args(const int argc, char** argv) {
    CliOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--passes") {
            if (index + 1 >= argc) {
                std::cerr << "--passes requires a value.\n";
                return std::nullopt;
            }
            options.passes = static_cast<std::uint32_t>((std::max)(1, std::stoi(argv[++index])));
            continue;
        }
        if (arg == "--ollama-model") {
            if (index + 1 >= argc) {
                std::cerr << "--ollama-model requires a value.\n";
                return std::nullopt;
            }
            options.ollama_model = argv[++index];
            continue;
        }
        if (arg == "--show-ops") {
            options.show_operations = true;
            continue;
        }
        if (!arg.empty() && arg.front() == '-') {
            std::cerr << "Unknown option: " << arg << '\n';
            return std::nullopt;
        }
        if (!options.ollama_model.empty() || !options.manifest_path.empty()) {
            std::cerr << "Choose one input source only.\n";
            return std::nullopt;
        }
        options.manifest_path = arg;
    }

    if (options.manifest_path.empty() && options.ollama_model.empty()) {
        print_usage();
        return std::nullopt;
    }
    return options;
}

std::vector<UINT> to_dims_4d(
    const std::uint64_t n,
    const std::uint64_t c,
    const std::uint64_t h,
    const std::uint64_t w) {
    return {
        static_cast<UINT>(std::max<std::uint64_t>(1u, n)),
        static_cast<UINT>(std::max<std::uint64_t>(1u, c)),
        static_cast<UINT>(std::max<std::uint64_t>(1u, h)),
        static_cast<UINT>(std::max<std::uint64_t>(1u, w))};
}

DML_TENSOR_DESC make_buffer_tensor_desc(
    const std::vector<UINT>& sizes,
    const std::uint64_t total_bytes,
    DML_BUFFER_TENSOR_DESC* buffer_desc) {
    buffer_desc->DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    buffer_desc->Flags = DML_TENSOR_FLAG_NONE;
    buffer_desc->DimensionCount = static_cast<UINT>(sizes.size());
    buffer_desc->Sizes = sizes.data();
    buffer_desc->Strides = nullptr;
    buffer_desc->TotalTensorSizeInBytes = total_bytes;
    buffer_desc->GuaranteedBaseOffsetAlignment = 0u;
    return DML_TENSOR_DESC{DML_TENSOR_TYPE_BUFFER, buffer_desc};
}

ComPtr<ID3D12Resource> create_buffer(
    ID3D12Device* device,
    const std::uint64_t size_bytes) {
    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resource_desc{};
    resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resource_desc.Width = std::max<std::uint64_t>(size_bytes, 4u);
    resource_desc.Height = 1;
    resource_desc.DepthOrArraySize = 1;
    resource_desc.MipLevels = 1;
    resource_desc.SampleDesc.Count = 1;
    resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resource_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ComPtr<ID3D12Resource> resource;
    throw_if_failed(
        device->CreateCommittedResource(
            &heap_properties,
            D3D12_HEAP_FLAG_NONE,
            &resource_desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&resource)),
        "CreateCommittedResource");
    return resource;
}

void submit_and_wait(DmlContext& context) {
    throw_if_failed(context.command_list->Close(), "Close command list");
    ID3D12CommandList* lists[] = {context.command_list.Get()};
    context.queue->ExecuteCommandLists(1u, lists);
    throw_if_failed(context.queue->Signal(context.fence.Get(), context.fence_value), "Signal fence");
    if (context.fence->GetCompletedValue() < context.fence_value) {
        throw_if_failed(
            context.fence->SetEventOnCompletion(context.fence_value, context.fence_event.handle),
            "SetEventOnCompletion");
        WaitForSingleObject(context.fence_event.handle, INFINITE);
    }
    throw_if_failed(context.device->GetDeviceRemovedReason(), "DirectML device removed");
    ++context.fence_value;
    throw_if_failed(context.allocator->Reset(), "Reset allocator");
    throw_if_failed(context.command_list->Reset(context.allocator.Get(), nullptr), "Reset command list");
}

DmlContext create_context() {
    DmlContext context;
    throw_if_failed(CreateDXGIFactory2(0u, IID_PPV_ARGS(&context.factory)), "CreateDXGIFactory2");

    for (UINT index = 0; ; ++index) {
        ComPtr<IDXGIAdapter1> candidate;
        if (context.factory->EnumAdapters1(index, &candidate) == DXGI_ERROR_NOT_FOUND) {
            break;
        }

        DXGI_ADAPTER_DESC1 desc{};
        candidate->GetDesc1(&desc);
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            continue;
        }
        if (FAILED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
            continue;
        }
        context.adapter = candidate;
        break;
    }

    if (!context.adapter) {
        throw std::runtime_error("No hardware D3D12 adapter found for DirectML benchmark.");
    }

    throw_if_failed(
        D3D12CreateDevice(context.adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&context.device)),
        "D3D12CreateDevice");

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    throw_if_failed(context.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&context.queue)), "CreateCommandQueue");
    throw_if_failed(
        context.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&context.allocator)),
        "CreateCommandAllocator");
    throw_if_failed(
        context.device->CreateCommandList(
            0u,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            context.allocator.Get(),
            nullptr,
            IID_PPV_ARGS(&context.command_list)),
        "CreateCommandList");
    throw_if_failed(context.command_list->Close(), "Close initial command list");
    throw_if_failed(context.allocator->Reset(), "Reset initial allocator");
    throw_if_failed(context.command_list->Reset(context.allocator.Get(), nullptr), "Reset initial command list");
    throw_if_failed(context.device->CreateFence(0u, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&context.fence)), "CreateFence");
    throw_if_failed(
        DMLCreateDevice1(
            context.device.Get(),
            DML_CREATE_DEVICE_FLAG_NONE,
            DML_FEATURE_LEVEL_2_0,
            IID_PPV_ARGS(&context.dml_device)),
        "DMLCreateDevice1");
    throw_if_failed(context.dml_device->CreateCommandRecorder(IID_PPV_ARGS(&context.recorder)), "CreateCommandRecorder");
    return context;
}

double dispatch_operator(
    DmlContext& context,
    IDMLCompiledOperator* compiled_operator,
    const std::vector<DML_BINDING_DESC>& input_bindings,
    const std::vector<DML_BINDING_DESC>& output_bindings) {
    const auto exec_props = compiled_operator->GetBindingProperties();

    ComPtr<IDMLOperatorInitializer> initializer;
    IDMLCompiledOperator* compiled_ops[] = {compiled_operator};
    throw_if_failed(
        context.dml_device->CreateOperatorInitializer(1u, compiled_ops, IID_PPV_ARGS(&initializer)),
        "CreateOperatorInitializer");
    const auto init_props = initializer->GetBindingProperties();

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors =
        (std::max)(1u, (std::max)(init_props.RequiredDescriptorCount, exec_props.RequiredDescriptorCount));
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> descriptor_heap;
    throw_if_failed(context.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&descriptor_heap)), "CreateDescriptorHeap");

    const auto persistent_size = (std::max)(init_props.PersistentResourceSize, exec_props.PersistentResourceSize);
    const auto init_temp_size = init_props.TemporaryResourceSize;
    const auto exec_temp_size = exec_props.TemporaryResourceSize;

    ComPtr<ID3D12Resource> persistent_resource;
    if (persistent_size > 0u) {
        persistent_resource = create_buffer(context.device.Get(), persistent_size);
    }
    ComPtr<ID3D12Resource> init_temp_resource;
    if (init_temp_size > 0u) {
        init_temp_resource = create_buffer(context.device.Get(), init_temp_size);
    }
    ComPtr<ID3D12Resource> exec_temp_resource;
    if (exec_temp_size > 0u) {
        exec_temp_resource = create_buffer(context.device.Get(), exec_temp_size);
    }

    DML_BINDING_TABLE_DESC table_desc{};
    table_desc.Dispatchable = initializer.Get();
    table_desc.CPUDescriptorHandle = descriptor_heap->GetCPUDescriptorHandleForHeapStart();
    table_desc.GPUDescriptorHandle = descriptor_heap->GetGPUDescriptorHandleForHeapStart();
    table_desc.SizeInDescriptors = heap_desc.NumDescriptors;

    ComPtr<IDMLBindingTable> binding_table;
    throw_if_failed(context.dml_device->CreateBindingTable(&table_desc, IID_PPV_ARGS(&binding_table)), "CreateBindingTable");

    ID3D12DescriptorHeap* heaps[] = {descriptor_heap.Get()};
    context.command_list->SetDescriptorHeaps(1u, heaps);

    DML_BINDING_DESC none_binding{DML_BINDING_TYPE_NONE, nullptr};
    DML_BUFFER_BINDING persistent_buffer_binding{persistent_resource.Get(), 0u, persistent_size};
    DML_BINDING_DESC persistent_binding{
        persistent_resource ? DML_BINDING_TYPE_BUFFER : DML_BINDING_TYPE_NONE,
        persistent_resource ? static_cast<const void*>(&persistent_buffer_binding) : nullptr};
    DML_BUFFER_BINDING init_temp_binding{init_temp_resource.Get(), 0u, init_temp_size};
    DML_BINDING_DESC init_temp_desc{
        init_temp_resource ? DML_BINDING_TYPE_BUFFER : DML_BINDING_TYPE_NONE,
        init_temp_resource ? static_cast<const void*>(&init_temp_binding) : nullptr};

    binding_table->BindTemporaryResource(init_temp_resource ? &init_temp_desc : &none_binding);
    binding_table->BindPersistentResource(persistent_resource ? &persistent_binding : &none_binding);
    context.recorder->RecordDispatch(context.command_list.Get(), initializer.Get(), binding_table.Get());
    submit_and_wait(context);

    table_desc.Dispatchable = compiled_operator;
    binding_table->Reset(&table_desc);

    DML_BUFFER_BINDING exec_temp_binding{exec_temp_resource.Get(), 0u, exec_temp_size};
    DML_BINDING_DESC exec_temp_desc{
        exec_temp_resource ? DML_BINDING_TYPE_BUFFER : DML_BINDING_TYPE_NONE,
        exec_temp_resource ? static_cast<const void*>(&exec_temp_binding) : nullptr};

    binding_table->BindTemporaryResource(exec_temp_resource ? &exec_temp_desc : &none_binding);
    binding_table->BindPersistentResource(persistent_resource ? &persistent_binding : &none_binding);
    binding_table->BindInputs(static_cast<UINT>(input_bindings.size()), input_bindings.data());
    binding_table->BindOutputs(static_cast<UINT>(output_bindings.size()), output_bindings.data());

    const auto start = std::chrono::steady_clock::now();
    context.command_list->SetDescriptorHeaps(1u, heaps);
    context.recorder->RecordDispatch(context.command_list.Get(), compiled_operator, binding_table.Get());
    submit_and_wait(context);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

double benchmark_elementwise(DmlContext& context, const jakal::OperationSpec& operation) {
    const auto elements = std::max<std::uint64_t>(1u, operation.output_bytes / sizeof(float));
    const auto sizes = to_dims_4d(1u, 1u, 1u, elements);
    DML_BUFFER_TENSOR_DESC input_buffer{};
    DML_BUFFER_TENSOR_DESC output_buffer{};
    auto input_tensor = make_buffer_tensor_desc(sizes, elements * sizeof(float), &input_buffer);
    auto output_tensor = make_buffer_tensor_desc(sizes, elements * sizeof(float), &output_buffer);

    DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc{};
    identity_desc.InputTensor = &input_tensor;
    identity_desc.OutputTensor = &output_tensor;
    identity_desc.ScaleBias = nullptr;

    DML_OPERATOR_DESC operator_desc{DML_OPERATOR_ELEMENT_WISE_IDENTITY, &identity_desc};
    ComPtr<IDMLOperator> op;
    ComPtr<IDMLCompiledOperator> compiled;
    throw_if_failed(context.dml_device->CreateOperator(&operator_desc, IID_PPV_ARGS(&op)), "Create identity operator");
    throw_if_failed(
        context.dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE, IID_PPV_ARGS(&compiled)),
        "Compile identity operator");

    auto input_resource = create_buffer(context.device.Get(), elements * sizeof(float));
    auto output_resource = create_buffer(context.device.Get(), elements * sizeof(float));
    DML_BUFFER_BINDING input_binding{input_resource.Get(), 0u, elements * sizeof(float)};
    DML_BUFFER_BINDING output_binding{output_resource.Get(), 0u, elements * sizeof(float)};
    const DML_BINDING_DESC input_desc{DML_BINDING_TYPE_BUFFER, &input_binding};
    const DML_BINDING_DESC output_desc{DML_BINDING_TYPE_BUFFER, &output_binding};

    return dispatch_operator(context, compiled.Get(), {input_desc}, {output_desc});
}

double benchmark_reduction(DmlContext& context, const jakal::OperationSpec& operation) {
    const auto elements = std::max<std::uint64_t>(1u, operation.input_bytes / sizeof(float));
    const auto input_sizes = to_dims_4d(1u, 1u, 1u, elements);
    const auto output_sizes = to_dims_4d(1u, 1u, 1u, 1u);
    const UINT axes[] = {3u};
    DML_BUFFER_TENSOR_DESC input_buffer{};
    DML_BUFFER_TENSOR_DESC output_buffer{};
    auto input_tensor = make_buffer_tensor_desc(input_sizes, elements * sizeof(float), &input_buffer);
    auto output_tensor = make_buffer_tensor_desc(output_sizes, sizeof(float), &output_buffer);

    DML_REDUCE_OPERATOR_DESC reduce_desc{};
    reduce_desc.Function = DML_REDUCE_FUNCTION_SUM;
    reduce_desc.InputTensor = &input_tensor;
    reduce_desc.OutputTensor = &output_tensor;
    reduce_desc.AxisCount = 1u;
    reduce_desc.Axes = axes;

    DML_OPERATOR_DESC operator_desc{DML_OPERATOR_REDUCE, &reduce_desc};
    ComPtr<IDMLOperator> op;
    ComPtr<IDMLCompiledOperator> compiled;
    throw_if_failed(context.dml_device->CreateOperator(&operator_desc, IID_PPV_ARGS(&op)), "Create reduce operator");
    throw_if_failed(
        context.dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE, IID_PPV_ARGS(&compiled)),
        "Compile reduce operator");

    auto input_resource = create_buffer(context.device.Get(), elements * sizeof(float));
    auto output_resource = create_buffer(context.device.Get(), sizeof(float));
    DML_BUFFER_BINDING input_binding{input_resource.Get(), 0u, elements * sizeof(float)};
    DML_BUFFER_BINDING output_binding{output_resource.Get(), 0u, sizeof(float)};
    const DML_BINDING_DESC input_desc{DML_BINDING_TYPE_BUFFER, &input_binding};
    const DML_BINDING_DESC output_desc{DML_BINDING_TYPE_BUFFER, &output_binding};

    return dispatch_operator(context, compiled.Get(), {input_desc}, {output_desc});
}

double benchmark_matmul(DmlContext& context, const jakal::OperationSpec& operation) {
    const auto rows = static_cast<UINT>(operation.extents.size() > 0u ? std::max<std::uint64_t>(1u, operation.extents[0]) : 1u);
    const auto columns = static_cast<UINT>(operation.extents.size() > 1u ? std::max<std::uint64_t>(1u, operation.extents[1]) : 1u);
    const auto depth = static_cast<UINT>(operation.extents.size() > 2u ? std::max<std::uint64_t>(1u, operation.extents[2]) : 1u);
    const auto a_sizes = to_dims_4d(1u, 1u, rows, depth);
    const auto b_sizes = to_dims_4d(1u, 1u, depth, columns);
    const auto out_sizes = to_dims_4d(1u, 1u, rows, columns);

    DML_BUFFER_TENSOR_DESC a_buffer{};
    DML_BUFFER_TENSOR_DESC b_buffer{};
    DML_BUFFER_TENSOR_DESC out_buffer{};
    auto a_tensor = make_buffer_tensor_desc(a_sizes, static_cast<std::uint64_t>(rows) * depth * sizeof(float), &a_buffer);
    auto b_tensor = make_buffer_tensor_desc(b_sizes, static_cast<std::uint64_t>(depth) * columns * sizeof(float), &b_buffer);
    auto out_tensor = make_buffer_tensor_desc(out_sizes, static_cast<std::uint64_t>(rows) * columns * sizeof(float), &out_buffer);

    DML_GEMM_OPERATOR_DESC gemm_desc{};
    gemm_desc.ATensor = &a_tensor;
    gemm_desc.BTensor = &b_tensor;
    gemm_desc.CTensor = nullptr;
    gemm_desc.OutputTensor = &out_tensor;
    gemm_desc.TransA = DML_MATRIX_TRANSFORM_NONE;
    gemm_desc.TransB = DML_MATRIX_TRANSFORM_NONE;
    gemm_desc.Alpha = 1.0f;
    gemm_desc.Beta = 0.0f;
    gemm_desc.FusedActivation = nullptr;

    DML_OPERATOR_DESC operator_desc{DML_OPERATOR_GEMM, &gemm_desc};
    ComPtr<IDMLOperator> op;
    ComPtr<IDMLCompiledOperator> compiled;
    throw_if_failed(context.dml_device->CreateOperator(&operator_desc, IID_PPV_ARGS(&op)), "Create gemm operator");
    throw_if_failed(
        context.dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE, IID_PPV_ARGS(&compiled)),
        "Compile gemm operator");

    auto a_resource = create_buffer(context.device.Get(), static_cast<std::uint64_t>(rows) * depth * sizeof(float));
    auto b_resource = create_buffer(context.device.Get(), static_cast<std::uint64_t>(depth) * columns * sizeof(float));
    auto out_resource = create_buffer(context.device.Get(), static_cast<std::uint64_t>(rows) * columns * sizeof(float));
    DML_BUFFER_BINDING a_binding{a_resource.Get(), 0u, static_cast<std::uint64_t>(rows) * depth * sizeof(float)};
    DML_BUFFER_BINDING b_binding{b_resource.Get(), 0u, static_cast<std::uint64_t>(depth) * columns * sizeof(float)};
    DML_BUFFER_BINDING out_binding{out_resource.Get(), 0u, static_cast<std::uint64_t>(rows) * columns * sizeof(float)};
    const DML_BINDING_DESC input_bindings[] = {
        {DML_BINDING_TYPE_BUFFER, &a_binding},
        {DML_BINDING_TYPE_BUFFER, &b_binding}};
    const DML_BINDING_DESC output_bindings[] = {
        {DML_BINDING_TYPE_BUFFER, &out_binding}};

    return dispatch_operator(context, compiled.Get(), {input_bindings[0], input_bindings[1]}, {output_bindings[0]});
}

double benchmark_operation(DmlContext& context, const jakal::OperationSpec& operation) {
    switch (operation.op_class) {
    case jakal::OperationClass::matmul:
        return benchmark_matmul(context, operation);
    case jakal::OperationClass::reduction:
        return benchmark_reduction(context, operation);
    case jakal::OperationClass::elementwise_map:
    default:
        return benchmark_elementwise(context, operation);
    }
}

std::string adapter_name(IDXGIAdapter1* adapter) {
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    char name[128]{};
    std::wcstombs(name, desc.Description, sizeof(name) - 1u);
    return name;
}

std::string format_extents(const std::vector<std::uint64_t>& extents) {
    std::ostringstream stream;
    for (std::size_t index = 0; index < extents.size(); ++index) {
        if (index != 0u) {
            stream << 'x';
        }
        stream << extents[index];
    }
    return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto parsed = parse_args(argc, argv);
        if (!parsed.has_value()) {
            return 1;
        }

        auto options = *parsed;
        if (!options.ollama_model.empty()) {
            const auto resolved = resolve_ollama_blob_path(options.ollama_model);
            if (!resolved.has_value()) {
                std::cerr << "Unable to resolve Ollama model: " << options.ollama_model << '\n';
                return 1;
            }
            options.manifest_path = *resolved;
        }

        if (!std::filesystem::exists(options.manifest_path)) {
            std::cerr << "Input path does not exist: " << options.manifest_path << '\n';
            return 1;
        }

        const auto manifest = jakal::load_workload_source(options.manifest_path);
        auto context = create_context();

        std::cout << "DirectML manifest bench\n";
        if (!options.ollama_model.empty()) {
            std::cout << "  ollama_model=" << options.ollama_model << '\n';
        }
        std::cout << "  source=" << options.manifest_path.string() << '\n';
        std::cout << "  adapter=" << adapter_name(context.adapter.Get()) << '\n';
        std::cout << "  format=" << manifest.source_format
                  << " imported=" << (manifest.imported ? "yes" : "no")
                  << " ops=" << manifest.graph.operations.size()
                  << " tensors=" << manifest.graph.tensors.size()
                  << '\n';

        for (std::uint32_t pass = 1; pass <= options.passes; ++pass) {
            double total_runtime_us = 0.0;
            std::size_t executed_ops = 0;
            std::size_t skipped_ops = 0;
            std::cout << "pass=" << pass;
            for (const auto& operation : manifest.graph.operations) {
                if (operation.op_class == jakal::OperationClass::convolution_2d ||
                    operation.op_class == jakal::OperationClass::resample_2d) {
                    ++skipped_ops;
                    continue;
                }
                double runtime_us = 0.0;
                try {
                    auto op_context = create_context();
                    runtime_us = benchmark_operation(op_context, operation);
                } catch (const std::exception& error) {
                    std::cerr << "\nDirectML op failed: name=" << operation.name
                              << " class=" << static_cast<int>(operation.op_class)
                              << " extents=" << format_extents(operation.extents)
                              << " input_bytes=" << operation.input_bytes
                              << " output_bytes=" << operation.output_bytes
                              << " error=" << error.what() << '\n';
                    throw;
                }
                total_runtime_us += runtime_us;
                ++executed_ops;
                if (options.show_operations) {
                    std::cout << "\n  op=" << std::setw(24) << std::left << operation.name
                              << " class=" << static_cast<int>(operation.op_class)
                              << " extents=" << format_extents(operation.extents)
                              << " runtime_us=" << std::fixed << std::setprecision(3) << runtime_us;
                }
            }
            std::cout << " total_us=" << std::fixed << std::setprecision(3) << total_runtime_us
                      << " executed_ops=" << executed_ops
                      << " skipped_ops=" << skipped_ops
                      << '\n';
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DirectML benchmark failed: " << error.what() << '\n';
        return 1;
    }
}

#else

#include <iostream>

int main() {
    std::cerr << "DirectML benchmark is only supported on Windows.\n";
    return 1;
}

#endif
