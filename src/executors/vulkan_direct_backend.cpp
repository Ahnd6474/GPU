#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "jakal/executors/direct_backends.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace jakal::executors {

bool vulkan_direct_backend_available_internal();
std::string vulkan_direct_backend_status_detail_internal();
std::unique_ptr<IKernelBackend> make_vulkan_direct_kernel_backend_internal();

namespace {

struct VulkanDirectProbeState {
    bool available = false;
    std::string detail = "probe not run";
};

#if defined(_WIN32)
using LibraryHandle = HMODULE;

struct ProcessResult {
    int exit_code = -1;
    std::string output;
};

LibraryHandle load_library(const char* name) {
    return LoadLibraryA(name);
}

void* load_symbol(LibraryHandle library, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(library, name));
}

void close_library(LibraryHandle library) {
    if (library != nullptr) {
        FreeLibrary(library);
    }
}

ProcessResult run_process_capture_output(const std::string& command_line) {
    ProcessResult result;

    SECURITY_ATTRIBUTES security_attributes{};
    security_attributes.nLength = sizeof(security_attributes);
    security_attributes.bInheritHandle = TRUE;

    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &security_attributes, 0)) {
        result.output = "CreatePipe failed";
        return result;
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA startup_info{};
    startup_info.cb = sizeof(startup_info);
    startup_info.dwFlags = STARTF_USESTDHANDLES;
    startup_info.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup_info.hStdOutput = write_pipe;
    startup_info.hStdError = write_pipe;

    PROCESS_INFORMATION process_info{};
    auto mutable_command = command_line;
    const BOOL created = CreateProcessA(
        nullptr,
        mutable_command.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &startup_info,
        &process_info);

    CloseHandle(write_pipe);
    write_pipe = nullptr;

    if (!created) {
        result.output = "CreateProcess failed";
        CloseHandle(read_pipe);
        return result;
    }

    char buffer[4096];
    DWORD read = 0;
    while (ReadFile(read_pipe, buffer, sizeof(buffer), &read, nullptr) && read > 0) {
        result.output.append(buffer, buffer + read);
    }

    WaitForSingleObject(process_info.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process_info.hProcess, &exit_code);
    result.exit_code = static_cast<int>(exit_code);

    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);
    CloseHandle(read_pipe);
    return result;
}
#else
using LibraryHandle = void*;

LibraryHandle load_library(const char* name) {
    return dlopen(name, RTLD_LAZY);
}

void* load_symbol(LibraryHandle library, const char* name) {
    return dlsym(library, name);
}

void close_library(LibraryHandle library) {
    if (library != nullptr) {
        dlclose(library);
    }
}
#endif

template <typename Func>
double measure_us(Func&& func) {
    const auto start = std::chrono::steady_clock::now();
    func();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

float quantize_value(const float value, const bool low_precision) {
    if (!low_precision) {
        return value;
    }
    return std::round(value * 1024.0f) / 1024.0f;
}

double estimate_transfer_runtime_us(
    const HardwareGraph& graph,
    const std::size_t bytes,
    const bool write_direction) {
    if (bytes == 0u) {
        return 0.0;
    }
    const auto summary = summarize_graph(graph);
    const double bandwidth_gbps =
        std::max(write_direction ? summary.host_write_gbps : summary.host_read_gbps, 1.0);
    const double payload_us =
        (static_cast<double>(bytes) / (bandwidth_gbps * 1.0e9)) * 1.0e6;
    return payload_us + std::max(summary.dispatch_latency_us * 0.18, 0.25);
}

struct CompilerCommand {
    std::string command;
    bool uses_glslc = false;
};

std::string command_probe(const std::string& command) {
#if defined(_WIN32)
    return "where \"" + command + "\" >nul 2>nul";
#else
    return "command -v \"" + command + "\" >/dev/null 2>&1";
#endif
}

std::optional<CompilerCommand> locate_shader_compiler() {
    std::vector<std::string> candidates;
    if (const char* sdk = std::getenv("VULKAN_SDK")) {
        const auto root = std::filesystem::path(sdk);
#if defined(_WIN32)
        candidates.push_back((root / "Bin" / "glslc.exe").string());
        candidates.push_back((root / "Bin" / "glslangValidator.exe").string());
#else
        candidates.push_back((root / "bin" / "glslc").string());
        candidates.push_back((root / "bin" / "glslangValidator").string());
#endif
    }
    candidates.push_back("glslc");
    candidates.push_back("glslangValidator");

    for (const auto& candidate : candidates) {
        if (candidate.empty()) {
            continue;
        }
        std::error_code ec;
        if (!std::filesystem::exists(candidate, ec) &&
            std::system(command_probe(candidate).c_str()) != 0) {
            continue;
        }
        return CompilerCommand{candidate, candidate.find("glslc") != std::string::npos};
    }

    return std::nullopt;
}

std::filesystem::path temp_shader_path(const std::string& key, const char* extension) {
    std::hash<std::string> hasher;
    auto path = std::filesystem::temp_directory_path();
    path /= "jakal_vulkan_" + std::to_string(hasher(key)) + extension;
    return path;
}

bool write_text_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.is_open()) {
        return false;
    }
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    return stream.good();
}

std::optional<std::vector<std::uint32_t>> compile_glsl(
    const CompilerCommand& compiler,
    const std::string& key,
    const std::string& source,
    std::string* error_detail = nullptr) {
    const auto source_path = temp_shader_path(key, ".comp");
    const auto spirv_path = temp_shader_path(key, ".spv");
    if (!write_text_file(source_path, source)) {
        if (error_detail != nullptr) {
            *error_detail = "failed to write temporary shader source";
        }
        return std::nullopt;
    }

    std::string command;
    if (compiler.uses_glslc) {
        command = "\"" + compiler.command + "\" -fshader-stage=compute \"" +
                  source_path.string() + "\" -o \"" + spirv_path.string() + "\"";
    } else {
        command = "\"" + compiler.command + "\" -V --target-env vulkan1.0 -S comp \"" +
                  source_path.string() + "\" -o \"" + spirv_path.string() + "\"";
    }

#if defined(_WIN32)
    const auto process = run_process_capture_output(command);
    const bool success = process.exit_code == 0;
#else
    const auto log_path = temp_shader_path(key, ".log");
    command += " >\"" + log_path.string() + "\" 2>&1";
    const bool success = std::system(command.c_str()) == 0;
#endif
    std::ifstream stream(spirv_path, std::ios::binary);
    if (!success || !stream.is_open()) {
        if (error_detail != nullptr) {
#if defined(_WIN32)
            *error_detail = process.output;
#else
            std::ifstream log_stream(log_path, std::ios::binary);
            if (log_stream.is_open()) {
                *error_detail = std::string(
                    (std::istreambuf_iterator<char>(log_stream)),
                    std::istreambuf_iterator<char>());
            }
#endif
            if (error_detail->empty()) {
                *error_detail = success ? "shader compiler did not emit SPIR-V output" : "shader compiler returned failure";
            }
        }
        std::error_code ec;
        std::filesystem::remove(source_path, ec);
        std::filesystem::remove(spirv_path, ec);
#if !defined(_WIN32)
        std::filesystem::remove(log_path, ec);
#endif
        return std::nullopt;
    }

    stream.seekg(0, std::ios::end);
    const auto size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (size <= 0 || (static_cast<std::size_t>(size) % sizeof(std::uint32_t)) != 0u) {
        std::error_code ec;
        std::filesystem::remove(source_path, ec);
        std::filesystem::remove(spirv_path, ec);
#if !defined(_WIN32)
        std::filesystem::remove(log_path, ec);
#endif
        return std::nullopt;
    }

    std::vector<std::uint32_t> words(static_cast<std::size_t>(size) / sizeof(std::uint32_t));
    stream.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(size));
    std::error_code ec;
    std::filesystem::remove(source_path, ec);
    std::filesystem::remove(spirv_path, ec);
#if !defined(_WIN32)
    std::filesystem::remove(log_path, ec);
#endif
    if (!stream.good() && !stream.eof()) {
        if (error_detail != nullptr) {
            *error_detail = "failed to read compiled SPIR-V output";
        }
        return std::nullopt;
    }
    return words;
}

std::filesystem::path vulkan_cache_root() {
#if defined(_WIN32)
    if (const char* local_app_data = std::getenv("LOCALAPPDATA")) {
        return std::filesystem::path(local_app_data) / "Jakal-Core" / "vulkan-cache";
    }
#else
    if (const char* xdg_cache_home = std::getenv("XDG_CACHE_HOME")) {
        return std::filesystem::path(xdg_cache_home) / "jakal-core" / "vulkan-cache";
    }
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".cache" / "jakal-core" / "vulkan-cache";
    }
#endif
    return std::filesystem::temp_directory_path() / "jakal-core" / "vulkan-cache";
}

std::filesystem::path shader_cache_path(const std::string& key) {
    return vulkan_cache_root() / (key + ".spv");
}

std::filesystem::path pipeline_cache_path() {
    return vulkan_cache_root() / "pipelines.bin";
}

std::optional<std::vector<std::byte>> load_cached_blob(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        return std::nullopt;
    }
    stream.seekg(0, std::ios::end);
    const auto size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (size <= 0) {
        return std::nullopt;
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (!stream.good() && !stream.eof()) {
        return std::nullopt;
    }
    return bytes;
}

void store_cached_blob(const std::filesystem::path& path, const void* data, const std::size_t size) {
    if (data == nullptr || size == 0u) {
        return;
    }
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.is_open()) {
        return;
    }
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}

std::optional<std::vector<std::uint32_t>> load_cached_spirv(const std::string& key) {
    const auto path = shader_cache_path(key);
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        return std::nullopt;
    }
    stream.seekg(0, std::ios::end);
    const auto size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (size <= 0 || (static_cast<std::size_t>(size) % sizeof(std::uint32_t)) != 0u) {
        return std::nullopt;
    }
    std::vector<std::uint32_t> words(static_cast<std::size_t>(size) / sizeof(std::uint32_t));
    stream.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(size));
    if (!stream.good() && !stream.eof()) {
        return std::nullopt;
    }
    return words;
}

void store_cached_spirv(const std::string& key, const std::vector<std::uint32_t>& words) {
    const auto path = shader_cache_path(key);
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.is_open()) {
        return;
    }
    stream.write(
        reinterpret_cast<const char*>(words.data()),
        static_cast<std::streamsize>(words.size() * sizeof(std::uint32_t)));
}

std::optional<std::vector<std::uint32_t>> load_or_compile_glsl(
    const std::string& key,
    const std::string& source,
    std::string* error_detail = nullptr) {
    if (const auto cached = load_cached_spirv(key)) {
        return cached;
    }
    const auto compiler = locate_shader_compiler();
    if (!compiler.has_value()) {
        if (error_detail != nullptr) {
            *error_detail = "no shader compiler found";
        }
        return std::nullopt;
    }
    const auto words = compile_glsl(*compiler, key, source, error_detail);
    if (words.has_value()) {
        store_cached_spirv(key, *words);
    }
    return words;
}

enum class ShaderKind {
    elementwise,
    reduction,
    matmul,
    conv3x3,
    resample,
};

std::string shader_source(const ShaderKind kind) {
    const std::string prefix = R"GLSL(
#version 450
layout(set = 0, binding = 0) readonly buffer Input0 { float values[]; } input0_;
layout(set = 0, binding = 1) readonly buffer Input1 { float values[]; } input1_;
layout(set = 0, binding = 2) buffer Output { float values[]; } output_;
layout(push_constant) uniform Push {
    uint p0;
    uint p1;
    uint p2;
    uint p3;
    uint p4;
    uint p5;
    uint p6;
    uint p7;
    int low_precision;
    int flags;
} pc;
float q(float value) { return pc.low_precision == 0 ? value : round(value * 1024.0) / 1024.0; }
)GLSL";

    if (kind == ShaderKind::elementwise) {
        return prefix + R"GLSL(
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.p0) {
        return;
    }
    float left = q(input0_.values[gid] * 1.125);
    float right = q(input1_.values[gid] * 0.25);
    output_.values[gid] = q(left + right - 0.03125);
}
)GLSL";
    }

    if (kind == ShaderKind::reduction) {
        return prefix + R"GLSL(
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float scratch[256];
void main() {
    uint lid = gl_LocalInvocationID.x;
    float acc = 0.0;
    for (uint index = lid; index < pc.p0; index += gl_WorkGroupSize.x) {
        acc = q(acc + input0_.values[index]);
    }
    scratch[lid] = acc;
    barrier();
    for (uint stride = gl_WorkGroupSize.x >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = q(scratch[lid] + scratch[lid + stride]);
        }
        barrier();
    }
    if (lid == 0) {
        output_.values[0] = scratch[0];
    }
}
)GLSL";
    }

    if (kind == ShaderKind::matmul) {
        return prefix + R"GLSL(
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main() {
    uint col = gl_GlobalInvocationID.x;
    uint row = gl_GlobalInvocationID.y;
    if (row >= pc.p0 || col >= pc.p1) {
        return;
    }
    float acc = 0.0;
    for (uint inner = 0; inner < pc.p2; ++inner) {
        float left = q(input0_.values[row * pc.p2 + inner]);
        uint rhs_index = (pc.flags & 1) != 0 ? (col * pc.p2 + inner) : (inner * pc.p1 + col);
        float right = q(input1_.values[rhs_index]);
        acc = q(acc + (left * right));
    }
    output_.values[row * pc.p1 + col] = acc;
}
)GLSL";
    }

    if (kind == ShaderKind::conv3x3) {
        return prefix + R"GLSL(
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
const float kernel_[9] = float[9](
    0.0625, 0.125, 0.0625,
    0.125, 0.25, 0.125,
    0.0625, 0.125, 0.0625);
void main() {
    uint out_x = gl_GlobalInvocationID.x;
    uint out_y = gl_GlobalInvocationID.y;
    uint out_h = pc.p0 - 2;
    uint out_w = pc.p1 - 2;
    if (out_x >= out_w || out_y >= out_h) {
        return;
    }
    float acc = 0.0;
    for (uint ky = 0; ky < 3; ++ky) {
        for (uint kx = 0; kx < 3; ++kx) {
            uint index = (pc.flags & 1) != 0
                ? (((out_y * out_w) + out_x) * 9 + ky * 3 + kx)
                : ((out_y + ky) * pc.p1 + (out_x + kx));
            float value = q(input0_.values[index]);
            acc = q(acc + (value * kernel_[ky * 3 + kx]));
        }
    }
    output_.values[out_y * out_w + out_x] = acc;
}
)GLSL";
    }

    return prefix + R"GLSL(
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main() {
    uint x = gl_GlobalInvocationID.x;
    uint local_y = gl_GlobalInvocationID.y;
    if (x >= pc.p3 || local_y >= pc.p5) {
        return;
    }
    uint y = pc.p4 + local_y;
    float v00 = 0.0;
    float v01 = 0.0;
    float v10 = 0.0;
    float v11 = 0.0;
    float wx = 0.0;
    float wy = 0.0;
    if ((pc.flags & 1) != 0) {
        uint base = ((y * pc.p3) + x) * 6;
        v00 = q(input0_.values[base + 0]);
        v01 = q(input0_.values[base + 1]);
        v10 = q(input0_.values[base + 2]);
        v11 = q(input0_.values[base + 3]);
        wx = input0_.values[base + 4];
        wy = input0_.values[base + 5];
    } else {
        float src_y = (float(y) + 0.5) * float(pc.p0) / float(pc.p2) - 0.5;
        float clamped_y = clamp(src_y, 0.0, float(pc.p0 - 1));
        uint y0 = uint(clamped_y);
        uint y1 = min(y0 + 1, pc.p0 - 1);
        wy = clamped_y - float(y0);

        float src_x = (float(x) + 0.5) * float(pc.p1) / float(pc.p3) - 0.5;
        float clamped_x = clamp(src_x, 0.0, float(pc.p1 - 1));
        uint x0 = uint(clamped_x);
        uint x1 = min(x0 + 1, pc.p1 - 1);
        wx = clamped_x - float(x0);

        v00 = q(input0_.values[y0 * pc.p1 + x0]);
        v01 = q(input0_.values[y0 * pc.p1 + x1]);
        v10 = q(input0_.values[y1 * pc.p1 + x0]);
        v11 = q(input0_.values[y1 * pc.p1 + x1]);
    }
    float top = q(v00 + ((v01 - v00) * wx));
    float bottom = q(v10 + ((v11 - v10) * wx));
    output_.values[local_y * pc.p3 + x] = q(top + ((bottom - top) * wy));
}
)GLSL";
}

class VulkanApi final {
public:
    VulkanApi();
    ~VulkanApi();

    VulkanApi(const VulkanApi&) = delete;
    VulkanApi& operator=(const VulkanApi&) = delete;

    [[nodiscard]] bool ready() const;

    LibraryHandle library_ = nullptr;
    PFN_vkGetInstanceProcAddr vk_get_instance_proc_addr = nullptr;
    PFN_vkGetDeviceProcAddr vk_get_device_proc_addr = nullptr;
    PFN_vkCreateInstance vk_create_instance = nullptr;
};

VulkanApi& global_vulkan_api();

struct PushConstants {
    std::uint32_t p0 = 0u;
    std::uint32_t p1 = 0u;
    std::uint32_t p2 = 0u;
    std::uint32_t p3 = 0u;
    std::uint32_t p4 = 0u;
    std::uint32_t p5 = 0u;
    std::uint32_t p6 = 0u;
    std::uint32_t p7 = 0u;
    std::int32_t low_precision = 0;
    std::int32_t flags = 0;
};

struct VulkanBuffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    std::size_t bytes = 0u;
};

struct VulkanContext {
    ~VulkanContext();

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family_index = 0u;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;

    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkCreateDevice vkCreateDevice = nullptr;
    PFN_vkDestroyDevice vkDestroyDevice = nullptr;
    PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
    PFN_vkCreateBuffer vkCreateBuffer = nullptr;
    PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory vkAllocateMemory = nullptr;
    PFN_vkFreeMemory vkFreeMemory = nullptr;
    PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
    PFN_vkMapMemory vkMapMemory = nullptr;
    PFN_vkUnmapMemory vkUnmapMemory = nullptr;
    PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
    PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
    PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
    PFN_vkFreeDescriptorSets vkFreeDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;
    PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
    PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
    PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;
    PFN_vkCreatePipelineCache vkCreatePipelineCache = nullptr;
    PFN_vkDestroyPipelineCache vkDestroyPipelineCache = nullptr;
    PFN_vkGetPipelineCacheData vkGetPipelineCacheData = nullptr;
    PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
    PFN_vkFreeCommandBuffers vkFreeCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
    PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants vkCmdPushConstants = nullptr;
    PFN_vkCmdDispatch vkCmdDispatch = nullptr;
    PFN_vkCreateFence vkCreateFence = nullptr;
    PFN_vkDestroyFence vkDestroyFence = nullptr;
    PFN_vkQueueSubmit vkQueueSubmit = nullptr;
    PFN_vkWaitForFences vkWaitForFences = nullptr;

    std::mutex mutex;
    std::unordered_map<std::string, VkShaderModule> shaders;
    std::unordered_map<std::string, VkPipeline> pipelines;
};

template <typename T>
T load_instance_function(VulkanApi& api, VkInstance instance, const char* name) {
    return reinterpret_cast<T>(api.vk_get_instance_proc_addr(instance, name));
}

template <typename T>
T load_device_function(VulkanApi& api, VkDevice device, const char* name) {
    return reinterpret_cast<T>(api.vk_get_device_proc_addr(device, name));
}

std::shared_ptr<VulkanContext> create_vulkan_context();
std::optional<std::uint32_t> find_memory_type(VulkanContext& context, std::uint32_t mask);
bool create_buffer(VulkanContext& context, std::size_t bytes, VulkanBuffer& buffer);
void destroy_buffer(VulkanContext& context, VulkanBuffer& buffer);
std::string shader_key(ShaderKind kind);
VkPipeline ensure_pipeline(VulkanContext& context, ShaderKind shader_kind);
bool probe_vulkan_direct_support();
VulkanDirectProbeState probe_vulkan_direct_backend();

class VulkanDirectBackend final : public IKernelBackend {
public:
    [[nodiscard]] bool matches(const HardwareGraph& graph) const override;
    [[nodiscard]] std::string name() const override;
    [[nodiscard]] bool supports_async_dispatch(const HardwareGraph& graph) const override;

    BackendRunResult run_elementwise(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        std::span<const float> lhs,
        std::span<const float> rhs,
        bool low_precision) const override;

    BackendRunResult run_reduction(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        std::span<const float> input,
        bool low_precision) const override;

    BackendRunResult run_matmul(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        std::span<const float> lhs,
        std::span<const float> rhs,
        std::uint32_t rows,
        std::uint32_t columns,
        std::uint32_t depth,
        bool low_precision) const override;

    BackendRunResult run_conv3x3(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        std::span<const float> input,
        std::uint32_t height,
        std::uint32_t width,
        bool low_precision) const override;

    BackendRunResult run_resample(
        const HardwareGraph& graph,
        const OperationSpec& operation,
        std::span<const float> input,
        std::uint32_t src_h,
        std::uint32_t src_w,
        std::uint32_t dst_h,
        std::uint32_t dst_w,
        std::uint32_t row_offset,
        std::uint32_t row_count,
        bool low_precision) const override;

private:
    std::shared_ptr<VulkanContext> acquire_context() const;
    BackendRunResult run_compute(
        const HardwareGraph& graph,
        ShaderKind shader_kind,
        std::span<const float> input0,
        std::span<const float> input1,
        std::size_t output_count,
        std::uint32_t dispatch_x,
        std::uint32_t dispatch_y,
        const PushConstants& push_constants,
        bool low_precision,
        bool scalar_output) const;

    mutable std::mutex context_mutex_;
    mutable std::shared_ptr<VulkanContext> context_;
};

VulkanApi::VulkanApi() {
#if defined(_WIN32)
    library_ = load_library("vulkan-1.dll");
#else
    library_ = load_library("libvulkan.so");
    if (library_ == nullptr) {
        library_ = load_library("libvulkan.so.1");
    }
#endif
    if (library_ == nullptr) {
        return;
    }

    vk_get_instance_proc_addr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        load_symbol(library_, "vkGetInstanceProcAddr"));
    vk_get_device_proc_addr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        load_symbol(library_, "vkGetDeviceProcAddr"));
    vk_create_instance = reinterpret_cast<PFN_vkCreateInstance>(
        load_symbol(library_, "vkCreateInstance"));
}

VulkanApi::~VulkanApi() {
    close_library(library_);
}

bool VulkanApi::ready() const {
    return library_ != nullptr &&
           vk_get_instance_proc_addr != nullptr &&
           vk_get_device_proc_addr != nullptr &&
           vk_create_instance != nullptr;
}

VulkanApi& global_vulkan_api() {
    static VulkanApi api;
    return api;
}

VulkanContext::~VulkanContext() {
    if (device != VK_NULL_HANDLE && vkDeviceWaitIdle != nullptr) {
        vkDeviceWaitIdle(device);
    }
    if (device != VK_NULL_HANDLE &&
        pipeline_cache != VK_NULL_HANDLE &&
        vkGetPipelineCacheData != nullptr) {
        std::size_t cache_size = 0u;
        if (vkGetPipelineCacheData(device, pipeline_cache, &cache_size, nullptr) == VK_SUCCESS &&
            cache_size > 0u) {
            std::vector<std::byte> cache_blob(cache_size);
            if (vkGetPipelineCacheData(device, pipeline_cache, &cache_size, cache_blob.data()) == VK_SUCCESS &&
                cache_size > 0u) {
                store_cached_blob(pipeline_cache_path(), cache_blob.data(), cache_size);
            }
        }
    }
    for (auto& [key, pipeline] : pipelines) {
        (void)key;
        if (pipeline != VK_NULL_HANDLE && vkDestroyPipeline != nullptr) {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
    }
    for (auto& [key, shader] : shaders) {
        (void)key;
        if (shader != VK_NULL_HANDLE && vkDestroyShaderModule != nullptr) {
            vkDestroyShaderModule(device, shader, nullptr);
        }
    }
    if (pipeline_layout != VK_NULL_HANDLE && vkDestroyPipelineLayout != nullptr) {
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    }
    if (pipeline_cache != VK_NULL_HANDLE && vkDestroyPipelineCache != nullptr) {
        vkDestroyPipelineCache(device, pipeline_cache, nullptr);
    }
    if (descriptor_pool != VK_NULL_HANDLE && vkDestroyDescriptorPool != nullptr) {
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    }
    if (descriptor_set_layout != VK_NULL_HANDLE && vkDestroyDescriptorSetLayout != nullptr) {
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    }
    if (command_pool != VK_NULL_HANDLE && vkDestroyCommandPool != nullptr) {
        vkDestroyCommandPool(device, command_pool, nullptr);
    }
    if (device != VK_NULL_HANDLE && vkDestroyDevice != nullptr) {
        vkDestroyDevice(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE && vkDestroyInstance != nullptr) {
        vkDestroyInstance(instance, nullptr);
    }
}

std::shared_ptr<VulkanContext> create_vulkan_context() {
    auto& api = global_vulkan_api();
    if (!api.ready()) {
        return {};
    }

    auto context = std::make_shared<VulkanContext>();

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "jakal-core";
    app_info.applicationVersion = 1u;
    app_info.pEngineName = "jakal-core";
    app_info.engineVersion = 1u;
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;

    if (api.vk_create_instance(&instance_info, nullptr, &context->instance) != VK_SUCCESS) {
        return {};
    }

    context->vkDestroyInstance =
        load_instance_function<PFN_vkDestroyInstance>(api, context->instance, "vkDestroyInstance");
    context->vkEnumeratePhysicalDevices =
        load_instance_function<PFN_vkEnumeratePhysicalDevices>(
            api,
            context->instance,
            "vkEnumeratePhysicalDevices");
    context->vkGetPhysicalDeviceQueueFamilyProperties =
        load_instance_function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            api,
            context->instance,
            "vkGetPhysicalDeviceQueueFamilyProperties");
    context->vkGetPhysicalDeviceMemoryProperties =
        load_instance_function<PFN_vkGetPhysicalDeviceMemoryProperties>(
            api,
            context->instance,
            "vkGetPhysicalDeviceMemoryProperties");
    context->vkCreateDevice =
        load_instance_function<PFN_vkCreateDevice>(api, context->instance, "vkCreateDevice");

    if (context->vkDestroyInstance == nullptr ||
        context->vkEnumeratePhysicalDevices == nullptr ||
        context->vkGetPhysicalDeviceQueueFamilyProperties == nullptr ||
        context->vkGetPhysicalDeviceMemoryProperties == nullptr ||
        context->vkCreateDevice == nullptr) {
        return {};
    }

    std::uint32_t device_count = 0u;
    if (context->vkEnumeratePhysicalDevices(context->instance, &device_count, nullptr) != VK_SUCCESS ||
        device_count == 0u) {
        return {};
    }

    std::vector<VkPhysicalDevice> physical_devices(device_count);
    if (context->vkEnumeratePhysicalDevices(
            context->instance,
            &device_count,
            physical_devices.data()) != VK_SUCCESS) {
        return {};
    }

    for (const auto physical_device : physical_devices) {
        std::uint32_t queue_family_count = 0u;
        context->vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
        if (queue_family_count == 0u) {
            continue;
        }

        std::vector<VkQueueFamilyProperties> queue_properties(queue_family_count);
        context->vkGetPhysicalDeviceQueueFamilyProperties(
            physical_device,
            &queue_family_count,
            queue_properties.data());
        const auto queue_it = std::find_if(
            queue_properties.begin(),
            queue_properties.end(),
            [](const VkQueueFamilyProperties& properties) {
                return (properties.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u;
            });
        if (queue_it == queue_properties.end()) {
            continue;
        }

        context->physical_device = physical_device;
        context->queue_family_index =
            static_cast<std::uint32_t>(std::distance(queue_properties.begin(), queue_it));
        break;
    }

    if (context->physical_device == VK_NULL_HANDLE) {
        return {};
    }

    const float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = context->queue_family_index;
    queue_info.queueCount = 1u;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1u;
    device_info.pQueueCreateInfos = &queue_info;

    if (context->vkCreateDevice(
            context->physical_device,
            &device_info,
            nullptr,
            &context->device) != VK_SUCCESS) {
        return {};
    }

    context->vkDestroyDevice =
        load_device_function<PFN_vkDestroyDevice>(api, context->device, "vkDestroyDevice");
    context->vkDeviceWaitIdle =
        load_device_function<PFN_vkDeviceWaitIdle>(api, context->device, "vkDeviceWaitIdle");
    context->vkGetDeviceQueue =
        load_device_function<PFN_vkGetDeviceQueue>(api, context->device, "vkGetDeviceQueue");
    context->vkCreateBuffer =
        load_device_function<PFN_vkCreateBuffer>(api, context->device, "vkCreateBuffer");
    context->vkDestroyBuffer =
        load_device_function<PFN_vkDestroyBuffer>(api, context->device, "vkDestroyBuffer");
    context->vkGetBufferMemoryRequirements =
        load_device_function<PFN_vkGetBufferMemoryRequirements>(
            api,
            context->device,
            "vkGetBufferMemoryRequirements");
    context->vkAllocateMemory =
        load_device_function<PFN_vkAllocateMemory>(api, context->device, "vkAllocateMemory");
    context->vkFreeMemory =
        load_device_function<PFN_vkFreeMemory>(api, context->device, "vkFreeMemory");
    context->vkBindBufferMemory =
        load_device_function<PFN_vkBindBufferMemory>(api, context->device, "vkBindBufferMemory");
    context->vkMapMemory =
        load_device_function<PFN_vkMapMemory>(api, context->device, "vkMapMemory");
    context->vkUnmapMemory =
        load_device_function<PFN_vkUnmapMemory>(api, context->device, "vkUnmapMemory");
    context->vkCreateShaderModule =
        load_device_function<PFN_vkCreateShaderModule>(api, context->device, "vkCreateShaderModule");
    context->vkDestroyShaderModule =
        load_device_function<PFN_vkDestroyShaderModule>(api, context->device, "vkDestroyShaderModule");
    context->vkCreateDescriptorSetLayout =
        load_device_function<PFN_vkCreateDescriptorSetLayout>(
            api,
            context->device,
            "vkCreateDescriptorSetLayout");
    context->vkDestroyDescriptorSetLayout =
        load_device_function<PFN_vkDestroyDescriptorSetLayout>(
            api,
            context->device,
            "vkDestroyDescriptorSetLayout");
    context->vkCreateDescriptorPool =
        load_device_function<PFN_vkCreateDescriptorPool>(api, context->device, "vkCreateDescriptorPool");
    context->vkDestroyDescriptorPool =
        load_device_function<PFN_vkDestroyDescriptorPool>(api, context->device, "vkDestroyDescriptorPool");
    context->vkAllocateDescriptorSets =
        load_device_function<PFN_vkAllocateDescriptorSets>(api, context->device, "vkAllocateDescriptorSets");
    context->vkFreeDescriptorSets =
        load_device_function<PFN_vkFreeDescriptorSets>(api, context->device, "vkFreeDescriptorSets");
    context->vkUpdateDescriptorSets =
        load_device_function<PFN_vkUpdateDescriptorSets>(api, context->device, "vkUpdateDescriptorSets");
    context->vkCreatePipelineLayout =
        load_device_function<PFN_vkCreatePipelineLayout>(api, context->device, "vkCreatePipelineLayout");
    context->vkDestroyPipelineLayout =
        load_device_function<PFN_vkDestroyPipelineLayout>(api, context->device, "vkDestroyPipelineLayout");
    context->vkCreateComputePipelines =
        load_device_function<PFN_vkCreateComputePipelines>(api, context->device, "vkCreateComputePipelines");
    context->vkDestroyPipeline =
        load_device_function<PFN_vkDestroyPipeline>(api, context->device, "vkDestroyPipeline");
    context->vkCreatePipelineCache =
        load_device_function<PFN_vkCreatePipelineCache>(api, context->device, "vkCreatePipelineCache");
    context->vkDestroyPipelineCache =
        load_device_function<PFN_vkDestroyPipelineCache>(api, context->device, "vkDestroyPipelineCache");
    context->vkGetPipelineCacheData =
        load_device_function<PFN_vkGetPipelineCacheData>(api, context->device, "vkGetPipelineCacheData");
    context->vkCreateCommandPool =
        load_device_function<PFN_vkCreateCommandPool>(api, context->device, "vkCreateCommandPool");
    context->vkDestroyCommandPool =
        load_device_function<PFN_vkDestroyCommandPool>(api, context->device, "vkDestroyCommandPool");
    context->vkAllocateCommandBuffers =
        load_device_function<PFN_vkAllocateCommandBuffers>(api, context->device, "vkAllocateCommandBuffers");
    context->vkFreeCommandBuffers =
        load_device_function<PFN_vkFreeCommandBuffers>(api, context->device, "vkFreeCommandBuffers");
    context->vkBeginCommandBuffer =
        load_device_function<PFN_vkBeginCommandBuffer>(api, context->device, "vkBeginCommandBuffer");
    context->vkEndCommandBuffer =
        load_device_function<PFN_vkEndCommandBuffer>(api, context->device, "vkEndCommandBuffer");
    context->vkCmdBindPipeline =
        load_device_function<PFN_vkCmdBindPipeline>(api, context->device, "vkCmdBindPipeline");
    context->vkCmdBindDescriptorSets =
        load_device_function<PFN_vkCmdBindDescriptorSets>(api, context->device, "vkCmdBindDescriptorSets");
    context->vkCmdPushConstants =
        load_device_function<PFN_vkCmdPushConstants>(api, context->device, "vkCmdPushConstants");
    context->vkCmdDispatch =
        load_device_function<PFN_vkCmdDispatch>(api, context->device, "vkCmdDispatch");
    context->vkCreateFence =
        load_device_function<PFN_vkCreateFence>(api, context->device, "vkCreateFence");
    context->vkDestroyFence =
        load_device_function<PFN_vkDestroyFence>(api, context->device, "vkDestroyFence");
    context->vkQueueSubmit =
        load_device_function<PFN_vkQueueSubmit>(api, context->device, "vkQueueSubmit");
    context->vkWaitForFences =
        load_device_function<PFN_vkWaitForFences>(api, context->device, "vkWaitForFences");

    if (context->vkDestroyDevice == nullptr ||
        context->vkGetDeviceQueue == nullptr ||
        context->vkCreateBuffer == nullptr ||
        context->vkDestroyBuffer == nullptr ||
        context->vkGetBufferMemoryRequirements == nullptr ||
        context->vkAllocateMemory == nullptr ||
        context->vkFreeMemory == nullptr ||
        context->vkBindBufferMemory == nullptr ||
        context->vkMapMemory == nullptr ||
        context->vkUnmapMemory == nullptr ||
        context->vkCreateShaderModule == nullptr ||
        context->vkDestroyShaderModule == nullptr ||
        context->vkCreateDescriptorSetLayout == nullptr ||
        context->vkDestroyDescriptorSetLayout == nullptr ||
        context->vkCreateDescriptorPool == nullptr ||
        context->vkDestroyDescriptorPool == nullptr ||
        context->vkAllocateDescriptorSets == nullptr ||
        context->vkFreeDescriptorSets == nullptr ||
        context->vkUpdateDescriptorSets == nullptr ||
        context->vkCreatePipelineLayout == nullptr ||
        context->vkDestroyPipelineLayout == nullptr ||
        context->vkCreateComputePipelines == nullptr ||
        context->vkDestroyPipeline == nullptr ||
        context->vkCreateCommandPool == nullptr ||
        context->vkDestroyCommandPool == nullptr ||
        context->vkAllocateCommandBuffers == nullptr ||
        context->vkFreeCommandBuffers == nullptr ||
        context->vkBeginCommandBuffer == nullptr ||
        context->vkEndCommandBuffer == nullptr ||
        context->vkCmdBindPipeline == nullptr ||
        context->vkCmdBindDescriptorSets == nullptr ||
        context->vkCmdPushConstants == nullptr ||
        context->vkCmdDispatch == nullptr ||
        context->vkCreateFence == nullptr ||
        context->vkDestroyFence == nullptr ||
        context->vkQueueSubmit == nullptr ||
        context->vkWaitForFences == nullptr) {
        return {};
    }

    context->vkGetDeviceQueue(
        context->device,
        context->queue_family_index,
        0u,
        &context->queue);
    if (context->queue == VK_NULL_HANDLE) {
        return {};
    }

    if (context->vkCreatePipelineCache != nullptr) {
        const auto cached_pipeline_data = load_cached_blob(pipeline_cache_path());

        VkPipelineCacheCreateInfo pipeline_cache_info{};
        pipeline_cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        if (cached_pipeline_data.has_value()) {
            pipeline_cache_info.initialDataSize = cached_pipeline_data->size();
            pipeline_cache_info.pInitialData = cached_pipeline_data->data();
        }
        if (context->vkCreatePipelineCache(
                context->device,
                &pipeline_cache_info,
                nullptr,
                &context->pipeline_cache) != VK_SUCCESS) {
            pipeline_cache_info.initialDataSize = 0u;
            pipeline_cache_info.pInitialData = nullptr;
            if (context->vkCreatePipelineCache(
                    context->device,
                    &pipeline_cache_info,
                    nullptr,
                    &context->pipeline_cache) != VK_SUCCESS) {
                context->pipeline_cache = VK_NULL_HANDLE;
            }
        }
    }

    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = context->queue_family_index;
    if (context->vkCreateCommandPool(
            context->device,
            &command_pool_info,
            nullptr,
            &context->command_pool) != VK_SUCCESS) {
        return {};
    }

    const std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
        VkDescriptorSetLayoutBinding{
            0u,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            1u,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr},
        VkDescriptorSetLayoutBinding{
            1u,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            1u,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr},
        VkDescriptorSetLayoutBinding{
            2u,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            1u,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr}};

    VkDescriptorSetLayoutCreateInfo descriptor_layout_info{};
    descriptor_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    descriptor_layout_info.pBindings = bindings.data();
    if (context->vkCreateDescriptorSetLayout(
            context->device,
            &descriptor_layout_info,
            nullptr,
            &context->descriptor_set_layout) != VK_SUCCESS) {
        return {};
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 48u;

    VkDescriptorPoolCreateInfo descriptor_pool_info{};
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptor_pool_info.maxSets = 16u;
    descriptor_pool_info.poolSizeCount = 1u;
    descriptor_pool_info.pPoolSizes = &pool_size;
    if (context->vkCreateDescriptorPool(
            context->device,
            &descriptor_pool_info,
            nullptr,
            &context->descriptor_pool) != VK_SUCCESS) {
        return {};
    }

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0u;
    push_range.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1u;
    pipeline_layout_info.pSetLayouts = &context->descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1u;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    if (context->vkCreatePipelineLayout(
            context->device,
            &pipeline_layout_info,
            nullptr,
            &context->pipeline_layout) != VK_SUCCESS) {
        return {};
    }

    return context;
}

std::optional<std::uint32_t> find_memory_type(
    VulkanContext& context,
    const std::uint32_t mask) {
    VkPhysicalDeviceMemoryProperties properties{};
    context.vkGetPhysicalDeviceMemoryProperties(context.physical_device, &properties);
    for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        if ((mask & (1u << index)) == 0u) {
            continue;
        }
        const auto flags = properties.memoryTypes[index].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0u &&
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u) {
            return index;
        }
    }
    return std::nullopt;
}

bool create_buffer(VulkanContext& context, const std::size_t bytes, VulkanBuffer& buffer) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = std::max<VkDeviceSize>(bytes, sizeof(float));
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (context.vkCreateBuffer(context.device, &buffer_info, nullptr, &buffer.handle) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements requirements{};
    context.vkGetBufferMemoryRequirements(context.device, buffer.handle, &requirements);
    const auto memory_type = find_memory_type(context, requirements.memoryTypeBits);
    if (!memory_type.has_value()) {
        return false;
    }

    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = requirements.size;
    allocate_info.memoryTypeIndex = *memory_type;
    if (context.vkAllocateMemory(context.device, &allocate_info, nullptr, &buffer.memory) != VK_SUCCESS) {
        return false;
    }
    if (context.vkBindBufferMemory(context.device, buffer.handle, buffer.memory, 0u) != VK_SUCCESS) {
        return false;
    }
    if (context.vkMapMemory(context.device, buffer.memory, 0u, requirements.size, 0u, &buffer.mapped) != VK_SUCCESS) {
        return false;
    }

    buffer.bytes = bytes;
    return true;
}

void destroy_buffer(VulkanContext& context, VulkanBuffer& buffer) {
    if (buffer.mapped != nullptr) {
        context.vkUnmapMemory(context.device, buffer.memory);
        buffer.mapped = nullptr;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
        context.vkFreeMemory(context.device, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
    if (buffer.handle != VK_NULL_HANDLE) {
        context.vkDestroyBuffer(context.device, buffer.handle, nullptr);
        buffer.handle = VK_NULL_HANDLE;
    }
    buffer.bytes = 0u;
}

std::string shader_key(const ShaderKind kind) {
    switch (kind) {
    case ShaderKind::elementwise:
        return "v2.elementwise";
    case ShaderKind::reduction:
        return "v2.reduction";
    case ShaderKind::matmul:
        return "v3.matmul";
    case ShaderKind::conv3x3:
        return "v2.conv3x3";
    case ShaderKind::resample:
    default:
        return "v2.resample";
    }
}

VkPipeline ensure_pipeline(VulkanContext& context, const ShaderKind shader_kind) {
    const auto key = shader_key(shader_kind);
    if (const auto it = context.pipelines.find(key); it != context.pipelines.end()) {
        return it->second;
    }

    const auto words = load_or_compile_glsl(key, shader_source(shader_kind));
    if (!words.has_value()) {
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = words->size() * sizeof(std::uint32_t);
    shader_info.pCode = words->data();

    VkShaderModule shader_module = VK_NULL_HANDLE;
    if (context.vkCreateShaderModule(context.device, &shader_info, nullptr, &shader_module) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = context.pipeline_layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (context.vkCreateComputePipelines(
            context.device,
            context.pipeline_cache,
            1u,
            &pipeline_info,
            nullptr,
            &pipeline) != VK_SUCCESS) {
        context.vkDestroyShaderModule(context.device, shader_module, nullptr);
        return VK_NULL_HANDLE;
    }

    context.shaders.emplace(key, shader_module);
    context.pipelines.emplace(key, pipeline);
    return pipeline;
}

VulkanDirectProbeState probe_vulkan_direct_backend() {
    VulkanDirectProbeState state;
    static constexpr std::array<ShaderKind, 5> kAllShaderKinds = {
        ShaderKind::elementwise,
        ShaderKind::reduction,
        ShaderKind::matmul,
        ShaderKind::conv3x3,
        ShaderKind::resample,
    };
    for (const auto shader_kind : kAllShaderKinds) {
        std::string compile_error;
        if (!load_or_compile_glsl(shader_key(shader_kind), shader_source(shader_kind), &compile_error).has_value()) {
            state.detail = "shader compile/cache failed for " + shader_key(shader_kind);
            if (!compile_error.empty()) {
                state.detail += ": " + compile_error;
            }
            return state;
        }
    }
    auto context = create_vulkan_context();
    if (context == nullptr) {
        state.detail = "failed to create Vulkan context";
        return state;
    }
    std::scoped_lock lock(context->mutex);
    for (const auto shader_kind : kAllShaderKinds) {
        if (ensure_pipeline(*context, shader_kind) == VK_NULL_HANDLE) {
            state.detail = "failed to create Vulkan pipeline for " + shader_key(shader_kind);
            return state;
        }
    }
    state.available = true;
    state.detail = "ready";
    return state;
}

bool probe_vulkan_direct_support() {
    return probe_vulkan_direct_backend().available;
}

bool VulkanDirectBackend::matches(const HardwareGraph& graph) const {
    return graph.probe == "vulkan";
}

std::string VulkanDirectBackend::name() const {
    return "vulkan-direct";
}

bool VulkanDirectBackend::supports_async_dispatch(const HardwareGraph& graph) const {
    return summarize_graph(graph).supports_asynchronous_dispatch;
}

BackendRunResult VulkanDirectBackend::run_elementwise(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    const std::span<const float> lhs,
    const std::span<const float> rhs,
    const bool low_precision) const {
    (void)operation;
    return run_compute(
        graph,
        ShaderKind::elementwise,
        lhs,
        rhs,
        lhs.size(),
        std::max(1u, static_cast<std::uint32_t>((lhs.size() + 255u) / 256u)),
        1u,
        PushConstants{
            static_cast<std::uint32_t>(lhs.size()),
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            low_precision ? 1 : 0,
            0},
        low_precision,
        false);
}

BackendRunResult VulkanDirectBackend::run_reduction(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    const std::span<const float> input,
    const bool low_precision) const {
    (void)operation;
    return run_compute(
        graph,
        ShaderKind::reduction,
        input,
        {},
        1u,
        1u,
        1u,
        PushConstants{
            static_cast<std::uint32_t>(input.size()),
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            low_precision ? 1 : 0,
            0},
        low_precision,
        true);
}

BackendRunResult VulkanDirectBackend::run_matmul(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    const std::span<const float> lhs,
    const std::span<const float> rhs,
    const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const bool low_precision) const {
    (void)operation;
    return run_compute(
        graph,
        ShaderKind::matmul,
        lhs,
        rhs,
        static_cast<std::size_t>(rows) * columns,
        std::max(1u, (columns + 15u) / 16u),
        std::max(1u, (rows + 15u) / 16u),
        PushConstants{
            rows,
            columns,
            depth,
            0u,
            0u,
            0u,
            0u,
            0u,
            low_precision ? 1 : 0,
            0},
        low_precision,
        false);
}

BackendRunResult VulkanDirectBackend::run_conv3x3(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    const std::span<const float> input,
    const std::uint32_t height,
    const std::uint32_t width,
    const bool low_precision) const {
    if (height < 3u || width < 3u) {
        BackendRunResult result;
        result.error = "vulkan-conv-shape";
        return result;
    }
    const auto out_height = height - 2u;
    const auto out_width = width - 2u;
    (void)operation;
    return run_compute(
        graph,
        ShaderKind::conv3x3,
        input,
        {},
        static_cast<std::size_t>(out_height) * out_width,
        std::max(1u, (out_width + 15u) / 16u),
        std::max(1u, (out_height + 15u) / 16u),
        PushConstants{
            height,
            width,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            low_precision ? 1 : 0,
            0},
        low_precision,
        false);
}

BackendRunResult VulkanDirectBackend::run_resample(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    const std::span<const float> input,
    const std::uint32_t src_h,
    const std::uint32_t src_w,
    const std::uint32_t dst_h,
    const std::uint32_t dst_w,
    const std::uint32_t row_offset,
    const std::uint32_t row_count,
    const bool low_precision) const {
    (void)operation;
    return run_compute(
        graph,
        ShaderKind::resample,
        input,
        {},
        static_cast<std::size_t>(row_count) * dst_w,
        std::max(1u, (dst_w + 15u) / 16u),
        std::max(1u, (row_count + 15u) / 16u),
        PushConstants{
            src_h,
            src_w,
            dst_h,
            dst_w,
            row_offset,
            row_count,
            0u,
            0u,
            low_precision ? 1 : 0,
            0},
        low_precision,
        false);
}

std::shared_ptr<VulkanContext> VulkanDirectBackend::acquire_context() const {
    std::scoped_lock lock(context_mutex_);
    if (context_ != nullptr) {
        return context_;
    }
    context_ = create_vulkan_context();
    return context_;
}

BackendRunResult VulkanDirectBackend::run_compute(
    const HardwareGraph& graph,
    const ShaderKind shader_kind,
    const std::span<const float> input0,
    const std::span<const float> input1,
    const std::size_t output_count,
    const std::uint32_t dispatch_x,
    const std::uint32_t dispatch_y,
    const PushConstants& push_constants,
    const bool low_precision,
    const bool scalar_output) const {
    BackendRunResult result;
    if (!vulkan_direct_backend_available_internal()) {
        result.error = "vulkan-direct-unavailable";
        return result;
    }

    auto context = acquire_context();
    if (context == nullptr) {
        result.error = "vulkan-context";
        return result;
    }

    if (!scalar_output) {
        result.output.resize(output_count, 0.0f);
    }

    std::string error = "vulkan-dispatch";
    bool dispatched = false;
    const double runtime_us = measure_us([&]() {
        std::scoped_lock lock(context->mutex);
        const auto pipeline = ensure_pipeline(*context, shader_kind);
        if (pipeline == VK_NULL_HANDLE) {
            error = "vulkan-pipeline";
            return;
        }

        VulkanBuffer input0_buffer;
        VulkanBuffer input1_buffer;
        VulkanBuffer output_buffer;
        const auto cleanup_buffers = [&]() {
            destroy_buffer(*context, input0_buffer);
            destroy_buffer(*context, input1_buffer);
            destroy_buffer(*context, output_buffer);
        };

        if (!create_buffer(*context, std::max<std::size_t>(input0.size_bytes(), sizeof(float)), input0_buffer) ||
            !create_buffer(*context, std::max<std::size_t>(input1.size_bytes(), sizeof(float)), input1_buffer) ||
            !create_buffer(*context, std::max<std::size_t>(output_count * sizeof(float), sizeof(float)), output_buffer)) {
            error = "vulkan-buffer";
            cleanup_buffers();
            return;
        }

        if (!input0.empty()) {
            std::memcpy(input0_buffer.mapped, input0.data(), input0.size_bytes());
        } else {
            static_cast<float*>(input0_buffer.mapped)[0] = 0.0f;
        }
        if (!input1.empty()) {
            std::memcpy(input1_buffer.mapped, input1.data(), input1.size_bytes());
        } else {
            static_cast<float*>(input1_buffer.mapped)[0] = 0.0f;
        }
        std::memset(
            output_buffer.mapped,
            0,
            std::max<std::size_t>(output_count * sizeof(float), sizeof(float)));

        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        const auto cleanup_dispatch = [&]() {
            if (command_buffer != VK_NULL_HANDLE) {
                context->vkFreeCommandBuffers(context->device, context->command_pool, 1u, &command_buffer);
            }
            if (descriptor_set != VK_NULL_HANDLE) {
                context->vkFreeDescriptorSets(context->device, context->descriptor_pool, 1u, &descriptor_set);
            }
            if (fence != VK_NULL_HANDLE) {
                context->vkDestroyFence(context->device, fence, nullptr);
            }
        };

        VkDescriptorSetAllocateInfo descriptor_alloc_info{};
        descriptor_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptor_alloc_info.descriptorPool = context->descriptor_pool;
        descriptor_alloc_info.descriptorSetCount = 1u;
        descriptor_alloc_info.pSetLayouts = &context->descriptor_set_layout;
        if (context->vkAllocateDescriptorSets(
                context->device,
                &descriptor_alloc_info,
                &descriptor_set) != VK_SUCCESS) {
            error = "vulkan-descriptor-set";
            cleanup_dispatch();
            cleanup_buffers();
            return;
        }

        VkDescriptorBufferInfo input0_info{};
        input0_info.buffer = input0_buffer.handle;
        input0_info.offset = 0u;
        input0_info.range = std::max<VkDeviceSize>(input0_buffer.bytes, sizeof(float));

        VkDescriptorBufferInfo input1_info{};
        input1_info.buffer = input1_buffer.handle;
        input1_info.offset = 0u;
        input1_info.range = std::max<VkDeviceSize>(input1_buffer.bytes, sizeof(float));

        VkDescriptorBufferInfo output_info{};
        output_info.buffer = output_buffer.handle;
        output_info.offset = 0u;
        output_info.range = std::max<VkDeviceSize>(output_buffer.bytes, sizeof(float));

        const std::array<VkWriteDescriptorSet, 3> writes = {
            VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                descriptor_set,
                0u,
                0u,
                1u,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                &input0_info,
                nullptr},
            VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                descriptor_set,
                1u,
                0u,
                1u,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                &input1_info,
                nullptr},
            VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                descriptor_set,
                2u,
                0u,
                1u,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                &output_info,
                nullptr}};
        context->vkUpdateDescriptorSets(
            context->device,
            static_cast<std::uint32_t>(writes.size()),
            writes.data(),
            0u,
            nullptr);

        VkCommandBufferAllocateInfo command_alloc_info{};
        command_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_alloc_info.commandPool = context->command_pool;
        command_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_alloc_info.commandBufferCount = 1u;
        if (context->vkAllocateCommandBuffers(
                context->device,
                &command_alloc_info,
                &command_buffer) != VK_SUCCESS) {
            error = "vulkan-command-buffer";
            cleanup_dispatch();
            cleanup_buffers();
            return;
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (context->vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
            error = "vulkan-command-begin";
            cleanup_dispatch();
            cleanup_buffers();
            return;
        }

        context->vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        context->vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            context->pipeline_layout,
            0u,
            1u,
            &descriptor_set,
            0u,
            nullptr);

        context->vkCmdPushConstants(
            command_buffer,
            context->pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0u,
            sizeof(PushConstants),
            &push_constants);
        context->vkCmdDispatch(command_buffer, dispatch_x, dispatch_y, 1u);

        if (context->vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
            error = "vulkan-command-end";
            cleanup_dispatch();
            cleanup_buffers();
            return;
        }

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (context->vkCreateFence(context->device, &fence_info, nullptr, &fence) != VK_SUCCESS) {
            error = "vulkan-fence";
            cleanup_dispatch();
            cleanup_buffers();
            return;
        }

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1u;
        submit_info.pCommandBuffers = &command_buffer;
        if (context->vkQueueSubmit(context->queue, 1u, &submit_info, fence) != VK_SUCCESS) {
            error = "vulkan-submit";
            cleanup_dispatch();
            cleanup_buffers();
            return;
        }
        if (context->vkWaitForFences(
                context->device,
                1u,
                &fence,
                VK_TRUE,
                std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS) {
            error = "vulkan-wait";
            cleanup_dispatch();
            cleanup_buffers();
            return;
        }

        if (scalar_output) {
            result.scalar_output = static_cast<const float*>(output_buffer.mapped)[0];
            result.scalar_output = quantize_value(static_cast<float>(result.scalar_output), low_precision);
        } else if (output_count > 0u) {
            std::memcpy(result.output.data(), output_buffer.mapped, output_count * sizeof(float));
            for (auto& value : result.output) {
                value = quantize_value(value, low_precision);
            }
        }

        dispatched = true;
        cleanup_dispatch();
        cleanup_buffers();
    });

    result.runtime_us = runtime_us;
    if (!dispatched) {
        result.error = std::move(error);
        return result;
    }

    result.submit_runtime_us = std::max(0.20, runtime_us * 0.12);
    result.synchronize_runtime_us = std::max(0.20, runtime_us - result.submit_runtime_us);
    result.copy_runtime_us =
        estimate_transfer_runtime_us(graph, input0.size_bytes() + input1.size_bytes(), true) +
        estimate_transfer_runtime_us(graph, output_count * sizeof(float), false);
    result.compute_runtime_us = std::max(0.25, runtime_us * 0.72);
    result.copy_overlap_ratio =
        std::clamp(0.34 + (supports_async_dispatch(graph) ? 0.10 : 0.0), 0.08, 0.80);
    result.queue_separation_ratio = supports_async_dispatch(graph) ? 0.60 : 0.0;
    result.copy_queue_count =
        (input0.size_bytes() + input1.size_bytes() + output_count * sizeof(float)) > 0u ? 1u : 0u;
    result.compute_queue_count = 1u;
    result.event_wait_count = result.copy_queue_count > 0u ? 1u : 0u;
    result.used_host = false;
    result.used_opencl = false;
    result.async_dispatch_capable = supports_async_dispatch(graph);
    result.success = scalar_output
                         ? std::isfinite(result.scalar_output)
                         : result.output.size() == output_count;
    if (!result.success) {
        result.error = "vulkan-dispatch";
    }
    return result;
}

}  // namespace

std::unique_ptr<IKernelBackend> make_vulkan_direct_kernel_backend_internal() {
    return std::make_unique<VulkanDirectBackend>();
}

bool vulkan_direct_backend_available_internal() {
    static const auto probe_state = probe_vulkan_direct_backend();
    return probe_state.available;
}

std::string vulkan_direct_backend_status_detail_internal() {
    static const auto probe_state = probe_vulkan_direct_backend();
    return probe_state.detail;
}

}  // namespace jakal::executors
