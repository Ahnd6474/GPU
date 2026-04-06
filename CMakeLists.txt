cmake_minimum_required(VERSION 3.20)

project(GPU
    VERSION 0.1.0
    DESCRIPTION "Structural-descriptor runtime skeleton for heterogeneous compute"
    LANGUAGES CXX)

option(GPU_BUILD_EXAMPLES "Build example applications" ON)
option(GPU_BUILD_TESTS "Build test targets" ON)

add_library(gpu_runtime
    src/device.cpp
    src/execution.cpp
    src/executor.cpp
    src/planner.cpp
    src/runtime.cpp
    src/c_api.cpp
    src/backends/cpu_backend.cpp
    src/backends/opencl_backend.cpp)

add_library(gpu::runtime ALIAS gpu_runtime)

target_include_directories(gpu_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(gpu_runtime PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(gpu_runtime PRIVATE /W4 /permissive-)
    target_compile_definitions(gpu_runtime PRIVATE NOMINMAX)
else()
    target_compile_options(gpu_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(gpu_runtime PRIVATE dl)
endif()

if(GPU_BUILD_EXAMPLES)
    add_executable(gpu_inspect examples/inspect_runtime.cpp)
    target_link_libraries(gpu_inspect PRIVATE gpu::runtime)
endif()

if(GPU_BUILD_TESTS)
    enable_testing()

    add_executable(gpu_smoke tests/smoke.cpp)
    target_link_libraries(gpu_smoke PRIVATE gpu::runtime)

    add_test(NAME gpu_smoke COMMAND gpu_smoke)

    add_executable(gpu_optimization tests/optimization.cpp)
    target_link_libraries(gpu_optimization PRIVATE gpu::runtime)

    add_test(NAME gpu_optimization COMMAND gpu_optimization)
endif()
