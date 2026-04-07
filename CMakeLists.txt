cmake_minimum_required(VERSION 3.20)

project(JakalCore
    VERSION 0.1.0
    DESCRIPTION "Structural-descriptor runtime skeleton for heterogeneous compute"
    LANGUAGES CXX)

option(JAKAL_CORE_BUILD_EXAMPLES "Build example applications" ON)
option(JAKAL_CORE_BUILD_TESTS "Build test targets" ON)

add_library(jakal_core
    src/device.cpp
    src/execution.cpp
    src/executor.cpp
    src/jakal_l0.cpp
    src/jakal_toolkit.cpp
    src/executors/common_scheduler.cpp
    src/executors/direct_backends.cpp
    src/executors/native_gpu_backends.cpp
    src/planner.cpp
    src/workloads.cpp
    src/runtime.cpp
    src/c_api.cpp
    src/backends/cpu_backend.cpp
    src/backends/native_gpu_probes.cpp
    src/backends/opencl_backend.cpp)

add_library(jakal::core ALIAS jakal_core)
add_library(jakal::runtime ALIAS jakal_core)

target_include_directories(jakal_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(jakal_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(jakal_core PRIVATE /W4 /permissive-)
    target_compile_definitions(jakal_core PRIVATE NOMINMAX)
else()
    target_compile_options(jakal_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(jakal_core PRIVATE dl)
endif()

if(JAKAL_CORE_BUILD_EXAMPLES)
    add_executable(jakal_inspect examples/inspect_runtime.cpp)
    target_link_libraries(jakal_inspect PRIVATE jakal::core)

    add_executable(jakal_profile_workloads examples/profile_workloads.cpp)
    target_link_libraries(jakal_profile_workloads PRIVATE jakal::core)

    add_executable(jakal_explore_cpu_dl examples/explore_cpu_dl.cpp)
    target_link_libraries(jakal_explore_cpu_dl PRIVATE jakal::core)

    add_executable(jakal_partition_roles examples/partition_roles.cpp)
    target_link_libraries(jakal_partition_roles PRIVATE jakal::core)
endif()

if(JAKAL_CORE_BUILD_TESTS)
    enable_testing()

    add_executable(jakal_smoke tests/smoke.cpp)
    target_link_libraries(jakal_smoke PRIVATE jakal::core)

    add_test(NAME jakal_smoke COMMAND jakal_smoke)

    add_executable(jakal_optimization tests/optimization.cpp)
    target_link_libraries(jakal_optimization PRIVATE jakal::core)

    add_test(NAME jakal_optimization COMMAND jakal_optimization)
endif()

