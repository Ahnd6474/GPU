cmake_minimum_required(VERSION 3.20)

project(JakalCore
    VERSION 0.1.0
    DESCRIPTION "Structural-descriptor runtime skeleton for heterogeneous compute"
    LANGUAGES CXX)

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(JAKAL_CORE_BUILD_EXAMPLES "Build example applications" ON)
option(JAKAL_CORE_BUILD_TESTS "Build test targets" ON)
option(JAKAL_CORE_BUILD_PRODUCT_TOOLS "Build bootstrap and maintenance tools" ON)
option(JAKAL_CORE_ENABLE_CODE_SIGNING "Sign installable Windows executables after build" OFF)

set(JAKAL_CORE_SIGNTOOL_PATH "" CACHE FILEPATH "Path to signtool.exe")
set(JAKAL_CORE_CODESIGN_CERT_SHA1 "" CACHE STRING "Certificate thumbprint used for Windows code signing")
set(JAKAL_CORE_CODESIGN_TIMESTAMP_URL "http://timestamp.digicert.com" CACHE STRING "RFC3161 timestamp URL for Windows code signing")
set(JAKAL_CORE_VERSION_SUFFIX "" CACHE STRING "Optional prerelease/build suffix appended to packaged artifact versions, for example -beta.1")

if(JAKAL_CORE_VERSION_SUFFIX AND NOT JAKAL_CORE_VERSION_SUFFIX MATCHES "^[-+][0-9A-Za-z.-]+$")
    message(FATAL_ERROR "JAKAL_CORE_VERSION_SUFFIX must start with '-' or '+' and contain only [0-9A-Za-z.-]")
endif()

set(JAKAL_CORE_PACKAGE_VERSION "${PROJECT_VERSION}${JAKAL_CORE_VERSION_SUFFIX}")

set(JAKAL_CORE_BRAND_SOURCE_PNG "${CMAKE_CURRENT_SOURCE_DIR}/Jakal_core_logo.png")
set(JAKAL_CORE_BRAND_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/branding")
set(JAKAL_CORE_BRAND_RUNTIME_PNG "")
set(JAKAL_CORE_BRAND_RUNTIME_ICO "")
set(JAKAL_CORE_BRAND_NSIS_HEADER_BMP "")
set(JAKAL_CORE_BRAND_NSIS_WIZARD_BMP "")

if(EXISTS "${JAKAL_CORE_BRAND_SOURCE_PNG}")
    if(WIN32)
        find_program(JAKAL_CORE_POWERSHELL_EXECUTABLE NAMES powershell pwsh REQUIRED)
        file(MAKE_DIRECTORY "${JAKAL_CORE_BRAND_OUTPUT_DIR}")
        execute_process(
            COMMAND
                "${JAKAL_CORE_POWERSHELL_EXECUTABLE}"
                -ExecutionPolicy Bypass
                -File "${CMAKE_CURRENT_SOURCE_DIR}/packaging/generate-brand-assets.ps1"
                -SourcePng "${JAKAL_CORE_BRAND_SOURCE_PNG}"
                -OutputDir "${JAKAL_CORE_BRAND_OUTPUT_DIR}"
                -BaseName "jakal-core-logo"
            RESULT_VARIABLE JAKAL_CORE_BRAND_GENERATION_RESULT
        )
        if(NOT JAKAL_CORE_BRAND_GENERATION_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to generate Windows branding assets from ${JAKAL_CORE_BRAND_SOURCE_PNG}")
        endif()

        set(JAKAL_CORE_BRAND_RUNTIME_PNG "${JAKAL_CORE_BRAND_OUTPUT_DIR}/jakal-core-logo.png")
        set(JAKAL_CORE_BRAND_RUNTIME_ICO "${JAKAL_CORE_BRAND_OUTPUT_DIR}/jakal-core-logo.ico")
        set(JAKAL_CORE_BRAND_NSIS_HEADER_BMP "${JAKAL_CORE_BRAND_OUTPUT_DIR}/jakal-core-logo-header.bmp")
        set(JAKAL_CORE_BRAND_NSIS_WIZARD_BMP "${JAKAL_CORE_BRAND_OUTPUT_DIR}/jakal-core-logo-wizard.bmp")
    else()
        set(JAKAL_CORE_BRAND_RUNTIME_PNG "${JAKAL_CORE_BRAND_SOURCE_PNG}")
    endif()
endif()

add_library(jakal_core
    src/device.cpp
    src/execution.cpp
    src/executor.cpp
    src/jakal_l0.cpp
    src/jakal_toolkit.cpp
    src/operation_variant_registry.cpp
    src/executors/common_scheduler.cpp
    src/executors/direct_backends.cpp
    src/executors/host_native_kernels.cpp
    src/executors/host_native_kernels_avx512.cpp
    src/executors/host_native_kernels_bf16.cpp
    src/executors/native_gpu_backends.cpp
    src/executors/vulkan_direct_backend.cpp
    src/planner.cpp
    src/workloads.cpp
    src/runtime.cpp
    src/backends/cpu_backend.cpp
    src/backends/native_gpu_probes.cpp
    src/backends/opencl_backend.cpp)

add_library(jakal::core ALIAS jakal_core)

target_include_directories(jakal_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

FetchContent_Declare(
    vulkan_headers
    URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.321.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(vulkan_headers)
target_include_directories(jakal_core PRIVATE ${vulkan_headers_SOURCE_DIR}/include)

target_compile_features(jakal_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(jakal_core PRIVATE /W4 /permissive-)
    target_compile_definitions(jakal_core PRIVATE NOMINMAX)
else()
    target_compile_options(jakal_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32 AND JAKAL_CORE_ENABLE_CODE_SIGNING)
    if(NOT JAKAL_CORE_SIGNTOOL_PATH)
        find_program(JAKAL_CORE_SIGNTOOL_PATH signtool)
    endif()
    if(NOT JAKAL_CORE_SIGNTOOL_PATH)
        message(FATAL_ERROR "JAKAL_CORE_ENABLE_CODE_SIGNING requires signtool.exe")
    endif()
    if(NOT JAKAL_CORE_CODESIGN_CERT_SHA1)
        message(FATAL_ERROR "JAKAL_CORE_ENABLE_CODE_SIGNING requires JAKAL_CORE_CODESIGN_CERT_SHA1")
    endif()
endif()

function(jakal_enable_codesign target_name)
    if(WIN32 AND JAKAL_CORE_ENABLE_CODE_SIGNING)
        add_custom_command(
            TARGET ${target_name}
            POST_BUILD
            COMMAND "${JAKAL_CORE_SIGNTOOL_PATH}" sign
                /fd SHA256
                /sha1 "${JAKAL_CORE_CODESIGN_CERT_SHA1}"
                /tr "${JAKAL_CORE_CODESIGN_TIMESTAMP_URL}"
                /td SHA256
                "$<TARGET_FILE:${target_name}>"
            VERBATIM)
    endif()
endfunction()

if(MSVC)
    set_source_files_properties(
        src/executors/host_native_kernels.cpp
        PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(
        src/executors/host_native_kernels_avx512.cpp
        PROPERTIES
            COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(
        src/executors/host_native_kernels_bf16.cpp
        PROPERTIES
            COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(
        src/executors/host_native_kernels.cpp
        PROPERTIES
            COMPILE_OPTIONS "-mavx2;-mf16c")
    set_source_files_properties(
        src/executors/host_native_kernels_avx512.cpp
        PROPERTIES
            COMPILE_OPTIONS "-mavx512f")
    set_source_files_properties(
        src/executors/host_native_kernels_bf16.cpp
        PROPERTIES
            COMPILE_OPTIONS "-mavx512f;-mavx512bf16")
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(jakal_core PRIVATE dl)
endif()

add_library(jakal_runtime SHARED src/c_api.cpp)
add_library(jakal::runtime ALIAS jakal_runtime)
target_link_libraries(jakal_runtime PUBLIC jakal_core)
target_include_directories(jakal_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(jakal_runtime PUBLIC cxx_std_20)
target_compile_definitions(jakal_runtime PRIVATE JAKAL_CORE_C_API_EXPORTS)
target_compile_definitions(jakal_runtime INTERFACE JAKAL_CORE_C_API_IMPORTS)
if(MSVC)
    target_compile_options(jakal_runtime PRIVATE /W4 /permissive-)
    target_compile_definitions(jakal_runtime PRIVATE NOMINMAX)
else()
    target_compile_options(jakal_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(jakal_core_cli src/jakal_core_cli.cpp)
target_link_libraries(jakal_core_cli PRIVATE jakal_core)
jakal_enable_codesign(jakal_core_cli)

if(JAKAL_CORE_BUILD_PRODUCT_TOOLS)
    add_executable(jakal_bootstrap examples/bootstrap_runtime.cpp)
    target_link_libraries(jakal_bootstrap PRIVATE jakal::core)
    jakal_enable_codesign(jakal_bootstrap)
endif()

if(JAKAL_CORE_BUILD_EXAMPLES)
    add_executable(jakal_inspect examples/inspect_runtime.cpp)
    target_link_libraries(jakal_inspect PRIVATE jakal::core)
    jakal_enable_codesign(jakal_inspect)

    add_executable(jakal_profile_workloads examples/profile_workloads.cpp)
    target_link_libraries(jakal_profile_workloads PRIVATE jakal::core)

    add_executable(jakal_explore_cpu_dl examples/explore_cpu_dl.cpp)
    target_link_libraries(jakal_explore_cpu_dl PRIVATE jakal::core)

    add_executable(jakal_partition_roles examples/partition_roles.cpp)
    target_link_libraries(jakal_partition_roles PRIVATE jakal::core)

    add_executable(jakal_profile_manifest examples/profile_manifest.cpp)
    target_link_libraries(jakal_profile_manifest PRIVATE jakal::core)

    if(WIN32)
        add_executable(jakal_directml_manifest_bench examples/directml_manifest_bench.cpp)
        target_link_libraries(jakal_directml_manifest_bench PRIVATE jakal::core d3d12 dxgi dxguid directml)
    endif()
endif()

if(JAKAL_CORE_BUILD_TESTS)
    enable_testing()

    if(JAKAL_CORE_BUILD_PRODUCT_TOOLS)
        add_test(NAME jakal_bootstrap_status COMMAND jakal_bootstrap --status --no-persist)
    endif()

    add_executable(jakal_smoke tests/smoke.cpp)
    target_link_libraries(jakal_smoke PRIVATE jakal::core jakal::runtime)

    add_test(NAME jakal_smoke COMMAND jakal_smoke)

    add_executable(jakal_optimization tests/optimization.cpp)
    target_link_libraries(jakal_optimization PRIVATE jakal::core)

    add_test(NAME jakal_optimization COMMAND jakal_optimization --fast)
    add_test(NAME jakal_optimization_long COMMAND jakal_optimization --long)
    set_tests_properties(jakal_optimization_long PROPERTIES TIMEOUT 900)

    add_executable(jakal_planner_learning
        tests/planner_strategy_learning.cpp
        src/device.cpp
        src/planner.cpp)
    target_include_directories(jakal_planner_learning PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(jakal_planner_learning PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(jakal_planner_learning PRIVATE /W4 /permissive-)
        target_compile_definitions(jakal_planner_learning PRIVATE NOMINMAX)
    else()
        target_compile_options(jakal_planner_learning PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME jakal_planner_learning COMMAND jakal_planner_learning)

    add_executable(jakal_partition_strategies tests/partition_strategies.cpp)
    target_link_libraries(jakal_partition_strategies PRIVATE jakal::core)

    add_test(NAME jakal_partition_strategies COMMAND jakal_partition_strategies)

    add_executable(jakal_runtime_product tests/runtime_product.cpp)
    target_link_libraries(jakal_runtime_product PRIVATE jakal::core)

    add_test(NAME jakal_runtime_product COMMAND jakal_runtime_product)

    add_executable(jakal_workload_import_adapters tests/workload_import_adapters.cpp)
    target_link_libraries(jakal_workload_import_adapters PRIVATE jakal::core)

    add_test(NAME jakal_workload_import_adapters COMMAND jakal_workload_import_adapters)

    add_executable(jakal_backend_contracts tests/backend_contracts.cpp)
    target_link_libraries(jakal_backend_contracts PRIVATE jakal::core)

    add_test(NAME jakal_backend_contracts COMMAND jakal_backend_contracts)

    add_executable(jakal_live_backend_smoke tests/live_backend_smoke.cpp)
    target_link_libraries(jakal_live_backend_smoke PRIVATE jakal::core)

    add_test(NAME jakal_live_backend_smoke COMMAND jakal_live_backend_smoke --host-only)

    add_executable(jakal_preset_execution_diag tests/preset_execution_diag.cpp)
    target_link_libraries(jakal_preset_execution_diag PRIVATE jakal::core)

    add_executable(jakal_runtime_install_smoke tests/runtime_install_smoke.cpp)
    target_link_libraries(jakal_runtime_install_smoke PRIVATE jakal::runtime)
    add_test(NAME jakal_runtime_install_smoke COMMAND jakal_runtime_install_smoke)

    if(WIN32)
        find_program(JAKAL_CORE_POWERSHELL_EXECUTABLE NAMES powershell pwsh REQUIRED)
        add_test(
            NAME jakal_package_smoke
            COMMAND
                "${JAKAL_CORE_POWERSHELL_EXECUTABLE}"
                -ExecutionPolicy Bypass
                -File "${CMAKE_CURRENT_SOURCE_DIR}/packaging/package-smoke.ps1"
                -BuildDir "${CMAKE_CURRENT_BINARY_DIR}"
                -BuildConfig "$<CONFIG>")
        set_tests_properties(jakal_package_smoke PROPERTIES TIMEOUT 900)
    endif()

    add_executable(jakal_workload_bench tests/workload_bench.cpp)
    target_link_libraries(jakal_workload_bench PRIVATE jakal::core)
    add_test(NAME jakal_workload_bench COMMAND jakal_workload_bench --smoke --host-only)

    add_test(NAME jakal_core_cli_doctor_json COMMAND jakal_core_cli doctor --json --host-only)
endif()

set(jakal_installable_runtime_targets jakal_core)
list(APPEND jakal_installable_runtime_targets jakal_runtime jakal_core_cli)
if(TARGET jakal_bootstrap)
    list(APPEND jakal_installable_runtime_targets jakal_bootstrap)
endif()
if(TARGET jakal_inspect)
    list(APPEND jakal_installable_runtime_targets jakal_inspect)
endif()

install(
    TARGETS ${jakal_installable_runtime_targets}
    EXPORT JakalCoreTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT development
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
    DIRECTORY include/jakal
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT development)

install(
    FILES
        README.md
        LICENSE
        docs/distribution.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    COMPONENT docs)

install(
    FILES
        packaging/install-jakal-core.ps1
        packaging/build-nsis-package.ps1
        packaging/sign-and-verify-artifact.ps1
        packaging/jakal-prerequisites.json
    DESTINATION ${CMAKE_INSTALL_DATADIR}/jakal-core/install
    COMPONENT maintenance)

install(
    DIRECTORY
        packaging/prereqs/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/jakal-core/install/prereqs
    COMPONENT maintenance)

install(
    FILES
        packaging/update-jakal-core.ps1
        packaging/sign-and-verify-artifact.ps1
    DESTINATION ${CMAKE_INSTALL_DATADIR}/jakal-core/update
    COMPONENT maintenance)

install(
    FILES
        packaging/remove-jakal-core.ps1
    DESTINATION ${CMAKE_INSTALL_DATADIR}/jakal-core/remove
    COMPONENT maintenance)

install(
    FILES
        packaging/jakal-runtime-config.ini
    DESTINATION ${CMAKE_INSTALL_DATADIR}/jakal-core/config
    COMPONENT runtime)

if(JAKAL_CORE_BRAND_RUNTIME_PNG)
    install(
        FILES
            "${JAKAL_CORE_BRAND_RUNTIME_PNG}"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/jakal-core/branding
        COMPONENT runtime)
endif()

if(JAKAL_CORE_BRAND_RUNTIME_ICO)
    install(
        FILES
            "${JAKAL_CORE_BRAND_RUNTIME_ICO}"
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT runtime)

    install(
        FILES
            "${JAKAL_CORE_BRAND_RUNTIME_ICO}"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/jakal-core/branding
        COMPONENT runtime)
endif()

install(
    FILES
        python/jakal_runtime.py
        python/jakal_app_adapter.py
    DESTINATION ${CMAKE_INSTALL_DATADIR}/jakal-core/python
    COMPONENT runtime)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/JakalCoreConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/JakalCoreConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/JakalCore)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/JakalCoreConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(
    EXPORT JakalCoreTargets
    FILE JakalCoreTargets.cmake
    NAMESPACE jakal::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/JakalCore
    COMPONENT development)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/JakalCoreConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/JakalCoreConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/JakalCore
    COMPONENT development)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
    IMMEDIATE
    @ONLY)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/packaging/launch-jakal-hardware-setup.cmd.in
    ${CMAKE_CURRENT_BINARY_DIR}/launch-jakal-hardware-setup.cmd
    @ONLY)

install(
    PROGRAMS
        ${CMAKE_CURRENT_BINARY_DIR}/launch-jakal-hardware-setup.cmd
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT runtime)

add_custom_target(
    uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

set(CPACK_PACKAGE_NAME "Jakal-Core")
set(CPACK_PACKAGE_VENDOR "Jakal")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Graph-first heterogeneous compute runtime skeleton")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Jakal-Core")
set(CPACK_PACKAGE_VERSION ${JAKAL_CORE_PACKAGE_VERSION})
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_COMPONENTS_ALL runtime development docs maintenance)
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "Runtime")
set(CPACK_COMPONENT_DEVELOPMENT_DISPLAY_NAME "Development Files")
set(CPACK_COMPONENT_DOCS_DISPLAY_NAME "Documentation")
set(CPACK_COMPONENT_MAINTENANCE_DISPLAY_NAME "Maintenance Tools")

if(WIN32)
    set(_jakal_cpack_generators ZIP)
    find_program(JAKAL_CORE_MAKENSIS_EXECUTABLE makensis)
    if(JAKAL_CORE_MAKENSIS_EXECUTABLE)
        list(APPEND _jakal_cpack_generators NSIS)
        set(CPACK_NSIS_DISPLAY_NAME "Jakal-Core")
        set(CPACK_NSIS_PACKAGE_NAME "Jakal-Core")
        set(CPACK_NSIS_EXECUTABLE "${JAKAL_CORE_MAKENSIS_EXECUTABLE}")
        set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
        set(CPACK_NSIS_MODIFY_PATH OFF)
        set(CPACK_NSIS_MUI_FINISHPAGE_RUN "bin\\\\launch-jakal-hardware-setup.cmd")
        if(JAKAL_CORE_BRAND_RUNTIME_ICO)
            string(REPLACE "/" "\\\\" _jakal_brand_ico_nsis "${JAKAL_CORE_BRAND_RUNTIME_ICO}")
            set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE
"!define MUI_ICON \\\"${_jakal_brand_ico_nsis}\\\"
!define MUI_UNICON \\\"${_jakal_brand_ico_nsis}\\\"")
            set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\jakal-core-logo.ico")
            set(CPACK_NSIS_CREATE_ICONS_EXTRA
                "CreateShortCut \\\"$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\Jakal Hardware Setup.lnk\\\" \\\"$INSTDIR\\\\bin\\\\launch-jakal-hardware-setup.cmd\\\" \\\"\\\" \\\"$INSTDIR\\\\bin\\\\jakal-core-logo.ico\\\" 0")
        else()
            set(CPACK_NSIS_CREATE_ICONS_EXTRA
                "CreateShortCut \\\"$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\Jakal Hardware Setup.lnk\\\" \\\"$INSTDIR\\\\bin\\\\launch-jakal-hardware-setup.cmd\\\"")
        endif()
        if(JAKAL_CORE_BRAND_NSIS_HEADER_BMP OR JAKAL_CORE_BRAND_NSIS_WIZARD_BMP)
            set(_jakal_branding_code "")
            if(JAKAL_CORE_BRAND_NSIS_HEADER_BMP)
                string(REPLACE "/" "\\\\" _jakal_brand_header_nsis "${JAKAL_CORE_BRAND_NSIS_HEADER_BMP}")
                string(APPEND _jakal_branding_code
"!define MUI_HEADERIMAGE_BITMAP \\\"${_jakal_brand_header_nsis}\\\"
!define MUI_HEADERIMAGE_UNBITMAP \\\"${_jakal_brand_header_nsis}\\\"
")
            endif()
            if(JAKAL_CORE_BRAND_NSIS_WIZARD_BMP)
                string(REPLACE "/" "\\\\" _jakal_brand_wizard_nsis "${JAKAL_CORE_BRAND_NSIS_WIZARD_BMP}")
                string(APPEND _jakal_branding_code
"!define MUI_WELCOMEFINISHPAGE_BITMAP \\\"${_jakal_brand_wizard_nsis}\\\"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP \\\"${_jakal_brand_wizard_nsis}\\\"
")
            endif()
            set(CPACK_NSIS_INSTALLER_MUI_WELCOMEFINISH_CODE "${_jakal_branding_code}")
        endif()
        set(CPACK_NSIS_DELETE_ICONS_EXTRA
            "Delete \\\"$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\Jakal Hardware Setup.lnk\\\"")
        if(TARGET jakal_bootstrap)
            set(CPACK_NSIS_MENU_LINKS
                "bin\\\\jakal_bootstrap.exe" "Jakal Bootstrap"
                "bin\\\\launch-jakal-hardware-setup.cmd" "Jakal Hardware Setup"
                "share\\\\doc\\\\JakalCore\\\\README.md" "README")
        else()
            set(CPACK_NSIS_MENU_LINKS
                "bin\\\\launch-jakal-hardware-setup.cmd" "Jakal Hardware Setup"
                "share\\\\doc\\\\JakalCore\\\\README.md" "README")
        endif()
    endif()
    set(CPACK_GENERATOR "${_jakal_cpack_generators}")
else()
    set(CPACK_GENERATOR "TGZ;ZIP")
endif()

include(CPack)

