# CMake generated Testfile for 
# Source directory: D:/GitHub/Jakal-Core
# Build directory: D:/GitHub/Jakal-Core/build_ninja
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[jakal_bootstrap_status]=] "D:/GitHub/Jakal-Core/build_ninja/jakal_bootstrap.exe" "--status" "--no-persist")
set_tests_properties([=[jakal_bootstrap_status]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/GitHub/Jakal-Core/CMakeLists.txt;151;add_test;D:/GitHub/Jakal-Core/CMakeLists.txt;0;")
add_test([=[jakal_smoke]=] "D:/GitHub/Jakal-Core/build_ninja/jakal_smoke.exe")
set_tests_properties([=[jakal_smoke]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/GitHub/Jakal-Core/CMakeLists.txt;157;add_test;D:/GitHub/Jakal-Core/CMakeLists.txt;0;")
add_test([=[jakal_optimization]=] "D:/GitHub/Jakal-Core/build_ninja/jakal_optimization.exe" "--fast")
set_tests_properties([=[jakal_optimization]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/GitHub/Jakal-Core/CMakeLists.txt;162;add_test;D:/GitHub/Jakal-Core/CMakeLists.txt;0;")
add_test([=[jakal_optimization_long]=] "D:/GitHub/Jakal-Core/build_ninja/jakal_optimization.exe" "--long")
set_tests_properties([=[jakal_optimization_long]=] PROPERTIES  TIMEOUT "900" _BACKTRACE_TRIPLES "D:/GitHub/Jakal-Core/CMakeLists.txt;163;add_test;D:/GitHub/Jakal-Core/CMakeLists.txt;0;")
add_test([=[jakal_planner_learning]=] "D:/GitHub/Jakal-Core/build_ninja/jakal_planner_learning.exe")
set_tests_properties([=[jakal_planner_learning]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/GitHub/Jakal-Core/CMakeLists.txt;178;add_test;D:/GitHub/Jakal-Core/CMakeLists.txt;0;")
add_test([=[jakal_partition_strategies]=] "D:/GitHub/Jakal-Core/build_ninja/jakal_partition_strategies.exe")
set_tests_properties([=[jakal_partition_strategies]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/GitHub/Jakal-Core/CMakeLists.txt;183;add_test;D:/GitHub/Jakal-Core/CMakeLists.txt;0;")
add_test([=[jakal_runtime_product]=] "D:/GitHub/Jakal-Core/build_ninja/jakal_runtime_product.exe")
set_tests_properties([=[jakal_runtime_product]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/GitHub/Jakal-Core/CMakeLists.txt;188;add_test;D:/GitHub/Jakal-Core/CMakeLists.txt;0;")
add_test([=[jakal_workload_import_adapters]=] "D:/GitHub/Jakal-Core/build_ninja/jakal_workload_import_adapters.exe")
set_tests_properties([=[jakal_workload_import_adapters]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/GitHub/Jakal-Core/CMakeLists.txt;193;add_test;D:/GitHub/Jakal-Core/CMakeLists.txt;0;")
add_test([=[jakal_backend_contracts]=] "D:/GitHub/Jakal-Core/build_ninja/jakal_backend_contracts.exe")
set_tests_properties([=[jakal_backend_contracts]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/GitHub/Jakal-Core/CMakeLists.txt;198;add_test;D:/GitHub/Jakal-Core/CMakeLists.txt;0;")
add_test([=[jakal_live_backend_smoke]=] "D:/GitHub/Jakal-Core/build_ninja/jakal_live_backend_smoke.exe" "--host-only")
set_tests_properties([=[jakal_live_backend_smoke]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/GitHub/Jakal-Core/CMakeLists.txt;203;add_test;D:/GitHub/Jakal-Core/CMakeLists.txt;0;")
subdirs("_deps/vulkan_headers-build")
