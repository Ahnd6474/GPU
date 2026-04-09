#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "jakal::jakal_core" for configuration "Debug"
set_property(TARGET jakal::jakal_core APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(jakal::jakal_core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/jakal_core.lib"
  )

list(APPEND _cmake_import_check_targets jakal::jakal_core )
list(APPEND _cmake_import_check_files_for_jakal::jakal_core "${_IMPORT_PREFIX}/lib/jakal_core.lib" )

# Import target "jakal::jakal_runtime" for configuration "Debug"
set_property(TARGET jakal::jakal_runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(jakal::jakal_runtime PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/jakal_runtime.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/jakal_runtime.dll"
  )

list(APPEND _cmake_import_check_targets jakal::jakal_runtime )
list(APPEND _cmake_import_check_files_for_jakal::jakal_runtime "${_IMPORT_PREFIX}/lib/jakal_runtime.lib" "${_IMPORT_PREFIX}/bin/jakal_runtime.dll" )

# Import target "jakal::jakal_core_cli" for configuration "Debug"
set_property(TARGET jakal::jakal_core_cli APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(jakal::jakal_core_cli PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/jakal_core_cli.exe"
  )

list(APPEND _cmake_import_check_targets jakal::jakal_core_cli )
list(APPEND _cmake_import_check_files_for_jakal::jakal_core_cli "${_IMPORT_PREFIX}/bin/jakal_core_cli.exe" )

# Import target "jakal::jakal_bootstrap" for configuration "Debug"
set_property(TARGET jakal::jakal_bootstrap APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(jakal::jakal_bootstrap PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/jakal_bootstrap.exe"
  )

list(APPEND _cmake_import_check_targets jakal::jakal_bootstrap )
list(APPEND _cmake_import_check_files_for_jakal::jakal_bootstrap "${_IMPORT_PREFIX}/bin/jakal_bootstrap.exe" )

# Import target "jakal::jakal_inspect" for configuration "Debug"
set_property(TARGET jakal::jakal_inspect APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(jakal::jakal_inspect PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/jakal_inspect.exe"
  )

list(APPEND _cmake_import_check_targets jakal::jakal_inspect )
list(APPEND _cmake_import_check_files_for_jakal::jakal_inspect "${_IMPORT_PREFIX}/bin/jakal_inspect.exe" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
