if(NOT EXISTS "D:/GitHub/Jakal-Core/build_ninja/install_manifest.txt")
    message(FATAL_ERROR "Cannot find install manifest: D:/GitHub/Jakal-Core/build_ninja/install_manifest.txt")
endif()

file(READ "D:/GitHub/Jakal-Core/build_ninja/install_manifest.txt" installed_files)
string(REPLACE "\n" ";" installed_files "${installed_files}")

foreach(installed_file ${installed_files})
    if(installed_file STREQUAL "")
        continue()
    endif()

    if(EXISTS "${installed_file}" OR IS_SYMLINK "${installed_file}")
        message(STATUS "Removing ${installed_file}")
        file(REMOVE "${installed_file}")
    endif()
endforeach()
