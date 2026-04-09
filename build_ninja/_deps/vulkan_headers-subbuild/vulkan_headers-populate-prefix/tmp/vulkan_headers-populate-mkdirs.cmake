# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "D:/GitHub/Jakal-Core/build_ninja/_deps/vulkan_headers-src")
  file(MAKE_DIRECTORY "D:/GitHub/Jakal-Core/build_ninja/_deps/vulkan_headers-src")
endif()
file(MAKE_DIRECTORY
  "D:/GitHub/Jakal-Core/build_ninja/_deps/vulkan_headers-build"
  "D:/GitHub/Jakal-Core/build_ninja/_deps/vulkan_headers-subbuild/vulkan_headers-populate-prefix"
  "D:/GitHub/Jakal-Core/build_ninja/_deps/vulkan_headers-subbuild/vulkan_headers-populate-prefix/tmp"
  "D:/GitHub/Jakal-Core/build_ninja/_deps/vulkan_headers-subbuild/vulkan_headers-populate-prefix/src/vulkan_headers-populate-stamp"
  "D:/GitHub/Jakal-Core/build_ninja/_deps/vulkan_headers-subbuild/vulkan_headers-populate-prefix/src"
  "D:/GitHub/Jakal-Core/build_ninja/_deps/vulkan_headers-subbuild/vulkan_headers-populate-prefix/src/vulkan_headers-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/GitHub/Jakal-Core/build_ninja/_deps/vulkan_headers-subbuild/vulkan_headers-populate-prefix/src/vulkan_headers-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/GitHub/Jakal-Core/build_ninja/_deps/vulkan_headers-subbuild/vulkan_headers-populate-prefix/src/vulkan_headers-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
