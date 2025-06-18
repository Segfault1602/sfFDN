cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 23)

project(sfFDN)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SF_FDN_USE_VDSP "Use VDSP for DSP operations" OFF)

set(COMMON_CXX_FLAGS $<$<CONFIG:Release>:-O3> -fno-omit-frame-pointer)

include(./fetch_deps.cmake)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(py_binding)
