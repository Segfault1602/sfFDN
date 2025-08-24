cmake_minimum_required(VERSION 3.31)

project(
    sfFDN
    LANGUAGES CXX C
    VERSION 0.1.1
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SFFDN_BUILD_TESTS "Build tests" ON)
option(SFFDN_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building for macOS")
    option(SFFDN_USE_VDSP "Use vDSP for matrix multiplication" OFF)
endif()

include(./fetch_deps.cmake)

add_subdirectory(src)

if(SFFDN_BUILD_PYTHON_BINDINGS)
    add_subdirectory(py_binding)
endif()

if(SFFDN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
