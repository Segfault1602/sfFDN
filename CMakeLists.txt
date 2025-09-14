cmake_minimum_required(VERSION 3.31)

project(
    sfFDN
    DESCRIPTION "A C++ library for Feedback Delay Networks (FDNs) for artificial reverberation"
    LANGUAGES CXX C
    VERSION 0.1.1
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SFFDN_BUILD_TESTS "Enable building tests. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF}."
       ${PROJECT_IS_TOP_LEVEL}
)
option(SFFDN_BUILD_PYTHON_BINDINGS "Build Python bindings" ${PROJECT_IS_TOP_LEVEL})

option(SFFDN_USE_VDSP "Use vDSP for dsp operations" OFF)

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
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
