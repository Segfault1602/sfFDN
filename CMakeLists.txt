cmake_minimum_required(VERSION 3.31)

project(
    sfFDN
    DESCRIPTION "A C++ library for Feedback Delay Networks (FDNs) for artificial reverberation"
    LANGUAGES CXX C
    VERSION 0.1.1
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_CLANG_TIDY "clang-tidy")

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0)
endif()

option(SFFDN_BUILD_TESTS "Enable building tests. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF}."
       ${PROJECT_IS_TOP_LEVEL}
)
option(SFFDN_BUILD_PYTHON_BINDINGS "Build Python bindings" ${PROJECT_IS_TOP_LEVEL})

option(SFFDN_USE_SANITIZER "Use AddressSanitizer" OFF)
option(SFFDN_ENABLE_HARDENING "Enable hardening" OFF)

option(SFFDN_USE_VDSP "Use vDSP for dsp operations" OFF)

include(./fetch_deps.cmake)

include(cmake/project-warnings.cmake)
include(cmake/project-options.cmake)

add_subdirectory(src)

# if(SFFDN_BUILD_PYTHON_BINDINGS) add_subdirectory(py_binding) endif()

if(SFFDN_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
