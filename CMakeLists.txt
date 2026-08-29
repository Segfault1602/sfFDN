cmake_minimum_required(VERSION 3.31)

project(
    sfFDN
    DESCRIPTION "A C++ library for Feedback Delay Networks (FDNs) for artificial reverberation"
    LANGUAGES CXX C
    VERSION 1.0.0
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SFFDN_BUILD_TESTS "Enable building tests. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF}."
       ${PROJECT_IS_TOP_LEVEL}
)
option(SFFDN_USE_SANITIZER "Use AddressSanitizer and UndefinedBehaviorSanitizer" ON)
option(SFFDN_USE_RTSAN "Use Clang RealtimeSanitizer" OFF)
option(SFFDN_ENABLE_HARDENING "Enable hardening" OFF)
option(SFFDN_ENABLE_STATIC_ANALYSIS "Enable static analysis with clang-tidy" OFF)
option(SFFDN_ENABLE_PCH "Enable precompiled headers for sfFDN targets" OFF)
option(
    SFFDN_ENABLE_LTO
    "Enable link-time optimization for sfFDN targets in non-Debug configurations"
    ON
)
option(SFFDN_USE_AVX2 "Compile sfFDN with AVX2/FMA; the binary requires a compatible x86 CPU." OFF)
option(SFFDN_USE_VDSP "Use vDSP for dsp operations" "${APPLE}")
option(SFFDN_ENABLE_FUNCTION_EFFECTS "Enable Clang Function Effect Analysis" OFF)
option(
    SFFDN_TREAT_FUNCTION_EFFECTS_AS_ERRORS
    "Treat Clang Function Effect Analysis diagnostics as errors"
    OFF
)

if(SFFDN_USE_RTSAN AND SFFDN_USE_SANITIZER)
    message(
        FATAL_ERROR
        "SFFDN_USE_RTSAN and SFFDN_USE_SANITIZER are mutually exclusive; disable one sanitizer configuration"
    )
endif()

if(SFFDN_TREAT_FUNCTION_EFFECTS_AS_ERRORS AND NOT SFFDN_ENABLE_FUNCTION_EFFECTS)
    message(
        FATAL_ERROR
        "SFFDN_TREAT_FUNCTION_EFFECTS_AS_ERRORS requires SFFDN_ENABLE_FUNCTION_EFFECTS=ON"
    )
endif()

include(./fetch_deps.cmake)

include(cmake/project-warnings.cmake)
include(cmake/project-options.cmake)

add_subdirectory(src)

if(SFFDN_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
