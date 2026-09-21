# ======== Find Dependencies ========

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

cpmaddpackage(
    NAME
    libsndfile
    GIT_TAG
    master
    GIT_REPOSITORY
    "https://github.com/libsndfile/libsndfile"
    OPTIONS
    "BUILD_PROGRAMS OFF"
    "BUILD_EXAMPLES OFF"
    "BUILD_REGTEST OFF"
    "ENABLE_EXTERNAL_LIBS OFF"
    "BUILD_TESTING OFF"
    "ENABLE_MPEG OFF"
    "ENABLE_CPACK OFF"
    "ENABLE_PACKAGE_CONFIG OFF"
    "INSTALL_PKGCONFIG_MODULE OFF"
)

cpmaddpackage(
    NAME
    Catch2
    GITHUB_REPOSITORY
    catchorg/Catch2
    VERSION
    3.14.0
)

cpmaddpackage(
    NAME
    nanobench
    GITHUB_REPOSITORY
    Segfault1602/nanobench
    GIT_TAG
    master
)

get_target_property(_catch2_inc Catch2 INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(Catch2 SYSTEM INTERFACE ${_catch2_inc})

get_target_property(_nanobench_inc nanobench INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(nanobench SYSTEM INTERFACE ${_nanobench_inc})

# ====================

add_executable(sfFDN.tests)
sffdn_enable_lto(sfFDN.tests)
target_sources(
    sfFDN.tests
    PRIVATE allocation_counter.cpp
            audio_buffer_tests.cpp
            audio_processor_chain_tests.cpp
            array_math_tests.cpp
            channel_matrix_tests.cpp
            dattorro_delay_tests.cpp
            delay_tests.cpp
            filter_tests.cpp
            filter_bank_tests.cpp
            kronecker_feedback_matrix_tests.cpp
            kronecker_stereo_signal_tests.cpp
            mixmat_tests.cpp
            matrix_multiplication_tests.cpp
            nonlinear_tests.cpp
            fdn_tests.cpp
            filter_design_tests.cpp
            fft_tests.cpp
            fdn_config_tests.cpp
            processor_option_validation_tests.cpp
            oscillator_tests.cpp
            upols_tests.cpp
            nupols_tests.cpp
            serialization_tests.cpp
            parallel_gains_tests.cpp
            sincos_tests.cpp
            time_varying_feedback_matrix_tests.cpp
            time_varying_kronecker_feedback_matrix_tests.cpp
            time_varying_fdn_signal_tests.cpp
            time_varying_schroeder_allpass_tests.cpp
            signal_test_utils.cpp
            test_utils.cpp
)
target_link_libraries(
    sfFDN.tests
    PRIVATE sfFDN::sfFDN
            sfFDN::serialization
            Catch2::Catch2WithMain
            SndFile::sndfile
            Eigen3::Eigen
            sfFDN::sfFDN_warnings
            sfFDN::sfFDN_options
)

set(SFFDN_INTERNAL_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../src)

target_include_directories(sfFDN.tests PRIVATE ${SFFDN_INTERNAL_INCLUDE_DIRS})
target_compile_features(sfFDN.tests PRIVATE cxx_std_23)

add_library(sfFDN.public_config_api OBJECT public_config_api_compile.cpp)
target_link_libraries(sfFDN.public_config_api PRIVATE sfFDN::sfFDN)
set_target_properties(sfFDN.public_config_api PROPERTIES DISABLE_PRECOMPILE_HEADERS ON)

add_executable(sfFDN.public_serialization_api public_serialization_api.cpp)
sffdn_enable_lto(sfFDN.public_serialization_api)
target_link_libraries(sfFDN.public_serialization_api PRIVATE sfFDN::serialization)

add_dependencies(sfFDN.tests sfFDN.public_config_api sfFDN.public_serialization_api)

if(SFFDN_ENABLE_PCH)
    target_precompile_headers(
        sfFDN.tests
        PRIVATE
            <catch2/catch_test_macros.hpp>
            <catch2/matchers/catch_matchers_floating_point.hpp>
            <sffdn/sffdn.h>
            <algorithm>
            <array>
            <iostream>
            <memory>
            <span>
            <vector>
    )
endif()

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(sfFDN.tests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
add_test(NAME PublicSerializationAPI COMMAND sfFDN.public_serialization_api)
add_test(
    NAME RealtimeExceptionInventory
    COMMAND
        ${CMAKE_COMMAND}
        -DSFFDN_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${PROJECT_SOURCE_DIR}/cmake/check-realtime-exceptions.cmake
)

add_executable(sfFDN.perf)
sffdn_enable_lto(sfFDN.perf)
set(
    SFFDN_PROCESSOR_PERF_SOURCES
    fdn_perf.cpp
    cascaded_biquads_perf.cpp
    fir_perf.cpp
    sparse_fir_perf.cpp
    filter_bank_perf.cpp
    iir_filter_bank_perf.cpp
    schroeder_allpass_section_perf.cpp
    time_varying_schroeder_allpass_section_perf.cpp
    one_pole_filter_perf.cpp
    allpass_filter_perf.cpp
    delay_interp_perf.cpp
    delay_time_varying_perf.cpp
    delay_bank_perf.cpp
    delay_bank_time_varying_perf.cpp
    dattorro_delay_perf.cpp
    filter_feedback_matrix_perf.cpp
    delay_matrix_perf.cpp
    scalar_feedback_matrix_perf.cpp
    controllable_full_wave_rectifier_perf.cpp
    signal_dependent_fractional_delay_perf.cpp
    ring_modulator_perf.cpp
    pass_through_perf.cpp
    audio_processor_chain_perf.cpp
    parallel_gains_perf.cpp
    time_varying_parallel_gains_perf.cpp
    partitioned_convolver_perf.cpp
    time_varying_feedback_matrix_perf.cpp
    kronecker_feedback_matrix_perf.cpp
    time_varying_kronecker_feedback_matrix_perf.cpp
    channel_matrix_perf.cpp
)
set(
    SFFDN_COMPARISON_PERF_SOURCES
    production_perf.cpp
    delay_fir_comparison_perf.cpp
    filter_bank_comparison_perf.cpp
    feedback_matrix_comparison_perf.cpp
    channel_matrix_comparison_perf.cpp
    processor_comparison_perf.cpp
)
set(
    SFFDN_LOW_LEVEL_PERF_SOURCES
    array_math_perf.cpp
    filter_design_perf.cpp
    matrix_multiplication_perf.cpp
    oscillator_perf.cpp
    sincos_perf.cpp
    upols_perf.cpp
)
set(SFFDN_PERF_SUPPORT_SOURCES production_workloads.cpp test_utils.cpp)

target_sources(
    sfFDN.perf
    PRIVATE ${SFFDN_PROCESSOR_PERF_SOURCES}
            ${SFFDN_COMPARISON_PERF_SOURCES}
            ${SFFDN_LOW_LEVEL_PERF_SOURCES}
            ${SFFDN_PERF_SUPPORT_SOURCES}
)
target_link_libraries(
    sfFDN.perf
    PRIVATE sfFDN.sfFDN
            nanobench::nanobench
            Catch2::Catch2WithMain
            SndFile::sndfile
            Eigen3::Eigen
            sfFDN::sfFDN_warnings
            sfFDN::sfFDN_options
)

if(APPLE)
    target_link_libraries(sfFDN.perf PRIVATE "-framework Accelerate")
endif()

target_include_directories(sfFDN.perf PRIVATE ${SFFDN_INTERNAL_INCLUDE_DIRS})
target_compile_features(sfFDN.perf PRIVATE cxx_std_23)

add_executable(sfFDN.callback_latency fdn_callback_latency.cpp production_workloads.cpp)
sffdn_enable_lto(sfFDN.callback_latency)
target_link_libraries(
    sfFDN.callback_latency
    PRIVATE sfFDN.sfFDN
            sfFDN::sfFDN_warnings
            sfFDN::sfFDN_options
)
target_include_directories(sfFDN.callback_latency PRIVATE ${SFFDN_INTERNAL_INCLUDE_DIRS})
target_compile_features(sfFDN.callback_latency PRIVATE cxx_std_23)

if(SFFDN_ENABLE_PCH)
    target_precompile_headers(
        sfFDN.perf
        PRIVATE
            <catch2/catch_test_macros.hpp>
            <nanobench.h>
            <sffdn/sffdn.h>
            <algorithm>
            <array>
            <iostream>
            <memory>
            <span>
            <vector>
    )
endif()
