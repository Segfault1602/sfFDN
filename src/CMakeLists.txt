find_package(IPP QUIET)

if(SFFDN_ENABLE_STATIC_ANALYSIS)
    message(STATUS "Enabling static analysis with clang-tidy")
    find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};--enable-check-profile")
    else()
        message(FATAL_ERROR "clang-tidy not found but SFFDN_ENABLE_STATIC_ANALYSIS is ON")
    endif()
endif()

add_library(sfFDN.sfFDN)
add_library(sfFDN::sfFDN ALIAS sfFDN.sfFDN)
sffdn_enable_lto(sfFDN.sfFDN)

set(SFFDN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_sources(
    sfFDN.sfFDN
    PRIVATE audio_buffer.cpp
            audio_processor.cpp
            array_math.cpp
            cascaded_filter.cpp
            channel_matrix.cpp
            channel_matrix_option_validation.cpp
            circular_buffer.cpp
            dattorro_delay.cpp
            delay.cpp
            delay_option_validation.cpp
            delaybank.cpp
            delaybank_time_varying.cpp
            delay_interp.cpp
            delay_matrix.cpp
            delay_time_varying.cpp
            delay_utils.cpp
            feedback_matrix.cpp
            matrix_option_validation.cpp
            fdn.cpp
            config_validation.cpp
            fdn_config.cpp
            fir.cpp
            fft.cpp
            filter.cpp
            filter_option_validation.cpp
            filterbank.cpp
            filter_design.cpp
            filter_feedback_matrix.cpp
            iirfilterbank.cpp
            kronecker_feedback_matrix.cpp
            math_utils.cpp
            matrix_data.cpp
            matrix_gallery.cpp
            matrix_multiplication.cpp
            nonlinear.cpp
            partitioned_convolver.cpp
            oscillator.cpp
            parallel_gains.cpp
            gain_option_validation.cpp
            processor_factory.cpp
            rng.cpp
            schroeder_allpass.cpp
            sparse_fir.cpp
            time_varying_feedback_matrix.cpp
            time_varying_kronecker_feedback_matrix.cpp
            time_varying_parallel_gains.cpp
            upols.cpp
)

target_sources(
    sfFDN.sfFDN
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${SFFDN_INCLUDE_DIRS}
           FILES
           ${SFFDN_INCLUDE_DIRS}/sffdn/attributes.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/audio_buffer.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/audio_processor.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/channel_matrix.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/dattorro_delay.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delay_matrix.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delay_utils.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delay.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delay_interp.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delay_time_varying.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delaybank.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delaybank_time_varying.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/config_diagnostics.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/fdn.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/fdn_config.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/feedback_matrix.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/filter_design.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/filter_feedback_matrix.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/filter.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/filterbank.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/kronecker_feedback_matrix.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/matrix_data.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/matrix_gallery.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/nonlinear.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/parallel_gains.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/partitioned_convolver.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/schroeder_allpass.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/sffdn.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/time_varying_feedback_matrix.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/time_varying_kronecker_feedback_matrix.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/types.h
)

target_include_directories(sfFDN.sfFDN PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${pffft_SOURCE_DIR})

target_link_libraries(
    sfFDN.sfFDN
    PRIVATE Eigen3::Eigen
            pffft::pffft
            kissfft::kissfft-float
            sfFDN::sfFDN_warnings
            sfFDN::sfFDN_options
)

target_compile_features(sfFDN.sfFDN PUBLIC cxx_std_23)

sffdn_enable_lto(sfFDN.sfFDN)

if(SFFDN_ENABLE_PCH)
    target_precompile_headers(
        sfFDN.sfFDN
        PRIVATE
            <Eigen/Core>
            <algorithm>
            <array>
            <cassert>
            <cmath>
            <cstdint>
            <memory>
            <optional>
            <span>
            <string>
            <utility>
            <variant>
            <vector>
    )
endif()

add_library(sfFDN.serialization)
add_library(sfFDN::serialization ALIAS sfFDN.serialization)
sffdn_enable_lto(sfFDN.serialization)

target_sources(
    sfFDN.serialization
    PRIVATE fdn_config_json.cpp
            json_helper.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${SFFDN_INCLUDE_DIRS}
           FILES
           ${SFFDN_INCLUDE_DIRS}/sffdn/serialization.h
)

target_include_directories(sfFDN.serialization PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    sfFDN.serialization
    PUBLIC sfFDN::sfFDN
           nlohmann_json::nlohmann_json
    PRIVATE sfFDN::sfFDN_warnings
            sfFDN::sfFDN_options
)

if(SFFDN_USE_VDSP)
    target_link_libraries(sfFDN.sfFDN PRIVATE "-framework Accelerate")
endif()

if(IPP_FOUND)
    message(STATUS "Using IPP optimizations")
    target_compile_definitions(sfFDN.sfFDN PRIVATE SFFDN_USE_IPP)
    target_link_libraries(sfFDN.sfFDN PRIVATE IPP::ippcore IPP::ipps IPP::ippvm)
endif()
