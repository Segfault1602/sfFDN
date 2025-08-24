add_library(sfFDN)

set(SFFDN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_sources(
    sfFDN
    PRIVATE audio_buffer.cpp
            audio_processor.cpp
            array_math.cpp
            cascaded_filter.cpp
            circular_buffer.cpp
            delay.cpp
            delaybank.cpp
            delaya.cpp
            delay_matrix.cpp
            delay_time_varying.cpp
            delay_utils.cpp
            feedback_matrix.cpp
            fdn.cpp
            fft.cpp
            filter.cpp
            filterbank.cpp
            filter_design.cpp
            filter_feedback_matrix.cpp
            math_utils.cpp
            matrix_gallery.cpp
            matrix_multiplication.cpp
            partitioned_convolver.cpp
            oscillator.cpp
            parallel_gains.cpp
            rng.cpp
            schroeder_allpass.cpp
            upols.cpp
    PUBLIC FILE_SET
           "sffdn_headers"
           TYPE
           HEADERS
           BASE_DIRS
           ${SFFDN_INCLUDE_DIRS}
           FILES
           ${SFFDN_INCLUDE_DIRS}/sffdn/audio_buffer.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/audio_processor.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delay_matrix.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delay_utils.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delay.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delaya.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delaybank.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/fdn.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/feedback_matrix.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/filter_design.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/filter_feedback_matrix.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/filter.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/filterbank.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/matrix_gallery.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/parallel_gains.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/partitioned_convolver.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/schroeder_allpass.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/sffdn.h
)

target_precompile_headers(sfFDN PRIVATE pch.h)
target_include_directories(sfFDN PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${pffft_SOURCE_DIR})
target_link_libraries(sfFDN PRIVATE Eigen3::Eigen pffft::pffft kissfft::kissfft-float)
target_compile_options(sfFDN PRIVATE ${SFFDN_CXX_COMPILE_OPTIONS})
target_compile_features(sfFDN PRIVATE cxx_std_23)
target_link_options(sfFDN PUBLIC ${SFFDN_LINK_OPTIONS})

if(${SFFDN_USE_VDSP})
    target_link_libraries(sfFDN PRIVATE "-framework Accelerate")
    target_compile_definitions(sfFDN PRIVATE SFFDN_USE_VDSP)
endif()
