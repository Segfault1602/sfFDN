find_package(IPP)


add_library(sfFDN.sfFDN)
add_library(sfFDN::sfFDN ALIAS sfFDN.sfFDN)

set(SFFDN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_sources(
    sfFDN.sfFDN
    PRIVATE audio_buffer.cpp
            audio_processor.cpp
            array_math.cpp
            # vdsp_array_math.cpp
            cascaded_filter.cpp
            circular_buffer.cpp
            delay.cpp
            delaybank.cpp
            delaybank_time_varying.cpp
            delaya.cpp
            delay_interp.cpp
            delay_matrix.cpp
            delay_time_varying.cpp
            delay_utils.cpp
            feedback_matrix.cpp
            fdn.cpp
            fir.cpp
            fft.cpp
            filter.cpp
            filterbank.cpp
            filter_design.cpp
            filter_feedback_matrix.cpp
            iirfilterbank.cpp
            math_utils.cpp
            matrix_gallery.cpp
            matrix_multiplication.cpp
            partitioned_convolver.cpp
            oscillator.cpp
            parallel_gains.cpp
            rng.cpp
            schroeder_allpass.cpp
            sparse_fir.cpp
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
           ${SFFDN_INCLUDE_DIRS}/sffdn/audio_buffer.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/audio_processor.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delay_matrix.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delay_utils.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delay.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delaya.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delay_interp.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delay_time_varying.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delaybank.h
           ${SFFDN_INCLUDE_DIRS}/sffdn/delaybank_time_varying.h
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

target_include_directories(sfFDN.sfFDN PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${pffft_SOURCE_DIR})

target_link_libraries(sfFDN.sfFDN PRIVATE Eigen3::Eigen pffft::pffft kissfft::kissfft-float)
target_compile_options(sfFDN.sfFDN PRIVATE ${SFFDN_CXX_COMPILE_OPTIONS})
target_compile_features(sfFDN.sfFDN PRIVATE cxx_std_23)
target_link_options(sfFDN.sfFDN PUBLIC ${SFFDN_LINK_OPTIONS})
target_compile_definitions(sfFDN.sfFDN PRIVATE ${SFFDN_COMPILE_DEFINITION})

if(SFFDN_USE_VDSP)
    target_link_libraries(sfFDN.sfFDN PRIVATE "-framework Accelerate")
endif()

if (IPP_FOUND)
    message(STATUS "Using IPP optimizations")
    target_compile_definitions(sfFDN.sfFDN PRIVATE SFFDN_USE_IPP)
    target_link_libraries(sfFDN.sfFDN PRIVATE IPP::ippcore IPP::ipps)
endif()
