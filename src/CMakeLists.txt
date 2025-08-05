set(FDN_SOURCE
    audio_buffer.cpp
    audio_processor.cpp
    cascaded_filter.cpp
    circular_buffer.cpp
    delay.cpp
    delaya.cpp
    delaybank.cpp
    delay_matrix.cpp
    delay_utils.cpp
    delay_time_varying.cpp
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
    )

add_library(sfFDN)

if(SF_FDN_USE_VDSP)
    message(STATUS "Using VDSP for DSP operations")
    set(FDN_SOURCE ${FDN_SOURCE} vdsp_array_math.cpp )
    target_link_libraries(sfFDN PRIVATE "-framework Accelerate")
else()
    message(STATUS "Using vanilla C++ for DSP operations")
    set(FDN_SOURCE ${FDN_SOURCE} array_math.cpp )
endif()

set(SFFDN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../include)
set(SFFDN_HEADERS ${SFFDN_INCLUDE_DIRS}/sffdn/audio_buffer.h
        ${SFFDN_INCLUDE_DIRS}/sffdn/audio_processor.h
        ${SFFDN_INCLUDE_DIRS}/sffdn/delay_matrix.h
        ${SFFDN_INCLUDE_DIRS}/sffdn/delay_utils.h
        ${SFFDN_INCLUDE_DIRS}/sffdn//delay.h
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
        ${SFFDN_INCLUDE_DIRS}/sffdn/sffdn.h)


target_sources(sfFDN
    PRIVATE
        ${FDN_SOURCE}
    PUBLIC
        FILE_SET "sffdn_headers"
        TYPE HEADERS
        BASE_DIRS ${SFFDN_INCLUDE_DIRS}
        FILES ${SFFDN_HEADERS}
        )
target_include_directories(sfFDN PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${pffft_SOURCE_DIR})
target_link_libraries(sfFDN PRIVATE Eigen3::Eigen PFFFT kissfft)
target_compile_definitions(sfFDN PRIVATE -DEIGEN_RUNTIME_NO_MALLOC)
target_compile_options(sfFDN PRIVATE -fno-omit-frame-pointer)

if (NOT WIN32)
    target_compile_options(sfFDN PUBLIC $<$<CONFIG:Debug>:-fsanitize=address> -fno-omit-frame-pointer)
    target_link_options(sfFDN PUBLIC $<$<CONFIG:Debug>:-fsanitize=address>)
endif()
