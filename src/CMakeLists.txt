set(FDN_SOURCE
    audio_buffer.cpp
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
    filter_utils.cpp
    math_utils.cpp
    matrix_gallery.cpp
    matrix_multiplication.cpp
    nupols.cpp
    parallel_gains.cpp
    schroeder_allpass.cpp
    upols.cpp
    )

if(SF_FDN_USE_VDSP)
    message(STATUS "Using VDSP for DSP operations")
    set(FDN_SOURCE ${FDN_SOURCE} vdsp_array_math.cpp )
else()
    message(STATUS "Using vanilla C++ for DSP operations")
    set(FDN_SOURCE ${FDN_SOURCE} array_math.cpp )
endif()

add_library(sfFDN ${FDN_SOURCE})
target_include_directories(sfFDN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${pffft_SOURCE_DIR})
target_link_libraries(sfFDN PUBLIC Eigen3::Eigen PFFFT kissfft)

if(SF_FDN_USE_VDSP)
    target_link_libraries(sfFDN PUBLIC "-framework Accelerate")
endif()

target_compile_options(sfFDN PRIVATE ${COMMON_CXX_FLAGS})

if (NOT WIN32)
    target_compile_options(sfFDN PRIVATE $<$<CONFIG:Debug>:-fsanitize=address> -fno-omit-frame-pointer)
    target_link_options(sfFDN PUBLIC $<$<CONFIG:Debug>:-fsanitize=address>)
endif()
