set(FDN_SOURCE
    delaya.cpp
    delay.cpp
    delay_matrix.cpp
    delay_time_varying.cpp
    delaybank.cpp
    filter.cpp
    filter_feedback_matrix.cpp
    mixing_matrix.cpp
    parallel_gains.cpp
    filterbank.cpp
    fdn.cpp
    schroeder_allpass.cpp)

if(SF_FDN_USE_VDSP)
    message(STATUS "Using VDSP for DSP operations")
    set(FDN_SOURCE ${FDN_SOURCE} vdsp_array_math.cpp vdsp_cascaded_filter.cpp)
else()
    message(STATUS "Using vanilla C++ for DSP operations")
    set(FDN_SOURCE ${FDN_SOURCE} array_math.cpp cascaded_filter.cpp)
endif()

add_library(fdn ${FDN_SOURCE})
target_include_directories(fdn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fdn PUBLIC Eigen3::Eigen)

if(SF_FDN_USE_VDSP)
    target_link_libraries(fdn PUBLIC "-framework Accelerate")
endif()

target_compile_options(fdn PRIVATE -Wno-deprecated $<$<CONFIG:Debug>:-fsanitize=address>
                                   $<$<CONFIG:Release>:-O3>)
target_link_options(fdn PUBLIC $<$<CONFIG:Debug>:-fsanitize=address>)
