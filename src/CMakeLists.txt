set(FDN_SOURCE
    delaya.cpp
    delay.cpp
    delay_matrix.cpp
    delay_time_varying.cpp
    delaybank.cpp
    filter.cpp
    filter_feedback_matrix.cpp
    mixing_matrix.cpp
    filterbank.cpp
    fdn.cpp
    schroeder_allpass.cpp)

add_subdirectory(CMSIS)

add_library(fdn ${FDN_SOURCE})
target_include_directories(fdn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fdn PUBLIC Eigen3::Eigen "-framework Accelerate")
target_compile_options(fdn PRIVATE -Wno-deprecated)
