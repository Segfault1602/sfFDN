find_package(Catch2 3 REQUIRED)

set(TEST_SOURCE
    audio_processor_chain_tests.cpp
    array_math_tests.cpp
    delay_tests.cpp
    filter_tests.cpp
    mixmat_tests.cpp
    matrix_multiplication_tests.cpp
    fdn_tests.cpp
    filter_design_tests.cpp
    fft_tests.cpp
    oscillator_tests.cpp
    upols_tests.cpp
    nupols_tests.cpp
    parallel_gains_tests.cpp
    test_utils.cpp)

add_executable(tests ${TEST_SOURCE})
target_link_libraries(tests PRIVATE sfFDN Catch2::Catch2WithMain SndFile::sndfile Eigen3::Eigen)

set(SFFDN_INTERNAL_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../src)

target_include_directories(tests PRIVATE ${SFFDN_INTERNAL_INCLUDE_DIRS})
target_compile_options(tests PRIVATE -fno-omit-frame-pointer)
target_compile_features(tests PRIVATE cxx_std_23)

if(NOT WIN32)
    target_compile_options(tests PRIVATE -fno-omit-frame-pointer -fsanitize=address)
    target_link_options(tests PRIVATE -fsanitize=address)
endif()

include(CTest)
include(Catch)
catch_discover_tests(tests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

set(PERF_SOURCE
    fdn_perf.cpp
    filter_perf.cpp
    filter_design_perf.cpp
    delay_perf.cpp
    array_math_perf.cpp
    mixmat_perf.cpp
    matrix_multiplication_perf.cpp
    parallel_gains_perf.cpp
    upols_perf.cpp
    nupols_perf.cpp
    test_utils.cpp)
add_executable(perf ${PERF_SOURCE})
target_link_libraries(perf PRIVATE sfFDN nanobench Catch2::Catch2WithMain SndFile::sndfile Eigen3::Eigen)

target_include_directories(perf PRIVATE ${SFFDN_INTERNAL_INCLUDE_DIRS})
target_compile_options(perf PRIVATE -fno-omit-frame-pointer)
target_compile_features(perf PRIVATE cxx_std_23)

target_compile_options(perf PRIVATE -O3 $<$<CONFIG:Debug>:-fsanitize=address>)
target_link_options(perf PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
