# ======== Find Dependencies ========
find_package(Catch2 3 REQUIRED)
find_package(SndFile CONFIG REQUIRED)
find_package(nanobench CONFIG REQUIRED)

# ====================

add_executable(sfFDN.tests)
target_sources(
    sfFDN.tests
    PRIVATE audio_buffer_tests.cpp
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
            test_utils.cpp
)
target_link_libraries(
    sfFDN.tests
    PRIVATE sfFDN.sfFDN
            Catch2::Catch2WithMain
            SndFile::sndfile
            Eigen3::Eigen
)

set(SFFDN_INTERNAL_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../src)

target_include_directories(sfFDN.tests PRIVATE ${SFFDN_INTERNAL_INCLUDE_DIRS})
target_compile_options(sfFDN.tests PRIVATE ${SFFDN_CXX_COMPILE_OPTIONS})
target_link_options(sfFDN.tests PRIVATE ${SFFDN_LINK_OPTIONS})
target_compile_features(sfFDN.tests PRIVATE cxx_std_23)
target_compile_definitions(sfFDN.tests PRIVATE ${SFFDN_COMPILE_DEFINITION})

include(Catch)
catch_discover_tests(sfFDN.tests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(sfFDN.perf)
target_sources(
    sfFDN.perf
    PRIVATE fdn_perf.cpp
            filter_perf.cpp
            filter_design_perf.cpp
            delay_perf.cpp
            array_math_perf.cpp
            mixmat_perf.cpp
            matrix_multiplication_perf.cpp
            parallel_gains_perf.cpp
            upols_perf.cpp
            nupols_perf.cpp
            oscillator_perf.cpp
            test_utils.cpp
)
target_link_libraries(
    sfFDN.perf
    PRIVATE sfFDN.sfFDN
            nanobench::nanobench
            Catch2::Catch2WithMain
            SndFile::sndfile
            Eigen3::Eigen
            "-framework Accelerate"
)

target_include_directories(sfFDN.perf PRIVATE ${SFFDN_INTERNAL_INCLUDE_DIRS})
target_compile_features(sfFDN.perf PRIVATE cxx_std_23)
target_compile_options(sfFDN.perf PRIVATE ${SFFDN_COMPILE_OPTIONS})
target_compile_definitions(sfFDN.perf PRIVATE ${SFFDN_COMPILE_DEFINITION})
