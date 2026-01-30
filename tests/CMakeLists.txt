# ======== Find Dependencies ========
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
    3.12.0
)

cpmaddpackage(
    NAME
    nanobench
    GITHUB_REPOSITORY
    martinus/nanobench
    VERSION
    4.3.11
)

get_target_property(_catch2_inc Catch2 INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(Catch2 SYSTEM INTERFACE ${_catch2_inc})

get_target_property(_nanobench_inc nanobench INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(nanobench SYSTEM INTERFACE ${_nanobench_inc})

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
            Eigen
            sfFDN::sfFDN_warnings
            sfFDN::sfFDN_options
)

set(SFFDN_INTERNAL_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../src)

target_include_directories(sfFDN.tests PRIVATE ${SFFDN_INTERNAL_INCLUDE_DIRS})
target_compile_features(sfFDN.tests PRIVATE cxx_std_23)

add_test(NAME sfFDN.tests COMMAND sfFDN.tests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

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
            Eigen
            sfFDN::sfFDN_warnings
            sfFDN::sfFDN_options
)

if(APPLE)
    target_link_libraries(sfFDN.perf PRIVATE "-framework Accelerate")
endif()

target_include_directories(sfFDN.perf PRIVATE ${SFFDN_INTERNAL_INCLUDE_DIRS})
target_compile_features(sfFDN.perf PRIVATE cxx_std_23)
