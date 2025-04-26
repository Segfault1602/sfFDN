set(TEST_SOURCE
    main.cpp
    array_math_tests.cpp
    delay_tests.cpp
    filter_tests.cpp
    mixmat_tests.cpp
    fdn_tests.cpp
    parallel_gains_tests.cpp
    test_utils.cpp)

add_executable(tests ${TEST_SOURCE})
target_link_libraries(tests PRIVATE fdn doctest sndfile)

if(SF_FDN_USE_VDSP)
    target_link_libraries(tests PRIVATE "-framework Accelerate")
endif()

target_include_directories(tests PRIVATE ${doctest_SOURCE_DIR}/doctest)
target_compile_options(tests PRIVATE ${COMMON_CXX_FLAGS})


if (NOT WIN32)
    target_compile_options(tests PRIVATE ${COMMON_CXX_FLAGS} -fsanitize=address)
    target_link_options(tests PRIVATE -fsanitize=address)
endif()

set(PERF_SOURCE
    main.cpp
    fdn_perf.cpp
    filter_perf.cpp
    array_math_perf.cpp
    mixmat_perf.cpp
    parallel_gains_perf.cpp
    test_utils.cpp)
add_executable(perf ${PERF_SOURCE})
target_link_libraries(perf PRIVATE fdn nanobench sndfile)

if(SF_FDN_USE_VDSP)
    target_link_libraries(perf PRIVATE "-framework Accelerate")
endif()

target_include_directories(perf PRIVATE ${doctest_SOURCE_DIR}/doctest)
target_compile_options(perf PRIVATE ${COMMON_CXX_FLAGS})
