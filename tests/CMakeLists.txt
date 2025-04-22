set(TEST_SOURCE main.cpp delay_tests.cpp filter_tests.cpp mixmat_tests.cpp fdn_tests.cpp
                test_utils.cpp)

add_executable(tests ${TEST_SOURCE})
target_link_libraries(tests PRIVATE fdn cmsis doctest sndfile "-framework Accelerate")
target_include_directories(tests PRIVATE ${doctest_SOURCE_DIR}/doctest)
target_compile_options(tests PRIVATE -Wno-deprecated)

set(PERF_SOURCE main.cpp fdn_perf.cpp filter_perf.cpp mixmat_perf.cpp test_utils.cpp)
add_executable(perf ${PERF_SOURCE})
target_link_libraries(perf PRIVATE fdn cmsis nanobench sndfile "-framework Accelerate")
target_include_directories(perf PRIVATE ${doctest_SOURCE_DIR}/doctest)
target_compile_options(perf PRIVATE -Wno-deprecated)
