cmake_minimum_required(VERSION 3.31)

project(sfFDN LANGUAGES CXX C VERSION 0.1.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SF_FDN_USE_VDSP "Use VDSP for DSP operations" OFF)
option(BUILD_TESTING "Build tests" ON)
option(SFFDN_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

include(./fetch_deps.cmake)

add_subdirectory(src)

if(SFFDN_BUILD_PYTHON_BINDINGS)
    add_subdirectory(py_binding)
endif()

# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/sfFDNConfigVersion.cmake" VERSION ${PROJECT_VERSION} COMPATIBILITY AnyNewerVersion
# )

# install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sfFDNConfigVersion.cmake" DESTINATION "${CMAKE_INSTALL_LIBDIR}/sfFDN")

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
