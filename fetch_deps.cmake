include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE)

FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.1.0
    GIT_SHALLOW TRUE)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
    GIT_SHALLOW TRUE)

FetchContent_Declare(
    libsndfile
    GIT_REPOSITORY https://github.com/libsndfile/libsndfile.git
    GIT_TAG 1.2.2)

set(BUILD_PROGRAMS
    OFF
    CACHE BOOL "Don't build libsndfile programs!")
set(BUILD_EXAMPLES
    OFF
    CACHE BOOL "Don't build libsndfile examples!")
set(BUILD_REGTEST
    OFF
    CACHE BOOL "Don't build libsndfile regtest!")
set(BUILD_PROGRAMS
    OFF
    CACHE BOOL "Don't build libsndfile programs!" FORCE)
set(ENABLE_EXTERNAL_LIBS
    OFF
    CACHE BOOL "Disable external libs support!" FORCE)
set(BUILD_TESTING
    OFF
    CACHE BOOL "Disable libsndfile tests!" FORCE)
set(ENABLE_MPEG
    OFF
    CACHE BOOL "Disable MPEG support!" FORCE)
set(ENABLE_CPACK
    OFF
    CACHE BOOL "Disable CPACK!" FORCE)
set(ENABLE_PACKAGE_CONFIG
    OFF
    CACHE BOOL "Disable package config!" FORCE)
set(INSTALL_PKGCONFIG_MODULE
    OFF
    CACHE BOOL "Disable pkgconfig module!" FORCE)

FetchContent_MakeAvailable(eigen nanobench doctest libsndfile)
