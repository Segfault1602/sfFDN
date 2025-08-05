include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG nightly
)

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

set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_BTL OFF)
set(EIGEN_BUILD_TESTING OFF)
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_LEAVE_TEST_IN_ALL_TARGET OFF)


if (WIN32)
    # Seems to be required to build on windows
    set(EIGEN_TEST_CXX11 ON CACHE BOOL "Enable C++11 tests")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.25)
endif()

FetchContent_Declare(pffft GIT_REPOSITORY https://bitbucket.org/jpommier/pffft.git)

FetchContent_Declare(kissfft
    GIT_REPOSITORY https://github.com/mborgerding/kissfft.git)
set(KISSFFT_PKGCONFIG
    OFF
    CACHE BOOL "Disable kissfft pkgconfig" FORCE)
set(KISSFFT_TEST OFF CACHE BOOL "Disable kissfft tests" FORCE)
set(KISSFFT_TOOLS OFF CACHE BOOL "Disable kissfft tools" FORCE)
set(KISSFFT_STATIC ON CACHE BOOL "Build kissfft as static library" FORCE)

FetchContent_MakeAvailable(eigen nanobench doctest libsndfile pffft kissfft)

add_library(PFFFT STATIC ${pffft_SOURCE_DIR}/pffft.c)
target_compile_options(sndfile PRIVATE "-Wno-deprecated")
target_compile_options(eigen INTERFACE "-Wno-deprecated")
