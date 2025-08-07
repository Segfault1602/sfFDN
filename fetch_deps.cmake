include(FetchContent)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(SndFile)

FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.1.0
    GIT_SHALLOW TRUE)

FetchContent_Declare(
    pffft
    GIT_REPOSITORY https://bitbucket.org/jpommier/pffft.git)

FetchContent_Declare(
    kissfft
    GIT_REPOSITORY https://github.com/mborgerding/kissfft.git)

set(KISSFFT_PKGCONFIG OFF CACHE BOOL "Disable kissfft pkgconfig" FORCE)
set(KISSFFT_TEST OFF CACHE BOOL "Disable kissfft tests" FORCE)
set(KISSFFT_TOOLS OFF CACHE BOOL "Disable kissfft tools" FORCE)
set(KISSFFT_STATIC ON CACHE BOOL "Build kissfft as static library" FORCE)

FetchContent_MakeAvailable(nanobench pffft kissfft)

add_library(PFFFT STATIC ${pffft_SOURCE_DIR}/pffft.c)
