include(FetchContent)

FetchContent_Declare(CPM GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake GIT_TAG v0.42.1)
FetchContent_MakeAvailable(CPM)
include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)

# Eigen is used for feedback matrix operations
cpmaddpackage(
    NAME
    Eigen
    GIT_TAG
    5.0.1
    GIT_REPOSITORY
    https://gitlab.com/libeigen/eigen
)

cpmaddpackage(
    NAME
    pffft
    GIT_TAG
    master
    GIT_REPOSITORY
    https://bitbucket.org/jpommier/pffft/src/master/
)

if(NOT TARGET pffft)

    if(pffft_ADDED)
        add_library(pffft STATIC ${pffft_SOURCE_DIR}/pffft.c)
        add_library(pffft::pffft ALIAS pffft)
        target_compile_definitions(pffft PRIVATE -D_USE_MATH_DEFINES)
        target_include_directories(pffft PUBLIC ${pffft_SOURCE_DIR})
    endif()
endif()

# PFFFT does not support small FFT sizes (< 32) so KISSFFT is used for those cases. So far, this is only used for
# building Circulant Matrix. find_package(kissfft REQUIRED)
cpmaddpackage(
    URI
    "gh:mborgerding/kissfft#131.2.0"
    OPTIONS
    "KISSFFT_STATIC ON"
    "KISSFFT_TEST OFF"
    "KISSFFT_TOOLS OFF"
)

cpmaddpackage("gh:nlohmann/json@3.12.0")
