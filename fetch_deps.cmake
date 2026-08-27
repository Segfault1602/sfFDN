include(FetchContent)

FetchContent_Declare(CPM GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake GIT_TAG v0.42.1)
FetchContent_MakeAvailable(CPM)
include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)

# Eigen is used for feedback matrix operations
if(DEFINED CMAKE_WARN_DEPRECATED)
    set(_SFFDN_CMAKE_WARN_DEPRECATED_DEFINED TRUE)
    set(_SFFDN_CMAKE_WARN_DEPRECATED ${CMAKE_WARN_DEPRECATED})
endif()
set(CMAKE_WARN_DEPRECATED OFF)
cpmaddpackage(
    NAME
    Eigen
    GIT_TAG
    5.0.1
    GIT_REPOSITORY
    https://gitlab.com/libeigen/eigen
)
if(_SFFDN_CMAKE_WARN_DEPRECATED_DEFINED)
    set(CMAKE_WARN_DEPRECATED ${_SFFDN_CMAKE_WARN_DEPRECATED})
else()
    unset(CMAKE_WARN_DEPRECATED)
endif()
unset(_SFFDN_CMAKE_WARN_DEPRECATED)
unset(_SFFDN_CMAKE_WARN_DEPRECATED_DEFINED)

if(Eigen_ADDED)
    get_target_property(_eigen_inc eigen INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(eigen SYSTEM INTERFACE ${_eigen_inc})
endif()

cpmaddpackage(
    NAME
    pffft
    GIT_TAG
    09796885cd5b9da5692242de2df0d81e5e1f3d21
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

if(TARGET kissfft AND CMAKE_C_COMPILER_ID MATCHES ".*Clang")
    target_compile_options(kissfft PRIVATE -Wno-cast-align)
endif()

cpmaddpackage("gh:nlohmann/json@3.12.0")
