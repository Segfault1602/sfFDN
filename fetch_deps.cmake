include(FetchContent)

# Eigen is used for feedback matrix operations
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${Eigen3_DIR}")

endif()

# PFFFT is used in the partitioned convolution code
FetchContent_Declare(pffft GIT_REPOSITORY https://bitbucket.org/jpommier/pffft.git)
FetchContent_MakeAvailable(pffft)
add_library(PFFFT STATIC ${pffft_SOURCE_DIR}/pffft.c)

# PFFFT does not support small FFT sizes (< 32) so KISSFFT is used for those cases. So far, this is only used for
# building Circulant Matrix.
FetchContent_Declare(kissfft GIT_REPOSITORY https://github.com/mborgerding/kissfft.git)
set(KISSFFT_PKGCONFIG OFF CACHE BOOL "Disable kissfft pkgconfig" FORCE)
set(KISSFFT_TEST OFF CACHE BOOL "Disable kissfft tests" FORCE)
set(KISSFFT_TOOLS OFF CACHE BOOL "Disable kissfft tools" FORCE)
set(KISSFFT_STATIC ON CACHE BOOL "Build kissfft as static library" FORCE)
FetchContent_MakeAvailable(kissfft)
