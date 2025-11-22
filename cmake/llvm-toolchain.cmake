include_guard(GLOBAL)

if(APPLE)
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm@20/bin/clang++")
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm@20/bin/clang")
else()
    set(CMAKE_CXX_COMPILER "clang++")
    set(CMAKE_C_COMPILER "clang")
endif()

if(APPLE)
    set(SFFDN_USE_SANITIZER $<$<CONFIG:Debug>:ON>)
endif()

set(SFFDN_COMPILE_DEFINITION
    $<$<CONFIG:Debug>:-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG>
    $<$<CONFIG:RelWithDebInfo>:-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG>
    $<$<CONFIG:Debug>:-DEIGEN_RUNTIME_NO_MALLOC>
)

if(APPLE)
    set(SFFDN_USE_VDSP ON)
    set(SFFDN_COMPILE_DEFINITION ${SFFDN_COMPILE_DEFINITION} -DSFFDN_USE_VDSP)
endif()

if(SFFDN_USE_SANITIZER)
    set(SFFDN_SANITIZER -fsanitize=address)
endif()

set(SFFDN_CXX_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wno-sign-compare
    -Wunsafe-buffer-usage
    -fno-omit-frame-pointer
    -march=native
    ${SFFDN_SANITIZER}
)

set(SFFDN_LINK_OPTIONS ${SFFDN_SANITIZER})

include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
