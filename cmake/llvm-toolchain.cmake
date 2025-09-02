include_guard(GLOBAL)

set(CMAKE_CXX_COMPILER "/Users/alex/llvm_install/bin/clang++")
set(CMAKE_C_COMPILER "/Users/alex/llvm_install/bin/clang")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(SFFDN_SANITIZER -fsanitize=address)
    set(SFFDN_COMPILE_DEFINITION -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(SFFDN_COMPILE_DEFINITION -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST)
endif()

set(SFFDN_CXX_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wno-sign-compare
    -Wunsafe-buffer-usage
    -fno-omit-frame-pointer
    ${SFFDN_SANITIZER}
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(SFFDN_CXX_COMPILE_OPTIONS ${SFFDN_CXX_COMPILE_OPTIONS})
endif()

set(SFFDN_LINK_OPTIONS ${SFFDN_SANITIZER})

include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
