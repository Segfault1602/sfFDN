include_guard(GLOBAL)

set(CMAKE_CXX_COMPILER cl)
set(CMAKE_C_COMPILER cl)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(SFFDN_SANITIZER /fsanitize=address)
    set(SFFDN_COMPILE_DEFINITION -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(SFFDN_COMPILE_DEFINITION -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST)
endif()

set(SFFDN_CXX_COMPILE_OPTIONS
    /W3
    /WX
    /wd4018
    /wd4244
    /wd4068
    /wd4267
    /wd5030
    /wd4305
    /GS-
    /fp:fast
    /Gw
    /Gy
    /GL
    /favor:INTEL64
    ${SFFDN_SANITIZER}
)

set(SFFDN_LINK_OPTIONS ${SFFDN_SANITIZER} /LTCG)

include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
