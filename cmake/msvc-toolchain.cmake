include_guard(GLOBAL)

set(CMAKE_CXX_COMPILER cl)
set(CMAKE_C_COMPILER cl)

# set(SFFDN_SANITIZER $<$<CONFIG:Debug>:/fsanitize=address>)

set(SFFDN_COMPILE_DEFINITION
    $<$<CONFIG:Debug>:-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG>
    $<$<CONFIG:RelWithDebInfo>:-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG>
    $<$<CONFIG:Debug>:-DEIGEN_RUNTIME_NO_MALLOC>
)

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
)

include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
