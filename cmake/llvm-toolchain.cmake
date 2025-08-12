include_guard(GLOBAL)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_C_COMPILER "clang")

set(SFFDN_CXX_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wno-sign-compare
    $<$<CONFIG:Debug>:-fsanitize=address>
    $<$<CONFIG:Debug>:-fsanitize=undefined>
)

set(SFFDN_LINK_OPTIONS $<$<CONFIG:Debug>:-fsanitize=address> $<$<CONFIG:Debug>:-fsanitize=undefined>)
